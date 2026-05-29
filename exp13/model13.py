"""exp13/model13.py — PCVRFusionFormer13

TokenFormer-inspired BFTS (Bottom-Full / Top-Sliding) architecture.

Changes from model12.py (exp6 + PEPNet):
  1. Unified Event Stream: 4 domains concatenated into single event_tokens (B, L_total, D)
     instead of CrossDomainInterestTransfer pooling to 4 domain tokens.
     all_tokens: [target(1), user(7), item(3), event_tokens(L_total)]
  2. BFTS blocks: bottom half = BottomFullBlock (Full Self-Attention on all_tokens),
                  top half   = TopSlidingBlock (Sliding Window Attention on event_tokens only).
  3. NLIR FFN: replaces GELU 4× FFN with multiplicative gate: x * sigmoid(Linear(x)).
     Reduces FFN params from ~132K to ~16K per block.
  4. Sinusoidal PE applied only to event_tokens (static tokens receive no positional encoding).
  5. Scoring Head: CrossAttn(target_token vs user+item) → ui_repr,
     masked_mean pooling per domain over event_refined → domain_sim; PEPNet gate retained.
  6. EventEncoder.log_lambda receives no gradient (decay_weights computed but discarded).

Alias:
  PCVRHyFormer = PCVRFusionFormer13  (for infer.py dynamic import)
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from model import ModelInput, CrossAttention


# ── Aligned pair specs (identical to model12) ────────────────────────────────
_ALIGNED_A = [
    (62, 17, 30,  5, 256,  5),
    (63, 18, 35, 11, 261, 11),
    (64, 19, 46, 18, 272, 18),
    (65, 20, 64, 49, 290, 49),
    (66, 21, 113, 66, 339, 66),
]
_ALIGNED_B = [
    (89, 25, 186, 10, 725, 10, 755, 10),
    (90, 26, 196, 10, 735, 10, 765, 10),
    (91, 27, 206, 10, 745, 10, 775, 10),
]
_ALIGNED_SKIP_INDICES = frozenset({17, 18, 19, 20, 21, 25, 26, 27})
_PRETRAINED_DIM = 256 + 320   # 576


# ── StructuredUserNSTokenizer (identical to model12) ─────────────────────────

class StructuredUserNSTokenizer(nn.Module):
    N_FIXED = 3

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        emb_dim: int,
        d_model: int,
        num_sparse_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.num_sparse_tokens = num_sparse_tokens
        self.has_recon = user_dense_dim >= 785

        self.pretrained_norm = nn.LayerNorm(_PRETRAINED_DIM)
        self.pretrained_proj = nn.Sequential(
            nn.Linear(_PRETRAINED_DIM, d_model),
            nn.LayerNorm(d_model),
        )

        self.aligned_a_embs = nn.ModuleList([
            nn.Embedding(int(feature_specs[int_idx][0]) + 1, emb_dim, padding_idx=0)
            for (_, int_idx, *_) in _ALIGNED_A
        ])
        self.aligned_a_proj = nn.Sequential(
            nn.Linear(len(_ALIGNED_A) * emb_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.aligned_b_embs = nn.ModuleList([
            nn.Embedding(int(feature_specs[int_idx][0]) + 1, emb_dim, padding_idx=0)
            for (_, int_idx, *_) in _ALIGNED_B
        ])
        if self.has_recon:
            self.recon_projs = nn.ModuleList([
                nn.Linear(10, emb_dim, bias=False)
                for _ in _ALIGNED_B
            ])
        self.aligned_b_proj = nn.Sequential(
            nn.Linear(len(_ALIGNED_B) * emb_dim, d_model),
            nn.LayerNorm(d_model),
        )

        sparse_fid_specs = [
            (idx, spec)
            for idx, spec in enumerate(feature_specs)
            if idx not in _ALIGNED_SKIP_INDICES
        ]
        self._sparse_specs = sparse_fid_specs

        embs_list, self._sparse_emb_index = [], []
        real_idx = 0
        for _, (vs, offset, length) in sparse_fid_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                self._sparse_emb_index.append(-1)
            else:
                self._sparse_emb_index.append(real_idx)
                embs_list.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
                real_idx += 1
        self.sparse_embs = nn.ModuleList(embs_list)

        n_sparse = len(sparse_fid_specs)
        total_sparse_dim = n_sparse * emb_dim
        self.chunk_dim = math.ceil(total_sparse_dim / num_sparse_tokens)
        self._sparse_pad = self.chunk_dim * num_sparse_tokens - total_sparse_dim

        self.sparse_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(self.chunk_dim, d_model), nn.LayerNorm(d_model))
            for _ in range(num_sparse_tokens)
        ])

        logging.info(
            f"StructuredUserNSTokenizer: 3 fixed + {num_sparse_tokens} sparse = "
            f"{self.num_ns_tokens} user NS tokens | has_recon={self.has_recon} | "
            f"n_sparse_fids={n_sparse}"
        )

    @property
    def num_ns_tokens(self) -> int:
        return self.N_FIXED + self.num_sparse_tokens

    def _weighted_emb(self, emb, int_vals, coeff):
        e = emb(int_vals.long())
        mask = (int_vals != 0).float()
        shifted = coeff - coeff.max(dim=-1, keepdim=True)[0]
        shifted = shifted + (1.0 - mask) * (-1e9)
        w = torch.softmax(shifted, dim=-1)
        return (e * w.unsqueeze(-1)).sum(dim=1)

    def forward(self, user_int_feats, user_dense_feats):
        B = user_int_feats.shape[0]
        tokens: List[torch.Tensor] = []

        fid61 = user_dense_feats[:, 0:256]
        fid87 = user_dense_feats[:, 405:725]
        pretrained = self.pretrained_norm(torch.cat([fid61, fid87], dim=-1))
        tokens.append(F.silu(self.pretrained_proj(pretrained)).unsqueeze(1))

        agg_a: List[torch.Tensor] = []
        for emb, (_, _, int_off, int_len, d_off, d_len) in zip(self.aligned_a_embs, _ALIGNED_A):
            iv = user_int_feats[:, int_off:int_off + int_len]
            co = user_dense_feats[:, d_off:d_off + d_len]
            agg_a.append(self._weighted_emb(emb, iv, co))
        tokens.append(F.silu(self.aligned_a_proj(torch.cat(agg_a, dim=-1))).unsqueeze(1))

        agg_b: List[torch.Tensor] = []
        for j, (emb, (_, _, int_off, int_len, d_off, d_len, r_off, r_len)) in enumerate(
            zip(self.aligned_b_embs, _ALIGNED_B)
        ):
            iv = user_int_feats[:, int_off:int_off + int_len]
            co = user_dense_feats[:, d_off:d_off + d_len]
            v = self._weighted_emb(emb, iv, co)
            if self.has_recon:
                recon = user_dense_feats[:, r_off:r_off + r_len]
                v = v + self.recon_projs[j](recon)
            agg_b.append(v)
        tokens.append(F.silu(self.aligned_b_proj(torch.cat(agg_b, dim=-1))).unsqueeze(1))

        sparse_embs: List[torch.Tensor] = []
        for (_, (vs, offset, length)), emb_ri in zip(self._sparse_specs, self._sparse_emb_index):
            if emb_ri == -1:
                sparse_embs.append(user_int_feats.new_zeros(B, self.emb_dim, dtype=torch.float))
            else:
                el = self.sparse_embs[emb_ri]
                if length == 1:
                    e = el(user_int_feats[:, offset].long())
                else:
                    vals = user_int_feats[:, offset:offset + length].long()
                    ea = el(vals)
                    mk = (vals != 0).float().unsqueeze(-1)
                    e = (ea * mk).sum(dim=1) / mk.sum(dim=1).clamp(min=1)
                sparse_embs.append(e)

        cat_s = torch.cat(sparse_embs, dim=-1)
        if self._sparse_pad > 0:
            cat_s = F.pad(cat_s, (0, self._sparse_pad))
        for chunk, proj in zip(cat_s.split(self.chunk_dim, dim=-1), self.sparse_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))

        return torch.cat(tokens, dim=1)  # (B, N_FIXED+K, d_model)


# ── ItemRankMixerTokenizer (identical to model12) ────────────────────────────

class ItemRankMixerTokenizer(nn.Module):

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        emb_dim: int,
        d_model: int,
        num_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.feature_specs = feature_specs

        embs_list, self._emb_index = [], []
        real_idx = 0
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                self._emb_index.append(-1)
            else:
                self._emb_index.append(real_idx)
                embs_list.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
                real_idx += 1
        self.embs = nn.ModuleList(embs_list)

        total_dim = len(feature_specs) * emb_dim
        self.chunk_dim = math.ceil(total_dim / num_tokens)
        self._pad = self.chunk_dim * num_tokens - total_dim

        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(self.chunk_dim, d_model), nn.LayerNorm(d_model))
            for _ in range(num_tokens)
        ])

    def forward(self, item_int_feats):
        B = item_int_feats.shape[0]
        all_embs: List[torch.Tensor] = []
        for (vs, offset, length), ri in zip(self.feature_specs, self._emb_index):
            if ri == -1:
                all_embs.append(item_int_feats.new_zeros(B, self.emb_dim, dtype=torch.float))
            else:
                el = self.embs[ri]
                if length == 1:
                    e = el(item_int_feats[:, offset].long())
                else:
                    vals = item_int_feats[:, offset:offset + length].long()
                    ea = el(vals)
                    mk = (vals != 0).float().unsqueeze(-1)
                    e = (ea * mk).sum(dim=1) / mk.sum(dim=1).clamp(min=1)
                all_embs.append(e)

        cat_e = torch.cat(all_embs, dim=-1)
        if self._pad > 0:
            cat_e = F.pad(cat_e, (0, self._pad))
        tokens = [F.silu(proj(c)).unsqueeze(1) for c, proj in zip(cat_e.split(self.chunk_dim, -1), self.projs)]
        return torch.cat(tokens, dim=1)


# ── EventEncoder (identical to model12) ──────────────────────────────────────

class EventEncoder(nn.Module):
    """Encodes sequence events → (event_embs, decay_weights).

    In exp13, decay_weights is discarded; log_lambda receives no gradient.
    """

    def __init__(
        self,
        vocab_sizes: List[int],
        emb_dim: int,
        d_model: int,
        num_time_buckets: int,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        S = len(vocab_sizes)

        embs_list, self._emb_index, self._is_id = [], [], []
        real_idx = 0
        for vs in vocab_sizes:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            self._is_id.append(int(vs) > seq_id_threshold)
            if skip:
                self._emb_index.append(-1)
            else:
                self._emb_index.append(real_idx)
                embs_list.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
                real_idx += 1
        self.embs = nn.ModuleList(embs_list)
        self.seq_id_dropout = nn.Dropout(dropout * 2)

        self.has_time_emb = num_time_buckets > 0
        if self.has_time_emb:
            self.time_emb = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        self.event_proj = nn.Sequential(
            nn.Linear(S * emb_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.log_lambda = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        seq_data: torch.Tensor,
        seq_lens: torch.Tensor,
        time_buckets: torch.Tensor,
        training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, L = seq_data.shape

        emb_list: List[torch.Tensor] = []
        for i in range(S):
            ri = self._emb_index[i]
            if ri == -1:
                emb_list.append(seq_data.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                el = self.embs[ri]
                e = el(seq_data[:, i, :])
                if self._is_id[i] and training:
                    e = self.seq_id_dropout(e)
                emb_list.append(e)

        cat_emb = torch.cat(emb_list, dim=-1)
        event_embs = F.gelu(self.event_proj(cat_emb))   # (B, L, d_model)

        if self.has_time_emb:
            event_embs = event_embs + self.time_emb(time_buckets)

        idx = torch.arange(L, device=seq_lens.device).unsqueeze(0)
        pad_mask = idx >= seq_lens.unsqueeze(1)

        lam = torch.exp(self.log_lambda)
        decay_weights = torch.exp(-lam * time_buckets.float() / 63.0)
        decay_weights = decay_weights.masked_fill(pad_mask, 0.0)

        return event_embs, decay_weights


# ── NLIR FFN ──────────────────────────────────────────────────────────────────

class NLIR(nn.Module):
    """Non-Linear Interaction Representation: x * sigmoid(Linear(x)).

    Replaces the GELU 4× FFN. Parameters: d_model² + d_model ≈ 16,512 (vs 131,712).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate(x))


# ── StaticFullBlock ───────────────────────────────────────────────────────────

class StaticFullBlock(nn.Module):
    """Pre-LN Full Self-Attention + NLIR FFN on static tokens (B, 11, D).

    11×11 attention: negligible memory cost.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.01) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.nlir = NLIR(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h
        h = self.norm2(x)
        return x + self.nlir(h)


# ── EventSlidingBlock ─────────────────────────────────────────────────────────

class EventSlidingBlock(nn.Module):
    """Pre-LN Sliding Window Self-Attention + NLIR FFN on event tokens (B, L_total, D).

    attn_mask: (B*num_heads, L_total, L_total) bool — True = blocked.
               Combines window mask + padding mask (precomputed by PCVRFusionFormer13).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.01) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.nlir = NLIR(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        h = self.norm2(x)
        return x + self.nlir(h)


# ── CrossBlock ────────────────────────────────────────────────────────────────

class CrossBlock(nn.Module):
    """Bidirectional cross-attention between static (11 tokens) and event (L_total tokens).

    Step 1 — static absorbs event: query=static, key/value=event  (11×L_total)
    Step 2 — event absorbs static: query=event,  key/value=static (L_total×11)
    Both with Pre-LN + NLIR FFN + residual.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.01) -> None:
        super().__init__()
        # Step 1: static queries event
        self.s_norm1  = nn.LayerNorm(d_model)
        self.e_norm1  = nn.LayerNorm(d_model)
        self.s2e_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.s_norm2  = nn.LayerNorm(d_model)
        self.s_nlir   = NLIR(d_model)

        # Step 2: event queries static
        self.e_norm3  = nn.LayerNorm(d_model)
        self.s_norm3  = nn.LayerNorm(d_model)
        self.e2s_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.e_norm4  = nn.LayerNorm(d_model)
        self.e_nlir   = NLIR(d_model)

    def forward(
        self,
        static_tokens: torch.Tensor,                      # (B, 11, D)
        event_tokens:  torch.Tensor,                      # (B, L_total, D)
        event_pad_mask: Optional[torch.Tensor] = None,    # (B, L_total) True=padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: static queries event (static absorbs event context)
        q_s  = self.s_norm1(static_tokens)
        kv_e = self.e_norm1(event_tokens)
        h, _ = self.s2e_attn(q_s, kv_e, kv_e,
                              key_padding_mask=event_pad_mask, need_weights=False)
        static_tokens = static_tokens + h
        static_tokens = static_tokens + self.s_nlir(self.s_norm2(static_tokens))

        # Step 2: event queries static (event absorbs static context)
        q_e  = self.e_norm3(event_tokens)
        kv_s = self.s_norm3(static_tokens)
        h, _ = self.e2s_attn(q_e, kv_s, kv_s, need_weights=False)
        event_tokens = event_tokens + h
        event_tokens = event_tokens + self.e_nlir(self.e_norm4(event_tokens))

        return static_tokens, event_tokens


# ── PCVRFusionFormer13 ────────────────────────────────────────────────────────

class PCVRFusionFormer13(nn.Module):
    """PCVRFusionFormer13: BFTS + Unified Event Stream + NLIR + PEPNet Gate.

    Token layout in all_tokens (B, NUM_STATIC+L_total, D):
      [0]       : target_token = item_tokens.mean(1, keepdim=True)
      [1:1+U]   : user_tokens  (U = 3 fixed + K sparse, default 7)
      [1+U:1+U+I]: item_tokens  (I = item_ns_tokens, default 3)
      [NUM_STATIC:]: event_tokens (L_total = sum of per-domain max_lens)

    NUM_STATIC = 1 + U + I (fixed constant per model instance).
    """

    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: dict,
        user_ns_groups: list,
        item_ns_groups: list,
        d_model: int = 128,
        emb_dim: int = 64,
        num_hyformer_blocks: int = 4,
        num_heads: int = 8,
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        action_num: int = 1,
        num_time_buckets: int = 64,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        user_ns_tokens: int = 4,
        item_ns_tokens: int = 3,
        window_size: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.num_sequences = len(self.seq_domains)

        _K = user_ns_tokens if user_ns_tokens > 0 else 4
        _M = item_ns_tokens if item_ns_tokens > 0 else 3

        # ── Tokenizers ───────────────────────────────────────────────────────
        self.user_tokenizer = StructuredUserNSTokenizer(
            feature_specs=user_int_feature_specs,
            user_dense_dim=user_dense_dim,
            emb_dim=emb_dim,
            d_model=d_model,
            num_sparse_tokens=_K,
            emb_skip_threshold=emb_skip_threshold,
        )
        self.num_user_ns = self.user_tokenizer.num_ns_tokens   # U

        self.item_tokenizer = ItemRankMixerTokenizer(
            feature_specs=item_int_feature_specs,
            emb_dim=emb_dim,
            d_model=d_model,
            num_tokens=_M,
            emb_skip_threshold=emb_skip_threshold,
        )
        self.num_item_ns = _M                                  # I
        self.num_ns = self.num_user_ns + self.num_item_ns

        # NUM_STATIC: 1 target + U user + I item (constant per model instance)
        self.NUM_STATIC = 1 + self.num_user_ns + self.num_item_ns

        # ── Sequence encoders ─────────────────────────────────────────────────
        self.seq_encoders = nn.ModuleDict({
            domain: EventEncoder(
                vocab_sizes=seq_vocab_sizes[domain],
                emb_dim=emb_dim,
                d_model=d_model,
                num_time_buckets=num_time_buckets,
                emb_skip_threshold=emb_skip_threshold,
                seq_id_threshold=seq_id_threshold,
                dropout=dropout_rate,
            )
            for domain in self.seq_domains
        })

        # ── BFTS blocks ───────────────────────────────────────────────────────
        # num_pairs = num_hyformer_blocks // 2
        # Each pair: StaticFullBlock + EventSlidingBlock (parallel) then CrossBlock
        num_pairs = num_hyformer_blocks // 2
        self.num_pairs = num_pairs

        self.static_blocks = nn.ModuleList([
            StaticFullBlock(d_model, num_heads, dropout_rate)
            for _ in range(num_pairs)
        ])
        self.event_blocks = nn.ModuleList([
            EventSlidingBlock(d_model, num_heads, dropout_rate)
            for _ in range(num_pairs)
        ])
        self.cross_blocks = nn.ModuleList([
            CrossBlock(d_model, num_heads, dropout_rate)
            for _ in range(num_pairs)
        ])

        # ── Sinusoidal PE for event tokens (no learned params) ────────────────
        _MAX_LEN = 2048  # covers seq_a:256+seq_b:256+seq_c:512+seq_d:512 = 1536
        _pos = torch.arange(_MAX_LEN).float().unsqueeze(1)
        _div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        _pe = torch.zeros(_MAX_LEN, d_model)
        _pe[:, 0::2] = torch.sin(_pos * _div)
        _pe[:, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('sinusoidal_pe', _pe)   # (2048, d_model), no grad

        # ── Sliding window mask (register_buffer, no grad) ───────────────────
        _i = torch.arange(_MAX_LEN).unsqueeze(1)   # (2048, 1)
        _j = torch.arange(_MAX_LEN).unsqueeze(0)   # (1, 2048)
        if window_size > 0:
            _half = window_size // 2
            _win = (_i - _j).abs() > _half          # True = blocked
        else:
            # window_size=0 → full attention (no window restriction)
            _win = torch.zeros(_MAX_LEN, _MAX_LEN, dtype=torch.bool)
        self.register_buffer('_window_block_mask', _win)  # (2048, 2048)

        # ── Scoring head ──────────────────────────────────────────────────────
        # CrossAttn: target_token queries user+item static tokens
        self.user_item_xattn = CrossAttention(d_model, num_heads, dropout_rate, ln_mode='pre')

        # Signal 1: ui_mlp
        self.ui_mlp = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, 1),
        )

        # Signal 3: rich_proj
        self.rich_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, 1),
        )

        # Gate: LayerNorm + PEPNet user context bias
        gate_in_dim = 1 + self.num_sequences + 1   # 6 when num_sequences=4
        self.gate_norm   = nn.LayerNorm(gate_in_dim)
        self.gate_linear = nn.Linear(gate_in_dim, 3)
        self.gate_bias_proj = nn.Linear(d_model, 3)  # PEPNet: user context → gate bias

        self.emb_dropout = nn.Dropout(dropout_rate)

        self._tb_logs: Optional[dict] = {}

        self._init_params()

        total = sum(p.numel() for p in self.parameters())
        logging.info(
            f"PCVRFusionFormer13: U={self.num_user_ns} I={self.num_item_ns} "
            f"NUM_STATIC={self.NUM_STATIC} "
            f"num_pairs={num_pairs} window_size={window_size} "
            f"d_model={d_model} total_params={total:,}"
        )

    def _init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)
                m.weight.data[0, :] = 0.0

    def _build_event_pad_mask(
        self,
        seq_lens: Dict[str, torch.Tensor],
        domain_offsets: Dict[str, Tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """Build (B, L_total) padding mask. True = padding (ignored)."""
        parts: List[torch.Tensor] = []
        for domain in self.seq_domains:
            _, L_d = domain_offsets[domain]
            lens = seq_lens[domain]                                  # (B,)
            pos  = torch.arange(L_d, device=device).unsqueeze(0)    # (1, L_d)
            pad  = pos >= lens.unsqueeze(1)                          # (B, L_d)
            parts.append(pad)
        return torch.cat(parts, dim=1)                               # (B, L_total)

    def _tokenize(
        self,
        inputs: ModelInput,
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Tuple[int, int]]]:
        """Returns (all_tokens, event_pad_mask, domain_offsets).

        all_tokens shape: (B, NUM_STATIC + L_total, d_model)
        event_pad_mask  : (B, L_total)  True = padding
        domain_offsets  : {domain: (offset_in_event_tokens, L_d)}
        """
        user_tok   = self.user_tokenizer(inputs.user_int_feats, inputs.user_dense_feats)
        item_tok   = self.item_tokenizer(inputs.item_int_feats)
        target_tok = item_tok.mean(dim=1, keepdim=True)   # (B, 1, d_model)

        # EventEncoder per domain — decay_weights discarded (log_lambda gets no gradient)
        event_embs_list: List[torch.Tensor] = []
        domain_offsets: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for domain in self.seq_domains:
            embs, _ = self.seq_encoders[domain](
                inputs.seq_data[domain],
                inputs.seq_lens[domain],
                inputs.seq_time_buckets[domain],
                training=training,
            )
            L_d = embs.shape[1]
            domain_offsets[domain] = (offset, L_d)
            event_embs_list.append(embs)
            offset += L_d

        event_tokens = torch.cat(event_embs_list, dim=1)   # (B, L_total, d_model)

        # Additive sinusoidal PE on event tokens only (static tokens: no PE)
        L_total = event_tokens.shape[1]
        event_tokens = event_tokens + self.sinusoidal_pe[:L_total]

        # Build event padding mask
        event_pad_mask = self._build_event_pad_mask(
            inputs.seq_lens, domain_offsets, event_tokens.device
        )

        # TB log: lambda values (informational; no gradient flows to log_lambda)
        if self._tb_logs is not None:
            with torch.no_grad():
                for domain in self.seq_domains:
                    lam = torch.exp(self.seq_encoders[domain].log_lambda).item()
                    self._tb_logs[f'lambda/{domain}'] = lam

        if training:
            user_tok     = self.emb_dropout(user_tok)
            item_tok     = self.emb_dropout(item_tok)
            event_tokens = self.emb_dropout(event_tokens)

        all_tokens = torch.cat([target_tok, user_tok, item_tok, event_tokens], dim=1)
        return all_tokens, event_pad_mask, domain_offsets

    def _build_event_attn_mask(
        self,
        event_tokens: torch.Tensor,
        event_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build combined (B*H, L_total, L_total) bool mask: window + padding. True = blocked."""
        B, L_total, _ = event_tokens.shape
        win      = self._window_block_mask[:L_total, :L_total].unsqueeze(0)        # (1, L, L)
        pad      = event_pad_mask.unsqueeze(1).expand(B, L_total, L_total)         # (B, L, L)
        combined = win | pad                                                        # (B, L, L)
        return combined.repeat_interleave(self.num_heads, dim=0)                   # (B*H, L, L)

    def _run_blocks(
        self,
        static_tokens: torch.Tensor,
        event_tokens:  torch.Tensor,
        event_pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Interleaved BFTS: (StaticFull + EventSliding) then CrossBlock, num_pairs times."""
        event_attn_mask = self._build_event_attn_mask(event_tokens, event_pad_mask)
        for i in range(self.num_pairs):
            static_tokens = self.static_blocks[i](static_tokens)
            event_tokens  = self.event_blocks[i](event_tokens, attn_mask=event_attn_mask)
            static_tokens, event_tokens = self.cross_blocks[i](
                static_tokens, event_tokens, event_pad_mask
            )
        return static_tokens, event_tokens

    def _score(
        self,
        static_tokens: torch.Tensor,
        event_tokens: torch.Tensor,
        domain_offsets: Dict[str, Tuple[int, int]],
        seq_lens: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        U = self.num_user_ns

        # Static token slices from CrossBlock output
        target_final = static_tokens[:, 0:1, :]                  # (B, 1, D)
        user_final   = static_tokens[:, 1:1 + U, :]              # (B, U, D)
        item_final   = static_tokens[:, 1 + U:self.NUM_STATIC, :] # (B, I, D)

        # CrossAttn: target queries user+item tokens
        kv     = torch.cat([user_final, item_final], dim=1)   # (B, U+I, D)
        ui_out = self.user_item_xattn(query=target_final, key_value=kv)  # (B, 1, D)
        ui_repr   = ui_out.squeeze(1)               # (B, D)
        user_repr = user_final.mean(1)              # (B, D)
        item_repr = item_final.mean(1)              # (B, D)

        scale = math.sqrt(self.d_model)

        # Signal 1: MLP over interaction features
        ui_input = torch.cat(
            [ui_repr, item_repr, ui_repr - item_repr, ui_repr * item_repr], dim=-1
        )
        ui_score = self.ui_mlp(ui_input)            # (B, 1)

        # Signal 2: domain_sim via masked_mean pooling of event_tokens per domain
        sim_list: List[torch.Tensor] = []
        for domain in self.seq_domains:
            off, L_d = domain_offsets[domain]
            event_d = event_tokens[:, off:off + L_d, :]    # (B, L_d, D)
            lens_d  = seq_lens[domain]                      # (B,)
            pos     = torch.arange(L_d, device=event_d.device).unsqueeze(0)
            valid   = (pos < lens_d.unsqueeze(1)).float().unsqueeze(-1)  # (B, L_d, 1)
            counts  = valid.sum(1).clamp(min=1.0)           # (B, 1)
            pool_d  = (event_d * valid).sum(1) / counts     # (B, D)
            # dot-product similarity
            sim_d = torch.bmm(
                pool_d.unsqueeze(1), ui_repr.unsqueeze(-1)
            ).squeeze(-1).squeeze(-1) / scale               # (B,)
            sim_list.append(sim_d.unsqueeze(1))
        domain_sim = torch.cat(sim_list, dim=1)             # (B, num_sequences)

        # Signal 3: rich context scalar
        rich_scalar = self.rich_proj(
            torch.cat([user_repr, item_repr, ui_repr], dim=-1)
        )                                                   # (B, 1)

        # Gate: LayerNorm + PEPNet user context bias
        gate_in   = torch.cat([ui_score, domain_sim, rich_scalar], dim=-1)  # (B, 1+S+1)
        gate_in   = self.gate_norm(gate_in)
        gate_bias = self.gate_bias_proj(user_repr)          # (B, 3) PEPNet
        gates     = torch.sigmoid(self.gate_linear(gate_in) + gate_bias)    # (B, 3)

        signals = torch.cat(
            [ui_score, domain_sim.mean(dim=-1, keepdim=True), rich_scalar],
            dim=-1,
        )                                                   # (B, 3)
        logits = (gates * signals).sum(dim=-1, keepdim=True)  # (B, 1)

        embedding = (user_repr + item_repr) / 2

        if self._tb_logs is not None:
            with torch.no_grad():
                g = gates.mean(0)
                for i in range(3):
                    self._tb_logs[f'gate/signal_{i}'] = g[i].item()
                self._tb_logs['signal/ui_score']    = ui_score.mean().item()
                self._tb_logs['signal/domain_sim']  = domain_sim.mean().item()
                self._tb_logs['signal/rich_scalar'] = rich_scalar.mean().item()

        return logits, embedding

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Returns logits (B, 1)."""
        self._tb_logs = {}
        all_tokens, event_pad_mask, domain_offsets = self._tokenize(inputs, training=True)
        static_tokens = all_tokens[:, :self.NUM_STATIC, :]
        event_tokens  = all_tokens[:, self.NUM_STATIC:, :]
        static_out, event_out = self._run_blocks(static_tokens, event_tokens, event_pad_mask)
        logits, _ = self._score(static_out, event_out, domain_offsets, inputs.seq_lens)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, embedding) — no dropout, no TB logging."""
        self._tb_logs = None
        all_tokens, event_pad_mask, domain_offsets = self._tokenize(inputs, training=False)
        static_tokens = all_tokens[:, :self.NUM_STATIC, :]
        event_tokens  = all_tokens[:, self.NUM_STATIC:, :]
        static_out, event_out = self._run_blocks(static_tokens, event_tokens, event_pad_mask)
        return self._score(static_out, event_out, domain_offsets, inputs.seq_lens)

    def get_sparse_params(self) -> List[nn.Parameter]:
        ptrs = {m.weight.data_ptr() for m in self.modules() if isinstance(m, nn.Embedding)}
        return [p for p in self.parameters() if p.data_ptr() in ptrs]

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def reinit_high_cardinality_params(self, cardinality_threshold: int = 10000) -> "set[int]":
        reinit_ptrs: "set[int]" = set()
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Embedding) and m.num_embeddings > cardinality_threshold + 1:
                nn.init.xavier_normal_(m.weight.data)
                m.weight.data[0, :] = 0.0
                reinit_ptrs.add(m.weight.data_ptr())
                count += 1
        logging.info(f"PCVRFusionFormer13: re-initialized {count} high-cardinality embeddings")
        return reinit_ptrs


# infer.py dynamic import requires PCVRHyFormer alias
PCVRHyFormer = PCVRFusionFormer13
