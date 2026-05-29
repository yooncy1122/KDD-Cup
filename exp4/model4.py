"""exp4/model4.py — PCVRFusionFormer4

Architecture:
  StructuredUserNSTokenizer : pretrained token + aligned-A token + aligned-B token + K sparse tokens
  ItemRankMixerTokenizer    : M sparse tokens from item_int
  TimeDecayEncoder × 4     : event-level embed + (content + time-decay) weighted pooling per domain
  UnifiedInteractionBlock × L : full self-attention over all tokens (NS+S) — bidirectional, stackable
  ScoringHead              : user-item cross-attn + domain similarity
                             → DCN-V2 Cross Network (parallel) + DNN → final linear → logit

Alias:
  PCVRHyFormer = PCVRFusionFormer4  (for infer.py dynamic import)
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from model import ModelInput, CrossAttention


# ── Aligned pair specs (derived from exp_1/schema_aligned.json) ──────────────
# Format: (fid, user_int_feat_idx, int_offset, int_len, dense_offset, dense_len)
_ALIGNED_A = [
    (62, 17, 30,  5, 256,  5),
    (63, 18, 35, 11, 261, 11),
    (64, 19, 46, 18, 272, 18),
    (65, 20, 64, 49, 290, 49),
    (66, 21, 113, 66, 339, 66),
]
# Format: (fid, user_int_feat_idx, int_offset, int_len, dense_offset, dense_len, recon_offset, recon_len)
_ALIGNED_B = [
    (89, 25, 186, 10, 725, 10, 755, 10),
    (90, 26, 196, 10, 735, 10, 765, 10),
    (91, 27, 206, 10, 745, 10, 775, 10),
]
_ALIGNED_SKIP_INDICES = frozenset({17, 18, 19, 20, 21, 25, 26, 27})

# fid61: offset 0, dim 256 | fid87: offset 405, dim 320
_PRETRAINED_DIM = 256 + 320   # 576


# ── StructuredUserNSTokenizer ─────────────────────────────────────────────────

class StructuredUserNSTokenizer(nn.Module):
    """Produces N_FIXED + K NS tokens from user_int/dense features.

    Token 1 — PretrainedEmbeddingToken : fid61(256d) + fid87(320d) → project
    Token 2 — AlignedPairTokenA        : fid62-66 weighted embeddings
    Token 3 — AlignedPairTokenB        : fid89-91 weighted + recon(111-113) fallback
    Tokens 4..3+K — SparseCatTokens    : remaining int fids via RankMixer split
    """

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
        # fid111-113 exist only when schema_aligned is used (total_dim=785)
        self.has_recon = user_dense_dim >= 785

        # Token 1 — pretrained projection
        self.pretrained_norm = nn.LayerNorm(_PRETRAINED_DIM)
        self.pretrained_proj = nn.Sequential(
            nn.Linear(_PRETRAINED_DIM, d_model),
            nn.LayerNorm(d_model),
        )

        # Token 2 — aligned pair A (fid62-66)
        self.aligned_a_embs = nn.ModuleList([
            nn.Embedding(int(feature_specs[int_idx][0]) + 1, emb_dim, padding_idx=0)
            for (_, int_idx, *_) in _ALIGNED_A
        ])
        self.aligned_a_proj = nn.Sequential(
            nn.Linear(len(_ALIGNED_A) * emb_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Token 3 — aligned pair B (fid89-91) + optional recon
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

        # Tokens 4..3+K — sparse remaining fids
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
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
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

    def _weighted_emb(
        self, emb: nn.Embedding, int_vals: torch.Tensor, coeff: torch.Tensor
    ) -> torch.Tensor:
        e = emb(int_vals.long())                                  # (B, L, emb_dim)
        mask = (int_vals != 0).float()                            # (B, L)
        shifted = coeff - coeff.max(dim=-1, keepdim=True)[0]
        shifted = shifted + (1.0 - mask) * (-1e9)
        w = torch.softmax(shifted, dim=-1)                        # (B, L)
        return (e * w.unsqueeze(-1)).sum(dim=1)                   # (B, emb_dim)

    def forward(
        self, user_int_feats: torch.Tensor, user_dense_feats: torch.Tensor
    ) -> torch.Tensor:
        B = user_int_feats.shape[0]
        tokens: List[torch.Tensor] = []

        # Token 1: pretrained
        fid61 = user_dense_feats[:, 0:256]
        fid87 = user_dense_feats[:, 405:725]
        pretrained = self.pretrained_norm(torch.cat([fid61, fid87], dim=-1))
        tokens.append(F.silu(self.pretrained_proj(pretrained)).unsqueeze(1))

        # Token 2: aligned A (fid62-66)
        agg_a: List[torch.Tensor] = []
        for emb, (_, _, int_off, int_len, d_off, d_len) in zip(self.aligned_a_embs, _ALIGNED_A):
            iv = user_int_feats[:, int_off:int_off + int_len]
            co = user_dense_feats[:, d_off:d_off + d_len]
            agg_a.append(self._weighted_emb(emb, iv, co))
        tokens.append(F.silu(self.aligned_a_proj(torch.cat(agg_a, dim=-1))).unsqueeze(1))

        # Token 3: aligned B (fid89-91) + optional recon (fid111-113)
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

        # Tokens 4..3+K: sparse
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


# ── ItemRankMixerTokenizer ────────────────────────────────────────────────────

class ItemRankMixerTokenizer(nn.Module):
    """RankMixer-style tokenizer for item_int features → M tokens."""

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
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_tokens)
        ])

    def forward(self, item_int_feats: torch.Tensor) -> torch.Tensor:
        """(B, item_int_dim) → (B, num_tokens, d_model)"""
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
        return torch.cat(tokens, dim=1)  # (B, num_tokens, d_model)


# ── TimeDecayEncoder ──────────────────────────────────────────────────────────

class TimeDecayEncoder(nn.Module):
    """Encodes a sequence domain → single domain repr via time-decay weighted pooling.

    Pooling weight: w[i] = softmax(content_score[i] + (-λ * time_bucket[i]))
    λ is a learnable per-domain parameter (log-parameterized for positivity).
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
        self.content_score = nn.Linear(d_model, 1, bias=True)
        self.log_lambda = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        seq_data: torch.Tensor,
        seq_lens: torch.Tensor,
        time_buckets: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        B, S, L = seq_data.shape

        emb_list: List[torch.Tensor] = []
        for i in range(S):
            ri = self._emb_index[i]
            if ri == -1:
                emb_list.append(seq_data.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                el = self.embs[ri]
                e = el(seq_data[:, i, :])                    # (B, L, emb_dim)
                if self._is_id[i] and training:
                    e = self.seq_id_dropout(e)
                emb_list.append(e)

        cat_emb = torch.cat(emb_list, dim=-1)                # (B, L, S*emb_dim)
        event_embs = F.gelu(self.event_proj(cat_emb))        # (B, L, d_model)

        if self.has_time_emb:
            event_embs = event_embs + self.time_emb(time_buckets)

        idx = torch.arange(L, device=seq_lens.device).unsqueeze(0)
        pad_mask = idx >= seq_lens.unsqueeze(1)              # (B, L), True=pad

        content = self.content_score(event_embs).squeeze(-1) # (B, L)
        lam = torch.exp(self.log_lambda)
        decay = -lam * time_buckets.float()                  # (B, L)
        score = content + decay
        score = score.masked_fill(pad_mask, -1e9)
        w = torch.softmax(score, dim=-1)                     # (B, L)

        return (event_embs * w.unsqueeze(-1)).sum(dim=1)     # (B, d_model)


# ── UnifiedInteractionBlock ───────────────────────────────────────────────────

class UnifiedInteractionBlock(nn.Module):
    """Pre-LN Transformer encoder block over all tokens (NS+S)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        hidden = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h)
        return x + h


# ── DCNv2CrossNet ─────────────────────────────────────────────────────────────

class DCNv2CrossNet(nn.Module):
    """DCN-V2 Cross Network (matrix form, full-rank).

    Cross layer: x_{l+1} = x0 ⊙ (W_l @ x_l + b_l) + x_l
    - x0: original input, fixed across all layers
    - W_l implemented as nn.Linear(d_in, d_in)
    """

    def __init__(self, d_in: int, num_layers: int = 3, dropout: float = 0.01) -> None:
        super().__init__()
        self.cross_layers = nn.ModuleList([
            nn.Linear(d_in, d_in)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d_in) → (B, d_in)"""
        x0 = x
        for linear in self.cross_layers:
            x = x0 * linear(x) + x
        return self.dropout(x)


# ── PCVRFusionFormer4 ─────────────────────────────────────────────────────────

class PCVRFusionFormer4(nn.Module):
    """PCVRFusionFormer4: structured NS tokenization + time-decay sequence encoding
    + unified full-attention interaction + DCN-V2 parallel scoring head.
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
        # actively used hyperparameters
        d_model: int = 128,
        emb_dim: int = 64,
        num_hyformer_blocks: int = 3,
        num_heads: int = 8,
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        action_num: int = 1,
        num_time_buckets: int = 64,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        user_ns_tokens: int = 4,
        item_ns_tokens: int = 3,
        num_cross_layers: int = 3,
        # absorb all other train.py flags
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model
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
        self.num_user_ns = self.user_tokenizer.num_ns_tokens

        self.item_tokenizer = ItemRankMixerTokenizer(
            feature_specs=item_int_feature_specs,
            emb_dim=emb_dim,
            d_model=d_model,
            num_tokens=_M,
            emb_skip_threshold=emb_skip_threshold,
        )
        self.num_item_ns = _M

        # num_ns for train.py compatibility log (line 344)
        self.num_ns = self.num_user_ns + self.num_item_ns

        # ── Sequence encoders ─────────────────────────────────────────────────
        self.seq_encoders = nn.ModuleDict({
            domain: TimeDecayEncoder(
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

        # ── Unified interaction blocks ────────────────────────────────────────
        self.blocks = nn.ModuleList([
            UnifiedInteractionBlock(d_model, num_heads, hidden_mult, dropout_rate)
            for _ in range(num_hyformer_blocks)
        ])

        # ── Scoring head (DCN-V2 parallel) ───────────────────────────────────
        self.user_item_xattn = CrossAttention(d_model, num_heads, dropout_rate, ln_mode='pre')

        clf_in = 3 * d_model + self.num_sequences  # 388 when d_model=128, S=4
        self.cross_net = DCNv2CrossNet(clf_in, num_layers=num_cross_layers, dropout=dropout_rate)
        self.deep_net = nn.Sequential(
            nn.Linear(clf_in, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.final_linear = nn.Linear(clf_in + d_model, action_num)

        self.emb_dropout = nn.Dropout(dropout_rate)

        self._init_params()

        total = sum(p.numel() for p in self.parameters())
        logging.info(
            f"PCVRFusionFormer4: U={self.num_user_ns} I={self.num_item_ns} "
            f"S={self.num_sequences} T={self.num_user_ns+self.num_item_ns+self.num_sequences} "
            f"d_model={d_model} blocks={num_hyformer_blocks} "
            f"num_cross_layers={num_cross_layers} clf_in={clf_in} "
            f"total_params={total:,}"
        )

    def _init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)
                m.weight.data[0, :] = 0.0

    def _tokenize(self, inputs: ModelInput, training: bool) -> torch.Tensor:
        user_tok = self.user_tokenizer(inputs.user_int_feats, inputs.user_dense_feats)
        item_tok = self.item_tokenizer(inputs.item_int_feats)

        domain_toks = []
        for domain in self.seq_domains:
            dr = self.seq_encoders[domain](
                inputs.seq_data[domain],
                inputs.seq_lens[domain],
                inputs.seq_time_buckets[domain],
                training=training,
            )
            domain_toks.append(dr.unsqueeze(1))
        domain_tok = torch.cat(domain_toks, dim=1)  # (B, 4, d_model)

        if training:
            user_tok = self.emb_dropout(user_tok)
            item_tok = self.emb_dropout(item_tok)
            domain_tok = self.emb_dropout(domain_tok)

        return torch.cat([user_tok, item_tok, domain_tok], dim=1)  # (B, T, d_model)

    def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def _score(self, all_tok: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        U, I = self.num_user_ns, self.num_item_ns

        user_final = all_tok[:, :U, :]          # (B, U, D)
        item_final = all_tok[:, U:U + I, :]     # (B, I, D)
        domain_final = all_tok[:, U + I:, :]    # (B, S, D)

        # User-item cross-attention: item attends to user
        ui = self.user_item_xattn(query=item_final, key_value=user_final)  # (B, I, D)
        ui_repr = ui.mean(dim=1)                                             # (B, D)
        user_repr = user_final.mean(dim=1)                                   # (B, D)
        item_repr = item_final.mean(dim=1)                                   # (B, D)

        # Domain similarity (dot product)
        scale = math.sqrt(self.d_model)
        sim = torch.bmm(
            ui_repr.unsqueeze(1), domain_final.transpose(1, 2)
        ).squeeze(1) / scale                                                  # (B, S)

        features = torch.cat([ui_repr, user_repr, item_repr, sim], dim=-1)   # (B, 3D+S)

        # DCN-V2 parallel: cross network + deep network
        cross_out = self.cross_net(features)                                  # (B, 3D+S)
        deep_out = self.deep_net(features)                                    # (B, d_model)
        logits = self.final_linear(
            torch.cat([cross_out, deep_out], dim=-1)
        )                                                                     # (B, action_num)

        embedding = (user_repr + item_repr) / 2                              # (B, D)
        return logits, embedding

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Returns logits (B, action_num)."""
        all_tok = self._tokenize(inputs, training=True)
        all_tok = self._run_blocks(all_tok)
        logits, _ = self._score(all_tok)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits (B, action_num), embedding (B, d_model)) — no dropout."""
        all_tok = self._tokenize(inputs, training=False)
        all_tok = self._run_blocks(all_tok)
        return self._score(all_tok)

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all Embedding weight parameters (for Adagrad)."""
        ptrs = {m.weight.data_ptr() for m in self.modules() if isinstance(m, nn.Embedding)}
        return [p for p in self.parameters() if p.data_ptr() in ptrs]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-Embedding parameters (for AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def reinit_high_cardinality_params(self, cardinality_threshold: int = 10000) -> "set[int]":
        """Re-initializes Embedding tables with vocab > threshold."""
        reinit_ptrs: "set[int]" = set()
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Embedding) and m.num_embeddings > cardinality_threshold + 1:
                nn.init.xavier_normal_(m.weight.data)
                m.weight.data[0, :] = 0.0
                reinit_ptrs.add(m.weight.data_ptr())
                count += 1
        logging.info(f"PCVRFusionFormer4: re-initialized {count} high-cardinality embeddings")
        return reinit_ptrs


# infer.py dynamic import requires PCVRHyFormer alias
PCVRHyFormer = PCVRFusionFormer4
