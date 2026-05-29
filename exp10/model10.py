"""exp10/model10.py — PCVRFusionFormer10

Base: exp6/model6.py (PCVRFusionFormer6)
Change from exp6:
  seq_d EventEncoder → SeqDStatisticalEncoder (from exp9/model9.py)

CrossDomainInterestTransfer: adds optional precomputed_repr_dict to
  bypass target-aware pooling for seq_d.

TB logging: entropy/target_attn_seq_d and lambda/seq_d are absent.

Alias:
  PCVRHyFormer = PCVRFusionFormer10  (for infer.py dynamic import)
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from model import ModelInput, CrossAttention


# ── Aligned pair specs ────────────────────────────────────────────────────────
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


# ── StructuredUserNSTokenizer ─────────────────────────────────────────────────

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


# ── ItemRankMixerTokenizer ────────────────────────────────────────────────────

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


# ── EventEncoder ──────────────────────────────────────────────────────────────

class EventEncoder(nn.Module):
    """Encodes sequence events → (event_embs, decay_weights), pooling deferred.

    Returns:
        event_embs   : (B, L, d_model)  — GELU projected + time_emb
        decay_weights: (B, L)           — exp(-λ * time_bucket / 63.0), NOT softmax-ed
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
        # log_lambda: learnable decay rate per domain, init=0 → λ=1
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
        pad_mask = idx >= seq_lens.unsqueeze(1)          # (B, L), True=pad

        lam = torch.exp(self.log_lambda)
        decay_weights = torch.exp(-lam * time_buckets.float() / 63.0)  # (B, L)
        decay_weights = decay_weights.masked_fill(pad_mask, 0.0)

        return event_embs, decay_weights


# ── SeqDStatisticalEncoder ────────────────────────────────────────────────────

class SeqDStatisticalEncoder(nn.Module):
    """Statistical feature extractor for seq_d (domain_d_seq).

    Replaces EventEncoder for seq_d. Computes 47-dim hand-crafted statistics
    from the full seq_d history (all L events, not truncated to seq_max_len
    in terms of counting — actual seq_data is still (B, 9, L)).

    S-index mapping (ts_fid=26 excluded from seq_data):
        S[0]=fid17(vocab=5)  S[1]=fid18(vocab=975)  S[2]=fid19(vocab=34190)
        S[3]=fid20           S[4]=fid21              S[5]=fid22(vocab=0,raw)
        S[6]=fid23(vocab=0)  S[7]=fid24(vocab=603)   S[8]=fid25(vocab=15)

    seq_time_buckets: inter-event gap bucket IDs (RELATIVE_BUCKET_BOUNDARIES).
        bucket 0 = padding.  ≤6h → bucket_id ≤ 32, ≤24h → ≤38, ≤72h → ≤42.
    """

    # S-dimension indices (schema_aligned.json seq_d.features, ts_fid=26 removed)
    S_FID17 = 0   # event type: values {1,2,3,4}
    S_FID18 = 1   # mid-card numeric: 2~974
    S_FID19 = 2   # high-card ID
    S_FID22 = 5   # sparse high-card (vocab=0, raw ints)
    S_FID23 = 6   # highest-card (vocab=0, raw ints)
    S_FID24 = 7   # 23-bucket category
    S_FID25 = 8   # action type: values {2,6,7,8,9,10,11,12,13,14}

    FID17_VALS: List[int] = [1, 2, 3, 4]
    FID24_VALS: List[int] = [
        30, 67, 102, 130, 179, 231, 276, 299, 306, 361,
        398, 401, 402, 403, 404, 465, 497, 511, 529, 548, 554, 567, 602,
    ]
    FID25_VALS: List[int] = [2, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # RELATIVE_BUCKET_BOUNDARIES thresholds: ≤6h/24h/72h
    RECENCY_BUCKETS: List[int] = [32, 38, 42]

    STAT_DIM: int = 47  # 1+1+4+10+23+1+1+1+1+1+3

    def __init__(self, d_model: int, dropout: float = 0.01) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(self.STAT_DIM, d_model),
            nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        seq_data: torch.Tensor,         # (B, 9, L)
        seq_lens: torch.Tensor,         # (B,)
        seq_time_buckets: torch.Tensor, # (B, L)
        training: bool = False,
    ) -> torch.Tensor:                  # (B, d_model)
        B, S, L = seq_data.shape

        valid_mask = (seq_data[:, 0, :] != 0)           # (B, L)
        raw_len = valid_mask.float().sum(-1)             # (B,)
        denom = raw_len.clamp(min=1)

        parts: List[torch.Tensor] = []

        # [1] seq_d_len (normalized)
        parts.append((raw_len / 1000.0).unsqueeze(-1))

        # [1] log1p_len (normalized)
        parts.append((torch.log1p(raw_len) / 10.0).unsqueeze(-1))

        # [4] fid17 class distribution
        vals17 = seq_data[:, self.S_FID17, :].long()
        dist17 = torch.stack(
            [(vals17 == v).float().sum(-1) / denom for v in self.FID17_VALS],
            dim=-1,
        )
        parts.append(dist17)

        # [10] fid25 action distribution
        vals25 = seq_data[:, self.S_FID25, :].long()
        dist25 = torch.stack(
            [(vals25 == v).float().sum(-1) / denom for v in self.FID25_VALS],
            dim=-1,
        )
        parts.append(dist25)

        # [23] fid24 category distribution
        vals24 = seq_data[:, self.S_FID24, :].long()
        dist24 = torch.stack(
            [(vals24 == v).float().sum(-1) / denom for v in self.FID24_VALS],
            dim=-1,
        )
        parts.append(dist24)

        # [1] fid18 mean (normalized)
        nz_mask18 = (seq_data[:, self.S_FID18, :] != 0) & valid_mask
        vals18 = seq_data[:, self.S_FID18, :].float() * nz_mask18.float()
        cnt18 = nz_mask18.float().sum(-1).clamp(min=1)
        mean18 = vals18.sum(-1) / cnt18 / 975.0
        parts.append(mean18.unsqueeze(-1))

        # [1] fid18 std (normalized)
        var18 = (
            (vals18 - (mean18 * 975.0).unsqueeze(-1)) ** 2 * nz_mask18.float()
        ).sum(-1) / cnt18
        std18 = var18.sqrt() / 975.0
        parts.append(std18.unsqueeze(-1))

        # [1] fid19 nonzero count (normalized)
        fid19_cnt = (seq_data[:, self.S_FID19, :] != 0).float() * valid_mask.float()
        parts.append((fid19_cnt.sum(-1) / 1500.0).unsqueeze(-1))

        # [1] fid22 nonzero count ratio
        cnt22 = (seq_data[:, self.S_FID22, :] != 0).float().sum(-1) / denom
        parts.append(cnt22.unsqueeze(-1))

        # [1] fid23 nonzero count (normalized)
        fid23_cnt = (seq_data[:, self.S_FID23, :] != 0).float() * valid_mask.float()
        parts.append((fid23_cnt.sum(-1) / 1000.0).unsqueeze(-1))

        # [3] recency features: fraction of gaps ≤ 6h/24h/72h
        valid_tb = (seq_time_buckets > 0)
        recency_denom = valid_tb.float().sum(-1).clamp(min=1)
        for thresh in self.RECENCY_BUCKETS:
            rec = ((seq_time_buckets > 0) & (seq_time_buckets <= thresh)).float().sum(-1)
            parts.append((rec / recency_denom).unsqueeze(-1))

        feat = torch.cat(parts, dim=-1)   # (B, 47)

        out = F.silu(self.proj(feat))     # (B, d_model)
        if training:
            out = self.dropout(out)
        return out


# ── CrossDomainInterestTransfer ───────────────────────────────────────────────

class CrossDomainInterestTransfer(nn.Module):
    """Two-step domain representation builder.

    Step A: Target-aware pooling — item_repr × decay_weights × attn_weights
            (skipped for domains in precomputed_repr_dict)
    Step B: Cross-domain MHA — domain representations attend to each other
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.01) -> None:
        super().__init__()
        self.d_model = d_model
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        item_repr: torch.Tensor,
        event_embs_dict: Dict[str, torch.Tensor],
        decay_weights_dict: Dict[str, torch.Tensor],
        seq_lens_dict: Dict[str, torch.Tensor],
        domains: List[str],
        precomputed_repr_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        scale = math.sqrt(self.d_model)
        domain_reprs: List[torch.Tensor] = []
        attn_w_list: List[Optional[torch.Tensor]] = []

        for domain in domains:
            # Precomputed path: skip target-aware pooling
            if precomputed_repr_dict is not None and domain in precomputed_repr_dict:
                domain_repr = precomputed_repr_dict[domain]   # (B, d_model)
                domain_reprs.append(domain_repr.unsqueeze(1))
                attn_w_list.append(None)   # sentinel: no entropy logging
                continue

            embs = event_embs_dict[domain]          # (B, L, d_model)
            decay_w = decay_weights_dict[domain]    # (B, L)
            seq_lens = seq_lens_dict[domain]        # (B,)

            B, L, _ = embs.shape
            pad_mask = (
                torch.arange(L, device=seq_lens.device).unsqueeze(0)
                >= seq_lens.unsqueeze(1)
            )  # (B, L), True=pad

            scores = torch.bmm(embs, item_repr.unsqueeze(-1)).squeeze(-1) / scale  # (B, L)
            scores = scores.masked_fill(pad_mask, -1e9)
            attn_w = torch.softmax(scores, dim=-1)   # (B, L)

            w = attn_w * decay_w                     # (B, L)
            w = w.masked_fill(pad_mask, 0.0)
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            domain_repr = (embs * w.unsqueeze(-1)).sum(dim=1)   # (B, d_model)
            domain_reprs.append(domain_repr.unsqueeze(1))
            attn_w_list.append(attn_w)

        domain_stacked = torch.cat(domain_reprs, dim=1)  # (B, num_domains, d_model)

        normed = self.norm(domain_stacked)
        cross_out, cross_attn_weights = self.cross_attn(
            normed, normed, normed,
            need_weights=True, average_attn_weights=True,
        )
        cross_domain_out = domain_stacked + self.dropout(cross_out)

        return cross_domain_out, cross_attn_weights, attn_w_list


# ── UnifiedInteractionBlock ───────────────────────────────────────────────────

class UnifiedInteractionBlock(nn.Module):

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


# ── PCVRFusionFormer10 ────────────────────────────────────────────────────────

class PCVRFusionFormer10(nn.Module):
    """PCVRFusionFormer10: exp6 base + seq_d EventEncoder → SeqDStatisticalEncoder.

    _score() is identical to model6 (MLP ui_score + LayerNorm gate).
    seq_a/b/c retain EventEncoder with target-aware pooling unchanged.
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
        self.num_ns = self.num_user_ns + self.num_item_ns

        # ── Sequence encoders ────────────────────────────────────────────────
        # seq_d: SeqDStatisticalEncoder; seq_a/b/c: EventEncoder
        self.seq_encoders = nn.ModuleDict()
        for domain in self.seq_domains:
            if domain == 'seq_d':
                self.seq_encoders[domain] = SeqDStatisticalEncoder(
                    d_model=d_model,
                    dropout=dropout_rate,
                )
            else:
                self.seq_encoders[domain] = EventEncoder(
                    vocab_sizes=seq_vocab_sizes[domain],
                    emb_dim=emb_dim,
                    d_model=d_model,
                    num_time_buckets=num_time_buckets,
                    emb_skip_threshold=emb_skip_threshold,
                    seq_id_threshold=seq_id_threshold,
                    dropout=dropout_rate,
                )

        # ── Cross-Domain Interest Transfer ───────────────────────────────────
        self.cross_domain_transfer = CrossDomainInterestTransfer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
        )

        # ── Unified interaction blocks ────────────────────────────────────────
        self.blocks = nn.ModuleList([
            UnifiedInteractionBlock(d_model, num_heads, hidden_mult, dropout_rate)
            for _ in range(num_hyformer_blocks)
        ])

        # ── Scoring head (identical to model6) ───────────────────────────────
        self.user_item_xattn = CrossAttention(d_model, num_heads, dropout_rate, ln_mode='pre')

        # Signal 1 (A): ui_mlp
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

        # Gate (B): LayerNorm before sigmoid
        gate_in_dim = 1 + self.num_sequences + 1
        self.gate_norm = nn.LayerNorm(gate_in_dim)
        self.gate_linear = nn.Linear(gate_in_dim, 3)

        self.emb_dropout = nn.Dropout(dropout_rate)

        self._tb_logs: Optional[dict] = {}

        self._init_params()

        total = sum(p.numel() for p in self.parameters())
        logging.info(
            f"PCVRFusionFormer10: U={self.num_user_ns} I={self.num_item_ns} "
            f"S={self.num_sequences} T={self.num_user_ns+self.num_item_ns+self.num_sequences} "
            f"d_model={d_model} blocks={num_hyformer_blocks} "
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

        item_repr = item_tok.mean(dim=1)   # (B, d_model)

        event_embs_dict: Dict[str, torch.Tensor] = {}
        decay_w_dict:    Dict[str, torch.Tensor] = {}
        precomputed_repr: Dict[str, torch.Tensor] = {}

        for domain in self.seq_domains:
            enc = self.seq_encoders[domain]
            if isinstance(enc, SeqDStatisticalEncoder):
                precomputed_repr[domain] = enc(
                    inputs.seq_data[domain],
                    inputs.seq_lens[domain],
                    inputs.seq_time_buckets[domain],
                    training=training,
                )
            else:
                embs, dw = enc(
                    inputs.seq_data[domain],
                    inputs.seq_lens[domain],
                    inputs.seq_time_buckets[domain],
                    training=training,
                )
                event_embs_dict[domain] = embs
                decay_w_dict[domain] = dw

        cross_domain_out, cross_attn_w, attn_w_list = self.cross_domain_transfer(
            item_repr=item_repr,
            event_embs_dict=event_embs_dict,
            decay_weights_dict=decay_w_dict,
            seq_lens_dict=inputs.seq_lens,
            domains=self.seq_domains,
            precomputed_repr_dict=precomputed_repr,
        )
        domain_tok = cross_domain_out   # (B, num_domains, d_model)

        if self._tb_logs is not None:
            with torch.no_grad():
                ca = cross_attn_w.mean(0)
                for i in range(ca.shape[0]):
                    for j in range(ca.shape[1]):
                        self._tb_logs[f'xdomain/attn_{i}{j}'] = ca[i, j].item()
                # entropy: skip seq_d (attn_w is None for StatisticalEncoder)
                for domain, attn_w in zip(self.seq_domains, attn_w_list):
                    if attn_w is None:
                        continue
                    ent = -(attn_w * (attn_w + 1e-9).log()).sum(-1).mean().item()
                    self._tb_logs[f'entropy/target_attn_{domain}'] = ent
                # lambda: only EventEncoder domains have log_lambda
                for domain in self.seq_domains:
                    enc = self.seq_encoders[domain]
                    if hasattr(enc, 'log_lambda'):
                        lam = torch.exp(enc.log_lambda).item()
                        self._tb_logs[f'lambda/{domain}'] = lam

        if training:
            user_tok = self.emb_dropout(user_tok)
            item_tok = self.emb_dropout(item_tok)
            domain_tok = self.emb_dropout(domain_tok)

        return torch.cat([user_tok, item_tok, domain_tok], dim=1)   # (B, T, d_model)

    def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def _score(self, all_tok: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        U, I = self.num_user_ns, self.num_item_ns

        user_final  = all_tok[:, :U, :]
        item_final  = all_tok[:, U:U + I, :]
        domain_final = all_tok[:, U + I:, :]

        ui = self.user_item_xattn(query=item_final, key_value=user_final)
        ui_repr    = ui.mean(dim=1)
        user_repr  = user_final.mean(dim=1)
        item_repr  = item_final.mean(dim=1)

        # Signal 1 (A): MLP over interaction features
        ui_input = torch.cat(
            [ui_repr, item_repr, ui_repr - item_repr, ui_repr * item_repr], dim=-1
        )
        ui_score = self.ui_mlp(ui_input)

        # Signal 2: domain similarity (dot product scaled by sqrt(d_model))
        domain_sim = torch.bmm(
            domain_final, ui_repr.unsqueeze(-1)
        ).squeeze(-1) / math.sqrt(self.d_model)

        # Signal 3: rich representation
        rich_scalar = self.rich_proj(
            torch.cat([user_repr, item_repr, ui_repr], dim=-1)
        )

        # Gate (B): LayerNorm before sigmoid
        gate_in = torch.cat([ui_score, domain_sim, rich_scalar], dim=-1)
        gate_in = self.gate_norm(gate_in)
        gates = torch.sigmoid(self.gate_linear(gate_in))

        signals = torch.cat(
            [ui_score, domain_sim.mean(dim=-1, keepdim=True), rich_scalar],
            dim=-1,
        )
        logits = (gates * signals).sum(dim=-1, keepdim=True)

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
        all_tok = self._tokenize(inputs, training=True)
        all_tok = self._run_blocks(all_tok)
        logits, _ = self._score(all_tok)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, embedding) — no dropout, no TB logging."""
        self._tb_logs = None
        all_tok = self._tokenize(inputs, training=False)
        all_tok = self._run_blocks(all_tok)
        return self._score(all_tok)

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
        logging.info(f"PCVRFusionFormer10: re-initialized {count} high-cardinality embeddings")
        return reinit_ptrs


# infer.py dynamic import requires PCVRHyFormer alias
PCVRHyFormer = PCVRFusionFormer10
