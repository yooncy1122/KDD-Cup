"""Weighted Embedding extension for PCVRHyFormer.

FID 62-66은 user_int(버킷 인덱스)와 user_dense(float 계수)가 위치 정렬된 aligned pair다.
RankMixerNSTokenizer의 uniform mean pooling 대신 float 계수로 softmax 정규화된
weighted sum을 적용해 쌍 관계를 직접 표현력에 반영한다.

변경 없는 파일: model.py, trainer.py, ModelInput, dataset.py, infer.py
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from model import (
    PCVRHyFormer,
    RankMixerNSTokenizer,
    ModelInput,
)

# user_int schema 내 FID 62-66의 fid_idx (0-based, schema_aligned.json 기준)
_WEIGHTED_FID_INDICES = {17, 18, 19, 20, 21}  # fid 62, 63, 64, 65, 66


class WeightedRankMixerNSTokenizer(RankMixerNSTokenizer):
    """RankMixerNSTokenizer with softmax-weighted sum for aligned-pair fids.

    aligned_weights: {fid_idx: (B, length) float tensor} — raw coefficients from user_dense.
    Fids not in aligned_weights fall back to uniform mean pooling.
    """

    def forward(
        self,
        int_feats: torch.Tensor,
        aligned_weights: Optional[dict] = None,
    ) -> torch.Tensor:
        """Embeds all features; uses weighted sum for fids in aligned_weights.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.
            aligned_weights: optional dict {fid_idx: (B, length) float tensor}.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]

                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()   # (B, length)
                        emb_all = emb_layer(vals)                            # (B, length, emb_dim)

                        use_weighted = (
                            aligned_weights is not None
                            and fid_idx in aligned_weights
                        )

                        if use_weighted:
                            w = aligned_weights[fid_idx].float()             # (B, length)
                            pad_mask = (vals != 0).float()                   # (B, length)
                            w = w * pad_mask                                 # 패딩 위치 0
                            # softmax: shift for numerical stability, then zero pad positions
                            w_shifted = w - w.max(dim=1, keepdim=True).values
                            exp_w = w_shifted.exp() * pad_mask
                            norm_w = exp_w / exp_w.sum(dim=1, keepdim=True).clamp(min=1e-9)
                            fid_emb = (emb_all * norm_w.unsqueeze(-1)).sum(dim=1)  # (B, emb_dim)
                        else:
                            # uniform mean (original behavior)
                            mask = (vals != 0).float().unsqueeze(-1)         # (B, length, 1)
                            count = mask.sum(dim=1).clamp(min=1)             # (B, 1)
                            fid_emb = (emb_all * mask).sum(dim=1) / count   # (B, emb_dim)

                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))

        chunks = cat_emb.split(self.chunk_dim, dim=-1)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class WeightedPCVRHyFormer(PCVRHyFormer):
    """PCVRHyFormer with weighted embedding for FID 62-66.

    Extra constructor argument:
        user_dense_aligned_plan: list of (fid_idx_in_user_int, dense_offset, dense_len)
            pre-computed from pcvr_dataset schemas by train.py.
    """

    def __init__(self, *args, user_dense_aligned_plan: Optional[List[Tuple[int, int, int]]] = None, **kwargs):
        super().__init__(*args, **kwargs)

        # In-place swap: replace RankMixerNSTokenizer with WeightedRankMixerNSTokenizer.
        # Safe because WeightedRankMixerNSTokenizer adds only a new forward() — no new __init__,
        # so all instance attributes set by the parent __init__ remain valid.
        if self.ns_tokenizer_type == 'rankmixer':
            self.user_ns_tokenizer.__class__ = WeightedRankMixerNSTokenizer

        self._aligned_plan: List[Tuple[int, int, int]] = user_dense_aligned_plan or []

    def _build_aligned_weights(self, user_dense_feats: torch.Tensor) -> dict:
        return {
            fid_idx: user_dense_feats[:, dense_off:dense_off + dense_len]
            for fid_idx, dense_off, dense_len in self._aligned_plan
        }

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        aligned_weights = self._build_aligned_weights(inputs.user_dense_feats)
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats, aligned_weights=aligned_weights)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training
        )

        logits = self.clsfier(output)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        aligned_weights = self._build_aligned_weights(inputs.user_dense_feats)
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats, aligned_weights=aligned_weights)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=False
        )

        logits = self.clsfier(output)
        return logits, output


# Re-export under the canonical name so train.py's
# `from model_weighted import PCVRHyFormer` works without modification.
PCVRHyFormer = WeightedPCVRHyFormer
