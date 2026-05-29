"""Aligned-pair dataset: adds sparse→dense reconstruction features for FID 89/90/91.

For each aligned pair (user_int fid X, user_dense fid X), the integer values are
bucket indices (0-9) and the float values are coefficients. This module reconstructs
a length-10 dense vector via scatter-add and appends it to user_dense.

FID mapping:
    89 → reconstructed at fid 111 (dims 755-764)
    90 → reconstructed at fid 112 (dims 765-774)
    91 → reconstructed at fid 113 (dims 775-784)

Requires schema_aligned.json (user_dense total_dim = 785).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    PCVRParquetDataset,
    FeatureSchema,
    NUM_TIME_BUCKETS,
    get_pcvr_data as _base_get_pcvr_data,
)

# Source fid → reconstructed fid mapping
_ALIGNED = {89: 111, 90: 112, 91: 113}


class PCVRParquetDatasetAligned(PCVRParquetDataset):
    """PCVRParquetDataset subclass that appends aligned-pair reconstruction features."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Filter out synthetic fids (111/112/113) that have no parquet column.
        # The base class zeros _buf_user_dense before filling; skipped slots stay 0
        # and are populated by _convert_batch via reconstruction below.
        self._user_dense_plan = [
            (ci, dim, offset)
            for ci, dim, offset in self._user_dense_plan
            if ci is not None
        ]

        self._recon_plan = []
        for src_fid, recon_fid in _ALIGNED.items():
            int_off, int_len = self.user_int_schema.get_offset_length(src_fid)
            dense_off, dense_len = self.user_dense_schema.get_offset_length(src_fid)
            recon_off, _ = self.user_dense_schema.get_offset_length(recon_fid)
            self._recon_plan.append((int_off, int_len, dense_off, dense_len, recon_off))

    def _convert_batch(self, batch):
        result = super()._convert_batch(batch)
        B = batch.num_rows
        user_int = result['user_int_feats']      # (B, 234) int64
        user_dense = result['user_dense_feats']  # (B, 785) float32

        for int_off, int_len, dense_off, dense_len, recon_off in self._recon_plan:
            idxs = user_int[:, int_off:int_off + int_len].numpy()    # (B, 10) int64
            coefs = user_dense[:, dense_off:dense_off + dense_len].numpy()  # (B, 10) float32

            recon = np.zeros((B, 10), dtype=np.float32)
            b_idx = np.repeat(np.arange(B)[:, None], int_len, axis=1)  # (B, 10)
            valid = idxs > 0  # index=0 is padding (matches _pad_varlen_int_column + Embedding(padding_idx=0))
            np.add.at(recon, (b_idx[valid], idxs[valid]), coefs[valid])

            user_dense[:, recon_off:recon_off + 10] = torch.from_numpy(recon)

        return result


def get_pcvr_data(
    data_dir,
    schema_path,
    batch_size=256,
    valid_ratio=0.1,
    train_ratio=1.0,
    num_workers=16,
    buffer_batches=20,
    shuffle_train=True,
    clip_vocab=True,
    seed=42,
    seq_max_lens=None,
):
    import os
    import glob as _glob
    import logging
    import pyarrow.parquet as pq

    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    train_dataset = PCVRParquetDatasetAligned(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDatasetAligned(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset
