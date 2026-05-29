"""Relative-time dataset: replaces request-relative time buckets with inter-event intervals.

Instead of encoding "how long ago did this event happen" (request_ts - event_ts),
each time_bucket now encodes the gap to the *previous* event in the same sequence
(ts[i] - ts[i-1]). This captures burst patterns and session boundaries.

Padding rules:
  - Position 0 (first event): bucket = 0  (no predecessor)
  - Current event is padding (ts == 0): bucket = 0
  - Previous event is padding (ts == 0): bucket = 0  (gap is undefined)

Inherits exp/dataset_aligned.py to preserve Exp 2a (FID 89/90/91 reconstruction).
Combine with --model_module model_weighted for full 2a + 2b + 3 stack.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_aligned import (
    PCVRParquetDatasetAligned,
    FeatureSchema,
)

# --- Bucket boundaries optimised for inter-event intervals ---
# Dense at short intervals (burst/session detection), coarser at long intervals.
# 63 boundaries → NUM_TIME_BUCKETS = 64  (same vocab size as baseline)
RELATIVE_BUCKET_BOUNDARIES = np.array([
    # 0–3 min: burst / within-session (12)
    2, 5, 10, 15, 20, 30, 40, 50, 60, 90, 120, 180,
    # 3–20 min: session browsing (8)
    240, 300, 360, 480, 600, 720, 900, 1200,
    # 20 min – 1 hr (6)
    1500, 1800, 2400, 3000, 3600, 5400,
    # 1.5–6 hr: within-day inter-session (6)
    7200, 9000, 10800, 14400, 18000, 21600,
    # 6–24 hr: session boundaries (6)
    28800, 36000, 43200, 57600, 72000, 86400,
    # 1–7 days (7)
    129600, 172800, 216000, 259200, 302400, 345600, 432000,
    # 1–4 weeks (6)
    518400, 604800, 864000, 1209600, 1728000, 2592000,
    # 1–3 months (5)
    3888000, 5184000, 6480000, 7776000, 9072000,
    # 3–12 months (4)
    10368000, 15552000, 23328000, 31536000,
    # >1 year (3)
    63072000, 94608000, 157680000,
], dtype=np.int64)

NUM_TIME_BUCKETS = len(RELATIVE_BUCKET_BOUNDARIES) + 1  # = 64


class PCVRParquetDatasetRelTime(PCVRParquetDatasetAligned):
    """PCVRParquetDatasetAligned subclass that uses inter-event time buckets."""

    def _convert_batch(self, batch):
        result = super()._convert_batch(batch)   # runs 2a reconstruction + base time buckets
        B = batch.num_rows

        for domain in self.seq_domains:
            _, ts_ci = self._seq_plan[domain]
            if ts_ci is None:
                continue
            max_len = self._seq_maxlen[domain]

            # Re-read the timestamp column (same padding logic as base class).
            ts_col = batch.column(ts_ci)
            ts_offs = ts_col.offsets.to_numpy()
            ts_vals = ts_col.values.to_numpy()
            ts_padded = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                s, e = int(ts_offs[i]), int(ts_offs[i + 1])
                ul = min(e - s, max_len)
                if ul > 0:
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

            # Events are stored newest-first: ts[0] >= ts[1] >= ...
            # diff[i] = ts[i-1] - ts[i]  =  gap between position i and the more-recent position i-1
            time_diff = np.zeros((B, max_len), dtype=np.int64)
            time_diff[:, 1:] = np.maximum(ts_padded[:, :-1] - ts_padded[:, 1:], 0)

            raw_buckets = np.clip(
                np.searchsorted(RELATIVE_BUCKET_BOUNDARIES, time_diff.ravel()),
                0, len(RELATIVE_BUCKET_BOUNDARIES) - 1,
            )
            buckets = raw_buckets.reshape(B, max_len) + 1

            # Padding rules
            buckets[:, 0] = 0                          # first event: no predecessor
            buckets[ts_padded == 0] = 0                # current event is padding
            prev_zero = np.zeros((B, max_len), dtype=bool)
            prev_zero[:, 1:] = (ts_padded[:, :-1] == 0)
            buckets[prev_zero] = 0                     # previous event is padding

            result[f'{domain}_time_bucket'] = torch.from_numpy(buckets)

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

    train_dataset = PCVRParquetDatasetRelTime(
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

    valid_dataset = PCVRParquetDatasetRelTime(
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


PCVRParquetDataset = PCVRParquetDatasetRelTime  # inference alias for infer.py
