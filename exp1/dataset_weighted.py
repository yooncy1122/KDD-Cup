"""Thin re-export: dataset_aligned + weighted embedding experiment.

dataset_aligned의 sparse→dense reconstruction (FID 89/90/91)과
model_weighted의 weighted embedding (FID 62-66)을 함께 사용한다.
"""

from dataset_aligned import (  # noqa: F401
    PCVRParquetDatasetAligned as PCVRParquetDataset,
    get_pcvr_data,
    FeatureSchema,
    NUM_TIME_BUCKETS,
)
