# TAAC 2026 — Tencent Advertising Algorithm Competition

## Task Overview

**Conversion Rate Prediction (pCVR)**: Given a specific user and an ad (item) shown to them, predict the probability that the user will click on the ad.

> "If we show *this ad* to *this user*, what is the probability they will click it?"

The model outputs a score between 0 and 1, representing the predicted Click-Through Rate (CTR / pCVR) for each user–item pair.

**Evaluation Metric**: AUC-ROC 

---

## Two Types of Input Data

The model leverages two fundamentally different types of information to make predictions:

### 1. Non-Sequential Data (Static Profile)
Features that do not change over short time horizons — background attributes and fixed specifications of the user or item.

- These are treated as **NS (Non-Sequential) tokens** in the unified architecture.

### 2. Sequential Data (Behavioral History)
Time-ordered records of user actions — what the user clicked, viewed, or interacted with across multiple behavioral domains over time.

- Captures the *temporal dynamics* of user interest
- Organized into multiple heterogeneous behavioral domains (Domain A / B / C / D)
- These are treated as **S (Sequential) tokens** in the unified architecture.

---

## Competition Goal: Unified Block Architecture

### Problem with Existing Approaches
Traditional recommendation systems handle the two data types separately:

```
Sequential Data   → Sequence Encoder  ──┐
                                         ├──→ Late Fusion
Non-Sequential Data → Feature Interaction ──┘
```

### What This Competition Asks For
Build a Unified Block architecture that processes both sequential and non-sequential features within a single backbone — not a decoupled two-stage pipeline.

- Non-Sequential (NS) tokens: static user/item profile features → converted to NS tokens
- Sequential (S) tokens: time-ordered user behavioral history → converted to S tokens
- Both token types must interact bidirectionally via attention inside the same block
- The unified block must be stackable (multiple layers) for deep representation learning

---

## Main Tasks

1. Unified Tokenization: map both sequential and non-sequential features into a shared embedding space (same dimension D)
2. Unified Block: feed S+NS tokens into one block; use attention so every token attends to every other token regardless of type; stack L such blocks
3. CVR Prediction: pass final representations through a prediction head; optimize with Binary Cross-Entropy Loss

---

## Dataset Feature Summary (TAAC 2026)
File: `data/demo_1000.parquet`  
Full feature definitions and data schema: `data/DATASET_README.md`  

---

## Key Design Constraints

1. **Single backbone**: Both sequential and non-sequential features must be processed within the same unified architecture — not in separate encoders that are fused later.
2. **Bidirectional interaction**: between static profile and behavioral history within each block
3. **Stackable**: The unified block must be designed to be stacked in multiple layers to enable deep representation learning.

---

## Coding Rule and extra info
- The training template includes a mandatory run.sh file as the execution entry point
- The script must be strictly named "infer.py" and must contain a main() function that takes no arguments
- If needed, it is available to plot various metric(ex. gradient, loss, embedding, auc, ...) based on tensor board system

---

## Experiment History

All AUC numbers are measured on demo data (1,000 rows, 700 train / 300 valid, 3 epochs, CPU).
Full details: `baseline/experiments.md`

### Experiment 1: time_to_action feature (TTA)
- **Idea**: compute `label_time - request_time` as a new scalar user_dense feature (fid 110)
- **Result**: AUC degraded (0.7535 vs baseline 0.7784)
- **Conclusion**: **FAILED — label leakage**. `label_time` is future information unavailable at inference time. The feature cannot be used in production.
- **Files**: `baseline/dataset_with_tta.py`, `data/schema_with_tta.json`

### Experiment 2a: FID 89/90/91 Sparse→Dense Reconstruction
- **Idea**: FID 89/90/91 are aligned pairs (int = bucket index 0-9, dense = coefficient). Reconstruct a length-10 dense vector via scatter-add and append to user_dense (total_dim 755 → 785).
- **Result**: AUC **0.7784 → 0.7901** (+0.0117)
- **Conclusion**: The reconstructed dense vector captures the marginal distribution of the sparse representation in a form the linear dense projection can directly leverage.
- **Files**: `baseline/dataset_aligned.py`, `data/schema_aligned.json`, `baseline/run_aligned.sh`

### Experiment 2b: FID 62-66 Weighted Embedding (cumulative with 2a)
- **Idea**: FID 62-66 are also aligned pairs. Replace `RankMixerNSTokenizer`'s uniform mean pooling with softmax-weighted sum using the float coefficients from user_dense.
- **Result**: AUC **0.7901 → 0.8001** (+0.0100)
- **Conclusion**: Letting the model attend to tokens proportionally to their importance (rather than equally) improves representation quality without adding parameters.
- **Files**: `exp/model_weighted.py`, `exp/dataset_weighted.py`, `exp/run_weighted.sh`

### Experiment 3: Relative Time Encoding (cumulative with 2a + 2b)
- **Idea**: Replace request-relative time buckets (`request_ts - event_ts`) with inter-event intervals (`ts[i-1] - ts[i]`, newest-first). Redesign `BUCKET_BOUNDARIES` (63 edges) with finer resolution at short intervals for burst/session detection.
- **Result**: AUC **0.8001 → 0.8006** (+0.0005)
- **Conclusion**: Marginal gain within noise range at demo scale. Inter-event temporal patterns are encoded but contribution is not clearly separable without full-scale data. Needs revalidation on the full competition dataset.
- **Files**: `exp/dataset_reltime.py`, `exp/run_reltime.sh`

---

## Current File Structure (baseline/)

| File | Role |
|------|------|
| `dataset.py` | Base dataset — original baseline |
| `dataset_with_tta.py` | Exp 1: adds time_to_action scalar (do not use — label leakage) |
| `dataset_aligned.py` | Exp 2a: adds FID 89/90/91 sparse→dense reconstruction vectors |
| `dataset_weighted.py` | Exp 2b: thin re-export of dataset_aligned (combined experiment) |
| `model.py` | Base model — original PCVRHyFormer |
| `model_weighted.py` | Exp 2b: WeightedRankMixerNSTokenizer + WeightedPCVRHyFormer |
| `run.sh` | Baseline run script |
| `run_aligned.sh` | Exp 2a run script (`--dataset_module dataset_aligned`) |
| `run_weighted.sh` | Exp 2b run script (`--dataset_module dataset_weighted --model_module model_weighted`) |
| `dataset_reltime.py` | Exp 3: inter-event relative time buckets (cumulative with 2a + 2b) |
| `run_reltime.sh` | Exp 3 run script (`--dataset_module dataset_reltime --model_module model_weighted`) |
| `train.py` | Shared entry point; supports `--dataset_module` and `--model_module` flags |

| Schema | total_dim | Notes |
|--------|-----------|-------|
| `data/schema.json` | 755 | Baseline |
| `data/schema_with_tta.json` | 756 | +fid 110 (TTA, deprecated) |
| `data/schema_aligned.json` | 785 | +fid 111/112/113 (reconstruction vectors) |

---

## Next Experiment Candidates

### Candidate A: FID 62-66 isolated effect verification
- Run `dataset_aligned` + `model_weighted` on the **full competition dataset** to confirm the demo-scale AUC gains hold at scale.
- Also run `model_weighted` **without** 2a (use `schema.json` instead of `schema_aligned.json`) to isolate the pure weighted-embedding contribution vs. the reconstruction contribution.

### Candidate B: Event structure exploitation (sequence domain design)
- The sequential domains (A/B/C/D) contain heterogeneous event types mixed together. A potential improvement is to split or re-weight events by type within each domain before sequence encoding.
- **Prerequisite**: verify that reshaping the sequence tensor does not conflict with the current `(B, S, L)` shape assumption in `_embed_seq_domain` and `MultiSeqHyFormerBlock`. A shape audit is needed before implementation.