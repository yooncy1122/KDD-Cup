# TAAC 2026 — Experiment Log

측정 환경: demo 데이터 1,000행 (700 train / 300 valid), 3 epochs, batch_size=64, CPU, seed=42  
비교 기준 스크립트: `run.sh` (baseline), 각 실험별 전용 run script

---

## Baseline

| 항목 | 값 |
|------|----|
| Schema | `data/schema.json` (user_dense total_dim = 755) |
| Dataset | `dataset.py` |
| Model | `model.py` (PCVRHyFormer) |
| Run script | `run.sh` |
| **3-epoch best AUC** | **0.7784** |

---

## Experiment 1: time_to_action (TTA) Feature

### 목적
레이블 발생까지 걸린 시간(`label_time - request_time`)을 user_dense scalar 피처(fid 110)로 추가해
전환 행동의 빠르기 정보를 모델에 제공한다.

### 구현 방식
- `request_time` 컬럼과 `label_time` 컬럼의 차이를 계산해 초 단위 float으로 변환
- `schema_with_tta.json`에 fid 110 (dim=1) 추가 → total_dim 756
- `dataset_with_tta.py`에서 `_convert_batch()` 내 해당 컬럼 읽어 채움

### 결과

| Epoch | AUC |
|-------|-----|
| 1 | 0.6904 |
| 2 | 0.7432 |
| 3 | 0.7535 |

**3-epoch best AUC: 0.7535 (baseline 0.7784 대비 -0.0249)**

### 결론 및 인사이트
- **FAILED — Label Leakage**: `label_time`은 전환이 실제로 발생했을 때만 기록되는 미래 정보.  
  추론 시점에 사용할 수 없으므로 프로덕션 배포 불가.
- negative sample의 `label_time`이 null이라 평균값 대체 등 imputation을 쓰더라도  
  label 여부가 피처에 직접 누출되는 구조적 결함이다.

### 관련 파일
- `baseline/dataset_with_tta.py`
- `data/schema_with_tta.json`

---

## Experiment 2a: FID 89/90/91 Sparse→Dense Reconstruction

### 목적
FID 89/90/91의 aligned pair 구조를 활용해 sparse 표현을 명시적인 dense 히스토그램 벡터로
복원하고 user_dense에 추가한다. 기존 `nn.Linear(755, d_model)` dense 프로젝션이
이 정보를 직접 학습할 수 있도록 한다.

### 구현 방식
- 각 FID에서 int 배열(버킷 인덱스 0-9)과 float 배열(계수)을 scatter-add로 합산
- 길이 10짜리 복원 벡터 3개(fid 111/112/113)를 user_dense 뒤에 append
- `schema_aligned.json`에 [111,10], [112,10], [113,10] 추가 → total_dim 785

### 결과

| Epoch | AUC |
|-------|-----|
| 1 | 0.6955 |
| 2 | 0.7603 |
| 3 | 0.7901 |

**3-epoch best AUC: 0.7901 (baseline 대비 +0.0117)**

### 결론 및 인사이트
- 복원된 벡터는 해당 피처의 주변 분포를 인덱스별로 명시적으로 표현한다.
- 기존 dense 프로젝션이 이 구조를 직접 학습하는 것보다, 이미 해석 가능한 형태로
  제공하는 것이 수렴을 가속한다.
- index=0 (padding)을 scatter-add에서 제외하는 것이 핵심: `valid = idxs > 0`

### 관련 파일
- `baseline/dataset_aligned.py` — `PCVRParquetDatasetAligned` 서브클래스
- `data/schema_aligned.json` — total_dim 785
- `baseline/run_aligned.sh` — `--dataset_module dataset_aligned`

---

## Experiment 2b: FID 62-66 Weighted Embedding (누적: 2a 포함)

### 목적
FID 62-66도 aligned pair 구조를 갖는다. `RankMixerNSTokenizer`의 uniform mean pooling을
float 계수 기반 softmax weighted sum으로 대체해, 더 중요한 버킷 인덱스의 임베딩에
가중치를 부여한다.

### 구현 방식
- `WeightedRankMixerNSTokenizer.forward()`에서 fid_idx 17-21(fid 62-66)에 대해
  `norm_w = softmax(w * pad_mask)` 계산 후 `fid_emb = (emb_all * norm_w).sum(dim=1)`
- `WeightedPCVRHyFormer.__init__()`에서 `__class__ swap`으로 토크나이저 in-place 교체
- `train.py`가 schema에서 오프셋을 자동 계산해 `user_dense_aligned_plan`으로 전달

### 결과

| Epoch | AUC |
|-------|-----|
| 1 | 0.6995 |
| 2 | 0.7740 |
| 3 | 0.8001 |

**3-epoch best AUC: 0.8001 (2a 대비 +0.0100, baseline 대비 +0.0217)**

### 결론 및 인사이트
- FID 62-66의 float 계수는 단순히 dense 프로젝션의 입력값으로 쓰이는 것보다
  임베딩 조합의 가중치로 활용될 때 더 효과적이다.
- 파라미터 추가 없이 구현 가능: 기존 임베딩 테이블과 프로젝션 레이어를 그대로 재사용.
- 주의: user_dense 오프셋이 fid 61(dim=256) 이후부터 시작함.
  fid 62의 실제 오프셋은 [256, 261) — 0이 아님.

### 관련 파일
- `baseline/model_weighted.py` — `WeightedRankMixerNSTokenizer`, `WeightedPCVRHyFormer`
- `baseline/dataset_weighted.py` — `dataset_aligned` re-export
- `baseline/run_weighted.sh` — `--dataset_module dataset_weighted --model_module model_weighted`
- `baseline/train.py` — `--model_module` 플래그, `user_dense_aligned_plan` 자동 계산

---

---

## Experiment 3: Relative Time Encoding (누적: 2a + 2b 포함)

### 목적
`time_bucket`이 인코딩하는 정보를 "이벤트가 request 기준으로 얼마나 오래됐는가"
(request_ts - event_ts)에서 "연속된 이벤트 사이의 간격"(inter-event interval)으로 교체한다.
세션 내 burst 패턴, 세션 경계, 사용자 활동 밀도를 time embedding에 직접 반영하는 것이 목표.

### 구현 방식
- `RELATIVE_BUCKET_BOUNDARIES` 새 설계 (63개 경계): 짧은 간격(2초~3분)에 밀도 집중,
  이후 세션 경계(6~24시간), 주간/월간 패턴까지 커버
- `NUM_TIME_BUCKETS = 64` (baseline과 동일 — 모델 가중치 크기 불변)
- 시퀀스가 **newest-first** 저장임을 확인 → `time_diff = ts[i-1] - ts[i]`
- 패딩 규칙: position 0 → bucket 0, 현재/이전 이벤트 padding → bucket 0
- `PCVRParquetDatasetRelTime(PCVRParquetDatasetAligned)`: `_convert_batch()`만 오버라이드

### 결과

| Epoch | AUC |
|-------|-----|
| 1 | 0.7011 |
| 2 | 0.7737 |
| 3 | 0.8006 |

**3-epoch best AUC: 0.8006 (Exp 2b 대비 +0.0005)**

### 결론 및 인사이트
- demo 규모(600행 train)에서 +0.0005는 랜덤 시드 노이즈 범위 내.
- inter-event interval 정보가 유익한지 여부를 판단하려면 전체 데이터 스케일 재검증이 필요.
- 기술적으로는 올바르게 구현됨: bucket 분포가 다양하고(max ≈ 59), padding 처리도 정확.
- 단독 효과 vs 절대 recency 효과 비교 실험(reltime만, weighted 없이)은 미수행.

### 관련 파일
- `exp/dataset_reltime.py` — `PCVRParquetDatasetRelTime`, `RELATIVE_BUCKET_BOUNDARIES`
- `exp/run_reltime.sh` — `--dataset_module dataset_reltime --model_module model_weighted`

---

## AUC 요약

| 실험 | 3-epoch best AUC | 전 단계 대비 | 관련 run script |
|------|-----------------|-------------|-----------------|
| Baseline | 0.7784 | — | `run.sh` |
| Exp 1: TTA | 0.7535 | **-0.0249** (실패) | `run_with_tta.sh` (deprecated) |
| Exp 2a: Recon | 0.7901 | +0.0117 | `run_aligned.sh` |
| Exp 2b: Weighted | 0.8001 | +0.0100 | `run_weighted.sh` |
| Exp 3: RelTime | **0.8006** | +0.0005 (노이즈 수준) | `run_reltime.sh` |

---

## 다음 실험 후보

### Candidate A: 전체 데이터에서 2a/2b/3 효과 검증
- demo 1,000행 결과가 전체 데이터 규모에서도 재현되는지 확인
- Exp 3(RelTime)의 +0.0005가 유효한지 전체 스케일에서 재측정
- 2a와 2b의 개별 기여도를 분리 측정 (2a 없이 2b만 적용 시 AUC)

### Candidate B: Event 구조 활용 (시퀀스 도메인 재설계)
- 각 sequential domain 내 이벤트 타입을 분리해 sub-domain으로 나누거나 타입별 가중치 부여
- **선행 작업**: `_embed_seq_domain()`과 `MultiSeqHyFormerBlock`의 `(B, S, L)` shape 제약 충돌 여부 확인

### Candidate C: Absolute + Relative 동시 사용
- position 0에 절대 recency(request - event_ts)를, 나머지에 inter-event interval을 결합
- 또는 두 임베딩을 concat하거나 가산하는 방식으로 양쪽 정보를 모두 보존
