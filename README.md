# TAAC 2026 — Tencent Advertising Algorithm Competition

## 대회 정보

| 항목 | 내용 |
|------|------|
| 대회명 | TAAC KDD Cup 2026 |
| 태스크 | 광고 노출 후 전환(CVR) 예측 — pCVR 예측 |
| 평가 지표 | AUC-ROC |
| 데이터 | 익명화된 멀티도메인 유저 행동 시퀀스 (user/item 정적 피처 + 4개 도메인 시퀀셜 피처) |

---

## 최종 성능

| 모델 | Test AUC |
|------|----------|
| **Best: exp6 (PCVRFusionFormer)** | **0.810362** |
| Baseline (exp2) | 0.8044 |

---

## 핵심 모델: PCVRFusionFormer (exp6)

### 1. StructuredUserNSTokenizer

피처 그룹별 성격에 맞게 다른 방식으로 토큰화한다.

- **Pretrained embedding** (fid61, fid87): concat → LayerNorm → Linear projection → 1 token
- **Aligned pair fid62~66**: activity weight 기반 softmax weighted sum 임베딩 → 1 token  
  float 계수를 임베딩 조합 가중치로 활용 (단순 pooling 대비 성능 향상)
- **Aligned pair fid89~91**: sparse→dense 히스토그램 복원 후 투영 → 1 token  
  버킷 인덱스별 계수를 scatter-add해 길이-10 dense 벡터로 복원
- **나머지 categorical**: RankMixer 스타일 K개 토큰 (청크 분할 후 각각 Linear projection)

### 2. Cross-Domain Interest Transfer

타겟 광고 아이템 관점에서 각 도메인의 관련 행동을 선택적으로 추출한다.

- **Step A**: `item_repr`을 query로 한 target-aware pooling  
  `w = softmax(score) × decay_weight` 조합으로 관련성 + 최신성을 동시 반영
- **Step B**: 4개 도메인 representation 간 cross self-attention으로 도메인 간 정보 전이

### 3. Gated Scoring Head

3가지 신호를 LayerNorm 정규화 후 동적 가중합으로 스코어링한다.

- **Signal 0 (ui_score)**: `[u, i, u−i, u⊙i]` concat → MLP → scalar
- **Signal 1 (domain_sim)**: `ui_repr`과 각 도메인 representation의 dot-product 유사도
- **Signal 2 (rich_scalar)**: `[user, item, ui]` concat → MLP → scalar
- **PEPNet gate** (exp12): `gate_bias = Linear(user_repr)` — 유저 컨텍스트 기반 개인화 게이팅

---

## 실험 히스토리

> 상세 실험 노트: [exp1/experiments.md](exp1/experiments.md)

### Demo 스케일 (1,000행 / 700 train / 300 valid / 3 epoch / CPU)

| 실험 | 핵심 변경 | 3-epoch best AUC | 비고 |
|------|-----------|-----------------|------|
| Baseline | 원본 PCVRHyFormer | 0.7784 | `baseline/run.sh` |
| Exp 1: TTA | label_time 피처 추가 | 0.7535 | **실패 — label leakage** |
| Exp 2a: Recon | fid89~91 sparse→dense 복원 | 0.7901 | +0.0117 |
| Exp 2b: Weighted | fid62~66 가중합 임베딩 | 0.8001 | +0.0100 |
| Exp 3: RelTime | inter-event interval time bucket | 0.8006 | +0.0005 (노이즈 수준) |
| Exp 6: FusionFormer | StructuredUserNSTokenizer + CDIT | — | Best Test AUC 0.810362 |
| Exp 12: PEPNet | gate_bias_proj(user_repr) 추가 | 0.7603* | PEPNet 개인화 게이팅 |
| Exp 13: BFTS | StaticFull + EventSliding + CrossBlock | 0.7519* | TokenFormer-inspired |

\* demo 스케일 수치. 전체 데이터 대비 경향이 다를 수 있음.

---

## 디렉토리 구조

```
TAAC_modeling/
├── baseline/               # 공유 인프라 (dataset, train, trainer, model base)
│   ├── dataset.py          # PCVRParquetDataset (IterableDataset)
│   ├── dataset_aligned.py  # Exp 2a: fid89~91 sparse→dense 복원
│   ├── dataset_reltime.py  # Exp 3: inter-event interval time bucket
│   ├── dataset_weighted.py # Exp 2b: dataset_aligned re-export
│   ├── model.py            # PCVRHyFormer base + ModelInput, CrossAttention
│   ├── model_weighted.py   # Exp 2b: WeightedRankMixerNSTokenizer
│   ├── train.py            # 공유 학습 엔트리포인트 (--dataset_module, --model_module)
│   ├── trainer.py          # Trainer 클래스
│   └── run.sh              # Baseline 실행 스크립트
│
├── exp1/                   # Exp 1~3 실험 (공유 인프라 + 스키마)
│   ├── schema_aligned.json # total_dim=785 (fid111~113 복원 벡터 추가)
│   ├── dataset_reltime.py
│   └── experiments.md      # 실험 상세 노트
│
├── exp2/ ~ exp13/          # 실험별 모델 (model{N}.py + run.sh + train.py + trainer.py)
│   └── model{N}.py         # 실험별 핵심 모델 구현
│
└── data/
    ├── demo_1000.parquet   # 1,000행 데모 데이터 (git 제외)
    ├── schema.json         # 원본 스키마 (total_dim=755)
    └── DATASET_README.md   # 피처 정의 및 데이터 스키마 설명
```

---

## 실행 방법

### 환경 설정

```bash
pip install torch pyarrow pandas scikit-learn tensorboard
```

### Baseline 학습

```bash
TRAIN_DATA_PATH=/path/to/data \
TRAIN_CKPT_PATH=/path/to/ckpt \
TRAIN_LOG_PATH=/path/to/log \
TRAIN_TF_EVENTS_PATH=/path/to/events \
bash baseline/run.sh
```

### Best 모델 (exp6) 학습

```bash
TRAIN_DATA_PATH=/path/to/data \
TRAIN_CKPT_PATH=/path/to/ckpt \
TRAIN_LOG_PATH=/path/to/log \
TRAIN_TF_EVENTS_PATH=/path/to/events \
bash exp6/run.sh \
    --seq_max_lens 'seq_a:256,seq_b:256,seq_c:512,seq_d:512' \
    --num_epochs 10
```

### 주요 공통 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--batch_size` | 배치 크기 | 128 |
| `--num_workers` | DataLoader 워커 수 | 8 |
| `--seq_max_lens` | 도메인별 시퀀스 최대 길이 | `seq_a:32,...` |
| `--valid_ratio` | 검증셋 비율 | 0.1 |
| `--num_epochs` | 학습 에폭 수 | 10 |
| `--device` | 학습 디바이스 | cuda (자동 감지) |

