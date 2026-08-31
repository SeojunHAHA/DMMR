# DMMR all-session LOSO 결과

## 실험 목적

SEED의 DE 특징을 이용해 DMMR을 논문의 session-1 전용 설정에서 session 1·2·3 전체를 사용하는 subject-independent LOSO로 확장했다. 한 fold에서 hold-out 피험자의 데이터는 학습에 전혀 사용하지 않았다.

## 프로토콜

- 데이터: `/media/NAS/nas_175/seojun/SEED_DE_MSMDA`
- 입력: 62채널 × 5 DE band = 310차원
- Fold: 15-fold leave-one-subject-out
- 학습: source 14명 각각의 session 1·2·3
- 평가: hold-out 1명의 session 1·2·3, 총 45 trials
- 모델: hold-out 피험자당 단일 DMMR 모델
- 정규화: 피험자·세션별 독립 min-max 정규화
- Sequence length: 30
- Batch size: 512
- Pretrain: 200 epochs, epoch당 7 iterations
- Finetune: 200 epochs, epoch당 source별 7 iterations
- Optimizer: Adam, learning rate 0.001, weight decay 0.0005
- DMMR beta: 0.05
- Seed: 3
- 실행 장치: NVIDIA TITAN RTX 8개, fold 병렬 실행
- 환경: Python 3.10, PyTorch 2.2.2+cu121
- 원본 결과: `/media/NAS/nas_175/seojun/DMMR/all_sessions_loso_200ep`

세 session은 별도 모델로 학습하지 않았다. source 피험자의 세 session window를 합쳐 하나의 모델을 학습하고, 같은 모델로 hold-out 피험자의 세 session 전체를 평가했다. 세션이 바뀔 때 trial ID에 offset을 적용해 총 45개 trial이 서로 겹치지 않도록 했다.

## 전체 결과 요약

분모가 `225`인 원 논문형 결과는 session 1만 평가했고, 분모가 `675`인 LOSO 결과는 session 1·2·3 전체, 분모가 `450`인 adaptation 결과는 adaptation에 쓴 session 1을 제외한 session 2·3 평가다.

| Encoder | 설정 | Validation/선택 | Trial Acc. | Trial Macro-F1 | 정답 Trials |
|---|---|---|---:|---:|---:|
| LSTM | 원 논문형 session-1 LOSO, 300+500 epochs | Final 500 | 77.33% | 75.30% | 174 / 225 |
| LSTM | Sessions 1·2·3 LOSO, 200+200 epochs | Final 200 | 74.37% | 72.76% | 502 / 675 |
| LSTM | Sessions 1·2·3 LOSO, 200+200 epochs | 피험자별 9-trial best validation | 74.96% | 73.57% | 506 / 675 |
| CNN | Sessions 1·2·3 LOSO, 200+200 epochs | Final 200 | 75.70% | 74.70% | 511 / 675 |
| CNN | Sessions 1·2 train, session 3 validation, 50+100 epochs | Best validation | 73.63% | 72.94% | 497 / 675 |
| CNN | Target session 1 adaptation | 없음 / 10 / 50 / 100 epochs | 72.22% / 76.00% / **77.11%** / 76.22% | 71.34% / 75.15% / **76.02%** / 74.94% | 325 / 342 / **347** / 343 of 450 |
| LSTM | Target session 1 adaptation | 없음 / 50 / 100 epochs | 75.78% / **79.11%** / **79.11%** | 74.07% / **78.20%** / 78.02% | 341 / **356** / **356** of 450 |

## 원 논문형 session-1 LOSO 기준선

초기 NPZ adapter 검증은 원 논문 공개 코드와 같이 session 1만 사용해 수행했다. 14명의 session 1로 학습하고 hold-out 피험자의 session 1, 15 trials를 평가했으며 target 데이터는 학습에 사용하지 않았다.

- Encoder: LSTM
- Pretrain 300 epochs, finetune 500 epochs
- Fixed final trial accuracy: **77.33% (174/225)**
- Fixed final trial macro-F1: **75.30%**
- Test-oracle trial accuracy: 88.00% (공식 수치로 사용 불가)
- 결과: `/media/NAS/nas_175/seojun/DMMR/outputs_npz_paper_loso`

이 결과는 session 1만 평가하므로 이후 session 1·2·3 LOSO의 `675 trials` 결과와 직접적인 우열 비교에는 사용하지 않는다.

## 최종 epoch 결과

| Subject | Sample Acc. | Sample Macro-F1 | Trial Acc. | Trial Macro-F1 |
|---:|---:|---:|---:|---:|
| 01 | 69.52% | 69.80% | 68.89% | 68.79% |
| 02 | 75.75% | 74.83% | 82.22% | 80.09% |
| 03 | 72.12% | 72.24% | 68.89% | 69.18% |
| 04 | 66.14% | 66.05% | 71.11% | 70.44% |
| 05 | 73.32% | 72.95% | 77.78% | 77.80% |
| 06 | 60.14% | 50.70% | 57.78% | 46.58% |
| 07 | 77.02% | 75.97% | 77.78% | 76.23% |
| 08 | 83.06% | 82.68% | 86.67% | 86.61% |
| 09 | 78.30% | 77.48% | 80.00% | 79.48% |
| 10 | 66.45% | 63.30% | 64.44% | 61.06% |
| 11 | 82.78% | 82.31% | 84.44% | 83.55% |
| 12 | 56.43% | 54.32% | 60.00% | 57.47% |
| 13 | 77.82% | 77.68% | 84.44% | 84.49% |
| 14 | 67.80% | 68.04% | 66.67% | 66.71% |
| 15 | 83.17% | 81.17% | 84.44% | 82.90% |
| **Mean** | **72.65%** | **71.30%** | **74.37%** | **72.76%** |
| **Population SD** | **7.93%p** | **9.24%p** | **9.25%p** | **10.99%p** |

공식 비교에는 위의 fixed final epoch 결과를 사용한다.

## Test-oracle 참고 결과

기존 실행은 매 finetune epoch마다 test 피험자를 평가했고, JSON의 `best_test_sample_oracle`은 test sample accuracy가 가장 높은 epoch를 사후 선택한 값이다. 이는 모델 선택에 test set을 사용하므로 공식 성능으로 보고하면 안 된다.

- Oracle sample accuracy 평균: 81.02%
- Oracle sample macro-F1 평균: 80.67%
- Oracle trial accuracy 평균: 83.11%
- Oracle trial macro-F1 평균: 82.97%

Oracle과 fixed-final trial accuracy의 차이는 약 8.74%p다. 이는 epoch 선택을 위한 독립 validation의 필요성을 보여준다.

## 해석 및 제한점

1. 기존 실험에는 독립 validation set이 없으므로 final epoch가 최적 epoch라는 보장이 없다.
2. 세션별 독립 정규화는 label을 사용하지 않지만, 해당 세션 전체의 feature 범위를 이용하는 transductive preprocessing이다.
3. DE 특징과 DMMR은 SEED 감정분류에 강한 spectral prior를 사용하므로 raw-EEG foundation model과 비교할 때 동일한 subject split, session 범위, trial aggregation을 맞춰야 한다.
4. 피험자별 편차가 크다. 특히 subject 06과 12가 전체 평균을 낮추므로 confusion matrix 및 class별 recall 확인이 필요하다.
5. 동일 trial의 인접 window는 반드시 같은 split에 있어야 하며, 이 실험은 피험자 단위 LOSO이므로 train/test 사이의 window leakage는 없다.

## 후속 validation 실험

후속 실험은 source 피험자마다 session 3을 validation으로 예약한다.

- Train: source 14명 × session 1·2
- Validation: 같은 source 14명 × session 3
- Test: hold-out 1명 × session 1·2·3
- 선택 기준: validation trial accuracy, 동률이면 validation trial macro-F1, 다시 동률이면 validation sample accuracy
- Test 피험자는 모델 선택에 사용하지 않음
- 출력: `/media/NAS/nas_175/seojun/DMMR/val_session3_loso_200ep`

이 설정은 session 3 일반화에 대한 validation이다. source 피험자가 train과 validation에 공통으로 등장하므로 완전한 cross-subject validation은 아니다. 더 엄격한 후속 실험으로는 inner validation subject를 따로 두는 nested LOSO가 적절하다. 또한 session 선택 편향을 확인하려면 validation session을 1, 2, 3으로 회전하는 세 실험이 필요하다.

### 채택한 9-trial validation

session 3 전체를 예약하는 실험은 중단하고, 최종적으로 피험자별 9-trial validation을 채택했다.

- 각 source 피험자의 각 session에서 감정별 1 trial을 fixed seed로 선택
- 피험자당 `3 sessions × 3 classes = 9 validation trials` (20%)
- 나머지 36 trials (80%)로 학습
- trial에 속한 모든 sequence를 함께 이동하여 인접-window leakage 방지
- 각 session의 min/max는 training trials로만 산출하고 validation trials에 적용
- hold-out 피험자의 session 1·2·3, 45 trials 전체는 최종 test로 완전히 격리
- 출력: `/media/NAS/nas_175/seojun/DMMR/val_9trials_loso_200ep`

## 9-trial validation 결과

15-fold가 오류 없이 완료됐다. 아래 test 결과는 validation trial accuracy를 우선 기준으로 선택한 epoch의 checkpoint에 대한 것이다.

| Subject | Selected epoch | Val trial acc. | Test sample acc. | Test sample F1 | Test trial acc. | Test trial F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 01 | 101 | 84.92% | 71.84% | 72.25% | 80.00% | 80.28% |
| 02 | 103 | 85.71% | 75.31% | 74.92% | 82.22% | 82.10% |
| 03 | 188 | 82.54% | 75.15% | 75.20% | 80.00% | 80.40% |
| 04 | 133 | 87.30% | 62.52% | 59.63% | 64.44% | 61.06% |
| 05 | 115 | 86.51% | 72.02% | 71.47% | 75.56% | 75.64% |
| 06 | 77 | 85.71% | 57.61% | 48.62% | 55.56% | 46.95% |
| 07 | 3 | 82.54% | 73.09% | 70.01% | 73.33% | 68.25% |
| 08 | 163 | 84.92% | 81.80% | 81.01% | 84.44% | 83.92% |
| 09 | 61 | 82.54% | 75.51% | 73.77% | 80.00% | 79.16% |
| 10 | 82 | 84.92% | 65.08% | 64.04% | 62.22% | 62.26% |
| 11 | 47 | 87.30% | 82.05% | 81.94% | 84.44% | 84.55% |
| 12 | 142 | 85.71% | 62.25% | 61.57% | 62.22% | 61.52% |
| 13 | 41 | 84.92% | 77.51% | 77.17% | 80.00% | 79.80% |
| 14 | 128 | 86.51% | 72.90% | 71.37% | 75.56% | 74.45% |
| 15 | 116 | 84.13% | 83.64% | 82.04% | 84.44% | 83.22% |
| **Mean** | — | — | **72.55%** | **71.00%** | **74.96%** | **73.57%** |
| **Population SD** | — | — | **7.45%p** | **8.91%p** | **9.09%p** | **10.73%p** |

동일 실행의 final epoch test trial accuracy는 74.81%였다. validation 선택 결과는 final epoch보다 +0.15%p, 기존 all-session fixed-final 74.37%보다 +0.59%p 높다. 차이는 작으므로 validation 선택이 큰 성능 향상을 만들었다고 해석하기보다는 test-oracle 없이 합법적으로 epoch를 선택할 수 있게 된 점이 핵심이다.

## CNN encoder, validation-free 결과

LSTM encoder를 파라미터 수가 유사한 temporal residual CNN으로 교체했다. source 14명의 session 1·2·3 전체로 학습하고, validation이나 test 기반 epoch 선택 없이 200번째 final epoch만 hold-out 피험자의 45 trials에 평가했다.

| Subject | Sample Acc. | Sample Macro-F1 | Trial Acc. | Trial Macro-F1 |
|---:|---:|---:|---:|---:|
| 01 | 62.36% | 61.73% | 64.44% | 64.41% |
| 02 | 77.04% | 76.33% | 82.22% | 81.07% |
| 03 | 76.88% | 76.92% | 77.78% | 77.87% |
| 04 | 60.73% | 57.42% | 64.44% | 60.03% |
| 05 | 75.53% | 75.50% | 77.78% | 78.03% |
| 06 | 59.51% | 54.72% | 60.00% | 56.34% |
| 07 | 73.83% | 72.06% | 77.78% | 75.00% |
| 08 | 86.22% | 85.90% | 91.11% | 91.07% |
| 09 | 77.41% | 76.10% | 80.00% | 79.19% |
| 10 | 67.33% | 67.09% | 68.89% | 68.57% |
| 11 | 90.26% | 90.21% | 88.89% | 88.94% |
| 12 | 64.91% | 64.48% | 64.44% | 64.31% |
| 13 | 78.89% | 78.93% | 77.78% | 77.94% |
| 14 | 67.88% | 66.85% | 75.56% | 74.71% |
| 15 | 84.05% | 82.13% | 84.44% | 83.01% |
| **Mean** | **73.52%** | **72.42%** | **75.70%** | **74.70%** |
| **Population SD** | **9.14%p** | **9.90%p** | **9.09%p** | **9.78%p** |

동일 validation-free final-epoch 프로토콜의 LSTM trial accuracy 74.37%와 비교하면 CNN은 75.70%로 **+1.33%p** 높다. Sample accuracy도 72.65%에서 73.52%로 +0.87%p 상승했다. 단일 seed 결과이므로 encoder 우열을 확정하려면 여러 seed 반복 또는 fold-wise paired 통계가 필요하다.

## CNN encoder, session-3 validation, 50+100 결과

- Train: source 14명의 session 1·2
- Validation: source 14명의 session 3 전체
- Test: hold-out 피험자의 session 1·2·3 전체
- Pretrain 50, finetune 100 epochs
- Cosine LR decay (`1e-3` → `1e-5`), gradient clipping 1.0
- Validation trial accuracy로 finetune checkpoint 선택

| Subject | Selected epoch | Val trial acc. | Test trial acc. | Test trial macro-F1 |
|---:|---:|---:|---:|---:|
| 01 | 53 | 81.43% | 57.78% | 58.49% |
| 02 | 88 | 81.43% | 77.78% | 76.88% |
| 03 | 44 | 84.29% | 84.44% | 84.54% |
| 04 | 47 | 83.33% | 73.33% | 74.00% |
| 05 | 29 | 82.38% | 80.00% | 79.96% |
| 06 | 62 | 82.86% | 62.22% | 58.76% |
| 07 | 52 | 80.95% | 73.33% | 71.78% |
| 08 | 85 | 80.95% | 77.78% | 77.68% |
| 09 | 58 | 81.90% | 75.56% | 74.93% |
| 10 | 44 | 82.86% | 64.44% | 63.08% |
| 11 | 45 | 80.95% | 88.89% | 89.03% |
| 12 | 7 | 82.86% | 64.44% | 64.13% |
| 13 | 42 | 83.33% | 73.33% | 72.85% |
| 14 | 12 | 80.48% | 71.11% | 70.70% |
| 15 | 56 | 80.48% | 80.00% | 77.32% |
| **Mean** | **48.3** | — | **73.63%** | **72.94%** |

Final-100 test trial accuracy는 73.04%였다. Session-3 validation 선택으로 +0.59%p 개선됐지만, all-session CNN no-validation final 결과 75.70%보다 2.07%p 낮다. 이는 source 학습에서 session 3 전체(33%)를 제외한 영향과 validation/test 분포 차이가 함께 반영된 결과다.

## CNN target session-1 supervised adaptation 결과

가장 최근 완료된 `val_session3_cnn_pre50_ft100_cosine`의 피험자별 best-validation checkpoint에서 이어서 학습했다.

- Adaptation: 각 target 피험자의 session 1 전체 15 trials와 라벨
- Test: 같은 target 피험자의 session 2·3, 총 30 trials
- Adaptation 범위: attention + CNN encoder + classifier
- Epoch: fixed 50; test 기반 epoch 선택 없음
- Optimizer: Adam, LR `1e-4`, weight decay `5e-4`
- Scheduler: cosine (`1e-4` → `1e-6`), gradient clipping 1.0
- Checkpoint: 5 epoch 간격
- Seed: 3
- 원본 결과: `/media/NAS/nas_175/seojun/DMMR/target_session1_adapt_cnn_50ep`

| Subject | Before adaptation | After 50 epochs | Change |
|---:|---:|---:|---:|
| 01 | 56.67% | 63.33% | +6.67%p |
| 02 | 76.67% | 83.33% | +6.67%p |
| 03 | 86.67% | 90.00% | +3.33%p |
| 04 | 66.67% | 63.33% | -3.33%p |
| 05 | 83.33% | 73.33% | -10.00%p |
| 06 | 70.00% | 86.67% | +16.67%p |
| 07 | 70.00% | 66.67% | -3.33%p |
| 08 | 76.67% | 96.67% | +20.00%p |
| 09 | 73.33% | 80.00% | +6.67%p |
| 10 | 63.33% | 56.67% | -6.67%p |
| 11 | 90.00% | 86.67% | -3.33%p |
| 12 | 50.00% | 66.67% | +16.67%p |
| 13 | 70.00% | 80.00% | +10.00%p |
| 14 | 66.67% | 70.00% | +3.33%p |
| 15 | 83.33% | 93.33% | +10.00%p |
| **Mean** | **72.22%** | **77.11%** | **+4.89%p** |

추가 집계:

- Trial macro-F1: 71.34% → 76.02% (`+4.69%p`)
- Sample accuracy: 70.95% → 74.41% (`+3.46%p`)
- Session 2 trial accuracy: 71.56% → 78.22% (`+6.67%p`)
- Session 3 trial accuracy: 72.89% → 76.00% (`+3.11%p`)
- 피험자별: 10명 개선, 5명 하락
- 모든 피험자의 adaptation-session trial accuracy는 final epoch에서 100%였다.

Target session 1의 라벨을 사용하는 supervised calibration이므로 이 결과는 완전한 unseen-subject LOSO 성능과 다른 설정이다. Session 2·3에 대한 cross-session personalization 성능으로 해석해야 한다. 평균 개선은 명확하지만 session 1에는 매우 빠르게 과적합했으며 5명의 성능은 하락했다. 따라서 50 epoch full adaptation을 최종 프로토콜로 확정하기 전에 head-only, 더 낮은 LR, 짧은 고정 epoch를 비교할 필요가 있다.

### 10-epoch 짧은 adaptation 비교

동일 설정에서 cosine scheduler의 전체 길이만 10 epoch로 줄여 별도로 15 folds를 실행했다. 결과는 `/media/NAS/nas_175/seojun/DMMR/target_session1_adapt_cnn_10ep`에 저장했다.

| Adaptation | Trial Acc. | Trial Macro-F1 | Sample Acc. | 개선/동률/하락 피험자 |
|---:|---:|---:|---:|---:|
| 없음 | 72.22% | 71.34% | 70.95% | – |
| 10 epochs | 76.00% | 75.15% | 73.80% | 8 / 2 / 5 |
| 50 epochs | **77.11%** | **76.02%** | **74.41%** | 10 / 0 / 5 |
| 100 epochs | 76.22% | 74.94% | **74.55%** | 9 / 1 / 5 |

10 epoch에서도 baseline보다 trial accuracy가 `+3.78%p` 높았다. Session 2는 `+5.33%p`, session 3은 `+2.22%p` 개선됐다. 50 epoch보다 평균 trial accuracy가 1.11%p 낮지만 계산량은 1/5이며, 피험자 01·02·06·12·15에서는 50 epoch와 같은 trial accuracy를 냈다. Subject 05·10·13은 오히려 10 epoch가 50 epoch보다 높았다. 고정된 하나의 epoch를 택해야 한다면 절대 성능은 50 epoch, 효율과 과적합 억제는 10 epoch가 유리하다.

### 100-epoch adaptation 비교

동일 설정을 100 epoch로 늘린 결과는 `/media/NAS/nas_175/seojun/DMMR/target_session1_adapt_cnn_100ep`에 저장했다. Trial accuracy는 76.22%로 baseline보다 `+4.00%p` 높았지만, 50 epoch보다 `-0.89%p` 낮았다. Session 2는 77.78% (`+6.22%p`), session 3은 74.67% (`+1.78%p`)였다. Sample accuracy는 74.55%로 세 설정 중 가장 높았지만, 공식 비교 지표인 trial accuracy와 trial macro-F1은 50 epoch가 가장 높았다. 특히 subject 05는 50 epoch 73.33%에서 100 epoch 63.33%로 하락했다. 따라서 현재 세 고정 길이 중에는 50 epoch를 채택하는 것이 타당하다.

## LSTM target session-1 supervised adaptation 결과

`val_9trials_loso_200ep`의 피험자별 best-validation LSTM checkpoint에서 새로 시작했다. 각 target 피험자의 session 1 전체 15 trials로 attention, LSTM encoder, classifier를 supervised adaptation하고, 분리된 session 2·3의 30 trials에서 고정 final epoch를 평가했다. Optimizer, LR, weight decay, cosine scheduler, gradient clipping, seed는 CNN adaptation과 동일하다.

| Adaptation | Trial Acc. | Trial Macro-F1 | 정답 Trials | 개선/동률/하락 피험자 |
|---:|---:|---:|---:|---:|
| 없음 | 75.78% | 74.07% | 341 / 450 | – |
| 50 epochs | **79.11%** | **78.20%** | **356 / 450** | 8 / 2 / 5 |
| 100 epochs | **79.11%** | 78.02% | **356 / 450** | 9 / 2 / 4 |

- 50-epoch 결과: `/media/NAS/nas_175/seojun/DMMR/target_session1_adapt_lstm_50ep`
- 100-epoch 결과: `/media/NAS/nas_175/seojun/DMMR/target_session1_adapt_lstm_100ep`
- 두 실험은 모두 원본 LSTM checkpoint에서 독립적으로 시작했다. 100 epoch 실험은 50 epoch checkpoint의 연장이 아니다.
- 100 epoch는 정확도 이득이 없고 macro-F1이 0.18%p 낮으므로 현재 고정 길이 중 50 epoch가 더 효율적이다.

### LSTM 50-epoch run의 사후 checkpoint 곡선

아래 값은 5 epoch 간격 checkpoint를 session 2·3 test에 사후 평가한 분석용 곡선이다. Test 기반 epoch 선택에 사용하면 안 되며, 공식 수치는 사전에 고정한 epoch 50 결과다.

| Epoch | Trial Acc. | Trial Macro-F1 | 정답 Trials |
|---:|---:|---:|---:|
| 0 | 75.78% | 74.07% | 341 / 450 |
| 5 | 74.67% | 71.88% | 336 / 450 |
| 10 | 77.11% | 75.59% | 347 / 450 |
| 15 | 77.11% | 75.57% | 347 / 450 |
| 20 | 78.44% | 77.11% | 353 / 450 |
| 25 | 78.22% | 76.89% | 352 / 450 |
| 30 | 78.67% | 77.57% | 354 / 450 |
| 35 | 79.11% | 78.17% | 356 / 450 |
| 40 | 78.67% | 77.68% | 354 / 450 |
| 45 | 79.11% | 78.20% | 356 / 450 |
| 50 | 79.11% | 78.20% | 356 / 450 |

## CNN과 LSTM complexity

30-step 입력에서 분류/adaptation에 실제 사용되는 attention + encoder + classifier 기준이다. MAC은 matrix multiplication과 convolution의 곱-누산만 센 근사치이며 softmax, batch normalization, activation은 제외했다.

| Encoder | 활성 파라미터 | 추론 MACs/sample | Pretraining 전체 파라미터 | Finetuning 객체 전체 파라미터 |
|---|---:|---:|---:|---:|
| LSTM | 197,085 | 약 5.76M | 1,823,260 | 3,457,363 |
| Temporal CNN | 193,629 | 약 5.42M | 1,819,804 | 3,453,907 |

CNN은 분류 경로에서 LSTM보다 파라미터가 약 1.75%, 근사 MAC이 약 5.9% 적다. 두 모델의 크기가 거의 같으므로 단일 seed에서 성능 차이가 크지 않은 것은 자연스럽다. 현재 target-session adaptation에서는 LSTM 50 epoch가 CNN 50 epoch보다 trial accuracy 기준 2.00%p 높지만, 서로 다른 기반 validation split에서 출발했으므로 encoder 자체의 우열로만 해석하면 안 된다.

## 과제 보고 범위와 후속 실험

- 현재까지 완료한 데이터/태스크는 SEED 기반 3-class 감정인식이다.
- OpenBMI 기반 의도 파악 태스크는 이 DMMR 실험 범위에 포함되지 않았고 후속 담당자가 이어서 진행해야 한다.
- 당해연도 정리 기준은 target subject의 session 1을 사용한 supervised adaptation과 session 2·3 평가다.
- REVE 모델로 얻은 설정/수치는 당해연도 보고에 사용하고, 본 문서의 DE 기반 추가 결과는 후속연도 연구 자료로 분리한다.
- Regularization 우선 비교안은 동일한 LSTM 50-epoch 고정 프로토콜에서 `(1) head-only`, `(2) full LR 3e-5`, `(3) full weight decay 1e-3`, `(4) 현재 full LR 1e-4 기준선`이다. 어떤 설정도 session 2·3 test 성능으로 epoch나 hyperparameter를 선택해서는 안 된다.
