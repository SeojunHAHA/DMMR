# DMMR

AAAI 2024 논문 [DMMR](https://ojs.aaai.org/index.php/AAAI/article/view/27819)의 공식 구현을 기반으로 한 SEED 재현 저장소다. Session-1/전체 session LOSO, LSTM/CNN, validation 및 target session adaptation을 지원한다.

## 환경과 데이터

```bash
conda env create -f environment.yml
conda activate dmmr
export DMMR_PYTHON="$(command -v python)"
```

기본 데이터 위치는 `/media/NAS/nas_175/seojun/SEED_DE_MSMDA`다.

```bash
python reproduction/audit_seed_npz.py \
  --data-root /media/NAS/nas_175/seojun/SEED_DE_MSMDA
```

## 실행

제공된 launcher는 15 folds를 GPU 8개에 분배한다.

```bash
# 전체 session LOSO
bash reproduction/launchers/run_npz_loso_8gpu.sh

# 9-trial validation LOSO
bash reproduction/launchers/run_npz_loso_val_9trials_8gpu.sh

# Session-3 validation LOSO
bash reproduction/launchers/run_npz_loso_val_session3_8gpu.sh

# Target session-1 adaptation
bash reproduction/launchers/run_target_session1_adapt_8gpu.sh
```

경로와 설정은 `DMMR_DATA_ROOT`, `DMMR_OUTPUT_DIR`, `DMMR_BASE_DIR`, `DMMR_ENCODER` 등의 환경변수로 덮어쓸 수 있다.

상세 설치, 단일-fold smoke test와 실험 옵션은 [REPRODUCTION.md](REPRODUCTION.md), 전체 결과는 [결과 문서](docs/RESULTS_ALL_SESSIONS_LOSO_200EP.md)를 참고한다.

## Citation

```bibtex
@inproceedings{wang2024dmmr,
  title={DMMR: Cross-Subject Domain Generalization for EEG-Based Emotion Recognition via Denoising Mixed Mutual Reconstruction},
  author={Wang, Yiming and Zhang, Bin and Tang, Yujiao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={628--636},
  year={2024}
}
```
