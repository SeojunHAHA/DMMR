# DMMR SEED LOSO reproduction

This workspace contains the reproducible NPZ-based pipeline used for the SEED
three-class emotion-recognition experiments. It supports the original LSTM
encoder, a temporal CNN encoder, validation-controlled LOSO training, and
supervised target-session adaptation.

Paper: [DMMR: Cross-Subject Domain Generalization for EEG-Based Emotion Recognition via Denoising Mixed Mutual Reconstruction](https://ojs.aaai.org/index.php/AAAI/article/view/27819)

## Files

| File | Purpose |
|---|---|
| `model.py` | DMMR attention, LSTM/CNN encoder, reconstruction decoders, classifiers |
| `GradientReverseLayer.py` | Domain-adversarial gradient reversal |
| `reproduction/run_npz_loso.py` | One-fold pretraining/fine-tuning/validation/evaluation runner |
| `reproduction/adapt_target_session.py` | One-fold supervised target-session adaptation runner |
| `reproduction/launchers/*.sh` | 15-fold, eight-GPU launchers and CUDA environment helper |
| `reproduction/summarize_training.py` | Aggregate epoch metrics and generate SVG curves |
| `reproduction/audit_seed_npz.py` | Verify NPZ structure and normalization |
| `environment.yml` | Reproduction conda environment |

## Environment

The NPZ reproduction pipeline uses the environment in `environment.yml`.
The repository's legacy `requirements.txt` describes the original paper code
and is **not** used for the reproduction commands in this document.

The recorded experiments used Linux, Python 3.10.20, PyTorch 2.2.2 with CUDA
12.1 and cuDNN 8.9.2, NumPy 1.26.4, SciPy 1.15.3, and scikit-learn 1.7.2.
`environment.yml` pins the main numerical and training dependencies to these
versions. An NVIDIA GPU is strongly recommended; CPU execution is supported
for checking the pipeline but full LOSO training is impractical on CPU.

### Prerequisites

- Conda or Mamba on a 64-bit Linux system
- An NVIDIA driver that supports the CUDA 12.1 runtime
- Enough storage for the NPZ dataset and experiment checkpoints
- Eight visible GPUs for the supplied `*_8gpu.sh` launchers, or one visible
  GPU for a direct single-fold run

The CUDA runtime is installed inside the Conda environment. A separate system
CUDA toolkit is not required, but `nvidia-smi` must work on the host.

### Create the environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate dmmr
export DMMR_PYTHON="$(command -v python)"
```

For an existing environment, synchronize it with the checked-in specification:

```bash
conda env update --name dmmr --file environment.yml --prune
conda activate dmmr
export DMMR_PYTHON="$(command -v python)"
```

Verify the installation before loading data:

```bash
python -c 'import numpy, scipy, sklearn, torch; print("numpy", numpy.__version__); print("scipy", scipy.__version__); print("sklearn", sklearn.__version__); print("torch", torch.__version__); print("CUDA runtime", torch.version.cuda); print("CUDA available", torch.cuda.is_available()); print("GPU count", torch.cuda.device_count())'
nvidia-smi
```

The expected core versions are NumPy 1.26.4, SciPy 1.15.3,
scikit-learn 1.7.2, PyTorch 2.2.2, and CUDA runtime 12.1. A `False` CUDA result
means GPU training will not work until the driver/device exposure is fixed.

### Configure machine-local paths

Do not edit the launchers for each machine. Override their defaults with these
environment variables:

```bash
export DMMR_DATA_ROOT=/path/to/SEED_DE_MSMDA
export DMMR_OUTPUT_DIR=/path/to/results/experiment_name
export DMMR_PYTHON="$(command -v python)"
```

The launchers source `reproduction/launchers/dmmr_cuda_env.sh` automatically so
the CUDA 12.1 libraries installed with PyTorch take priority over conflicting
host libraries. If the environment keeps its NVIDIA Python packages somewhere
other than the standard Python 3.10 `site-packages/nvidia` directory, set
`DMMR_NVIDIA_ROOT` to that directory.

### Validate with one fold

First audit the transferred dataset as described below. Then run a short
single-subject smoke test before starting all 15 folds:

```bash
python reproduction/run_npz_loso.py \
  --subject 1 --sessions 1 2 3 \
  --data-root "$DMMR_DATA_ROOT" \
  --encoder lstm \
  --pretrain-epochs 1 --finetune-epochs 1 --iteration 1 \
  --checkpoint-every 0 --device cuda:0 \
  --output-dir outputs_smoke
```

For a CPU-only functional check, replace `--device cuda:0` with
`--device cpu`. The full launchers map folds to GPU indices 0 through 7 and
therefore require eight visible GPUs without modification.

## Dataset

The runner expects the following layout. Each NPZ preserves session, subject,
trial, label, and sample boundaries.

```text
SEED_DE_MSMDA/
├── manifest.json
├── session_1/subject_01.npz ... subject_15.npz
├── session_2/subject_01.npz ... subject_15.npz
└── session_3/subject_01.npz ... subject_15.npz
```

Audit the transferred dataset before training:

```bash
python reproduction/audit_seed_npz.py --data-root "$DMMR_DATA_ROOT"
```

## Training

### Validation-free LOSO

```bash
DMMR_ENCODER=lstm \
DMMR_PRETRAIN_EPOCHS=200 \
DMMR_FINETUNE_EPOCHS=200 \
bash reproduction/launchers/run_npz_loso_8gpu.sh
```

Set `DMMR_ENCODER=cnn` for the temporal CNN.

### Nine-trial validation

This reserves three trials per session from every source subject, for nine
validation trials per source subject in each LOSO fold.

```bash
DMMR_ENCODER=lstm \
DMMR_PRETRAIN_EPOCHS=200 \
DMMR_FINETUNE_EPOCHS=200 \
bash reproduction/launchers/run_npz_loso_val_9trials_8gpu.sh
```

### Whole-session validation

```bash
DMMR_ENCODER=cnn \
DMMR_VALIDATION_SESSION=3 \
DMMR_PRETRAIN_EPOCHS=50 \
DMMR_FINETUNE_EPOCHS=100 \
DMMR_SCHEDULER=cosine \
DMMR_GRAD_CLIP=1 \
bash reproduction/launchers/run_npz_loso_val_session3_8gpu.sh
```

### Target session-1 adaptation

The target subject's labeled session 1 is used for supervised calibration;
sessions 2 and 3 are held out for the fixed-final evaluation.

```bash
DMMR_BASE_DIR=/path/to/val_9trials_loso_200ep \
DMMR_OUTPUT_DIR=/path/to/target_session1_adapt_lstm_50ep \
DMMR_ADAPT_EPOCHS=50 \
bash reproduction/launchers/run_target_session1_adapt_8gpu.sh
```

This is a personalization/cross-session protocol, not fully unseen-subject
LOSO. Do not select adaptation epochs using sessions 2/3 test accuracy.

## Reproducibility artifacts

New runs save per-epoch metrics and full checkpoints every five epochs by
default. Each `subject_XX_artifacts/` contains:

- `epoch_metrics.jsonl` and `epoch_metrics.csv`
- `checkpoints/*_epoch_XXXX.pt` with model, optimizer, scheduler, and RNG state
- `run_metadata.json` with arguments, versions, hashes, and device information
- `source_snapshot/` with the exact runner/model source
- `reproduce_command.sh`
- `packages.txt` for LOSO training runs

Completed subject JSON files are skipped when a launcher is resumed. Set
`DMMR_CHECKPOINT_EVERY=0` only when periodic recovery and epoch analysis are
not required.

Generate aggregate logs and curves:

```bash
python reproduction/summarize_training.py /path/to/experiment_output
```

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
