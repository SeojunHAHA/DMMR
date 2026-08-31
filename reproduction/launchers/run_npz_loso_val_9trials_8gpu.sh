#!/usr/bin/env bash
set -euo pipefail
launcher_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${launcher_dir}/../.." && pwd)"
cd "${project_root}"
source "${launcher_dir}/dmmr_cuda_env.sh"
python_bin="${DMMR_PYTHON:-/home/seojun/.conda/envs/dmmr/bin/python}"
data_root="${DMMR_DATA_ROOT:-/media/NAS/nas_175/seojun/SEED_DE_MSMDA}"
encoder="${DMMR_ENCODER:-lstm}"
pretrain_epochs="${DMMR_PRETRAIN_EPOCHS:-200}"
finetune_epochs="${DMMR_FINETUNE_EPOCHS:-200}"
scheduler="${DMMR_SCHEDULER:-none}"
grad_clip="${DMMR_GRAD_CLIP:-0}"
checkpoint_every="${DMMR_CHECKPOINT_EVERY:-5}"
output_dir="${DMMR_OUTPUT_DIR:-/media/NAS/nas_175/seojun/DMMR/val_9trials_${encoder}_loso_200ep}"
mkdir -p "${output_dir}"

run_subject() {
  local gpu="$1" subject="$2"
  test -s "${output_dir}/subject_$(printf '%02d' "${subject}").json" && return 0
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=1 "${python_bin}" -u reproduction/run_npz_loso.py \
    --subject "${subject}" --sessions 1 2 3 --data-root "${data_root}" \
    --validation-trials-per-class 3 --encoder "${encoder}" \
      --pretrain-epochs "${pretrain_epochs}" --finetune-epochs "${finetune_epochs}" \
      --scheduler "${scheduler}" --grad-clip "${grad_clip}" \
    --checkpoint-every "${checkpoint_every}" --device cuda:0 --output-dir "${output_dir}"
}

pids=()
for assignment in "0 1" "1 2" "2 3" "3 4" "4 5" "5 6" "6 7" "7 8" \
                  "0 9" "1 10" "2 11" "3 12" "4 13" "5 14" "6 15"; do
  read -r gpu subject <<< "${assignment}"
  run_subject "${gpu}" "${subject}" & pids+=("$!")
done
wait "${pids[@]}"
