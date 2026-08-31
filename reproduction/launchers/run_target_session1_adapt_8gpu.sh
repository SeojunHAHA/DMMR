#!/usr/bin/env bash
set -euo pipefail
launcher_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${launcher_dir}/../.." && pwd)"
cd "${project_root}"
source "${launcher_dir}/dmmr_cuda_env.sh"
python_bin="${DMMR_PYTHON:-/home/seojun/.conda/envs/dmmr/bin/python}"
data_root="${DMMR_DATA_ROOT:-/media/NAS/nas_175/seojun/SEED_DE_MSMDA}"
base_dir="${DMMR_BASE_DIR:-/media/NAS/nas_175/seojun/DMMR/val_session3_cnn_pre50_ft100_cosine}"
output_dir="${DMMR_OUTPUT_DIR:-/media/NAS/nas_175/seojun/DMMR/target_session1_adapt_cnn_50ep}"
adapt_epochs="${DMMR_ADAPT_EPOCHS:-50}"
mkdir -p "${output_dir}"

run_subject() {
  local gpu="$1" subject="$2"
  test -s "${output_dir}/subject_$(printf '%02d' "${subject}").json" && return 0
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=1 "${python_bin}" -u reproduction/adapt_target_session.py \
    --subject "${subject}" --base-dir "${base_dir}" --data-root "${data_root}" \
    --adapt-session 1 --test-sessions 2 3 \
    --adapt-epochs "${adapt_epochs}" --batch-size 512 --lr 1e-4 --scheduler cosine \
    --grad-clip 1 --adapt-scope full --checkpoint-every 5 \
    --device cuda:0 --output-dir "${output_dir}"
}

pids=()
for assignment in "0 1" "1 2" "2 3" "3 4" "4 5" "5 6" "6 7" "7 8" \
                  "0 9" "1 10" "2 11" "3 12" "4 13" "5 14" "6 15"; do
  read -r gpu subject <<< "${assignment}"
  run_subject "${gpu}" "${subject}" & pids+=("$!")
done
wait "${pids[@]}"
