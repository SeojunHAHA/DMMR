#!/usr/bin/env bash
# Prefer the CUDA 12.1 runtime bundled with the dmmr PyTorch environment over
# the host CUDA 12.4 libraries. Source this file before direct Python runs.
dmmr_python_bin="${DMMR_PYTHON:-/home/seojun/.conda/envs/dmmr/bin/python}"
dmmr_env_root="$(dirname "$(dirname "${dmmr_python_bin}")")"
dmmr_nvidia_root="${DMMR_NVIDIA_ROOT:-${dmmr_env_root}/lib/python3.10/site-packages/nvidia}"
dmmr_cuda_lib_paths=""
for dmmr_cuda_lib_dir in "${dmmr_nvidia_root}"/*/lib; do
  dmmr_cuda_lib_paths="${dmmr_cuda_lib_paths:+${dmmr_cuda_lib_paths}:}${dmmr_cuda_lib_dir}"
done
export LD_LIBRARY_PATH="${dmmr_cuda_lib_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
unset dmmr_python_bin dmmr_env_root dmmr_nvidia_root dmmr_cuda_lib_paths dmmr_cuda_lib_dir
