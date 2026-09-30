#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 OUTPUT_DIR" >&2
    exit 2
fi

repo_root="/mnt/sda/zhushihao/VS_Match/GDROS"
python_bin="/mnt/sda/zhushihao/VS_Match/.conda/gdros/bin/python"
dataset_root="/mnt/sda/zhushihao/VS_Match/datasets/extracted/gdros_official_e000_archive"
checkpoint="${repo_root}/checkpoints/GDROS-os512.pth"
provenance="${dataset_root}/E000_PROVENANCE.json"
reference="${repo_root}/experiments/sparsegeo/runs/E-000/run-01/equivalent-precheck/metrics.json"
checker="${repo_root}/experiments/sparsegeo/e000/check_gdros_e000_metrics.py"
provenance_gate="${repo_root}/experiments/sparsegeo/e001/evaluate_sparse_lsr.py"
output_dir="$(realpath -m "$1")"

case "${output_dir}" in
    "${repo_root}"/experiments/sparsegeo/runs/E-000/*)
        ;;
    *)
        echo "output must be under the E-000 run root: ${output_dir}" >&2
        exit 2
        ;;
esac

if [[ -L "${output_dir}" ]]; then
    echo "output path must not be a symlink: ${output_dir}" >&2
    exit 1
fi
if [[ -e "${output_dir}" && ! -d "${output_dir}" ]]; then
    echo "output path exists and is not a directory: ${output_dir}" >&2
    exit 1
fi
if [[ -d "${output_dir}" ]] && [[ -n "$(find "${output_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "refusing to reuse non-empty output directory: ${output_dir}" >&2
    exit 1
fi
if [[ ! -f "${provenance}" ]]; then
    echo "official E-000 provenance is missing: ${provenance}" >&2
    exit 1
fi

mkdir -p "${output_dir}"

test_command=(
    "${python_bin}"
    "${repo_root}/test.py"
    --path "${dataset_root}"
    --dstype testing
    --model "${checkpoint}"
    --Npoint 200000
    --seed 1234
    --log_every 25
    --output "${output_dir}/metrics.json"
)
check_command=(
    "${python_bin}"
    "${checker}"
    --metrics "${output_dir}/metrics.json"
    --reference "${reference}"
    --provenance "${provenance}"
    --checkpoint "${checkpoint}"
    --dataset-root "${dataset_root}"
    --repo-root "${repo_root}"
    --max-pair-delta 0.0001
    --output "${output_dir}/validation.json"
)

{
    printf 'test: '
    printf '%q ' "${test_command[@]}"
    printf '\ncheck: '
    printf '%q ' "${check_command[@]}"
    printf '\n'
} >"${output_dir}/command.txt"

{
    date -Iseconds
    hostname
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
    "${python_bin}" --version
    sha256sum \
        "${checkpoint}" \
        "${provenance}" \
        "${reference}" \
        "${checker}" \
        "${provenance_gate}" \
        "${repo_root}/test.py"
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} >"${output_dir}/environment.txt" 2>&1

"${test_command[@]}"
"${check_command[@]}"
