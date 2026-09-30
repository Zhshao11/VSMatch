#!/usr/bin/env bash
set -euo pipefail

archive="/mnt/sda/zhushihao/VS_Match/datasets/official_archives/gdros/testing.zip"
extract_parent="/mnt/sda/zhushihao/VS_Match/datasets/extracted"
target="${extract_parent}/gdros_official_e000_archive"
staging="${extract_parent}/.gdros_official_e000_archive.extracting-e9c29d64"
comparison="${extract_parent}/gdros_official"
lock_dir="${extract_parent}/.gdros-e000-extract.lock"
validator="/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e000/validate_gdros_e000_dataset.py"
python_bin="/mnt/sda/zhushihao/VS_Match/.conda/gdros/bin/python"
resume_existing=false

if [[ "$#" -eq 1 && "$1" == "--resume-existing-staging" ]]; then
    resume_existing=true
elif [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--resume-existing-staging]" >&2
    exit 2
fi

if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "extraction lock exists: ${lock_dir}" >&2
    exit 1
fi

cleanup() {
    rmdir "${lock_dir}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -e "${target}" || -L "${target}" ]]; then
    echo "refusing to overwrite existing target: ${target}" >&2
    exit 1
fi
if [[ ! -f "${archive}" ]]; then
    echo "official archive is missing: ${archive}" >&2
    exit 1
fi

"${python_bin}" "${validator}" --archive "${archive}" --archive-only
if [[ "${resume_existing}" == "true" ]]; then
    if [[ -L "${staging}" || ! -d "${staging}" ]]; then
        echo "resume requires the audited extraction staging directory: ${staging}" >&2
        exit 1
    fi
    if [[ -e "${staging}/E000_PROVENANCE.json" || -L "${staging}/E000_PROVENANCE.json" ]]; then
        echo "resume staging contains provenance and requires a separate audit: ${staging}" >&2
        exit 1
    fi
    echo "resuming audited extraction staging: ${staging}"
else
    if [[ -e "${staging}" || -L "${staging}" ]]; then
        echo "staging already exists and requires manual audit: ${staging}" >&2
        exit 1
    fi
    mkdir "${staging}"
    unzip -q "${archive}" -d "${staging}"
fi
"${python_bin}" "${validator}" \
    --archive "${archive}" \
    --dataset-root "${staging}" \
    --published-root "${target}" \
    --comparison-root "${comparison}" \
    --output "${staging}/E000_PROVENANCE.json"

mv -T -n "${staging}" "${target}"
if [[ -e "${staging}" || -L "${staging}" ]]; then
    echo "publication race detected; staging was not atomically renamed: ${staging}" >&2
    exit 1
fi
"${python_bin}" "${validator}" \
    --archive "${archive}" \
    --dataset-root "${target}" \
    --published-root "${target}" \
    --comparison-root "${comparison}" \
    --output "${target}/E000_PROVENANCE.json"
sha256sum "${target}/E000_PROVENANCE.json"
echo "official GDROS E-000 dataset published at ${target}"
