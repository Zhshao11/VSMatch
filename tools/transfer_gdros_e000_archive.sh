#!/usr/bin/env bash
set -euo pipefail

transfer_log="/Users/shihao/Documents/图像匹配之二/tmp/e000-transfer.log"
exec >>"${transfer_log}" 2>&1

local_parts_dir="/Users/shihao/Documents/图像匹配之二/tmp/e000-transfer-parts-8m-e9c29d64"
local_lock_dir="/Users/shihao/Documents/图像匹配之二/tmp/e000-transfer-e9c29d64.lock"
remote_host="bl_tunnel"
remote_dir="/mnt/sda/zhushihao/VS_Match/datasets/official_archives/gdros"
remote_parts_dir="${remote_dir}/e000-upload-parts-e9c29d64"
remote_lock_dir="${remote_dir}/.e000-transfer-e9c29d64.lock"
remote_parts_manifest="${remote_parts_dir}/PARTS-8m-e9c29d64.txt"
remote_stage="${remote_dir}/testing.zip.e9c29d64.uploading"
remote_final="${remote_dir}/testing.zip"
local_parts_manifest="${local_parts_dir}/PARTS.txt"
expected_archive_sha="e9c29d6459aa390677066e2da2cdd5927deeb4c9685a10d41c7b6ce44ad81218"
expected_archive_size="1973635350"
max_part_attempts=5

if ! mkdir "${local_lock_dir}" 2>/dev/null; then
    echo "local transfer lock exists: ${local_lock_dir}" >&2
    exit 1
fi

remote_lock_acquired=false
cleanup() {
    if [[ "${remote_lock_acquired}" == "true" ]]; then
        ssh "${remote_host}" rmdir "${remote_lock_dir}" 2>/dev/null || true
    fi
    rmdir "${local_lock_dir}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! ssh "${remote_host}" mkdir "${remote_lock_dir}"; then
    echo "remote transfer lock exists: ${remote_lock_dir}" >&2
    exit 1
fi
remote_lock_acquired=true

echo "transfer started at $(date -Iseconds)"
ssh "${remote_host}" mkdir -p "${remote_parts_dir}"

existing_final_size="$(ssh "${remote_host}" stat -c '%s' "${remote_final}" 2>/dev/null || true)"
existing_final_sha=""
if [[ "${existing_final_size}" == "${expected_archive_size}" ]]; then
    existing_final_sha="$(ssh "${remote_host}" sha256sum "${remote_final}" | awk '{print $1}')"
fi
if [[ "${existing_final_sha}" == "${expected_archive_sha}" ]]; then
    ssh "${remote_host}" unzip -tq "${remote_final}"
    echo "verified official archive already present"
    exit 0
fi

part_names=()
for local_part in "${local_parts_dir}"/testing.zip.part-*; do
    part_name="${local_part##*/}"
    part_names+=("${part_name}")
    remote_part="${remote_parts_dir}/${part_name}"
    remote_part_stage="${remote_part}.uploading"
    local_size="$(stat -f '%z' "${local_part}")"
    local_sha="$(shasum -a 256 "${local_part}" | awk '{print $1}')"

    remote_size="$(ssh "${remote_host}" stat -c '%s' "${remote_part}" 2>/dev/null || true)"
    remote_sha=""
    if [[ "${remote_size}" == "${local_size}" ]]; then
        remote_sha="$(ssh "${remote_host}" sha256sum "${remote_part}" | awk '{print $1}')"
    fi

    if [[ "${remote_sha}" == "${local_sha}" ]]; then
        echo "verified existing part: ${part_name}"
        continue
    fi

    part_verified=false
    for ((attempt = 1; attempt <= max_part_attempts; attempt++)); do
        echo "uploading part: ${part_name} (${local_size} bytes), attempt ${attempt}"
        if ! scp -O "${local_part}" "${remote_host}:${remote_part_stage}"; then
            echo "part transport interrupted: ${part_name}, attempt ${attempt}" >&2
            continue
        fi

        uploaded_size="$(ssh "${remote_host}" stat -c '%s' "${remote_part_stage}")"
        uploaded_sha="$(ssh "${remote_host}" sha256sum "${remote_part_stage}" | awk '{print $1}')"
        if [[ "${uploaded_size}" == "${local_size}" && "${uploaded_sha}" == "${local_sha}" ]]; then
            ssh "${remote_host}" mv "${remote_part_stage}" "${remote_part}"
            part_verified=true
            echo "verified uploaded part: ${part_name}"
            break
        fi
        echo "part verification failed: ${part_name}, attempt ${attempt}" >&2
    done

    if [[ "${part_verified}" != "true" ]]; then
        echo "exhausted retries for part: ${part_name}" >&2
        exit 1
    fi
done

printf '%s\n' "${part_names[@]}" >"${local_parts_manifest}"
scp -O "${local_parts_manifest}" "${remote_host}:${remote_parts_manifest}.uploading"
ssh "${remote_host}" mv "${remote_parts_manifest}.uploading" "${remote_parts_manifest}"

echo "assembling remote archive"
ssh "${remote_host}" "cd '${remote_parts_dir}' && xargs -a '${remote_parts_manifest}' cat -- > '${remote_stage}'"

archive_size="$(ssh "${remote_host}" stat -c '%s' "${remote_stage}")"
archive_sha="$(ssh "${remote_host}" sha256sum "${remote_stage}" | awk '{print $1}')"
if [[ "${archive_size}" != "${expected_archive_size}" || "${archive_sha}" != "${expected_archive_sha}" ]]; then
    echo "archive verification failed: size=${archive_size} sha256=${archive_sha}" >&2
    exit 1
fi

ssh "${remote_host}" unzip -tq "${remote_stage}"

if ssh "${remote_host}" test -e "${remote_final}"; then
    existing_sha="$(ssh "${remote_host}" sha256sum "${remote_final}" | awk '{print $1}')"
    if [[ "${existing_sha}" != "${expected_archive_sha}" ]]; then
        backup_time="$(ssh "${remote_host}" date '+%Y%m%dT%H%M%S%z')"
        backup_path="${remote_final}.incomplete-${backup_time}-${existing_sha:0:8}"
        ssh "${remote_host}" ln "${remote_final}" "${backup_path}"
        echo "preserved old incomplete archive: ${backup_path}"
    fi
fi

for sidecar in "${remote_dir}/testing.zip.aria2" "${remote_dir}/testing.zip.part.aria2"; do
    if ssh "${remote_host}" test -e "${sidecar}"; then
        sidecar_time="$(ssh "${remote_host}" date '+%Y%m%dT%H%M%S%z')"
        ssh "${remote_host}" mv "${sidecar}" "${sidecar}.incomplete-${sidecar_time}"
    fi
done

ssh "${remote_host}" mv -f "${remote_stage}" "${remote_final}"
published_size="$(ssh "${remote_host}" stat -c '%s' "${remote_final}")"
published_sha="$(ssh "${remote_host}" sha256sum "${remote_final}" | awk '{print $1}')"
if [[ "${published_size}" != "${expected_archive_size}" || "${published_sha}" != "${expected_archive_sha}" ]]; then
    echo "post-publish verification failed: size=${published_size} sha256=${published_sha}" >&2
    exit 1
fi
ssh "${remote_host}" unzip -tq "${remote_final}"
ssh "${remote_host}" sha256sum "${remote_final}"
echo "official GDROS archive transfer completed"
