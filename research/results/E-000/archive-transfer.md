# E-000 Official Archive Transfer

- Completed: `2026-08-04T03:06:45+08:00`
- Transfer screen: `51242.gdros-e000-transfer-v4`
- Script: `/Users/shihao/Documents/图像匹配之二/tools/transfer_gdros_e000_archive_parallel.sh`
- Script SHA256: `94329a3bfe98b1aa217f5ae991234f33d53d48d6ec4f0c9391cdb4cc5a8b97b1`
- Log: `/Users/shihao/Documents/图像匹配之二/tmp/e000-transfer-parallel.log`

## Result

- Source: `/Users/shihao/Downloads/os_dataset/testing.zip`
- Destination: `/mnt/sda/zhushihao/VS_Match/datasets/official_archives/gdros/testing.zip`
- Size: `1,973,635,350` bytes
- SHA256: `e9c29d6459aa390677066e2da2cdd5927deeb4c9685a10d41c7b6ce44ad81218`
- ZIP test: passed independently after publication
- Parts: `236/236` verified; 20 reused and 216 uploaded
- Upload attempts: 217; one transport interruption recovered on the second attempt
- Verification failures: 0
- Exhausted retries: 0
- Final state: screen/processes exited, local and remote locks released, no staging or `.uploading`
  files remain

The old invalid canonical archive was retained at
`/mnt/sda/zhushihao/VS_Match/datasets/official_archives/gdros/testing.zip.incomplete-20260804T030509+0800-7f261a36`.
It is `2,686,976` bytes with SHA256
`7f261a363bc8956b58d2bdc529e283b1352d26c15a090bdd1c6b5c6e9b6f52ba`.
The old sidecar was isolated as
`testing.zip.part.aria2.incomplete-20260804T030537+0800`.

## Next stage

Formal extraction and publication were submitted to watchdog as job
`20260804_030910_dcbf6bef`. Transfer completion alone does not mark E-000 complete.
