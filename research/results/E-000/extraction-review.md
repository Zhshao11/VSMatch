# E-000 Archive Publication Review

Date: `2026-08-04`

Status: `APPROVED FOR EXECUTION`

## Reviewed Artifacts

- Local validator: `tools/validate_gdros_e000_dataset.py`
- Remote validator SHA256:
  `5995d7a492083991fcce2bf869d4d725f2df17756759a263034d1918c9c6d408`
- Local extractor: `tools/extract_gdros_e000_archive.sh`
- Remote extractor SHA256:
  `87a8759d9bdb267568fec16428a28dc3d8b3a4a86279d3320510f6a5f7e5bf63`

## Closed Findings

1. Final provenance no longer points only to the pre-publication staging path. The archive
   is fully validated before publication and the published target is fully validated again.
2. Category manifests now cover exactly five directories with 424 files each instead of
   accidentally producing per-file categories.
3. A post-publication rerun accepts only the root `E000_PROVENANCE.json` metadata file and
   rejects all other extra files, directories, symlinks, dangling symlinks and special entries.
4. Publication uses `mv -T -n` and verifies that staging disappeared, so a target created in
   the race window is not overwritten or used as a nesting destination.
5. Final provenance includes the official archive SHA/size/CRC/member set, 424 pair count,
   2120 data-file manifest, five category manifests, the published root, post-publication
   flags and the 1272-file evaluation-content comparison.

The independently calculated official archive data manifest is:

`c129ae7147a7d9ad1c1a7ead35bb6da2d8c85ad8da73eef4723cf277e5ea7e96`

Bash syntax and remote Python compilation passed. The independent reviewer found no
remaining execution blocker. No extraction was performed during review.

## Final Resume Review And Outcome

The review above is retained as publication-chain history. The actual resume was executed only
after a later incremental review approved these final hashes:

- Validator: `da6d59041a926eba98bb59bdbc85aa1302c7744bc23bc6b4a58680df2a6ba609`
- Validator tests: `11c6a629e79463a6c5bc1d579088b24b3712912351d2882da29db3986702ef5c`
- Extractor: `e9c17149999e6f5bfcc1a36f457284f9afe8398f543ef7fc718662dc775c7f3e`

The final comparator is fail-closed to the verified PIL `L` and `RGB` modes. A real full staging
validation passed with 1272/1272 loader-equivalent inputs. Raw-byte matches were optical/SAR/FLO
`0/424/424`, recorded separately from logical content equality. Remote validator tests passed
`9/9`; a deliberately different FLO failed with a nonzero exit.

Watchdog job `20260804_035335_8b16c356` completed with return code 0. The published provenance is
bound to the target root, has SHA256
`bff6f565e9a8a12bb2675d54b97fccaeb3a88d39b49ae56a1571c57d8489aad9`, and records the pinned
2120-file manifest `c129ae7147a7d9ad1c1a7ead35bb6da2d8c85ad8da73eef4723cf277e5ea7e96`.
