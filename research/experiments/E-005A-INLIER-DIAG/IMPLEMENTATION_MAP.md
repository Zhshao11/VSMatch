# E-005A-INLIER-DIAG Implementation Map

- Frozen identity and 404 dev isolation -> `diagnostic.validate_provenance`, runtime SHA/cache/split/checkpoint checks.
- Public-200k `[10000:210000]`, full-512, matched-M256 -> `core.public_200k_indices`, `diagnostic.run`, threshold tests.
- Optical source to SAR target and target-source flow -> endpoint construction in `diagnostic.run`, affine direction unit test.
- GT/self-consensus raw and weighted ratios at 1/2/5/8/16 px -> `core.threshold_rows`, `core.aggregate_ratio_rows`.
- Source-only FPS with proposal-ID tie break -> `core.farthest_point_indices`, deterministic tie test.
- Exact K/rho and SHA-prefix seeds -> `core.controlled_observations`, exact-count and seed tests.
- Equal-weight and E-005A-weight FP32 ordinary affine -> `core.solve_pixel_affine_batch`, scalar/batch affine tests.
- Full-image no-mask AEPE and CMR curve aggregation -> `core.matrix_aepe_batch`, `diagnostic.CurveAccumulator`.
- Final active K128 coordinate/weight decomposition -> `diagnostic._oracle_rows`, explicit binary validity gates.
- GT isolation/no training -> output-before-`read_flow` ordering and source audit test.
- Raw CSV, summary JSON, PNG/PDF consistency -> `plotting.py`, artifact generation test and final PDF rendering QA.

The diagnostic uses NumPy PCG64 for slot permutation and with-replacement sampling. The seed is the first 64 bits, unsigned big-endian, of `sha256(pair_id|K|rho_two_decimals|rep_id)`.
