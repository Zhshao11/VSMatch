# E-006A P0 Method Review

Status: `APPROVE`

- No-GT region matching generates the eight method affine basins before oracle access.
- The live global, region-restricted and per-basin affine-local candidate paths are executed.
- Raw truth flow is read only after all method candidates and live budget checks are complete; P0 GT only selects a basin/candidate and local oracle offset.
- The discrete-center 8 px supervision radius and per-axis `[-4,4] px` offset are enforced.
- Optical-to-SAR direction, target-minus-source flow, token/pixel scale, full-image no-mask metrics and E-004A M256 parity are correct.
- Remote automatic tests passed `12/12` after the final evidence patch.

Reviewer scope: `p0.py`, candidate/region/proposal/geometry modules, core tests, and `p0-smoke-v3.json`.
