# E-003A Corrected Oracle Audit Review

The original `top8-gt-oracle` diagnostic incorrectly assigned a nearest candidate to all anchors, including
anchors whose frozen label was `recoverable=false`. This affected only the oracle summary and oracle gap;
learned-main, unary, C8, soft-EM, coverage and selective-risk were computed independently and are unchanged.

The result-level audit script SHA is
`31feaf9394e463b32930b5b18e84bcdab9e766e62bbaed77413af5ccc68154a8`. It assigns unit weight to the
nearest top8 candidate only for `recoverable=true` anchors and zero weight otherwise, never loads datum,
and runs only after formal method outputs. It reuses the full 512x512 unmasked evaluator.

Independent checks: 404 unique pairs, 404 valid solvers, all values finite, recoverable-rank histogram sum
18,060 equals the total recoverable-anchor count. Corrected oracle AEPE is `2.8409765 px`, PairLocalReady
is `100%`, grid mean is `2.9588739 px` and CMR@5 is `95.5446%`.

APPROVE
