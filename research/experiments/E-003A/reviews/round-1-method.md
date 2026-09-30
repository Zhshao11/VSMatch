# E-003A Review Round 1: Method and Training Safety

Reviewer role: independent method/coordinates/gradient reviewer

Final source manifest: `daaf3fea395180560109d74154bb1078e50958cbaacb337738fe9c1c86518482`

Final config: `325e2059da71b9c802670a89b58a20304b014c758da0ea8db1e0b62c152c5737`

Initial findings and closure:

- Real-checkpoint frozen/no-grad gate was missing. Added `test_frozen_frontend.py`; official GDROS
  checkpoint and a real pair now prove all frontend parameters and outputs are detached. PASS.
- Non-finite solver gate was missing. Added NaN injection and verified finite identity output,
  `solver_valid=false` and structured abstention. PASS.
- Per-epoch learning rate was absent. It is now recorded in training history and verified by real smoke.

The reviewer independently reran `16/16` tests. Optical-to-SAR direction, target-source flow, coordinate
normalization, raw truth-flow supervision, datum isolation, K128/top8/T2/width64, weighted solve,
confidence, optimizer/budget and no N2/hypothesis/fallback requirements match the frozen specification.
Runtime provenance changes do not enter the model or loss path.

APPROVE
