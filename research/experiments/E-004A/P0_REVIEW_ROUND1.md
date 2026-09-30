# E-004A P0 Review Round 1

Verdict: `CHANGES_REQUIRED`

1. Evaluator trusted candidate/label files without binding pair, split, current source, checkpoint and
   input/truth-flow identities. Fixed with per-pair strong validation and a content-hashed cache manifest.
2. Importing the E-003A metric helper transitively loaded consensus/triple-bank/reranker modules. Fixed by
   moving the frozen metric equations into dependency-isolated `e004a/evaluator.py` and adding a runtime
   negative import test.
3. Smoke evidence predated the reviewed source manifest. Fixed by rebuilding both artifacts; build and
   parity now record the same current source SHA.

Targeted fix rereview is required before the full dev cache may start.
