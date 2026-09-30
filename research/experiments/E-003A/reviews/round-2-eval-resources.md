# E-003A Review Round 2: Evaluation, Resources and Provenance

Reviewer role: independent metrics/resources/provenance reviewer

Final source manifest: `daaf3fea395180560109d74154bb1078e50958cbaacb337738fe9c1c86518482`

Final config: `325e2059da71b9c802670a89b58a20304b014c758da0ea8db1e0b62c152c5737`

Initial findings and closure:

- Solver-invalid outputs now remain explicit abstentions and are ranked last for fixed-90% risk.
- Risk-coverage, oracle gap, C8 abstention and fixed post-output diagnostic groups are reported.
- Slim and GDROS are measured sequentially in the same fresh process; five paired processes feed CV and speedup.
- Complexity is observed from exactly four live `[1,128,1024]` score chunks totaling `128x4096`, plus
  an observed `128x8` selector table and no-N2 flag.
- AST recursion closes 41 local source dependencies. Formal entrypoints enforce source/config/split/cache
  identities; train/eval rehash the cache files they consume; checkpoint identity and official GDROS SHA
  are enforced. The negative smoke correctly rejects an old unbound checkpoint.
- Cache validation binds opt/SAR/truth-flow SHA and all label shapes/finite values for 2,011 pairs.

The reviewer confirmed full 512x512 unmasked AEPE/CMR and PairLocalReady semantics and inspected the final
resource/evaluator smoke artifacts. No blocking findings remain.

APPROVE
