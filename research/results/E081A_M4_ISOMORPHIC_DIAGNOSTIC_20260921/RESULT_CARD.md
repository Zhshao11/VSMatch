# E081A-M4 WHU isomorphic diagnostic

Status: `COMPLETE_DIAGNOSTIC_ONLY`

700-pair iter32 M4 post-LSR: AEPE `1.905514`, CMR@1/2/5/10/16 `28.0000/72.0000/98.1429/99.5714/99.5714`.

Final >=5 px failures: `13`. Predictions and traces were fully materialized before the separate GT join.

E081A-M4 contains no F2/F3 or B/C branches. The available exact decomposition is backbone -> region refiner -> LSR; absent branch values were not inferred.

No training, new module, checkpoint selection, or LuckyR2 run was performed.
