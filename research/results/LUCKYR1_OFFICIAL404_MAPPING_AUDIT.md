# LuckyR1 official-404 mapping audit

Status: `PASS`

The historical E075A-M4 step80K materialized manifest contains 404 unique pair IDs. The raw dataset object exposes 424 items, while the historical evaluator used the ordered first 404.

The first 248 LuckyR1 exploratory-A predictions available during the audit matched the historical manifest pair IDs exactly, with no mismatch. Therefore the explicit `Subset(range(404))` reproduces the established formal evaluation view; the other 20 raw items are not silently included.
