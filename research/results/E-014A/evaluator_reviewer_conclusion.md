# E-014A Evaluator / Protocol Reviewer

Verdict: **APPROVE**

Blocking findings: none.

Evidence:

- P0 gate is enforced before interpretation; P0 fail maps to `COMPLETED / IMPLEMENTATION_BLOCKED`.
- E-011A top128 parity checks mean/P10/P50/P90/oracle AEPE.
- `materialize_scores_before_label_join()` computes and hashes all signal scores before labels/truth_flow are loaded.
- No training, finetune, checkpoint write, baseline overwrite, E-014B/rescue, datum, OS424, or confirm path was found.
- AUROC/AP/AUPRC/Brier/ECE and topK/quota metrics use endpoint hard-positive labels and valid masks.
- Random score, shuffled target, and swapped-direction controls are present.

Non-blocking note:

- Raw SAR patch semantics were corrected before approval: raw_patch now uses raw grayscale; normalized_patch applies log-SAR robust z-score.
