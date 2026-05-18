# Group 08 Audit (current 41/42)

**Verdict:** OK — score defensible, but flag one cross-group consistency tension.

## Specific issues

1. **Top-3 voxels deduction (−1) is correct.**
   - `code/interpretation/run_interpretation.py:400` defaults `top_k_voxels: int = 3`; argparse default same (`:631`).
   - `code/interpretation/submit_interpretation.sh` explicitly passes `--top-k-voxels 3`.
   - Report figures confirm: Fig 7 (Pluto) shows only rank-1 per subject; Figs 9-10 (Marry) show rank-1 and rank-2 only. No rank-3/4/5 figures anywhere. Spec asks for ~5 voxels per story. Deduction justified.

2. **Page-limit consistency vs group-07: tension, defensible.**
   - Group-08: 21 pages; p.21 contains LLM-usage statement + 4-item references (Conclusion ends p.20). No deduction.
   - Group-07: 21 pages; main body ends p.19, refs p.20, LLM/honesty appendix p.21. Got −1.
   - Both technically violate 20-page limit. Grader's rule (implicit): "if overflow page is purely admin/refs, no deduction" — but group-07's overflow page IS purely admin (LLM/honesty appendix on p.21, refs already on p.20). Strict reading: group-07's analytical body ends p.19 (better than group-08's p.20). Asymmetric treatment. Recommend either docking group-08 −1 OR refunding group-07 +1 for consistency. Lean toward refunding group-07 since both groups' analytical content fits within spec.

3. **SHAP/LIME wrapper pattern: acceptable.**
   - Uses `LimeTabularExplainer` not `LimeTextExplainer`, but `predict_fn` (`:468-487`) IS a genuine text-perturbation wrapper: binary mask → zero out word-TR contributions in pre-built X → re-run ridge prediction. This is the correct pattern (perturbing text, not feature columns directly). Calibration-acceptable.

## Bottom line
Score 41/42 stands. Flag page-limit inconsistency vs group-07 to Sean (recommend group-07 +1 for fairness, not group-08 −1).
