**Verdict:** OK

**Specific issues:**
1. 39.5/42 holds. Q=12: pre-trained BERT + LoRA pipeline confirmed in report §1.1 (loads `bert-base-uncased`, 5-embedding × 2-subject Table 1, gains BoW→BERT +38/+74%, BERT→LoRA +39/+29%). LoRA-only earns 6/6 per Rule 1.
2. S=7.5: CV-mismatch −0.5 (moderated from −1) is internally consistent — code uses `bootstrap_ridge(NBOOTS=1)` with `logspace(0,3,6)` but report §1.1 claims "5-fold CV ... α∈{1,10,100,1000,10000}". Moderation justified by differentiable LoRA + Lanczos sinc engineering.
3. U=10: Mask-on-X violation real (−2). Wrapper-pattern Rule 3 applied correctly. Voxel count: §3.3.2 confirms top-1% (then top-5) voxel selection per (subject, story), Figs 10-11 show 2 voxels per story (one per subject) — meets ≥2 floor per Rule 4. Refund of figure-coverage −0.5×2 applied correctly.
4. W=5: §4 stability check on temporal delay stack (6 configs, end-to-end rerun, Table 4) is depth-on-one-judgment-call — Rule 2 applied correctly.
5. Y=5: 17 pages, well under 20-page limit (Rule 5). Academic honesty + LLM-usage present in Appendix A. No issues.
6. Moderation (+0.5) applied to one gray-zone deduction only (CV-mismatch), not stacked — consistent with rule.
