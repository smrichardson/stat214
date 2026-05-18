**Verdict:** Possibly under-credited (+0.5 pt) on Y — gray-zone, defensible either way

**Specific issues:**
1. SHAP/LIME wrapper compliance verified: `shap.maskers.Text(r"\s+", mask_token="[MASK]")` + partition explainer; `LimeTextExplainer`; the predictor class re-runs full pipeline (BERT extraction → preprocessing → ridge) per perturbed text input. Rule 3 cleanly satisfied. Per Sean's clarification (group-09 audit2): position-vs-type Jaccard bug is a comparison-metric quirk, not a methodology error.
2. Voxel count: 10 per story × 2 stories × 2 subjects = 40 attributions, well above Rule 4 floor.
3. Above-and-beyond moderation candidate: Real MLM fine-tuning + LoRA at three ranks (over-delivers on Rules 1's "either-or" criterion), four-axis stability work (trimming, per-story, cross-embedding Jaccard, +2 TR interpretability shift), transparent in-line limitation disclosure (α reuse, ridge-coef storage workaround, SHAP-sign caveat), real wrapper SHAP/LIME. Current Y = 4.5/5 deducts 0.5 for: (a) Fig 4 / Figs 10-11 rotated unreadable tick labels, (b) only Subject 2 shown in SHAP/LIME body figures, (c) no side-by-side composite figure. These are minor figure-quality issues. Given above-and-beyond depth elsewhere, the 0.5 deduction is a gray-zone candidate for moderation softening (per the new rule, gray-zone deductions can be softened by 0.5).
4. Recommendation: Y → 5/5 (+0.5 on Y) for consistency with above-and-beyond moderation rule; new total 41/42. This is gray-zone — defensible to leave at 40.5 too.
5. Rules 1, 2, 4, 5 cleanly satisfied.
