**Verdict:** OK.

**Specific issues:**
1. SHAP/LIME wrapper IS correctly applied. `run_shap_lime.py:39-155` defines `predictor(texts)` that takes `list[str]` and returns scalar voxel BOLD predictions through the full pipeline (tokenize → BERT encode → subword→word averaging → lanczos downsample to TRs → trim → 4-delay → ridge weights). SHAP uses `shap.maskers.Text(r"\s+", mask_token="[MASK]")` + `shap.Explainer(predict_fn, masker)`; LIME uses `LimeTextExplainer(bow=False, mask_string="[MASK]", split_expression=r'\s+')` with `lime_predict_fn` wrapping the same predictor. Wrapper rule satisfied — no penalty triggered. U=10/12 stands.
2. 3 voxels per story confirmed: `--rank` arg has `choices=[1,2,3]`, reproduce script runs ranks 1-3 for both stories. Matches grader claim.
3. S=7/8 verified: bootstrap_ridge with `nboots=5, chunklen=15, nchunks=5`, `alphas=np.logspace(1,5,10)`, 80/20 story-level split (seed=42). Code is correct; -1 is for writeup vagueness only. Fair.
4. Cross-group consistency: group 08 also did 3 voxels per story but got 11/12 (-1). Group 13's extra -1 is justified by additional shortcomings beyond voxel count (single 300-word segment vs. multi-TR windows, max_evals=200 and num_samples=200 both low, Subject 2 only). Consistent.
