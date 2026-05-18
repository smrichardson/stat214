**Verdict:** OK.

**Specific issues:**

1. **Page-limit deduction (Y −1):** Warranted. Report runs to p.21 (main body to p.19, refs p.20, appendix p.21). Calibration treats 20 pages as the limit; 21 → −1 on readability is consistent with cross-group practice. The body itself technically ends at p.19 so this is a soft call, but the deduction is defensible and the grader flagged it for Sean's review.

2. **2 voxels per story (U 12/12):** OK to keep full credit. The reference guide / lab spec mentions "~5 top voxels" as guidance, but the group does 2 stories × 2 voxels × {SHAP + LIME + POS-aggregated SHAP} = 12 distinct analyses, exceeding the depth bar in other dimensions. POS-level SHAP (Figs 2-3) is bonus interpretability not required by spec. Per generous-partial-credit calibration, no deduction warranted; consistent with how other groups were scored.

3. **SHAP/LIME wrapper pattern:** Correctly applied. `run_shap.py:211-212` uses `shap.maskers.Text(r"\s+", mask_token="[MASK]")` + `shap.Explainer(predict_fn, masker)`; `run_lime.py:213` uses `LimeTextExplainer`. Both invoke `make_window_regression_predict_fn` over `fMRI_Predictor` (`fmri_predictor_wrapper.py:281,313,337`) which does text→tokenize→embed→downsample→delay→ridge — full text-perturbation pipeline. No vector-level perturbation.

4. **28 LoRA experiments:** Verified. `configs/` contains `config_bert_lora_exp001.yaml` through `exp028.yaml`. Tables 3-7 in report match. Citation is accurate.

Score 41/42 stands.
