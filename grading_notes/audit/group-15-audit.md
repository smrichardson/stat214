**Verdict:** OK.

**Specific issues:** None — grader's notes accurately reflect the submission.

**Verification:**
1. SHAP/LIME wrapper pattern correctly applied (`lab3_3_shap_lime.py:27-78`): `fMRI_Predictor.predict()` replays full BERT → downsample → trim (5/10) → delay (1-4 TRs) → ridge coefficient pipeline for each perturbed text. SHAP uses `shap.maskers.Text(r"\s+", mask_token="[MASK]")` (line 135); LIME uses `LimeTextExplainer` with `num_features=20`, `num_samples=500`, signed two-class wrapper.
2. Scope confirmed: `top_voxels[:5]` × two stories (`sweetaspie`, `thetriangleshirtwaistconnection`) — exactly 5 voxels × 2 stories.
3. Cross-method comparison §4.4 (p.16) is indeed prose-only — three paragraphs of LIME-vs-SHAP discussion, no paired bar chart, no overlap metric. Grader's 2-pt dock justified.
4. Triangle Shirtwaist signed-driver figure NOT embedded — only Sweetaspie's (Fig 9, p.12) is in the report; Triangle's `plot4_signed_drivers_*.pdf` exists in `figs/shap/` but does not appear in the PDF body. Grader correctly factored into Y dock.
5. LoRA via PEFT (`LoraConfig(r=16, lora_alpha=32, target_modules=["query","value"])`) confirmed — full 6/6 per calibration.
6. RidgeCV alpha grid ceiling at 10^5 selected uniformly — confirmed; S=7/8 dock appropriate.

All grader judgments hold; 38/42 is consistent.
