**Verdict:** Mixed — possibly over-credited by ~1 pt overall (net), with one clear under-credit on U.

**Specific issues:**

1. **U — Comparison 2/2 may be over-credited (Δ −0.5 to −1).** §4 (p.7) compares SHAP vs LIME on the *same single voxel/single instance*, both run in 3072-dim feature space. The lab's comparison item expects per-method tradeoffs grounded in proper raw-text attributions across stories/voxels; here the "comparison" is largely "both flag dim_308." The pre-trained-vs-fine-tuned BERT SHAP comparison on p.8 is a nice add but is a model-vs-model check, not a SHAP-vs-LIME methodological comparison. Suggest 1.5/2 or 1/2 — consistent with the grader's own −0.5 dock applied to SHAP and LIME for single-voxel.

2. **W — §5.3 α inconsistency is more serious than noted (Δ ~0).** §5.3 (p.8) claims "α = 1000 gave the best saved validation performance" for all four non-BoW methods, but Table 2 (p.4) lists best α as 10^5, 10^4, 10^6, 10^6, and Figure 1 (p.3) clearly peaks at 10^5–10^6 for BERT. This isn't a unit-mismatch — it's a factual error in the stability narrative undermining the α-stability claim. Already absorbed in the −1 W deduction, so no Δ, but worth flagging.

3. **S — Per-subject claim is unsupported (Δ ~0, already docked).** §5.1 (p.8) asserts "we applied the exact same modeling approach to a second subject... distribution of voxel correlations was remarkably similar" but no subject-2 figure/table/numbers appear anywhere. Grader already docked 0.5 on "Performance over voxels"; fine.

4. **U — Possibly under-credited by 0.5 on Story-1 SHAP (Δ +0.5).** SHAP is actually run on 100 held-out rows with 500 bg samples (Table 3 caption + influence.py), which is a real global summary — not a single instance like LIME. The grader's −0.5 for "single instance" on SHAP conflates SHAP's setup with LIME's. Suggest SHAP = 1.5/2, LIME = 1/2.

Net: roughly cancels; 31/42 is defensible but borderline 30–31.
