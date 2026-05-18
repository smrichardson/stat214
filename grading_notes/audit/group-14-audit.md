**Verdict:** OK.

**Specific issues:**
1. SHAP/LIME wrapper pattern is correctly applied. `fMRI_Predictor.predict` (lines 162-191) takes raw text, splits to words, runs full embed→downsample→delay→ridge pipeline, returning per-voxel scalars. SHAP uses `shap.maskers.Text(r"\s+", mask_token="[MASK]")` with partition explainer (line 220-221); LIME uses `LimeTextExplainer` with `num_features=20, num_samples=500` (line 229-241). Both are proper text-perturbation wrappers — no leakage of indices/embeddings. The 9.5/12 is not under-credited on wrapper correctness.
2. Voxel coverage: script does compute SHAP for 5 top voxels (N_TOP_VOXELS=5), and saves a 3-voxel comparison figure (`shap_voxels_{story}_{subject}.png`, lines 296-325). Grader correctly notes these aren't referenced inline in the PDF; main figs show only voxel 77204. Deduction reasoning stands.
3. Sum-over-TR aggregation (line 190) is real and acknowledged in §5.2 as deliberate. Loses per-TR granularity rubric reaches for. Deduction stands.
4. Subject 2 only (line 59) and fine-tuned BERT not pretrained (lines 72-89). Both confirmed in code. Deductions stand.
5. Stability check: two axes (CC mean + Jaccard voxel-identity dendrogram) at three LoRA ranks. Substantive and PCS-framed. 5/5 correct.

No over/under-credit. Score 39.5/42 stands.
