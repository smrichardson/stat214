**Verdict:** OK, U=5 stands (possibly even generous at the upper end).

**Specific issues:**

1. **Code confirmed.** `07_interpret_bert.py:234-241` (`mask_design_matrix`) does `x_masked[rows, :] = 0.0` on the already-computed design matrix X. `kernel_shap_scores` (:263-283) and `lime_scores` (:286-315) feed those zeroed-row matrices straight to `pred = x_masked @ voxel_weights` (:246). No re-tokenization, no BERT forward pass per perturbation. The reference guide's `fMRI_Predictor` wrapper pattern is bypassed entirely.

2. **Report framing is transparent but not self-critical.** p.13: "masking a word corresponded to removing the corresponding time rows from the processed feature matrix before recomputing the voxel prediction score." They consistently say "SHAP-style"/"LIME-style," so no false claim of canonical SHAP/LIME — but no acknowledgement that X-row masking misses BERT's contextual sensitivity or that per-TR features are Lanczos blends of multiple words (the exact watchout the reference guide flags).

3. **Scope confirmed.** Table 6 (p.13): 2 voxels × 2 stories × Subject 2 only = 4 attributions total. Spec target ~5 voxels/story; most peer groups did both subjects.

4. **Defending 6-8?** A defense at 6-7 is conceivable if you weight "they did paired SHAP+LIME with thoughtful comparison" heavily and treat the wrapper-skip as a single methodology dock (~−3) plus voxel/subject docks (~−2 to −3). 8 is hard to defend — that would require treating the X-row method as substantially equivalent to the wrapper, which it isn't (no contextual re-encoding, blended-TR confound). **5 is well-calibrated**; 6 would also be defensible. Recommend leaving at 5.
