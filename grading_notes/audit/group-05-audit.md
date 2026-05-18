**Verdict:** Possibly over-credited (Delta approx -4 pts on U).

**Specific issues:**
1. **U = 12/12 violates the locked SHAP/LIME wrapper-pattern rule.** The group does NOT use a `text -> BERT -> downsample -> ridge` wrapper. Their own method note (`interpret_story_voxels.py:720-724`) states "SHAP uses exact linear contributions for the fitted ridge model after z-scoring." `shap_tr_scores_for_voxel` (lines 398-419) just decomposes `x_z[:, k] * weights_v[k]` per delayed feature block — no perturbation, no BERT re-encoding. `lime_group_scores_for_voxel` (lines 463-505) perturbs by zeroing TR-blocks of the already-encoded delayed embedding matrix (`mask_group_in_z`, lines 447-460) and regresses on the predicted-response cosine, not on raw-text deletions through BERT. Lab reference guide explicitly requires the wrapper (lines 574-610) and warns (line 450) that word-level attribution only works because the wrapper re-runs the full pipeline per perturbation. Per the locked calibration rule, this is a substantive deduction — comparable to Group 03/04 expectations. Suggest U = 8/12 (lose 2 of 4 SHAP points and 2 of 4 LIME points across the two stories; comparison points retained since paired analysis was done).
2. S = 6/8 deduction magnitude is correct: `run_modeling_lora.sh:65-67` and `run_modeling_bert_pretrain.sh:72-74` confirm `--nboots 0 --alphas 10 --single-alpha`. No CV grid was run; held-out story split is not a substitute.
3. Q, W, Y verifications hold.

Revised: 36/42 (Q 12, S 6, U 8, W 5, Y 5).
