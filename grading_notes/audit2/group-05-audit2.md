**Verdict:** OK (post-revision)

**Specific issues:**
1. **U=8/12 correctly applies Rule 3** per previous audit's catch. Verified `interpret_story_voxels.py:398-419` (`shap_tr_scores_for_voxel`) is exact linear decomposition of `x_z @ weights_v` — no perturbation. `:463-505` (`lime_group_scores_for_voxel`) zero-masks already-encoded delayed embedding matrix rows, no BERT re-encoding. Both violate raw-text wrapper pattern. −1 per (SHAP, LIME) × 2 stories = −4 properly applied. Comparison points (2+2) retained because paired analysis IS thorough.
2. **S=6/8 correctly retained** — `run_modeling_lora.sh:65-67` confirms `--nboots 0 --alphas 10 --single-alpha`. No CV grid was executed; held-out story split is not a CV substitute. 0.5/2 on CV item is justified.
3. **Above-and-beyond moderation considered but not applied.** Group's stability section (threshold sweep with "what stays stable / what flexes" framing) and cross-subject analysis (§3.7) are unusually mature, AND the writing's "What we should not conclude" paragraph (§3.3 p.11) is exemplary. However: Rule 3 violation is methodological, not gray-zone, so moderation rule explicitly does not apply. S=6/8 is also methodological (no CV), not gray-zone. Cannot soften.
4. W=5/5 and Y=5/5 correctly applied per Rules 2 and 5 (19-page body under cap).

Final 36/42 stands.
