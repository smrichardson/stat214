**Verdict:** OK

**Specific issues:**
1. 38.5/42 holds. Q=12: all three BERT variants implemented (pretrained, full MLM FT, LoRA via PEFT r=16, α=32, Q/V); exceeds Rule 1.
2. S=7: ridge α-grid ceiling issue (selected 10^5 = grid max, transparently flagged in §5 conclusion) → −0.5 on CV item; missing BERT-variant per-voxel CC histogram → −0.5 on voxel-performance. Both reasonable, group flags the alpha issue honestly.
3. U=10: Wrapper pattern verified in audit1: `fMRI_Predictor` (lines 27-78) replays full BERT→downsample→trim→delay→ridge for each perturbed text. SHAP via `shap.maskers.Text` `[MASK]`; LIME via `LimeTextExplainer` with `num_features=20, num_samples=500`. Rule 3 satisfied. 5 voxels × 2 stories confirmed via top_voxels[:5] (Rule 4 met). Comparison −0.5×2 (prose-only, no paired bar chart / no overlap metric, leans on Triangle Shirtwaist) verified in §4.4 — reasonable.
4. W=5: Cross-subject stability check (embedding ranking via Fig 13 violin + LIME word-importance Figs 14-15). Rule 2 met.
5. Y=4.5 (moderated from 4): Missing Triangle Shirtwaist signed-driver figure (exists in `figs/shap/` but not embedded) was the basis for original −1; moderation (+0.5) given all three BERT variants implemented. Defensible per moderation rule — the figure existed, just wasn't embedded (borderline administrative). 19 pp under limit (Rule 5).
6. Moderation applied to one gray-zone deduction (figure embedding) only, not stacked.
