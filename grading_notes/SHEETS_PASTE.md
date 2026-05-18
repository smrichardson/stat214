# Lab 3 Spreadsheet Entry — Paste into Google Sheets

For each group below, click the indicated cell (column Q of the leader's row) in Google Sheets and paste the single comma-separated line. The 10 values will populate Q→Z automatically.

**Order: Q (3.2 BERT) · R (comment) · S (Modeling) · T (comment) · U (Interp) · V (comment) · W (Stability) · X (comment) · Y (Writing) · Z (comment).**

**Format:** comma-separated, with text fields wrapped in `"..."`. To paste in Google Sheets: click the target cell, then `Edit → Paste special → Paste comma-separated values` (or just paste — Sheets will auto-detect CSV). Internal `"` characters in comments are doubled (`""`) per CSV spec.

**Cohort: Mean 39.37/42 (93.7%) · Range 33–42 · Median 41 · 7 perfect scores (01, 04, 06, 07, 08, 12, 14).**

> Comments below are concise (1–2 sentences with specifics). Full detailed reasoning lives in `group-NN.md`.

---

## Group 01 — Row 3 — Alex Soh (`alexchsoh`) — **42/42**

Paste at **Q3**:

```
12,"Clean BERT and LoRA pipelines with sliding-window subtoken aggregation. Performance comparison covers 5 embeddings × 2 subjects × 4 metrics.",8,"Story-level CV (no leakage), per-voxel α-search on 8000-voxel subset, PCS-aware voxel analysis. CC distributions for all 5 embeddings × 2 subjects.",12,"5 voxels × 3 TRs × 2 stories with paired SHAP+LIME. Voxel filtering by predictability. Cross-method comparison addresses content-word vs function-word divergence.",5,"Stability check on BERT context window size (512/256/128). Voxel-overlap finding (0/5 at window 256, 2/5 at window 128) shows voxel-level instability while word-level patterns hold.",5,"Well-structured 8-section report. Clean modeling math, consistent figure conventions. Within page limit."
```

---

## Group 02 — Row 7 — Abhishek Shukla (`abss123`) — **33.5/42**

Paste at **Q7**:

```
12,"Solid BERT setup with bert-base-uncased, layer-9 extraction, chunk_size 200. PEFT/LoRA on Q/V (r=132). Performance comparison covers 5 embeddings × 5 metrics.",7,"Clean ridge with 5-fold CV. Best-model deep-dive thin; §5.1 claims subject-2 replication but no per-subject figures appear.",6,"SHAP/LIME run on a single voxel from a single model in 3072-d embedding-feature space rather than raw text via the wrapper pattern (Lab3 Reference Guide). No explicit second story. Partial credit for the cross-method comparison and word-projection table.",4,"Stability check on α-grid range. §5.3 says best α=1000 but Table 2 lists 10^4–10^6 — factual inconsistency.",4.5,"10-page report with strong PCS framing. Half-point off for figures with unreadable per-story stability labels (Figs 6-7). Several typos throughout."
```

---

## Group 03 — Row 11 — Akanshaa Borgohain (`Akanshaa18`) — **33/42**

Paste at **Q11**:

```
10.5,"Full MLM fine-tuning implemented (counts for LoRA per calibration). Pre-trained BERT extraction (bert_pipeline.py:39-50) passes each word to BERT as its own input sequence with no surrounding text, so self-attention has no context — BERT collapses to a quasi-static embedder, which likely explains the report's ""BERT ≈ GloVe"" finding.",7,"Bootstrap ridge with 5 iterations and per-voxel α selection. Voxel distribution analysis solid.",6,"2 voxels × 2 stories with paired SHAP+LIME. Wrapper-pattern violation: SHAP via shap.LinearExplainer on 3072-d tabular features; LIME via LimeTabularExplainer on same. Word attributions reconstructed post-hoc via vocab projection rather than text perturbation.",5,"Two judgment calls probed (delay window and seed split) with coefficients of variation. Embedding rankings preserved across seeds.",4.5,"17 figures, generally well-written. Half-point off for SHAP wordclouds (Figs 14, 15) losing signed attribution information."
```

---

## Group 04 — Row 15 — Thomas Lee (`cylee0815`) — **42/42**

Paste at **Q15**:

```
12,"Both full-parameter fine-tuning AND LoRA implemented. Pre-registered prediction ""BERT-Pre ≳ BERT-LoRA > BERT-Full FT"" entered before evaluation. Mean-pool justified.",8,"Two-SVD ridge precomputation, per-voxel α selection from 10 log-spaced candidates. 81/20 story-level train/test split.",12,"5 voxels × 2 stories with shap.PartitionExplainer + Text masker and LimeTextExplainer. Wrapper re-encodes perturbed text through BERT → Lanczos → make_delayed → ridge per perturbation.",5,"SHAP max_evals × 3 sweep + LIME num_samples / random_state × 3 sweep, used to gate which words enter the Discussion (LIME identified as noise-limited).",5,"Body exactly 20 pages, refs/honesty/LLM statement overflow. Strong figures."
```

---

## Group 05 — Row 19 — Chenfei Peng (`ChenfeiP`) — **36/42**

Paste at **Q19**:

```
12,"Clean BERT setup with sliding windows (128 words, 32 overlap). LoRA MLM on Q/K/V/dense. 5-embedding × 2-subject performance table.",6,"SLURM scripts run --nboots 0 --single-alpha, so no CV α-search actually performed; uses single shared α=10. Story-level train/test split is correct.",8,"5 voxels × 2 stories. Wrapper-pattern violation: SHAP method (shap_tr_scores_for_voxel) is exact linear decomposition with no perturbation; LIME zero-masks the encoded delayed embedding matrix. Strong cross-method discussion otherwise.",5,"Stability check on SHAP/LIME interpretation threshold (CC ≥ 0.05, 0.10, any positive). ""What we should not conclude"" framing in §3.3 is mature.",5,"Body 19 pages. Strong PCS framing throughout."
```

---

## Group 06 — Row 23 — Drew Galloro (`andreagalloro`) — **42/42**

Paste at **Q23**:

```
12,"8-method × 2-subject × 7-metric performance comparison. PEFT LoRA at r/L grid. Bonus BERT-layer ablation with validation-driven layer-8 selection per Toneva & Wehbe.",8,"Voxelwise permutation testing with Bonferroni resolution analysis, story-level CV, per-voxel α from logspace grid.",12,"5 voxels × 2 stories × 2 subjects with word-level wrapper (BERT pre-encoded with ±10 context window, then masked; full Lanczos/delay/ridge/CC rerun per perturbation).",5,"Five stability methods: per-story CC, 5-fold CV with σ, voxel-wise permutation, α-grid sensitivity, cross-subject Q-Q (Pearson ≥0.998). Each isolates a different judgment call.",5,"16 pages including LLM statement. Strong figures."
```

---

## Group 07 — Row 27 — Oleksandr Volkov (`AlexVolkov0404`) — **42/42**

Paste at **Q27**:

```
12,"28-experiment LoRA hyperparameter sweep across rank, α, LR, epochs, stride, target modules. Final-model selection with rationale. Pre-trained BERT-vs-BoW inversion noted.",8,"5-fold story-level CV, candidate α grid, per-voxel selection. Top-256 voxel restriction for downstream interpretation.",12,"2 voxels × 2 stories × {SHAP + LIME + POS-aggregated SHAP}. shap.maskers.Text + LimeTextExplainer via make_window_regression_predict_fn over fMRI_Predictor (full text → tokenize → embed → downsample → delay → ridge).",5,"Four stability probes: voxel subsetting, cross-subject Jaccard (~0.04), cross-embedding Jaccard (~0.76–0.87), seed reruns. Cross-subject finding leads to ""voxel-level interpretations don't generalize across individuals.""",5,"21 pages total, body ends p.19, refs/honesty overflow OK. Well-structured."
```

---

## Group 08 — Row 31 — Gongyao Xu (`eluxyl`) — **42/42**

Paste at **Q31**:

```
12,"8-method comparison with paired Wilcoxon significance (p<10^-15 for almost every step). Three LoRA variants (last4, improved-recipe, mid4). Empirical finding: mid-layer (5-8) extraction beats last-4 by +9% on subject 2.",8,"Two-SVD precomputation, sparse LU for BoW, per-voxel α selection. Wilcoxon significance testing throughout.",12,"Top-3 voxels × 2 stories with LimeTabularExplainer over binary masks + custom predict_fn mapping masks back to word-TR zeroing then re-running ridge. Story-anchored LIME windows from response traces (Figs 6, 8).",5,"Four-level cross-subject replication: story-level r=0.745, sorted bar plot, voxelwise distribution, and cross-method ratio (better embeddings shrink s3/s2 ratio).",5,"21 pages with p.21 = LLM + refs (admin overflow). Tables well-formatted."
```

---

## Group 09 — Row 35 — Destiny Ogu (`destinyogu`) — **36/42**

Paste at **Q35**:

```
12,"Pre-trained BERT with 512-token overlapping chunks, stride 64. PEFT LoRA on Q/V with MLM fine-tuning and 90/10 split. 5-embedding × 2-subject performance table.",8,"Bootstrap CV (5 iterations, 5×40-TR held-out chunks), per-voxel α selection. PCS-aware framing of voxel-level performance.",7,"2 voxels × 2 stories on Subject 2 only. Wrapper-pattern violation: hand-rolled ""KernelSHAP-style"" and ""LIME-style"" perturbations zero-mask design-matrix rows for each word's TRs rather than re-encoding perturbed text through BERT.",5,"Stability check on α-grid range for GloVe ridge. Setup table, before/after metric comparison, PCS framing.",4,"19 pages. Section numbering inconsistent. Figure 4 (per-voxel-block boxplots) is dense."
```

---

## Group 10 — Row 39 — Canhui Huang (`CanhuiH`) — **41/42**

Paste at **Q39**:

```
12,"Sliding-window pre-trained extraction. Full MLM fine-tuning AND LoRA at three ranks (r=4/8/16). Table 2 separates contextual-vs-static gains from rank effects.",8,"Bootstrap CV, per-voxel α selection. α=10^5 hit grid ceiling on static embeddings; reused on BERT without re-running CV but disclosed honestly.",11,"10 voxels × 2 stories × 2 subjects (40 SHAP+LIME runs). Real text-perturbation wrapper. SHAP-vs-LIME Jaccard normalization bug: SHAP tokens carry trailing whitespace from shap.maskers.Text(r""\s+"") while LIME returns clean tokens, so cross-method intersection is 0 across all 40 voxels (within-method Jaccards are nonzero).",5,"Four-axis stability work: trimming convention, per-story CC, cross-embedding Jaccard, +2 TR interpretability shift. Cross-embedding Jaccard finding: LoRA shifts ~1/3 of top-5% voxel membership despite tiny aggregate CC change.",5,"11 figures + 4 tables. Minor visibility issues (rotated tick labels, Subject 2-only body figures) noted but not docked given depth elsewhere."
```

---

## Group 11 — Row 43 — Xiaoyang Xiao (`Anchor-bkl`) — **39.5/42**

Paste at **Q43**:

```
12,"LoRA fine-tuning trains end-to-end through Lanczos sinc interpolation + delay stack with differentiable subword pooling — a sophisticated implementation. Clean PEFT on Q/V.",7.5,"Per-voxel ridge via bootstrap_ridge. Methodology mismatch between report and code: report §1.1 says ""5-fold CV, α ∈ {1,10,100,1000,10000}""; code uses bootstrap_ridge(nboots=1) with np.logspace(0,3,6). Per-voxel α selection is sound, just described inaccurately.",10,"5 voxels × 2 stories × 2 subjects with shap.KernelExplainer + from-scratch LIME (textbook kernel weighting). Wrapper-pattern violation: perturbations zero-mask the already-encoded design matrix X_story_z by chunk rather than re-encoding perturbed text through BERT.",5,"Stability check on temporal delay stack (6 configurations: [1,2,3,4], [1,2,3], [2,3,4], [1,2], [0], [1,2,3,4,5,6]). Baseline empirically optimal; LoRA-BERT remains best across all; early lags carry more signal.",5,"17 pages including TOC and appendix. Strong overlay figures (BERT-vs-LoRA scatter, SHAP-vs-LIME bar overlays)."
```

---

## Group 12 — Row 47 — Eviana Pham (`evianaphm`) — **42/42**

Paste at **Q47**:

```
12,"Both full fine-tuning AND LoRA implemented (Q/V r=8 α=32). Strong Table 1 + Figure 1 performance comparison. Per-voxel delta histograms (Fig 3) answer ""does fine-tuning help?"" directly.",8,"Per-voxel ridge with story-level CV. PCS-aware voxel filtering for downstream interpretation.",12,"3 voxels × 2 subjects × 2 stories with story_X_from_words wrapper (perturb → BERT re-embed → downsample → trim/lag → z-score → ridge → story CC). Spearman ρ=0.13 SHAP-vs-LIME — quantitative cross-method check.",5,"Two PCS-framed stability checks: (1) contextual vs word-level BERT shows surrounding context contributes ~half the contextual lift; (2) LoRA rank sweep r∈{4,8,16} confirms result not artifact of chosen rank.",5,"19 pages, ~16 main body. LLM statement present (separate coding + writing notes)."
```

---

## Group 13 — Row 51 — Aarush Maddela (`aarushmaddela`) — **39/42**

Paste at **Q51**:

```
12,"Sliding-window BERT extraction. PEFT LoRA on Q/V. 5-embedding × 2-subject performance table.",7,"Bootstrap ridge with proper CV (nboots=5, chunklen=15, nchunks=5), logspace alphas, story-level 80/20 split. CV description in writing is vague but code is correct.",11,"Top-3 voxels per story with shap.maskers.Text + LimeTextExplainer (real text-perturbation wrapper). Single 300-word block per voxel rather than multiple TRs. SHAP max_evals=200 and LIME num_samples=200 on the low side. Cross-method comparison ties LIME's function-word bias to BERT positional encodings.",5,"LoRA hyperparameter sweep (3×3 rank × LR grid, val-loss heatmap in Fig 5). Data-driven r=8 / lr=5e-4 selection.",4,"18 pages. Several typos (""od"", ""wrods"", ""disagreeance""). SHAP/LIME bar charts use steel-blue/salmon with hard-to-scan highlighting; SHAP x-axis sometimes at tiny scales."
```

---

## Group 14 — Row 55 — Candy Zhang (`candiizy`) — **42/42**

Paste at **Q55**:

```
12,"All three BERT variants implemented (pretrained, full MLM fine-tuning, LoRA at three ranks r=4/8/16). Jaccard-distance dendrogram shows BERT vs static embeddings access fundamentally different brain regions.",8,"Bootstrap CV (nboots=5, chunklen=40, nchunks=5), per-voxel α selection on highest bootstrap correlation. Test set fixed across embeddings.",12,"3 voxels × 2 stories with correct text-perturbation wrapper (shap.maskers.Text + LimeTextExplainer). Cross-story finding: same voxel shows opposite SHAP signs for penpal vs adollshouse, interpreted as reflective vs confrontational narrative register.",5,"LoRA-rank stability tested on two axes (CC performance AND voxel identity via Jaccard distance over top-5% voxel sets). Strengthens the ""fine-tuning doesn't help"" conclusion.",5,"Honest limitations section. Strong overlay figures."
```

---

## Group 15 — Row 59 — Donjhai Holland (`DonjhaiH`) — **38.5/42**

Paste at **Q59**:

```
12,"All three BERT variants implemented (pre-trained, full MLM, LoRA via PEFT at r=16/α=32 on Q/V). Slurm-resilient training, merge_and_unload() adapter merging. 5-embedding × subject performance comparison.",7,"RidgeCV with α grid topped out at 10^5; α=10^5 selected for every BERT variant. Honestly flagged in conclusion but grid not extended.",10,"Top-5 voxels × 2 stories with shap.maskers.Text + LimeTextExplainer via real text-perturbation wrapper. Cross-method comparison stays prose-only — no paired bar charts, no quantitative overlap metric.",5,"Cross-subject stability check across both modeling (embedding ranking) and interpretability (LIME word importance). BoW→W2V/GloVe→BERT/LoRA hierarchy reproduces on Subject 3; top LIME word ""said"" stable; sign patterns shift.",4.5,"19 pages. Triangle Shirtwaist signed-driver figure exists on disk (figs/shap/) but not embedded in PDF; Sweetaspie equivalent is embedded."
```

---

## After all 15 paste operations

Don't forget to **propagate the Total (column AA) to the other 3 group members** in each group. Those members' rows have the same Total but blank score cells. The leader row is the only one with grades — but the Total formula needs to flow to the rest of the group so each student's final grade reflects the group score.

If column AA already has a formula like `=SUM(D:Y)` or similar, it should auto-update once you paste the leader's row. Verify after pasting.
