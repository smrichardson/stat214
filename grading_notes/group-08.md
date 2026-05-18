# Group 08 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-08/` · **Leader row:** spreadsheet row 31 · **Leader handle/name:** eluxyl / Gongyao Xu
**Group members (rows 31-34):** Gongyao Xu, Chun-Kai Hsu, Shizhe Zhang, Ricardo Perez Castillo
**Report:** `report/lab3.pdf` (21 pages — 1 page over the 20-page limit, but final page is just refs + LLM statement)

---

## Summary scores (Sean's columns)

| Col | Sub-category | Score | Max |
|---|---|---:|---:|
| Q | 3.2 BERT | **12** | 12 |
| S | 3.2 Modeling/Eval | **8** | 8 |
| U | 3.3 Interpretability | **8** | 12 |
| W | Stability Check | **5** | 5 |
| Y | Writing | **5** | 5 |
| | **Sean's subtotal** | **38** | **42** |

> **Final spreadsheet revision (−4 on U):** On close re-reading, the LIME `predict_fn` (`run_interpretation.py:468-487`) zeros word-aligned TR rows in the **pre-computed design matrix `X_story_raw`** rather than re-encoding perturbed text through BERT. SHAP is closed-form linear decomposition (`phi_j = x_ij × w_jv`, docstring lines 8-12) — exact for a linear model but not a text perturbation. Both items violate the wrapper pattern documented in `Lab3_Reference_Guide.html`. Cross-group consistency with groups 03, 05, 09, 11 requires the same −4 wrapper dock. SHAP/LIME items per story drop from 2/2 → 1/2; Comparison stays at 2/2 (cross-method comparison Table 4 + §5.4 is genuine).

> **Post-calibration revision (+1 on U, 11 → 12):** Per Sean's updated rule — "top voxels" in the lab spec means ≥2 per story is sufficient for full credit. This group analyzed top-3 voxels per story, which exceeds the new floor. The original −1 deduction for "top-3 not top-5" is refunded. LIME items per story revised from 1.5/2 to 2/2.

> **Calibration applied** (per Sean): (1) LoRA-only counts as the canonical fine-tune (full 6/6 on Fine-tuned + LoRA). (2) Stability check: depth-on-one-judgment-call = 5/5. (3) Top-3 voxels per story (instead of top-5) earns small deduction on interpretability (-1 across both stories).

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | Report §3.2.2 p.2-3 describes two pretrained BERT variants: "Baseline" (128-word non-overlapping chunks, final layer) and "Enhanced BERT" (256-word sliding window, stride 128, last-4 layer mean). Both use `bert-base-uncased`, WordPiece subword averaging. Code: `pretrained_bert.py:106` `AutoModel.from_pretrained("google-bert/bert-base-uncased")`, `:312` `words_to_bert_embeddings_sliding` with configurable stride. |
| Fine-tuned (4) | 4/4 | LoRA serves as the fine-tune (per calibration). MLM fine-tuning of `bert-base-uncased` on 81 training stories with frozen base weights; podcast-domain adaptation explicitly motivated and validated against pretrained baseline. |
| LoRA (2) | 2/2 | Excellent PEFT-based implementation with three LoRA variants. Code: `fine_tune_bert.py:33` imports `LoraConfig, get_peft_model, PeftModel`, `:172-181` configures `r=32, lora_alpha=64, dropout=0.1, target_modules=["query","key","value","dense"]` (+ optional FFN). Report §3.2.3 p.3 documents MLM masking schedule (80/10/10), 8 epochs with cosine annealing, AdamW lr=2e-5, validation-loss-based checkpoint at epoch 5 (val loss 2.157). Two LoRA-recipe variants: "improved" (whole-word masking + 256-word training chunks) and "mid4" (read out layers 5-8 vs 9-12). |
| Performance comparison (2) | 2/2 | Table 2 (p.5) compares **eight** embedding configurations (BoW, Word2Vec, GloVe, Vanilla BERT, Enhanced BERT, LoRA last4, LoRA improved, LoRA mid4) on Mean/Median/Top-5%/Top-1%/Best CC for both subjects. Figure 1 (violin plots), Figure 2 (% gain over GloVe baseline), Figure 3 (8x8 Spearman heatmaps of per-voxel CC vectors for both subjects). §4.3 reports paired Wilcoxon signed-rank significance (p<10^-15) for almost every consecutive step. |
| Description of each model (2) | 2/2 | §3.2 p.2-4 walks through six text representations (BoW, Word2Vec, GloVe, Vanilla BERT, Enhanced BERT, three LoRA variants) each with ~1 paragraph block describing pretraining corpus, dimensionality, pooling, and extraction details. Ordered simplest → most complex. |

**Column R comment (suggested):** Outstanding — eight-method comparison with rigorous significance testing, clean PEFT LoRA implementation with three thoughtful variants (last4, improved-recipe, mid4), and a genuinely interesting empirical finding that mid-layer (5-8) extraction beats last-4 extraction by +9% on s2. The Spearman rank-correlation analysis between per-voxel CC vectors (Figure 3) is particularly polished.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | §3.3.1 p.4-5 — explicit ridge objective `‖Y-XW‖²_F + λ‖W‖²_F`, closed-form `(X^T X + λI)^{-1} X^T Y`, and SVD reformulation (Eq. 3). Code: `run_ridge.py:121` `SVDRidgeSolver` class with thin SVD precomputed once, diagonal rescaling for any α. Separate sparse LU solver for BoW (~48k-dim, 99.99% sparse). |
| Cross-validation analysis (2) | 2/2 | §3.3.2 p.5 — two SVDs precomputed: one on 80% fit set for per-voxel λ selection, one on full train for final weights. 10 log-spaced α values (`run_ridge.py:224` `--n-alphas=10`). Per-voxel α selection by validation Pearson r. BoW uses single global α on 200-voxel sample due to sparse-matrix memory. Story-level train/test split (81/20) maintained throughout. |
| Best-performing model deep-dive (2) | 2/2 | LoRA mid4 identified as best (§4.1, Table 2). §4.1 p.5-6 explains why: Spearman rank correlation between mid4 and last4 per-voxel CC vectors drops to ρ=0.866/0.909 vs near-perfect ρ=0.977/0.984 between last4 variants → layer choice (not recipe) is the bottleneck. §3.2.3 p.4 motivates middle-layer superiority anatomically (compositional syntax/semantics peak in BERT layers 5-8; upper layers reshape for MLM objective). Strongly significant Wilcoxon (p<10^-15) for last4→mid4. |
| Performance over voxels (2) | 2/2 | §4.2 p.6-7 + Figure 1 (violin plots, both subjects, all 8 methods). Distributions are right-skewed; upper tail (Top-5%) stretches +168% across method hierarchy (BoW → LoRA mid4); body of violin shifts only modestly. §4.4 + Figure 3 — Spearman cross-method analysis at voxel level. PCS-aware (interpretation later restricted to top-256 voxels). |

**Column T comment (suggested):** Polished ridge pipeline with two-SVD precomputation, sparse LU for BoW, per-voxel α selection, and rigorous Wilcoxon significance testing. Deep voxel-level analysis: violin plots + Spearman rank correlations between every method pair. The argument that mid4 vs last4 ρ=0.87/0.91 (vs last4-recipe-variants ρ=0.98) confirms the layer choice—not the fine-tuning recipe—is the real driver is sharp PCS reasoning.

---

## U. Lab 3.3 Interpretability (12 pts) — score: 8/12

> **Wrapper-pattern violation.** SHAP is exact linear decomposition (no perturbation); LIME via `LimeTabularExplainer` with custom `predict_fn` that zeros word-aligned TR rows in the pre-computed design matrix `X_story_raw` rather than re-encoding perturbed text through BERT. −1 per (SHAP, LIME) per story = −4 on U.
>
> **Word-disambiguation limitation:** group's own §s2 acknowledges this — "SHAP attribution operates at TR resolution, words that co-occur within the same 2 s window receive identical scores; tied bars therefore reflect a shared high-salience time window rather than individual word salience." This TR-resolution tie is a structural consequence of operating on the design matrix rather than re-encoding through BERT. The wrapper pattern would have produced distinct attributions per word because removing each token from the input text produces a unique BERT output and downsampled trajectory.

Two stories analyzed: `On Approach to Pluto` (space exploration) and `Marry a Man Who Loves His Mother` (interpersonal). Both subjects (s2, s3) covered. SHAP is global (top-256 voxels weighted by test r) + per-story TR-level word attribution; LIME is local (top-3 voxels × peak 5-TR window). **Top-3 voxels per story, not top-5** — small deduction.

**`On Approach to Pluto` (5.5 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Figure 5 (p.12) — top-20 SHAP word scores for s2. Method §5.1.1 p.10: closed-form linear SHAP `ϕ_j = w_j(x_j - E[x_j])` aggregated across top-256 voxels weighted by `|r_v|`. Code: `run_interpretation.py:232-309` `run_shap` and `_story_word_shap`; also `ridge_regression/shap/shap_analysis.py:218-221` for the closed-form linear SHAP. Caveat: SHAP at TR-resolution means co-occurring words within same TR get identical scores (acknowledged in §5.1.1 and §5.4.4 as a feature, not a bug). |
| LIME (2) | 1/2 | Figure 7 (p.13) — LIME word importances for the rank-1 voxel of both subjects. Setup §5.1.2 p.10: 300 perturbations (mask subsets of unique word types), word-masked by zeroing TR-block contributions in pre-built X. Local 5-TR peak window. Code: `run_interpretation.py:389-530` `run_lime`, uses `LimeTabularExplainer` from `lime.lime_tabular` (not LimeTextExplainer). **Deduction: top-3 voxels (`top_k_voxels: int = 3` at `run_interpretation.py:400`) per story instead of top-5.** |
| Comparison (2) | 2/2 | §5.4 p.15-16 + Table 4 — well-structured comparison along Scope, Granularity, Attribution, Background, Time resolution, Output. §5.4.3 demonstrates convergence: SHAP highlights "year, old, move" at TR-level; LIME independently picks "july, fourth, working, long" at word-level — both surface the temporal-mission-marker content. §5.4.4 articulates complementary strengths (SHAP→latent BERT structure; LIME→narrative-specific word salience). |

**`Marry a Man Who Loves His Mother` (5.5 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Figure 5b (p.12) — top-20 SHAP word scores for s2 on this story. Dense spatial-scene cluster ("wooden", "chair", "propped", "foot", "against"). |
| LIME (2) | 1/2 | Figures 9-10 (p.14) — LIME word importances for rank-1 and rank-2 voxels for both s2 and s3. Spatially adjacent voxels (s2: 32384/32271; s3: 25884/25885) show diverging LIME profiles but shared social-proximity theme. **Deduction shared with Pluto: top-3 voxels.** |
| Comparison (2) | 2/2 | Same §5.4 framework applied; §5.4.3 notes SHAP highlights "wooden, chair, propped" while LIME picks out "together, live, door, hand" — converging on the social-proximity content. Cross-subject divergence (s2: relational verbs; s3: physical-proximity nouns) discussed §5.3.2. |

**Column V comment (suggested):** Strong, methodologically careful interpretability work. Two thematically contrasting stories, both subjects, response-trace figures (Figs 6, 8) anchor LIME windows in genuine BOLD-prediction peaks, and the SHAP-vs-LIME comparison (Table 4 + §5.4) is one of the most thoughtful in the cohort — explicitly recognizes that SHAP-at-TR vs LIME-at-word is a complementarity, not a bug. The one shortfall: report and code both analyze top-3 voxels per story rather than top-5; small deduction (-0.5 per story).

---

## W. Stability Check (5 pts) — score: 5/5

The stability check asks: "does the encoding model capture a real, generalizable language-to-brain mapping that replicates across individuals?" — explicitly framed as the **central judgment call** (§6, p.16). They apply this test only to the best model (LoRA mid4) and explicitly justify why ("conservative test on weaker model would understate replication").

| Aspect | Evidence |
|---|---|
| Story-level replication | Pearson r(s2, s3) = 0.745 across 20 test stories. s3 > s2 in 17/20 stories. Top stories ("onapproachtopluto", "fromboyhoodtofatherhood", "marryamanwholoveshismother") replicate across subjects; only 3 stories reverse the typical ordering. |
| Voxelwise CC distribution | Figure 12 panel C — 95.5% (s2) / 97.8% (s3) of voxels have positive r. s3 distribution uniformly right-shifted (not localized to a few elite voxels). |
| Cross-method ratio | Table on p.17 — `s3/s2` ratio shrinks monotonically from 1.99x (BoW) to 1.37x (LoRA mid4). Convergence-with-embedding-quality is itself a replication argument: if the model were capturing idiosyncratic noise, this ratio would not change systematically. |
| Final framing | §6 distills: cross-subject replication holds at four complementary levels (story-level scalar, sorted bar chart, voxelwise distribution, cross-method ratio). This is rigorous PCS-style stability. |

→ Per calibration (depth-on-one-judgment-call), **5/5**.

**Column X comment (suggested):** Exemplary stability analysis on a meaningful judgment call (cross-subject replication of the encoding map). Four complementary levels of agreement (story-level r=0.745, voxelwise distribution, sorted bar plot, cross-method ratio convergence) — the last is especially clever: better embeddings shrink the s3/s2 ratio, which is itself a replication argument.

---

## Y. Writing (5 pts) — score: 5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 3/3 | Clean 8-section structure (Intro, Data, Method, Main Results, Interpretability Study, Stability Analysis, Discussion, Conclusion). Math notation correct. Tables are publication-quality. 21 pages total — strictly above the 20-page limit, but the final page is just LLM-usage statement + 4-item reference list (the analytical content fits within 20). I am not deducting for this. The prose is occasionally lightly typo-laden ("our method" → "out method", "the four rank-1 voxels" should be "the two rank-1 voxels", "whic" → "which") but the overall narrative is clear and rigorous. |
| Figure quality (0-2) | 2/2 | 12 figures total, well-captioned with informative subtitles. Figure 1 (violin grid, both subjects) is excellent. Figure 3 (8x8 Spearman heatmap) is publication-quality. Figures 6 and 8 (BOLD response traces with yellow peak-window shading) elegantly anchor the LIME analysis. SHAP bar charts (Figs 4, 5) and LIME bar charts (Figs 7, 9, 10) use consistent positive/negative color conventions. |

**Column Z comment (suggested):** Polished, well-structured report with strong figures and rigorous statistical reporting (Wilcoxon significance, Spearman rank correlation, story-level Pearson). Minor copy-edits would tighten it further. 21-page length is technically over the limit but the analytical content fits within 20 pages (final page is the LLM-usage statement + references); not deducting.

---

## Calibration notes — resolved

Both rubric/spec tensions discussed with Sean before grading the rest of the cohort:

1. **Fine-tuned vs LoRA:** LoRA-only earns full credit (6/6). Locked.
2. **Stability check:** Lab spec ("one judgement call") wins over rubric ("1 pt per model"). A thoughtful depth-on-one-model check earns 5/5. Locked.

See `memory/feedback_stat214_lab3_calibration.md` for the durable rule.
