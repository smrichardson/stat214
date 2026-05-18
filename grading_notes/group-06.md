# Group 06 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-06/` · **Leader row:** spreadsheet row 23 · **Leader handle/name:** andreagalloro / Drew Galloro
**Group members (rows 23-26):** Drew Galloro, Benjamin Khothsombath, Rhea Zhang, Ruihang Wang
**Report:** `report/lab3.pdf` (16 pages, under 20-page limit)

---

## Summary scores (Sean's columns)

| Col | Sub-category | Score | Max |
|---|---|---:|---:|
| Q | 3.2 BERT | **12** | 12 |
| S | 3.2 Modeling/Eval | **8** | 8 |
| U | 3.3 Interpretability | **12** | 12 |
| W | Stability Check | **5** | 5 |
| Y | Writing | **5** | 5 |
| | **Sean's subtotal** | **42** | **42** |

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6 since lab spec presents LoRA as the canonical fine-tune. (2) Stability check follows lab spec ("stability of one judgement call"), not the rubric's "1 pt per model" — depth-on-one-judgment-call earns 5/5. This group did *five* distinct stability checks, well beyond what's needed.

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | Report §3.3 p.3 "Pre-Trained BERT" — uses `bert-base-uncased`, ±10-neighbor context window, `BertTokenizerFast`, averaged last-layer hidden states of subtoken positions matching `word_ids()` (avoids repeated-context duplicate matches). 768-d. Code: `run_bert_pretrained.py:140-141` `BertTokenizerFast.from_pretrained` + `BertModel.from_pretrained("bert-base-uncased")`; multi-layer extraction in `extract_bert_all_layers.py:50,140-141`. |
| Fine-tuned (4) | 4/4 | BERT-LoRA serves as the fine-tuned model (per calibration). MLM fine-tuning (15% mask: 80% [MASK]/10% random/10% unchanged) on training stories. 2×2 hyperparameter grid sweep over rank r ∈ {4,8} × MLM length L ∈ {128,256}. Reports MLM losses for all 4 configs in Table 1 p.3, and full downstream ridge CC for each LoRA config in Table 3 p.6. |
| LoRA (2) | 2/2 | Clean PEFT-based implementation. Code: `train_bert_lora.py:22` imports `LoraConfig, get_peft_model` from peft, `:301-307` builds `LoraConfig(r=lora_r, lora_alpha=LORA_ALPHA, lora_dropout=0.1, target_modules=["query","value"])`, `:309` wraps `mlm_model = get_peft_model(mlm_model, lora_config)`. α=2r, AdamW lr=2e-4, batch 16, 3 epochs (acknowledged as undertrained — §3.4 p.3). |
| Performance comparison (2) | 2/2 | Report Table 3 (p.6) is *exhaustive*: 8 embeddings × 2 subjects × 7 metrics (D_feat, Mean CC, Median, Top 5%, Top 1%, Max, fraction r>0). Figure 2 (p.6) adds top-N CC scaling curves across embeddings. Within-subject ordering BERT layer-8 > LoRA configs > BERT pretrained (L12) > Word2Vec ≈ GloVe > BoW is stated explicitly §5.1. |
| Description of each model (2) | 2/2 | §3.1-3.5 p.2-4 describes BoW, Word2Vec, GloVe, pre-trained BERT, LoRA fine-tuning, *and* a layer-ablation strategy (mid-layer hypothesis from Toneva & Wehbe). Each gets a dedicated subsection. Bonus: §3.5 pre-registers a 6-variant BERT-layer ablation (layer4/8/11/12 + concat_4_8_12 + avg_4_8_12). |

**Column R comment (suggested):** Outstanding — clean PEFT-based LoRA, exhaustive performance comparison across 8 embeddings × 2 subjects × 7 metrics. Bonus BERT-layer ablation with validation-driven selection of layer 8 as winner is exactly the methodology in Toneva & Wehbe. No deductions.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | §4.1 p.4 — per-voxel ridge with explicit objective `‖X_train w − Y_train[:,v]‖² + α²_v ‖w‖²`, closed-form SVD solution `ŵ_v = V diag(S/(S²+α²_v)) Uᵀ Y_train[:,v]`. Code: uses `ridge_utils.ridge.bootstrap_ridge`. |
| Cross-validation analysis (2) | 2/2 | §4.2 p.4 — two-tier α-selection strategy: (a) `bootstrap_ridge` with 5 bootstraps × 50 random TR-chunks of length 40 (~2000 TRs held out per bootstrap) for in-memory cases; (b) single-pass SVD on 74 inner-train / 12 validation stories for memory-bound BoW + pretrained BERT. α ∈ logspace(10⁻¹, 10⁶, 30), per-voxel α selection. Plus §7.2 5-fold story-level CV explicit. Code: `kfold_cv_ridge.py:46` `ALPHAS = np.logspace(-1, 6, 30)`. |
| Best-performing model deep-dive (2) | 2/2 | BERT layer-8 identified as winner via validation-driven selection §4.3 p.4. §5.3 p.8 shows full vs inner-train refit comparison (Figure 4). §5.4 p.8 + Figure 5 p.9 explicitly notes heavy right-tailed CC distribution: "bulk of mass sits near zero, ~1% of voxels reach r>0.30", ties this to anatomical specialization. PCS framing motivates restriction to high-CC voxels for interpretability. |
| Performance over voxels (2) | 2/2 | Figure 2 p.6 (top-N CC scaling, log-scale) + Figure 5 p.9 (per-voxel CC histograms) + Table 4 p.7 (per-story aggregation: mean/median/top-5%/top-1%/top-100/max per subject). Per-story CC reported in Table 4 (0.0730 S2, 0.0746 S3 — higher than concatenated). |

**Column T comment (suggested):** Exemplary ridge pipeline — two-tier α-selection (bootstrap_ridge for memory-friendly, single-pass SVD on inner-train/val for memory-bound), per-voxel α grid of 30 values, both concatenated-test and per-story aggregations reported. Full BERT-layer ablation with validation-driven winner selection.

---

## U. Lab 3.3 Interpretability (12 pts) — score: 12/12

Two stories analyzed: `adollshouse` and `birthofanation`, top 5 high-CC voxels per (subject, story) = 20 voxel-story pairs total, both subjects covered. Setup §6.1 p.9: full forward-function wrapping the downstream pipeline (word mask → embed → Lanczos → trim/delay → ridge predict → per-voxel CC), so attributions are at *word* level not embedding level — this is the careful version.

**`adollshouse` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | `shap.KernelExplainer` with 300 binary samples per story §6.1 p.9. Code: `interpret_shap_lime.py:239-244` `shap.KernelExplainer(predict_fn, background)` + `explainer.shap_values(...nsamples=SHAP_NSAMPLES)`. Top 5 voxels per subject covered in Table 6 p.10, |SHAP-only| counts reported per voxel. Word cloud Figure 6 p.10. |
| LIME (2) | 2/2 | `lime.lime_tabular.LimeTabularExplainer` with 400 samples §6.1 p.9. Code: `interpret_shap_lime.py:248-256`. |LIME-only| counts reported in Table 6. |
| Comparison (2) | 2/2 | Table 6 p.10 reports per-voxel SHAP∩LIME overlap (1-5 of top-20 shared, vs random expectation ~0.2). §6.2 p.9 explicitly notes high-importance words are content-bearing (concrete nouns, action verbs), not function words — content/function-word divergence framed. |

**`birthofanation` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Same setup; Table 6 p.10 reports top-5 voxels × both subjects with SHAP&LIME overlap counts (2-4) and SHAP-only/LIME-only counts. |
| LIME (2) | 2/2 | Same setup; per-voxel top-20 sets reported in Table 6 and pooled into word cloud Figure 6 p.10. |
| Comparison (2) | 2/2 | Same per-voxel SHAP/LIME overlap analysis, framed in §6.2 with content-vs-function word divergence theme. Figure 6 panels show overlapping content words across subjects per story. |

**Column V comment (suggested):** Strong interpretability section with word-level forward-function (forward function wraps the full downstream pipeline: word mask → embed → downsample → ridge → per-voxel CC — exactly the right framing per the lab spec). Two stories × both subjects × top-5 voxels each = 20 voxel-story pairs total. Per-voxel SHAP∩LIME overlap quantified against random-expectation baseline.

---

## W. Stability Check (5 pts) — score: 5/5

This group does FIVE distinct stability methods (§7 p.11): (i) per-test-story CC, (ii) 5-fold story-level CV, (iii) voxel-wise permutation testing, (iv) α-grid sensitivity, (v) cross-subject distribution agreement. Per calibration, any one done thoughtfully earns 5/5; this is wildly over-spec.

| Aspect | Evidence |
|---|---|
| Per-test-story CC | Figure 7 p.11 + §7.1: bar charts of mean / top-1% / max-voxel CC across 15 test stories per subject. Most predictable story: `fromboyhoodtofatherhood` (mean 0.122 S2 / 0.138 S3, peak voxel r=0.69/0.74). Least: `whenmothersbullyback` (S2) / `tetris` (S3 — outlier, rule-explanatory). |
| 5-fold story-level CV | Figure 8 p.12 + Table footnote §7.2: r̄ = 0.064±0.004 (S2), 0.067±0.004 (S3); single-split mean 0.069 sits within ~1.5σ. Confirms holdout choice is not anomalous. |
| Permutation significance | §7.3 p.12 + Figure 9 + Table 7 p.13: 1,000-shuffle null per voxel. 65.9% (S2) / 70.0% (S3) of voxels significant at uncorrected α=0.05. Honestly acknowledges Bonferroni resolution limit (need ≥2×10⁶ perms to test 0.05/V≈5×10⁻⁷), reports parametric Fisher-z equivalent. Acknowledges TR-shuffle null is stringent vs block-permutation; that's nuanced. |
| α-grid sensitivity | §7.4 p.13 + Figure 10 + Table 8 p.14: 4 grids (narrow_low [0.1,1e2], default [0.1,1e6], wide [1e-2,1e7], narrow_high [1e2,1e5]). Three grids that span α≥10² give indistinguishable mean+top-5% CC; narrow_low drops 23% in mean. Conclusion: as long as α grid contains ≥10², estimate is grid-insensitive. |
| Cross-subject agreement | §7.5 p.14 + Figure 11 + Table 9: Q-Q plots + KS + Pearson-on-quantiles. Pearson ≥ 0.998 across all embeddings → CC distribution shape is essentially identical between subjects (means differ slightly). Indicates model picks up cortical response pattern, not subject-specific noise. |
| Final framing | §8 p.15: "our five stability checks support the conclusion that the voxel-CC pattern is a real, replicable feature of the data, not a particular split artifact or α-grid coincidence." PCS-style framing. |

→ Per calibration, **5/5**.

**Column X comment (suggested):** Far above what was required — *five* distinct stability methods (per-story CC, 5-fold CV, permutation testing with honest Bonferroni resolution analysis, α-grid sensitivity, cross-subject distribution Q-Q matching). Each isolates a different judgment call. Exemplary PCS framing.

---

## Y. Writing (5 pts) — score: 5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 3/3 | 16-page, cleanly organized 9-section structure (Intro, Data/Preprocessing, Embedding Methods, Ridge Encoding, Results, Interpretability, Stability, Discussion, Conclusion) + Code/data + Academic honesty + References. Limitations section (§8.1) is candid (admits LoRA undertrained at 3 epochs; 15-story test small; TR-shuffle null slightly stringent; no anatomical voxel correspondence cross-subject). Math notation correct (closed-form SVD ridge solution written out). PCS framing explicit throughout. |
| Figure quality (0-2) | 2/2 | 11 figures + 9 tables, all numbered/captioned with descriptive titles. Figure 2 (top-N CC scaling, log-scale, multi-panel by subject) and Figure 11 (cross-subject Q-Q with diagonal reference) are particularly clean. Word cloud Figure 6 is well-styled. The "polished" naming convention in `figs/` reflects deliberate iteration. |
| LLM statement | Present | Academic honesty p.16: explicit statement "We used LLM assistants for grammar/wording edits and figure styling, consistent with the lab's stated LLM policy. Every figure in this report was generated by code we wrote and checked into the repository; every numeric value can be reproduced from the saved *.pkl models and the per-voxel CC arrays in results/metrics/." Reproducibility-aware. |

**Column Z comment (suggested):** Polished, comprehensive, reproducibility-focused report. Math notation correct, figure conventions consistent, limitations section honest. No deductions.

---

## Calibration notes — resolved

Both rubric/spec tensions:

1. **Fine-tuned vs LoRA:** LoRA-only earns full credit (6/6). Locked. (Group 06 explicitly fine-tunes via LoRA + hyperparameter grid.)
2. **Stability check:** Lab spec ("one judgement call") wins over rubric ("1 pt per model"). 5/5 for thoughtful depth. (Group 06 does five distinct methods — exceptional.)
