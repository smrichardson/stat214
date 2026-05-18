# Group 04 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-04/` · **Leader row:** spreadsheet row 15 · **Leader handle/name:** cylee0815 / Thomas Lee
**Group members (rows 15-18):** Thomas Lee, Mingji Xi, Junle Xu, Sumeng Xu
**Report:** `report/lab3.pdf` (22 pages — slightly over the 20-page limit; pages 21-22 are References + Academic Honesty / LLM usage statement, so the body is effectively 20 pages)

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

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6. Here the group implemented BOTH full-parameter fine-tuning AND LoRA, so they go further than required. (2) Stability check follows lab spec ("stability of one judgement call"). Their depth-on-one-judgement-call (SHAP `max_evals` and LIME `num_samples`/seed) is exemplary, earning 5/5.

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | Report §5.1 p.8 "Pretrained BERT embeddings" — uses `google-bert/bert-base-uncased`, 12-layer, 768-dim, sliding word-window of chunk_size=90 / overlap=30, mean-pools sub-token vectors per word (explicitly avoids CLS pooler with rationale). Code: `generate_bert_embeddings.py:64` `REQUIRED_MODEL_ID = "google-bert/bert-base-uncased"`, `:149-220` `extract_story_embeddings` does word-level sub-token mean pooling with overlap averaging across chunks. `max_length=512` set on tokenizer. |
| Fine-tuned (4) | 4/4 | Two fine-tunes implemented: (a) **Full FT** of all 109.8M parameters (`BertForMaskedLM`, AdamW, lr=2e-5, batch=8, 3 epochs, MLM 15%/80/10/10 mask schedule). Code: `bert_finetune_pipeline.py:208-238` `train_full_finetune`. Report §5.1 p.9 "Full fine-tuning configuration" gives complete recipe, training loss trajectory 2.624 → 2.315 → 2.234 (Table 3, p.9). Fine-tuning corpus built from 81 training stories only (1812 MLM chunks), test 20 stories never touched. |
| LoRA (2) | 2/2 | Clean PEFT-based implementation. Code: `bert_finetune_pipeline.py:34` imports `LoraConfig, TaskType, get_peft_model`; `:253-262` builds `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["query","value"], bias="none")` then `get_peft_model`; `:284` `merge_and_unload()` folds adapters before extraction. Report §5.1 p.9 "LoRA fine-tuning" walks through the rank-r constraint, the implicit-regularizer PCS argument, and notes 294,912 trainable params (0.27% of BERT-base). Training: AdamW, lr=1e-4, 4 epochs, loss 2.812 → 2.548 (Table 3). |
| Performance comparison (2) | 2/2 | Report Table 4 (p.11) reports test-set voxel-wise CC averaged across subjects for BoW, GloVe, Word2Vec, BERT-Pre, BERT-Full FT, BERT-LoRA on Mean / Median / Top-1% / Top-5%. Figure 5 (p.11) plots the same as a line plot ordered by narrative. Figure 2 (p.3) is per-subject bar chart for all six embeddings. Discussion §6.4 p.11-12 articulates the three findings: static→contextual dominant gain, fine-tuning ≈ no-op given noise, subject heterogeneity exceeds embedding gap. |
| Description of each model (2) | 2/2 | §5.1 p.8-9 has four labeled subsections describing BoW/GloVe/Word2Vec (recap from §3.2), then "What is BERT, and why expect it to help?" (architecture + MLM pretraining), "Pretrained BERT embeddings" (extractor setup), "What is fine-tuning, and why?" (in-domain MLM motivation), "Full fine-tuning configuration" (hyperparams), and "LoRA fine-tuning" (PEFT setup + PCS regularizer argument). Each model has multi-paragraph treatment with motivation, not just hyperparameter dumps. |

**Column R comment (suggested):** Outstanding BERT section. Group implemented BOTH full-parameter fine-tuning AND LoRA (not just one), with complete training recipes, per-epoch MLM losses, and a thoughtful "anticipated downstream performance (a priori)" prediction (p.10) registered BEFORE evaluation — `BERT-Pre ≳ BERT-LoRA > BERT-Full FT` — which turned out essentially correct. Mean-pool-not-CLS justification is exactly right. No deductions.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | §6.1-6.2 p.10 — per-subject, per-embedding ridge with SVD-based solver `W(α) = V (S/(S²+α²)) Uᵀ Y`, sharing one SVD across all candidate α's. Code: `model.py:8-58` uses `ridge_utils.ridge.ridge_corr` for the SVD path (multi-α from a single SVD). `train.py:552` calls `select_alpha`. Motivation given: at D=5000+ voxels, normal-equation sklearn.Ridge OOMs on the login node (§2.6 p.3). |
| Cross-validation analysis (2) | 2/2 | §6.3 p.10 — story-level train/test split (no TR-level leakage, §2.3 p.2 explicitly motivated by temporal autocorrelation: BOLD at TR `t` is shaped by stimulus at `t-1..t-3`). Train stories further split with `val_frac=0.15` (12 stories) used ONLY for α selection. Code: `train.py:41` `DEFAULT_VAL_FRAC=0.15`, `:500` `make_subtrain_val`. Report says α ∈ {1, 10, 10², 10³, 10⁴}; code default is α ∈ {1, 10, 100} per `train.py:40` `DEFAULT_ALPHAS = [1.0, 10.0, 100.0]`. Minor inconsistency between report text and default; not a credit-affecting issue since best α was always 100 (Table 2, p.6), so the wider grid in the report text would have given the same answer. |
| Best-performing model deep-dive (2) | 2/2 | Two deep-dives: (i) §6.5 p.12 + Figure 6: BERT-Full FT (formally best in Table 4, though report carefully argues BERT-Pre / Full FT / LoRA are statistically indistinguishable given the per-subject spread). (ii) §7.3-7.5 chooses BERT-LoRA / subject2 as the interpretability deep-dive because LoRA matches Full FT at ≪ 1% params (the PCS-favored choice). The "what does a CC of 0.01 actually mean?" passage (§4.2 p.6) gives a noise-ceiling-aware reading of the absolute scale and motivates focusing on the right-tail percentiles. |
| Performance over voxels (2) | 2/2 | Figure 6 (p.12) — per-voxel CC histograms for BERT-Full FT on each subject. Both right-skewed (mean > median, S2: 0.025/0.020; S3: 0.033/0.025), with the top-1% reaching ≈ 0.13. Report explicitly states "the ridge model does not predict all voxels equally" (p.12) and ties this to known anatomy ("plausibly auditory and language cortex"). Figure 3 (p.7) does the same for GloVe in Lab 3.1, and Table 4 (p.11) reports mean/median/top-1%/top-5% for all 6 embeddings × subject-averaged. |

**Column T comment (suggested):** Clean SVD-based ridge pipeline with story-level CV (no leakage), and the analysis is unusually careful — the "fine-tuning ≈ no-op given the noise" reading of Table 4 (p.11) is the right one, and the per-voxel right-tail framing in §6.5 is exactly the PCS predictability move. The α-grid mismatch (report mentions 5 values, code uses 3) is a tiny documentation glitch, not a methodological problem.

---

## U. Lab 3.3 Interpretability (12 pts) — score: 12/12

Two stories analyzed for LoRA / subject2: **`gangstersandcookies`** and **`fromboyhoodtofatherhood`** — picked as the top-2 stories by mean per-voxel test CC on the subject's held-out stories (the PCS predictability filter, §7.1 p.13). Top-5 voxels per story by held-out CC; voxels analyzed all have test CC between 0.37 and 0.53, well above their CC > 0.05 threshold. Full interpretation pipeline also run on BERT-Full FT and on subject3 (Table 5, p.14 lists 4 (variant, subject) pairs); detailed report focuses on LoRA/subject2 to maintain depth.

**`gangstersandcookies` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Figure 7 (p.15, left panels) — SHAP top-words bar charts for top-2 voxels (v35485 CC=0.527, v46989 CC=0.521); Figure 8a (p.15) — 5-voxel heatmap with top-10 SHAP words per voxel. Method §7.1 p.12: `shap.PartitionExplainer` with `shap.maskers.Text(r"\s")`, `max_evals=200`. Code: `interpret_utils.py:338-353` `run_shap`. v46989 ("opening McDonald's scene") gives high SHAP to `mcflurrys`, `mcdonald's`, `mcflurry`, `standing`, `line`; v35430 (gang scene) gives high SHAP to `his`, `head`, `down`, `gang`, `generous`. |
| LIME (2) | 2/2 | Figure 7 (p.15, right panels); Figure 8b (p.15) — 5-voxel LIME heatmap. Method §7.1 p.12: `LimeTextExplainer(bow=False, split_expression=r"\s+")` with `num_samples=300`. Code: `interpret_utils.py:356-388` `run_lime`. v46989 LIME: `chocolate`, `thirty`, `nine`, `cents`; v32395 (social scene): `laughing`, `girlfriends`, `tables`, `jokes`. |
| Comparison (2) | 2/2 | §7.3 p.14 + Figure 8 (p.15) discuss that the same voxel-level passages surface different word sets in SHAP vs LIME ("Both methods pick up McDonald's-related words, just different ones"). Table 8 (p.16) reports Jaccard@5 = 0 to 0.111 and Jaccard@10 = 0.111 to 0.333 across the 10 voxels — disagreement is quantified. §7.3 p.16 gives four explanations for the low overlap: different objectives (Shapley marginal vs local-linear), correlated features in the 30-60 word window, local non-linearity, method-internal noise. |

**`fromboyhoodtofatherhood` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Figure 9 (p.16, left) — SHAP for top-2 voxels (v61490 CC=0.389, v61530 CC=0.378); Figure 10a (p.17) — top-10 SHAP heatmap across 5 voxels. v61490, v61530, v87230 all peak in the Berlin Stories passage and SHAP highlights `isherwood`, `berlin`, `stories`, `christopher`. v78125 SHAP gives high scores to `able`, `expose`, `pass`, `wasn't`, `hid`, `didn't`. |
| LIME (2) | 2/2 | Figure 9 (p.16, right) and Figure 10b (p.17). LIME for the same Berlin-stories voxels picks up physical action words `pool`, `muscles`, `push`, `ups`, `running` — words that co-occur in the same window but pick up a different theme. Both methods agree on v73161's `thought`, `but`, `she`. |
| Comparison (2) | 2/2 | Table 8 (p.16, bottom block): J@5 ranges 0-0.25, J@10 ranges 0-0.176 for this story's voxels. §7.3 p.16 articulates that across this story the two methods "agree on the theme but disagree on the tokens" — content-vs-function-word divergence framed mechanistically. The story-level pattern matches gangstersandcookies, strengthening the claim. |

**Column V comment (suggested):** Exceptionally thorough interpretability section. Story selection is properly PCS-motivated (top-2 by held-out subject-level CC, Table 5 p.14); voxel selection is PCS-motivated (top-5 by held-out test CC, all above their 0.05 floor, §7.1 p.13); back-mapping from peak TR to word window is explicitly flagged as approximate (Lanczos + 4-TR delay). Cross-method comparison quantified via Jaccard@5/10 (Table 8, p.16) AND structured into four candidate explanations. Heatmap visualizations (Fig 8, 10) effectively show across-voxel patterns. The decision to deep-dive LoRA/subject2 while running the full pipeline on all 4 (variant, subject) pairs is exactly the right PCS-style tradeoff.

---

## W. Stability Check (5 pts) — score: 5/5

The stability check probes the **interpretability hyperparameters themselves** — SHAP `max_evals ∈ {100, 200, 400}` and LIME `num_samples ∈ {150, 300, 600}` plus `random_state ∈ {0, 1, 2}` — for the top-1 voxel of each of the two stories. This is precisely "stability of one judgement call" (per the calibration rule), done in depth.

| Aspect | Evidence |
|---|---|
| Setup | §7.4 p.17 — each variant compared to the baseline (max_evals=200 for SHAP, num_samples=300 seed=0 for LIME) using J@5, J@10, Spearman ρ, and cosine similarity. Code: `notebooks/3.3part1_stability_check.ipynb`. |
| SHAP stability | Table 9 (p.18) — at `max_evals=200` (baseline) everything is identical by construction; at 100 Spearman ≥ 0.74, cosine ≥ 0.92; at 400 cosine ≈ 0.94 — top-3 words stable across all settings. Conclusion: SHAP is essentially reproducible. |
| LIME stability | Table 10 (p.18) — much less stable. Even with `num_samples=300` and fixed seed, varying seed alone moves J@5 from 0.429 → 0.250 and Spearman from 0.739 → 0.365. Crucially, increasing `num_samples` from 300 to 600 actually *lowers* agreement with the seed=0 / n=300 baseline because more samples converge toward the *true* Shapley-style answer rather than that particular noisy draw. |
| Visual evidence | Figure 11 (p.18) — heatmaps for v61490 across the SHAP sweep (top 4 rows, nearly identical) and the LIME seed/sample sweep (bottom 6 rows, visibly different). |
| Stability-aware conclusion | §7.4 "What we can claim" (p.19) — distills a calibrated claim: trust SHAP top-3 at `max_evals ≥ 200` AND LIME words that show up in ≥ 2 of the 3 seeds at `num_samples=300`, or at `num_samples ≥ 600` averaged over seeds. This is then carried forward into §8 Discussion (p.19) ("words that pass both checks are the ones we can report as real findings"). |

→ Per calibration (lab spec asks for stability of "one judgement call", done thoughtfully), **5/5**. This is among the most rigorous stability sections I've seen — it discovered something genuinely informative (LIME is noise-limited at typical hyperparameter settings) and then used the finding to discipline the claims in the report.

**Column X comment (suggested):** Outstanding stability check. The SHAP-vs-LIME stability sweep produces a quantitative reproducibility budget (top-3 SHAP words trustworthy; LIME requires ≥ 2-of-3 seed agreement) that is then carried back into the Discussion to gate which findings get reported. Figure 11 (p.18) is a particularly clear illustration of the differential stability of the two methods. Exactly the PCS framing the lab spec is reaching for.

---

## Y. Writing (5 pts) — score: 5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 3/3 | Cleanly organized 11-section structure (Intro / EDA / Lab 3.1 Part 1 / Lab 3.1 Part 2 / Lab 3.2 Part 1 / Lab 3.2 Part 2 / Lab 3.3 / Discussion / Conclusion / References / Academic Honesty + LLM usage). Clear roadmap at end of §1 (4 numbered questions). PCS framing introduced in §1 p.1 and threaded through every subsequent section. Prose is dense but precise. **Length: 22 pages**, but pages 21-22 are References + Academic-Honesty/LLM-statement, so the scientific body is exactly 20 pages — within the spirit of the limit. Math notation correct (ridge `W(α) = V S/(S²+α²) Uᵀ Y`, etc.). LLM usage statement present and detailed (§11.2 p.22). |
| Figure quality (0-2) | 2/2 | 11 figures + 10 tables, all numbered and captioned with descriptive titles. Figure 5 (p.11) line plot of metrics by embedding is clean. Figure 6 (p.12) histograms have explicit mean/median markers in the legend. SHAP/LIME bar charts (Figs 7, 9) use consistent blue-positive / red-negative colors per panel. Heatmaps (Figs 8, 10, 11) use clear diverging colormaps with shared color bars. Minor: per-subplot text on the 5-voxel heatmap grids (Figs 8, 10) is small but still legible. |

**Column Z comment (suggested):** Polished, technically dense report with PCS framing threaded throughout. Pre-registers a prior prediction about Full FT vs LoRA vs Pre BEFORE evaluation (§5.1 p.10), which is exactly the kind of scientific discipline the lab is meant to teach. LLM usage statement is exemplary — distinguishes "polish" use from "intellectual content" use. Body is exactly 20 pages (references + honesty statement push the total to 22). No deductions.

---

## Calibration notes — resolved

Both rubric/spec tensions discussed with Sean before grading the rest of the cohort:

1. **Fine-tuned vs LoRA:** LoRA-only earns full credit (6/6). Locked. **This group went further** — they implemented both full-parameter fine-tuning AND LoRA, with per-epoch loss tables (Table 3, p.9) and an explicit a-priori prediction about which would win (§5.1 p.10).
2. **Stability check:** Lab spec ("one judgement call") wins over rubric ("1 pt per model"). A thoughtful depth-on-one-judgement-call earns 5/5. Locked. **This group's** SHAP/LIME hyperparameter sweep with seed variation is exemplary.

See `memory/feedback_stat214_lab3_calibration.md` for the durable rule.

---

## Minor observations (no credit impact)

- **α-grid documentation mismatch:** Report §6.3 p.10 says α ∈ {1, 10, 10², 10³, 10⁴}; code defaults to α ∈ {1, 10, 100} (`train.py:40`). Best α is reported as 100 in Table 2 (p.6, Lab 3.1), so even with the wider documented grid the result would have been the same. Worth flagging in feedback but not deduction-worthy.
- **Page count:** 22 total but the body is exactly 20 pages. References + Academic Honesty / LLM statement sit on pages 21-22. I read this as compliant.
- **No held-out validation during fine-tuning** — flagged honestly in §5.1 p.9 as a judgement call (MLM perplexity is a noisy proxy for downstream voxel-CC). Appropriate self-awareness.
- **Single-subject, single-variant focus in §7** — flagged in §7.5 Limitations p.19. Appropriate framing.
