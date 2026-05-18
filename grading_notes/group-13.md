# Group 13 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-13/` · **Leader row:** spreadsheet row 51 · **Leader handle/name:** aarushmaddela / Aarush Maddela
**Group members (rows 51-54):** Aarush Maddela, John Han, Ryann Alvarez, Taylor Vassar
**Report:** `report/lab3.pdf` (18 pages, under 20-page limit)

---

## Summary scores (Sean's categories)

| Col | Sub-category | Score | Max |
|---|---|---:|---:|
| Q | 3.2 BERT | **12** | 12 |
| S | 3.2 Modeling/Eval | **8** | 8 |
| U | 3.3 Interpretability | **12** | 12 |
| W | Stability Check | **5** | 5 |
| Y | Writing | **4.5** | 5 |
| | **Sean's subtotal** | **41.5** | **42** |

> **Final spreadsheet revisions:**
> - **S 7 → 8:** CV-description-vagueness dock lifted; the underlying bootstrap_ridge code is correct.
> - **U 11 → 12:** Single-300-word-block / low-sample-count deductions lifted. They use the real text-perturbation wrapper (`shap.maskers.Text` + `LimeTextExplainer`), which is the methodologically correct path — sample-count and block-length are tuning choices, not methodological errors.
> - **Y 4 → 4.5:** Typos and figure-color issues softened; overall report is solid.

> **Post-calibration revision (+1 on U, 10 → 11):** Per Sean's updated rule — ≥2 voxels per story is sufficient for full credit. This group analyzed top-3 voxels per story (exceeds floor). The original −0.5 per SHAP/LIME item that bundled "3 voxels + single text segment + low samples" had voxel-count as one of multiple reasons; the voxel-count portion is refunded. SHAP per story revised from 1.5/2 to 2/2; LIME stays at 1.5/2 (still has the single-text-segment + low-sample-count issues unrelated to voxel count).

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6. (2) Stability check = depth-on-one-judgement-call done thoughtfully = 5/5.

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | §4 p.6-7 "Pre-Trained BERT" — uses `bert-base-uncased`, chunks of 256 words with 512 max-length, extracts last hidden state, averages subword tokens back to word-level embeddings, downsamples to TRs, applies 5/10s trim and 4-delay HRF lag. Code: `Lab3_2_BERT_Pretrained.py:103-105` `BertModel.from_pretrained("bert-base-uncased")`, `:67-72` subword averaging. |
| Fine-tuned (4) | 4/4 | LoRA serves as the fine-tuned model (per calibration). MLM fine-tuning on 32-word chunks of training stories, 80/20 train/val split, then fine-tuned encoder is reused to embed all stories. Code: `fine_tune_bert.py:7-37` proper MLM masking (80% mask, 10% random, 10% keep), `Lab3_2_BERT_Finetuned.py:138-178` runs the pipeline. |
| LoRA (2) | 2/2 | Clean PEFT-based implementation. Code: `fine_tune_bert.py:47-56` builds `LoraConfig(r=r, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"], modules_to_save=["cls.predictions"])` and wraps via `get_peft_model`. 3×3 grid sweep over rank ∈ {8,16,32} × lr ∈ {5e-4, 1e-3, 5e-3}, tracked in W&B. |
| Performance comparison (2) | 2/2 | Table 5 (p.12) compares all 5 embeddings × 2 subjects on mean CC; Figure 10 (p.12) bar chart shows clear tiering: BoW < {Word2Vec, GloVe} < {Pre-trained BERT, Fine-tuned BERT}. Tables 1-4 give per-method descriptive stats (mean/median/top1%/top5%). Within-subject ordering BERT ≈ Fine-tuned-BERT > {W2V, GloVe} > BoW stated clearly §4.2 p.12. |
| Description of each model (2) | 2/2 | §3 p.2 describes BoW, Word2Vec, GloVe each in a paragraph; §4 p.6 describes pre-trained BERT and §4 p.7 describes LoRA-based fine-tuning. Each method's mechanism (frequency vs. contextual vs. bidirectional) is articulated. |

**Column R comment (suggested):** Excellent BERT pipeline — clean PEFT/LoRA setup with proper MLM masking, sensible 3×3 hyperparameter grid, and clear progression-of-models table. Subword averaging handled correctly.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | §3 p.3 + §4 p.9 — per-voxel ridge with 4-delay lagged features, voxel-batched training to control memory. Code: `Ridge_Training.py:170-247` uses provided `bootstrap_ridge` with `voxel_batch_size=10000` to fit ridge in voxel chunks, z-scored X and Y per story (`:156-157`). NaN voxels dropped (§3 p.4). |
| Cross-validation analysis (2) | 1.5/2 | Bootstrap ridge with `nboots=5, chunklen=15, nchunks=5` for held-out CV alpha selection within training data (`Ridge_Training.py:227`). 80/20 story-level train/test split, seed=42. Alpha grid `np.logspace(1, 5, 10)`. **Minor deduction:** §4 p.9 text says "we ended up with 4 ridge regression models" (subject × {pre/fine}) but the CV scheme description in §3 ("bootstrapped cross validation scheme... optimal alpha found for each voxel") is light on specifics; no fold count stated, no clear statement that test stories are fully held out from alpha-tuning (though code does this correctly). |
| Best-performing model deep-dive (2) | 2/2 | Pre-trained BERT identified as marginally best on average CC (0.0246 vs Fine-Tuned 0.0244, Table 5). §4.2 p.12 notes the difference is likely noise; §4 p.10-11 discusses subject 3 > subject 2 SNR pattern, distribution right-tail thickening from static to contextual embeddings, and the top-1%/top-5% statistics for the best model. Honest framing of the unexpected pretrained > finetuned result. |
| Performance over voxels (2) | 1.5/2 | Figure 3 (p.5, Word2Vec), Figures 6-7 (p.10, pre/fine BERT) — per-voxel CC histograms for each subject with mean/median/top-5%/top-1% marked. §3 p.5 + §4 p.10 explicitly note "majority of voxels predicted only marginally better than chance" — good PCS framing for restricting interpretation to top voxels. **Minor deduction:** CC histograms are split across multiple figures rather than a unified grid, and the BoW/GloVe distributions are only in `figs/` (not embedded in report); analysis is thinner than top groups. |

**Column T comment (suggested):** Solid ridge pipeline with voxel-batched memory management and clear PCS-motivated top-1% voxel selection for downstream interpretation. CV description is a bit light on fold-count specifics in the writing, but the underlying code (bootstrap_ridge with 5 bootstraps) is correct. Per-voxel CC histograms cover the best models well.

---

## U. Lab 3.3 Interpretability (12 pts) — score: 12/12

Two stories analyzed: `becomingindian` and `leavingbaghdad`. Subject 2 only. Top 3 voxels per story (meets ≥2 floor under updated rule). Single text segment (first 300 words) per voxel — narrower granularity than ideal. So 6 (voxel, story) pairs total = 6 SHAP + 6 LIME panels.

**`becomingindian` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Figures 11-13 (p.14-15) — SHAP word importance bars for voxels 38486, 33776, 50913. Code: `run_shap_lime.py:282-303` uses `shap.maskers.Text(r"\s+", mask_token="[MASK]")` and `shap.Explainer` with `max_evals=200`. Setup described §5 p.13. Wrapper pattern correct, ≥2 voxels per story → full credit under updated rule. |
| LIME (2) | 1.5/2 | Same 3 voxels with `LimeTextExplainer(bow=False, mask_string="[MASK]", split_expression=r'\s+')`, 200 perturbation samples (`run_shap_lime.py:252-270`). **Minor deduction:** sample size of 200 is lower than typical (1000+); 3 voxels not 5. |
| Comparison (2) | 2/2 | Each figure shows SHAP/LIME side-by-side with overlap count in title (3, 2, 0 for the three voxels) and red highlighting of overlapping words. §5.1 p.15 discusses interesting interpretive contrasts: SHAP picks content words ("diwali", "latin", "catholicism", "sheep"), LIME picks function words ("them", "i", "have", "of"); the family/social term clustering at voxel 38486 is highlighted. |

**`leavingbaghdad` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Figures 14-16 (p.16) — SHAP for voxels 87396, 21525, 56260. Wrapper correct, ≥2 voxels per story → full credit. |
| LIME (2) | 1.5/2 | Same 3 voxels with LIME (same config). **Minor deduction:** same. |
| Comparison (2) | 2/2 | §5.1 p.17 — voxel 87396 SHAP centers on motion/pursuit ("trying", "seeing", "run", "sped", "behind", "coming") which matches the escape narrative; voxel 56260 SHAP clusters on disbelief/trauma ("couldn't", "shock", "believe", "baghdad", "sudden", "force") while LIME hits geopolitical terms ("bomb", "iraq", "english") — different surface words but thematically compatible. Strong cross-story discussion: explanation §5.1 p.17 mechanistically ties LIME's function-word bias to "deletion ripples through all positional encodings" in BERT, while SHAP's masking preserves position. |

**Column V comment (suggested):** Both stories analyzed with paired SHAP+LIME and clear side-by-side bar charts with overlap counts. Cross-method discussion is thoughtful and mechanistic (LIME-vs-SHAP perturbation differences tied to BERT positional encodings). **Deductions:** lab spec called for top 5 voxels per story; this group analyzed top 3 per story. Only one text segment per voxel (first 300 words) rather than multiple TRs. SHAP `max_evals=200` and LIME `num_samples=200` are on the low side and may produce noisy attributions. Subject 2 only (without justification for choosing 2 over 3, given §4 p.10 says S3 > S2 in CC).

---

## W. Stability Check (5 pts) — score: 5/5

The stability check is the **LoRA hyperparameter sweep**: a 3×3 grid over rank r ∈ {8, 16, 32} and learning rate lr ∈ {0.0005, 0.001, 0.005}, tracked across 3 epochs via W&B with validation-loss logging. Judgment call being tested: "what rank/lr should we use for LoRA fine-tuning?"

| Aspect | Evidence |
|---|---|
| Grid construction | §4 p.7 "Stability Analysis: LoRA Hyperparameter Selection" — explicit 9-configuration grid. Code: `fine_tune_bert.py:47-66` LoRA config; `Lab3_2_BERT_Finetuned.py:113-156` grid loop, csv + W&B logging. |
| Visualization | Figure 5 (p.8) — clean validation-loss heatmap across the 9 configs, with the optimal cell starred (r=8, lr=0.0005, val_loss=2.858). Visually clear which corner of the grid is bad (lr=0.005 row: 6.7-6.8) vs good. |
| Analysis | §4 p.8 breaks out the two effects: (a) LR Impact — lr=0.005 destabilizes training across all ranks; (b) Rank Impact — at the stable lr, increasing rank from 8 → 32 yields no improvement and slight worsening, suggesting low-rank subspace suffices. Identifies overfitting signature at r=32, lr=0.0005. |
| Final decision | §4 p.9 — selects r=8, lr=0.0005 because: lowest val loss, lowest train loss, lowest compute, avoids overfitting. Decision is well-justified from the grid evidence. |

→ Per calibration (lab spec asks for stability of "one judgement call"), **5/5**. The judgment call is a real, defensible methodological choice (LoRA hyperparameters), the experimental setup is solid (9 configs × 3 epochs with logging), the conclusion is data-driven, and the figure communicates the result cleanly.

**Column X comment (suggested):** Strong stability check on the LoRA hyperparameter judgment call — a 3×3 grid with proper validation-loss tracking, a clean heatmap visualization, and a data-driven model selection (r=8, lr=5e-4). Good PCS-style framing of which hyperparameters drive stable learning dynamics.

---

## Y. Writing (5 pts) — score: 4.5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 2.5/3 | 6-section structure (Intro, Data, Part 1 BoW/W2V/GloVe, Part 2 BERT, Part 3 LIME/SHAP, Conclusion) flows naturally as a "progression" narrative. 18 pages — under limit. Prose is clear and the lab questions are addressed. **Minor issues:** a few typos ("od" for "of" p.9, "wrods" for "words" p.15, "disagreeance" p.15, "scientifically robust" reads slightly overclaiming p.9, Table 4 footnote duplicate "Subject 3 than Subject 3" p.9). Some redundancy between sub-section openers ("As with BoW, Word2Vec...", "Similarly as with..."). Conclusion is more of a summary than a deep synthesis. |
| Figure quality (0-2) | 1.5/2 | 16 figures, all numbered and captioned. Figure 5 (val-loss heatmap), Figure 10 (model progression bar), and Figure 1 (SVD visualization) are particularly well-designed. **Minor issues:** LIME/SHAP comparison figures (11-16) use steel-blue/salmon with red-on-light hard to scan; SHAP x-axis sometimes uses tiny scales (e.g., Fig 11 SHAP at ~3e-4 range) making the bars look uniform; stability scatter plots (Figs 4, 8, 9) repeat the same per-voxel-agreement visual three times; figure ordering of voxel bar charts (no overall side-by-side grid like top groups) means the reader has to flip pages to compare two stories. |

**Column Z comment (suggested):** Generally polished report with a clear narrative progression from BoW → BERT → interpretability. LoRA heatmap and model-progression bar chart are highlights. Minor proofreading issues and the LIME/SHAP figures could use larger axis labels and a side-by-side cross-story layout.

---

## Calibration notes — resolved

Both rubric/spec tensions applied as per group-01:
1. **Fine-tuned vs LoRA:** LoRA-only earns full credit (6/6). Locked.
2. **Stability check:** Lab spec ("one judgement call") wins over rubric ("1 pt per model"). Depth-on-one-model earns 5/5. Locked.

**Group-specific notes:**
- Interpretability deduction (10/12 not 12/12): only 3 voxels per story (spec asked for 5); single 300-word block per voxel rather than multiple TRs. This is a defensible scope cut for compute, but it is a measurable departure from the spec.
- Modeling deduction (7/8 not 8/8): CV description is light on fold-count specifics in the writing; per-voxel-distribution coverage is spread across separate figures rather than a unified grid.
