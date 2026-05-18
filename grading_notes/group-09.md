# Group 09 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-09/` · **Leader row:** spreadsheet row 35 · **Leader handle/name:** destinyogu / Destiny Ogu
**Group members (rows 35-38):** Mara Masic, Destiny Ogu, Xinpeng Chen, Dongyu Zhang
**Report:** `report/lab3.pdf` (19 pages, under 20-page limit)

---

## Summary scores (Sean's columns)

| Col | Sub-category | Score | Max |
|---|---|---:|---:|
| Q | 3.2 BERT | **12** | 12 |
| S | 3.2 Modeling/Eval | **8** | 8 |
| U | 3.3 Interpretability | **7** | 12 |
| W | Stability Check | **5** | 5 |
| Y | Writing | **4** | 5 |
| | **Sean's subtotal** | **36** | **42** |

> **Post-calibration revision (+2 on U, 5 → 7):** Per Sean's updated rule — "top voxels" means ≥2 per story is sufficient for full credit. This group analyzed 2 voxels per story, which now meets the floor. The voxel-count portion of the prior deduction (−0.5 on SHAP and −0.5 on LIME × 2 stories = −2) is refunded. The method portion (wrapper-pattern violation: hand-rolled SHAP/LIME on already-computed X rows rather than re-encoding through BERT) remains the dominant deduction. SHAP and LIME per story revised from 0.5/2 to 1/2.

> **Post-calibration revision (−4 on U, 9 → 5):** The reference guide (`Lab3_Reference_Guide.html` §Interpretation → The Wrapper Pattern) explicitly prescribes wrapping `text → BERT → downsample → ridge` and perturbing raw text. This group's hand-rolled "KernelSHAP-style" and "LIME-style" perturbations zero out a word's TR rows in the **already-computed design matrix** — they never re-encode perturbed text through BERT. This loses BERT's contextual sensitivity (removing word X cannot change the embedding of word Y), and the per-TR feature is a Lanczos blend of multiple words rather than a single word's contribution. Sean covered SHAP/LIME methodology in a full lecture and uploaded a specific reference guide, so this is a documented expectation. The deduction also reflects compounding sub-spec voxel count (2 per story vs spec ~5) and Subject 2 only.

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6 since lab spec presents LoRA as the canonical fine-tune. (2) Stability check follows lab spec ("stability of one judgement call"), not the rubric's "1 pt per model" — depth-on-one-model earns 5/5.

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | Report p.4 "Pre-trained BERT" — uses `google-bert/bert-base-uncased`, `is_split_into_words=True` to track subword-to-word mappings, 512-token overlapping chunks with stride 64, batch 4 on GPU. Subtoken averaging on final hidden states produces one 768-dim vector per original word; empty-token words replaced with `[UNK]`, missing words assigned zero vectors. Code: `02_build_bert_embeddings.py:56-95` (full word-level extraction loop with `word_ids` tracking). |
| Fine-tuned (4) | 4/4 | LoRA fine-tuned BERT serves as the fine-tuned model (per calibration). MLM fine-tuning on story transcripts with frozen base weights + trainable LoRA adapters on attention Q/V projections (`03_fine_tune_bert.py:142-148`). 90/10 train/val split (seed 42), 3 epochs, batch 8, LR 5e-5, AdamW. Mask probability 0.15, 80/10/10 mask/random/keep split. Re-encoder used for both training and held-out test stories with identical extraction pipeline. |
| LoRA (2) | 2/2 | Clean PEFT-based implementation. Code: `03_fine_tune_bert.py:142-148` builds `LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["query", "value"])`, then `get_peft_model`. Configuration tabulated in report Table 1 (p.5). Adapters merged via `merge_and_unload` after training so the saved checkpoint can be reloaded with `AutoModel.from_pretrained` for embedding extraction. |
| Performance comparison (2) | 2/2 | Report Table 2 (p.6) for Lab 3.1 + Table 5 (p.8) for Lab 3.2 give Mean/Median/Top-1%/Top-5% CC for all 5 embeddings × 2 subjects. Figure 3 (p.10) is a clean side-by-side bar chart (mean CC + top 5% CC) comparing all 5 embeddings × 2 subjects. The within-subject ordering BERT-LoRA ≈ BERT > {W2V, GloVe} >> BoW is stated explicitly with quantitative examples (Subject 3 top 1% CC: 0.087 GloVe → 0.137 BERT-LoRA). |
| Description of each model (2) | 2/2 | p.3-5 describes BoW, Word2Vec, GloVe, pre-trained BERT, and BERT-LoRA in 1-paragraph blocks. The LoRA section (p.4-5) describes the rationale (memory/compute efficiency) and the technical choice (rank-8 adapters on Q/V). |

**Column R comment (suggested):** Excellent — clear model descriptions, BERT-LoRA implemented cleanly via PEFT on attention Q/V projections, comprehensive performance table covering 5 embeddings × 2 subjects × 4 metrics. No deductions.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | Report p.5 "Ridge Regression Framework" — explicit objective `‖Y − XW‖²_F + α‖W‖²_F`, per-voxel α selection. Code: `04_fit_ridge.py:128-142` builds `np.logspace(alpha_min_exp, alpha_max_exp, alpha_count)` and calls `bootstrap_ridge` from `ridge_utils.ridge`. |
| Cross-validation analysis (2) | 2/2 | Report p.5 — bootstrap CV with 5 iterations, 5 held-out chunks of 40 consecutive TRs, candidate α ∈ logspace(10^1, 10^3, 10 values), best-α-per-voxel rule. Final weights computed on full training set using each voxel's optimal α. 80/20 story-level train/test split (seed 42). |
| Best-performing model deep-dive (2) | 2/2 | The deep-dive splits across two best models: GloVe (best Lab 3.1) gets Figure 1 (p.7) + Table 3 + Lab 3.1 stability discussion; BERT/BERT-LoRA (best Lab 3.2) gets Figure 2 (4-panel CC histograms, p.9), Figure 3 (bar chart, p.10), and §"Voxel-Level Performance" (p.11). Discussion correctly notes LoRA gains for Subject 2 but slight regression for Subject 3, and frames this as "fine-tuning had mixed effects across subjects." |
| Performance over voxels (2) | 2/2 | Figure 1 (p.7) for GloVe + Figure 2 (4-panel, p.9) for BERT — both show right-skewed CC distributions with most voxels near 0 and a small upper tail. Figure 4 (p.12) adds a comprehensive boxplot view across voxel chunks (10k-voxel slices × 2 subjects × 2 BERT variants). Report explicitly motivates filtering to top voxels for downstream interpretation under PCS: "the most meaningful interpretation should focus on voxels with reliable predictive performance rather than treating all voxels equally" (p.13). |

**Column T comment (suggested):** Solid ridge pipeline with bootstrap CV (5 iterations, 5×40-TR held-out chunks), per-voxel α selection, and PCS-aware framing of voxel-level performance. Histograms and per-chunk boxplots both presented.

---

## U. Lab 3.3 Interpretability (12 pts) — score: 7/12

> **Wrapper-pattern violation.** Hand-rolled KernelSHAP-style and LIME-style perturbations zero word-aligned rows of the pre-built design matrix `x` (`07_interpret_bert.py:234-241`, `mask_design_matrix`), then recompute prediction via `x_masked @ voxel_weights`. The math is textbook (kernel-SHAP coalition weighting, LIME exponential distance kernel) but operates on the wrong input space — BERT is never re-invoked. −1 per (SHAP, LIME) per story = −4 on U.
>
> **Word-disambiguation limitation:** `words_to_rows` maps each word to its TR rows; words that share TR rows are functionally indistinguishable under the perturbation (masking either zeros the same rows). The wrapper pattern would have disambiguated them via re-encoding. **Comparison docked −1 total** for the comparison narrative staying at prose level with no quantitative overlap metric.

Two stories analyzed: `gangstersandcookies` and `thetriangleshirtwaistconnection`, both Subject 2 only. 2 voxels per story (4 voxels total) — meets the ≥2 voxel-count floor per the post-calibration rule. Voxels selected by held-out story-level CC on the fine-tuned BERT ridge model (Table 6, p.13). SHAP/LIME implemented as custom KernelSHAP-style and LIME-style perturbations on the **already-computed design matrix** — they mask the rows of X corresponding to a word's TRs and recompute the voxel's predicted-vs-observed CC. The reference guide explicitly prescribes the text-perturbation wrapper pattern (re-encode through BERT per perturbation); this implementation skips that entirely. **Wrapper-pattern violation is the dominant deduction (−1 SHAP × 2 stories + −1 LIME × 2 stories = −4); voxel-count and single-subject no longer dock under the updated rule.**

**`gangstersandcookies` (6 pts → score 2.5/6):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 1/2 | Figure 5 (p.14) — KernelSHAP-style bar chart for voxel 11016; Figure 6 (p.15) for voxel 67419. Method described p.13-14: random word-coalition masks, kernel weighting, weighted linear surrogate on **X rows**, not text. Code: `07_interpret_bert.py:263-283` (`kernel_shap_scores`). −1 pt for skipping the text-perturbation wrapper. (Voxel-count deduction refunded post-calibration: 2 voxels per story ≥ minimum.) |
| LIME (2) | 1/2 | Figures 5-6 right panels — LIME-style bars using `keep_prob=0.8` random word-deletion masks on X rows. Code: `07_interpret_bert.py:286-315` (`lime_scores`). Same wrapper deduction as SHAP. |
| Comparison (2) | 1.5/2 | Per-voxel comparison text after each figure: voxel 11016 — both methods pick *order, mcflurrys, crap, guys, end, understanding* as shared top drivers; voxel 67419 — both pick *wearing, look, walking, break, while, tell*. −0.5 for no quantitative overlap plot. (Comparison is unaffected by the method-error since SHAP and LIME use the same wrong-input space, so the *relative* comparison is still informative.) |

**`thetriangleshirtwaistconnection` (6 pts → score 2.5/6):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 1/2 | Figure 7 (p.15) — voxel 5048; Figure 8 (p.16) — voxel 38488. Same wrapper deduction as story 1. |
| LIME (2) | 1/2 | Figures 7-8 right panels. Same wrapper deduction. |
| Comparison (2) | 1.5/2 | Voxel 5048 — SHAP emphasizes *international, important*; LIME emphasizes *says, ladies, security, safety*. Voxel 38488 — both pick *ticket, yeah, cuz, last, slept* but disagree on tail. §"Comparison of SHAP and LIME" (p.16) discusses divergence. −0.5 for prose-only comparison. |

**Column V comment (suggested):** Both stories analyzed with paired SHAP+LIME figures and thoughtful per-voxel discussion. Main deductions: (1) **Method.** The implementation is a hand-rolled KernelSHAP-style + LIME-style perturbation on the **design-matrix rows** corresponding to each word — not the text-perturbation wrapper pattern (re-encode `text → BERT → downsample → ridge` per perturbation) that the reference guide explicitly prescribes and that Sean covered in lecture. Because each TR's features are a Lanczos blend of nearby words, and because BERT's contextual sensitivity requires re-encoding to be measured, masking X rows does not actually answer "which words drive this voxel." (2) **Voxel count.** 2 voxels per story instead of ~5. (3) **Single-subject.** Subject 2 only.

---

## W. Stability Check (5 pts) — score: 5/5

The stability check varies the **ridge α grid** (original: 10 alphas in 10^1–10^3; perturbed: 15 alphas in 10^0–10^4) on the GloVe embedding and re-runs the full bootstrap ridge pipeline for both subjects. The judgment call being tested is "is the GloVe ridge result driven by the specific α search range we picked?" — a legitimate hyperparameter judgment call.

| Aspect | Evidence |
|---|---|
| Setup transparency | Table 7 (p.17) lists every setting that is held fixed (subjects, train/test split, bootstrap iterations, chunk length, held-out chunks, selection rule) and exactly what is perturbed (number of α values, α range). |
| Stability of summary CCs | Table 8 (p.17) reports mean/median/top-1%/top-5% CC for GloVe at both α grids, both subjects. Changes are small: Subject 2 mean CC 0.0117 → 0.0114; Subject 3 mean CC 0.0175 → 0.0173. Top-1% changes by ~0.003 in both subjects. |
| Conclusion | "GloVe remained a strong non-contextual embedding model and continued to outperform bag-of-words by a large margin" — same qualitative ranking under the wider grid. Mara's collaboration notes also describe an additional Lab-3.1 story-by-story stability check (Table 4, p.8) that confirms the GloVe ridge model is stable across 21 held-out stories per subject. |
| PCS framing | §"Stability Analysis" (p.17) explicitly invokes PCS — "we chose to evaluate the performance of the ridge regression model for the GloVe embedding using different alpha-grid values, which would allow us to evaluate if our conclusions were responsive to the ridge regularization search range." |

→ Per calibration (lab spec asks for stability of "one judgement call", not breadth across models), **5/5**.

**Column X comment (suggested):** Thoughtful single-judgment-call check (α-grid range widening for GloVe ridge) with explicit setup table, before/after metric comparison, and PCS framing. The conclusion is honest (small numerical movement, same qualitative ranking) and complements the Lab 3.1 story-by-story stability check (Table 4, p.8) that confirms positive CC on 41/42 held-out test stories across both subjects.

---

## Y. Writing (5 pts) — score: 4/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 2.5/3 | Clean structure (Intro → Data → Preprocessing → Lab 3.1 Embeddings → Lab 3.2 BERT → Ridge → Results → Lab 3.3 → Stability → Conclusion → Academic honesty). Math notation correct. Prose is mostly concise. 19 pages — just under the 20-page limit but a bit longer than necessary. Section numbering is inconsistent (most sections use unnumbered bold headings, but the Academic honesty section jumps to "1 Academic honesty / 1.1 LLM Usage"). The conclusion (p.18) is brief and treats the BERT/LoRA mixed-subject result as the headline rather than tying it back to the Lab 3.3 interpretability story. Half-point off for these minor structural roughness issues. |
| Figure quality (0-2) | 1.5/2 | Figures 1-3 (CC histograms, 4-panel BERT histograms, comparison bar chart) are clean and well-captioned with consistent color conventions. SHAP/LIME bar charts (Figs 5-8) are clean and use a consistent blue=SHAP / orange=LIME convention with overlapping word lists visible. Figure 4 (p.12) — the boxplot grid spanning ~46 rows of `bert base/finetuned subject2/3 voxels 0_10000` etc. — is dense and hard to read; the per-chunk decomposition adds visual noise without much insight beyond the histograms. Half-point off for Figure 4 clutter. |

**Column Z comment (suggested):** Well-organized report with a thorough writeup and consistent figure conventions on most plots. LLM usage statement is present (p.19, "Academic honesty / LLM Usage"). Minor docks: Figure 4 (per-voxel-block boxplots) is visually crowded, and the section numbering is inconsistent (most headings unnumbered, academic-honesty section uses §1/§1.1). 19-page length is just under the cap.

---

## Calibration notes — resolved

Both rubric/spec tensions discussed with Sean before grading the rest of the cohort:

1. **Fine-tuned vs LoRA:** LoRA-only earns full credit (6/6). Locked.
2. **Stability check:** Lab spec ("one judgement call") wins over rubric ("1 pt per model"). A thoughtful depth-on-one-model check earns 5/5. Locked.

See `memory/feedback_stat214_lab3_calibration.md` for the durable rule.

---

## Notes for Sean — items worth a personal look

1. **3.3 voxel count.** The lab spec asks for ~5 voxels per story; this group did 2 voxels per story × 2 stories = 4 voxels total. I deducted 0.5 per item (1.5/2 × 6 items = 9/12). If Sean wants to be more lenient given that the discussion per voxel is genuinely thoughtful, +1 to +2 would be defensible. If Sean wants to apply the rubric more strictly, leaving this at 9/12 (or even lower) is also defensible.
2. **3.3 SHAP/LIME implementation.** Group did NOT use the `shap` / `lime` Python libraries. Instead they hand-rolled KernelSHAP-style coalition weighting (`07_interpret_bert.py:263-283`) and LIME-style random-keep masks with exponential-distance kernel weighting (`:286-315`), with the "perturbation" being zeroing the design-matrix rows for that word's TRs. Mathematically reasonable but Sean may want to verify this matches his expectation for the task — the report calls them "SHAP-style" and "LIME-style" so this is at least transparent.
3. **3.3 subject coverage.** Only Subject 2 was interpreted. Lab spec doesn't strictly require both subjects, but most groups do both.
4. **Repo size.** This is the ~4GB repo Sequoia flagged — committed data/models. Not Sean's deduction.
