# Group 01 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-01/` · **Leader row:** spreadsheet row 3 · **Leader handle/name:** alexchsoh / Alex Soh
**Group members (rows 3-6):** Alex Soh, David Pan, Zhiyi Rao, Tobias Roemer
**Report:** `report/lab3.pdf` (15 pages, well under 20-page limit)

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

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6 since lab spec presents LoRA as the canonical fine-tune. (2) Stability check follows lab spec ("stability of one judgement call"), not the rubric's "1 pt per model" — depth-on-one-model earns 5/5.

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | Report §2.2 p.3 "Pre-trained BERT" — uses uncased `bert-base-uncased`, 768-dim, overlapping 512-token windows with stride 256, subtoken averaging + window-level averaging. Code: `generate_embeddings_bert.py:267` `AutoModel.from_pretrained(model_name)`. |
| Fine-tuned (4) | 4/4 | BERT-LoRA serves as the fine-tuned model (per calibration: LoRA = canonical fine-tune in lab spec). MLM fine-tuning on training stories with frozen base weights and trainable adapters on Q/V projections, then encoder reused to embed both training and held-out test stories. |
| LoRA (2) | 2/2 | Clean PEFT-based implementation. Code: `generate_embeddings_bert_lora.py:235` builds `LoraConfig`, `:242` wraps with `get_peft_model`, target modules configurable via CLI. Report explains adapter placement on attention sub-modules (Q, V at each layer). |
| Performance comparison (2) | 2/2 | Report Table 1 (p.5) compares all 5 embeddings × 2 subjects on Mean / Median / 95th / 99th CC. Section 4.1 text gives three concrete patterns (contextual > static, BoW ≈ static baselines, Subject 3 > Subject 2). Within-subject ordering BERT-LoRA > BERT > {BoW, W2V, GloVe} is stated explicitly. |
| Description of each model (2) | 2/2 | §2.2 p.2-3 describes BoW, Word2Vec, GloVe, pre-trained BERT, and BERT-LoRA each in 1-paragraph blocks, ordered "simplest to most complex". |

**Column R comment (suggested):** Excellent — clear model descriptions, BERT-LoRA implemented cleanly via PEFT on attention Q/V projections, comprehensive performance table covering 5 embeddings × 2 subjects × 4 metrics. No deductions.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | §3.1 p.3-4 — per-voxel ridge with explicit objective `‖y − Xw‖² + α‖w‖²`, 4d input from lagged features, motivation for L2 over OLS (wide design matrix, correlated lag features). |
| Cross-validation analysis (2) | 2/2 | 5-fold story-level CV (no story spans both halves of any fold), candidate α ∈ {0.1, 1, 10, 100}, alpha-search done on randomly sampled 8000-voxel subset then α transferred to full voxel grid. Final eval on 21 held-out test stories. Reports mean/median/95th/99th per subject-embedding (Table 1). |
| Best-performing model deep-dive (2) | 2/2 | BERT-LoRA identified as best; §4.1-4.2 walks through why contextual representations dominate, why Subject 3 has higher CCs than Subject 2 (subject-level SNR differences in BOLD), and §4.2 frames the right-skewed CC distribution against the PCS predictability bar. |
| Performance over voxels (2) | 2/2 | §4.2 p.5-6 + Figure 1 (10-panel CC histogram grid) — explicitly notes "mean CC remains under 0.02 for both subjects" despite a small upper tail reaching ≈0.2, ties this to known language-cortex anatomy ("majority of voxels not predicted at all"), motivates restricting interpretation to top voxels for SHAP/LIME under PCS. |

**Column T comment (suggested):** Clean ridge pipeline with story-level CV (no leakage), thoughtful α-search subsetting, and PCS-aware voxel analysis. Per-voxel CC distributions plotted for all 5 embeddings × 2 subjects.

---

## U. Lab 3.3 Interpretability (12 pts) — score: 12/12

Two stories analyzed: `avatar` and `treasureisland`, top 5 voxels × 3 TRs each = 30 attribution pairs total. Subject 3 only (the higher-SNR subject), which is well-motivated.

**`avatar` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Figure 5 (p.9) — SHAP top-5-voxel grid with signed bar charts. Method described §5.1: partition explainer with `Text` masker, 500 evaluations per (voxel, TR) pair. Code: `interpret_voxel.py:230-235`. |
| LIME (2) | 2/2 | Figure 4 (p.9) — LIME top-5-voxel grid. Setup §5.1: `LimeTextExplainer` with `bow=False` (so word order preserved), whitespace split, 1000 perturbation samples. Code: `interpret_voxel.py:205-207`. |
| Comparison (2) | 2/2 | Figure 2 (p.8) — head-to-head LIME-vs-SHAP bar chart at voxel 38105/TR5; both rank "tom" near the top. Cross-method discussion §5.2: methods agree on content words (proper nouns, salient verbs), disagree on function words ("the", "and", "to"); attributes disagreement to LIME's deletion-sampling picking up high-frequency tokens. |

**`treasureisland` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Figure 7 (p.10) — top-5-voxel SHAP grid for treasureisland. |
| LIME (2) | 2/2 | Figure 6 (p.10) — top-5-voxel LIME grid for treasureisland. |
| Comparison (2) | 2/2 | Figure 3 (p.8) — side-by-side LIME/SHAP for voxel 38161/TR148, both methods independently rank "kill" as strongest positive driver with near-identical magnitudes (+0.062 LIME, +0.064 SHAP). Threat-related content discussed. |

**Column V comment (suggested):** Both stories thoroughly analyzed with paired SHAP+LIME on top 5 voxels × 3 TRs each. Strong cross-method comparison, careful PCS-style voxel filtering (only voxels passing predictability bar are interpreted), and content-word-vs-function-word divergence explained mechanistically.

---

## W. Stability Check (5 pts) — score: 5/5

The stability check varies BERT context window size (512 → 256, 128) and reruns the full downstream pipeline: ridge fitting + SHAP/LIME interpretation. The judgment call being tested is "should we use 512-token windows for embedding extraction?" — a real preprocessing choice.

| Aspect | Evidence |
|---|---|
| Stability of held-out CC distributions | Table 2 (p.11) reports mean/median/top-5%/top-1% CC for BERT-LoRA at windows 128, 256 vs the original 512. Figure 8 (p.12) shows full CC histograms for all 4 stability variants. CCs are reasonably stable across windows (within ~10% on top-1%). |
| Stability of top-voxel identities | Table 3 (p.11): Window 256 retains 0/5 of the original top voxels; window 128 retains 2/5. → Voxel-level rankings are **unstable**. |
| Stability of word-level conclusions | Figures 9-10 (p.12-13) — different voxels surface at window 128/256, but the same kinds of words ("tom", "kill") still emerge as top drivers at matched TRs. → Word-level conclusions are **stable**. |
| Final framing | §7.2 distills a two-tier conclusion: voxel identity is sensitive to embedding choice, but word-level semantic content is robust. This is exactly the PCS-style framing the lab is reaching for. |

→ Per calibration (lab spec asks for stability of "one judgement call", not breadth across models), **5/5**.

**Column X comment (suggested):** Exemplary stability analysis — depth-on-one-judgment-call (BERT context window size) executed end-to-end including downstream interpretation. Table 3's voxel-overlap finding (0/5 at window 256, 2/5 at window 128) and the resulting word-level-stable / voxel-level-unstable conclusion is exactly the PCS framing the lab spec is reaching for.

---

## Y. Writing (5 pts) — score: 5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 3/3 | Cleanly organized 8-section structure (Intro, Data/Features, Modeling, Results, Interpretability, Stability, Discussion, Conclusions). Clear roadmap paragraph at end of intro. Sections build logically. Math notation correct. Prose is concise and the contributions are clearly stated. 15 pages — well under 20-page limit. |
| Figure quality (0-2) | 2/2 | 10 figures total, all numbered and captioned with descriptive titles. Figure 1 (CC histogram grid) and Figure 8 (stability histogram grid) are clean and consistent. The SHAP/LIME bar charts (Figs 2-7, 9, 10) use a consistent positive=blue / negative=red convention. Minor: grid figures (4, 5, 6, 7) have small per-subplot text; could be larger but is still readable. |

**Column Z comment (suggested):** Polished, well-structured report. Clean modeling math, well-captioned figures with consistent color conventions. No deductions.

---

## Calibration notes — resolved

Both rubric/spec tensions discussed with Sean before grading the rest of the cohort:

1. **Fine-tuned vs LoRA:** LoRA-only earns full credit (6/6). Locked.
2. **Stability check:** Lab spec ("one judgement call") wins over rubric ("1 pt per model"). A thoughtful depth-on-one-model check earns 5/5. Locked.

See `memory/feedback_stat214_lab3_calibration.md` for the durable rule.
