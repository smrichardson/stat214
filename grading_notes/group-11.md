# Group 11 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-11/` · **Leader row:** spreadsheet row 43 · **Leader handle/name:** Anchor-bkl / Xiaoyang Xiao
**Group members (rows 43-46):** Xiaoyang Xiao (Anchor), Tong An (Olivia), Ka Yan (Kelly), Quan
**Report:** `report/final_report.pdf` (17 pages w/ TOC + appendix; ~15 pages of body — within 20-page limit)

---

## Summary scores (Sean's categories)

| Col | Sub-category | Score | Max |
|---|---|---:|---:|
| Q | 3.2 BERT | **12** | 12 |
| S | 3.2 Modeling/Eval | **8** | 8 |
| U | 3.3 Interpretability | **8** | 12 |
| W | Stability Check | **5** | 5 |
| Y | Writing | **5** | 5 |
| | **Sean's subtotal** | **38** | **42** |

> **Final spreadsheet revisions:**
> - **S 7.5 → 8:** CV-mismatch dock fully refunded. The methodology description is inaccurate but the underlying per-voxel α selection is correct, and the engineering depth elsewhere (differentiable LoRA through Lanczos sinc interpolation) is exceptional.
> - **U 10 → 8:** Strict wrapper dock applied per cross-group consistency. The clean engineering on `shap.KernelExplainer` + from-scratch LIME doesn't change that the input is the wrong object (chunk-masking of `X_story_z` rather than text re-encoding). Matches groups 03/05/08 at −4 wrapper.

> **Above-and-beyond moderation (+0.5 on S):** The CV-mismatch deduction was for the *description* not matching the code (report claims 5-fold CV with α∈{1,10,100,1000,10000}; code uses `bootstrap_ridge(nboots=1)` with `np.logspace(0,3,6)`) — the underlying per-voxel α selection is correct, the writeup is just inaccurate. Given this group's exceptional engineering depth elsewhere (end-to-end differentiable LoRA fine-tuning through Lanczos sinc interpolation against PCA-200 fMRI targets — well above the lab's expected sophistication), the deduction is moderated from −1 to −0.5. Cross-validation analysis goes from 1/2 → 1.5/2.

> **Post-calibration revision (+1 on U, 9 → 10):** Per Sean's updated rule — ≥2 voxels per story is sufficient. This group analyzed top-5 voxels per (subject, story) and the figures show 1 voxel per subject = 2 voxels per story (across both subjects), which meets the ≥2 floor. The Comparison −0.5 per story for "figure shows 1 of 5 voxels" is refunded. The wrapper-pattern −2 (mask-on-X rather than re-encoding through BERT) remains.

> **Post-calibration revision (−2 on U, 11 → 9):** The reference guide (`Lab3_Reference_Guide.html` §Interpretation → The Wrapper Pattern) explicitly prescribes wrapping `text → BERT → downsample → ridge` and perturbing raw text per perturbation. This group's `shap.KernelExplainer` and from-scratch LIME run on the **processed design matrix `X_story_z`** by zero-ing chunk-aligned rows — they never re-encode perturbed text through BERT. The library calls and kernel-weighting math are textbook-clean (cleaner than groups 02 and 09), but the input is the wrong object. Sean covered this in lecture and the reference guide is explicit; smaller deduction than groups 02/09 because the engineering quality is genuinely high.

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6 since lab spec presents LoRA as the canonical fine-tune. (2) Stability check follows lab spec ("stability of one judgement call") — depth-on-one-judgement-call earns 5/5.

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | Report §1.1 p.3 names `bert-base-uncased`; §1.3 motivates pre-trained advantages (semantic compositionality, contextual disambiguation, world knowledge). Code: `code/extract_bert_embeddings.py:99-100` loads `BertModel.from_pretrained("google-bert/bert-base-uncased")`; per-word vectors built by mean-pooling subword tokens in 200-word windows under 512-token cap (`:54-77`). |
| Fine-tuned (4) | 4/4 | LoRA serves as the fine-tuned model (per calibration). Per-story supervised objective: MSE between linear head over delayed embeddings and PCA-200 fMRI targets. Both LoRA params and a linear regression head are trained, while the BERT backbone stays frozen. Code: `code/finetune_bert_lora.py:166-292`. |
| LoRA (2) | 2/2 | Clean PEFT setup: `LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["query","value"], bias="none", task_type=FEATURE_EXTRACTION)` (`finetune_bert_lora.py:207-212`) wrapped via `get_peft_model` (`:213`). Differentiable subword→word pooling via `scatter_add` (`:133-139`) ensures gradients reach LoRA layers through the Lanczos-sinc TR alignment and the delayed-feature stack. |
| Performance comparison (2) | 2/2 | Table 1 (p.3) compares all 5 embeddings × 2 subjects on Mean / Median / Top-1% / Top-5% CC. §1.2 p.4-5 gives quantitative contrasts: BoW→pre-trained BERT (+38%/+74% mean CC), pre-trained→LoRA-BERT (+39%/+29% mean CC); within-subject ordering LoRA-BERT > BERT > {BoW, Word2Vec, GloVe} is stated explicitly. |
| Description of each model (2) | 2/2 | §1.1 p.3 describes preprocessing common to all five embeddings (downsample, trim, delay); §1.3 p.4-5 gives a 4-bullet rationale for each model family (semantic compositionality for w2v/GloVe; contextual disambiguation for BERT; world knowledge from pre-training; domain adaptation via LoRA). Models are introduced in increasing capacity. |

**Column R comment (suggested):** Excellent BERT/LoRA pipeline. LoRA fine-tuning trains end-to-end through Lanczos interpolation + delay stack with a differentiable subword pooling — a sophisticated implementation. Clean PEFT setup on attention Q/V projections.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | Per-voxel L2 ridge via `bootstrap_ridge` from `ridge_utils/ridge.py:290`. Standardization on training stories then applied to test (`lab3_1_modeling.py:352-357`). |
| Cross-validation analysis (2) | 1.5/2 | **Mismatch between report and code.** Report §1.1 says "5-fold CV ... α ∈ {1, 10, 100, 1000, 10000}" — but `lab3_1_modeling.py:59-63` actually uses `ALPHAS = np.logspace(0, 3, 6)` and `bootstrap_ridge(NBOOTS=1, CHUNKLEN=5, NCHUNKS=2)` — a *single* bootstrap held-out fold per voxel, not 5-fold CV. Per-voxel alpha selection is sound, just described inaccurately. **−0.5** (moderated from −1 given above-and-beyond engineering depth elsewhere — see header revision note). |
| Best-performing model deep-dive (2) | 2/2 | LoRA-BERT identified as best; §2.1 p.5-6 frames right-skewed CC distribution against PCS, sets the CC > 0.1 interpretability bar, ties low-CC voxels to non-language cortex. §2.2 p.6-7 + Fig 6 (head-to-head BERT vs LoRA scatter) confirms broad gains (Pearson 0.78/0.83, mean Δ ≈ 0.006-0.007). |
| Performance over voxels (2) | 2/2 | §2.1-2.4 + Figures 3-7 — voxelwise CC distribution per embedding, stacked overlays, box plots, Jaccard top-voxel overlap table (Table 2: top-1% Jaccard 0.226 S2 / 0.312 S3 between all embedding pairs), and CC-band counts. Strong PCS framing throughout. |

**Column T comment (suggested):** Strong voxel-level analysis (Jaccard overlap across embedding pairs, CC-band partitioning, head-to-head BERT vs LoRA scatter). Small methodology-vs-report inconsistency: report claims 5-fold CV with α ∈ {1,10,100,1000,10000}, but code uses a single `bootstrap_ridge` iteration with `np.logspace(0,3,6)`. Per-voxel alpha selection is sound, just described inaccurately. **−1 pt for CV mismatch.**

---

## U. Lab 3.3 Interpretability (12 pts) — score: 8/12

> **Wrapper-pattern violation.** `shap.KernelExplainer` + from-scratch LIME operate on the **already-processed design matrix `X_story_z`** by zero-masking chunks of rows (`lab3_3_interpretation.py:326-379, 415-444, 394-412`). The library calls and kernel-weighting math are textbook-clean, but BERT is never re-invoked per perturbation. −1 per (SHAP, LIME) per story = −4 on U.
>
> **Word-disambiguation limitation (chunk-resolution):** unlike groups 03/05/08/09 which operate at TR-resolution (~2-3 words tied per group), group 11 uses 12-word chunks — all 12 words within a chunk receive the same importance score. The wrapper pattern (re-encode each text perturbation through BERT) would have provided per-word resolution because every single-word removal yields a distinct BERT output and downsampled trajectory.

Two stories analyzed: `sweetaspie` (Part 3.1: representative voxels) and `notontheusualtour` (Part 3.2: validation across stories). For each subject-story pair, top-5 voxels selected by per-story CC (`lab3_3_interpretation.py:207-233`); LIME and SHAP-Kernel run on each with chunk_size=12 words, n_perturb=50 (per §3.3.3 p.12).

**Method issue:** the explanation perturbations are **mask-on-X**, not text-perturbation on the BERT input. The group zero-masks the already-processed (downsampled, delayed) `X_story_z` rows by chunk, then computes importance of word chunks via the ridge-predicted voxel response (`lab3_3_interpretation.py:326-379`). The reference guide explicitly prescribes wrapping `text → BERT → downsample → ridge` so SHAP/LIME perturb raw text and the effect propagates through re-encoding. Mask-on-X bypasses BERT's contextual sensitivity — removing word X cannot change the embedding of word Y. Engineering is clean (proper `shap.KernelExplainer` + textbook LIME kernel weighting), but the input is the wrong object.

**`sweetaspie` (6 pts → score 5/6):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 1/2 | `run_shap_kernel` (`lab3_3_interpretation.py:415-444`) uses `shap.KernelExplainer` over binary chunk masks, with adaptive background/explain sample sizes. Figure 10 (p.13) shows SHAP scores for representative voxels (S2 vox 16006, S3 vox 67682). −0.5 for mask-on-X rather than re-encoding through BERT (per reference guide). |
| LIME (2) | 1/2 | `run_lime_style` (`:394-412`) implements LIME from scratch: binary masks with keep_prob=0.75, cosine-distance kernel with width = √n_chunks × 0.75, weighted linear regression — i.e., the standard LIME local-surrogate recipe. Figure 10 shows LIME bars. −0.5 for mask-on-X. |
| Comparison (2) | 2/2 | Figure 10 overlays LIME and SHAP for one voxel each per subject (2 voxels per story across both subjects — meets the ≥2 floor per the updated rule). §3.3.2 p.12 states "voxel 16006 from 'Sweet as Pie' showed strong agreement between SHAP and LIME." Jaccard overlap table is computed in code (`compute_overlap`, `:525-557`) but not shown in the report — minor visibility issue but not a deduction. |

**`notontheusualtour` (6 pts → score 5/6):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 1/2 | Same SHAP-Kernel methodology. Figure 11 (p.13) shows SHAP bars for representative voxels (S2 vox 35507, S3 vox 64460). −0.5 for mask-on-X. |
| LIME (2) | 1/2 | Same LIME methodology. Figure 11 shows LIME bars overlaid with SHAP. −0.5 for mask-on-X. |
| Comparison (2) | 2/2 | Same overlay format as Fig 10. §3.3.4 p.14 "Key Findings": (1) strong cross-method agreement on top words, (2) content words dominate function words, (3) robustness across stories. 2 voxels per story across both subjects meets the ≥2 floor per the updated rule. |

**Column V comment (suggested):** Both required stories analyzed with cleanly engineered SHAP (`shap.KernelExplainer`) and from-scratch LIME implementations. Main deduction: explanations are computed via masking processed-X rows by chunk rather than re-encoding perturbed text through BERT — the reference guide explicitly prescribes the wrapper pattern (`text → BERT → downsample → ridge`) so the effect of removing a word propagates through Lanczos blending and BERT's contextual mechanism. Mask-on-X misses both. Smaller deduction than groups 02/09 because the engineering quality is genuinely high. Minor secondary issue: figures only display one of the top-5 voxels per (subject, story); the per-voxel Jaccard table is computed in code but not surfaced in the report.

---

## W. Stability Check (5 pts) — score: 5/5

The stability check varies the **temporal delay stack** used to align word embeddings with fMRI response. Baseline is `[1,2,3,4]`; alternatives are `[1,2,3]`, `[2,3,4]`, `[1,2]`, `[0]`, `[1,2,3,4,5,6]` (§4 p.14-15). Whole pipeline rerun with each delay config.

| Aspect | Evidence |
|---|---|
| Why this judgement call matters | §4.1 p.14 — explicit motivation: BOLD signal peaks 4-6s after stimulus, but exact delay window is a modeling choice; if results depend heavily on one specific delay then the model is brittle. |
| Stability of held-out CC | Table 4 p.15 reports mean CC for both subjects across 6 delay configurations. Baseline [1,2,3,4] is best (S2: 0.0216, S3: 0.0308); no-delay [0] is worst (0.0134, 0.0196). Performance degrades gradually under misspecification (~15-40% drop). |
| Robustness of main claim | §4.3 p.15 — three clean conclusions: (1) baseline is empirically optimal, (2) LoRA-BERT remains best across all configurations, (3) early delays carry most usable signal (removing early lags hurts more than removing late ones, consistent with hemodynamic onset). |
| PCS framing | §4.3 p.15 closing paragraph: "main conclusion does not depend on a narrow hyperparameter setting." Solid PCS framing. |

→ Per calibration, **5/5**.

**Column X comment (suggested):** Strong stability check — depth on one judgment call (the temporal delay stack), with 6 perturbation variants, rerun end-to-end through ridge fitting, and a clean three-point conclusion. Tied directly back to neurobiological motivation (early lags carry the strongest semantic signal).

---

## Y. Writing (5 pts) — score: 5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 3/3 | Well-organized: §1 embedding comparison, §2 voxel-level analysis, §3 interpretability, §4 stability, §5 conclusion & limitations. Each section has clear roadmap/setup paragraphs. Limitations §5 honestly notes small absolute CC, post-hoc nature of SHAP/LIME, ridge linearity. 17 pages including TOC and academic-honesty appendix — well within 20 limit. Academic honesty + LLM-usage paragraphs present (App. A). |
| Figure quality (0-2) | 2/2 | 11 figures + 4 tables. Figure 1 (mean CC bar plots) clean and compact; Figure 5 (boxplots) shows IQR/whisker structure cleanly; Figure 6 (BERT-vs-LoRA scatter + ΔCC histogram) is a strong cross-model visualization; Figure 7 (CC-band counts) gives an interpretable categorical breakdown. SHAP-vs-LIME overlay format (Figs 10, 11) uses consistent color encoding (LIME blue, SHAP orange). Minor: a few bar charts in Fig 7's pie/bar pair are a bit small but readable. |

**Column Z comment (suggested):** Polished, well-organized report with thoughtful limitations section. Strong use of overlay figures (BERT-vs-LoRA scatter, SHAP-vs-LIME bar overlays). Honest about both methodology and limitations.

---

## Calibration notes — applied

1. **Fine-tuned vs LoRA:** LoRA-only earns full 6/6. Applied.
2. **Stability check:** Depth-on-one-judgement-call earns 5/5. Applied.

## Notes for Sean before approving

- **CV mismatch (−1 pt on S):** Report claims 5-fold CV with α ∈ {1, 10, 100, 1000, 10000}, but code is `bootstrap_ridge(nboots=1, chunklen=5, nchunks=2)` with `ALPHAS=np.logspace(0,3,6)`. The alpha selection is per-voxel and uses a held-out fold from training, so the core idea is correct — but the description is not faithful to the code. Worth flagging in feedback even if the deduction is small.
- **Mask-on-X interpretability (−2 on U, post-calibration):** Their LIME/SHAP perturb the processed feature matrix `X_story_z` by zero-ing chunk-aligned rows, rather than re-encoding perturbed text through BERT. Per the reference guide, the wrapper pattern is the canonical way; smaller dock than groups 02/09 because the SHAP/LIME engineering itself is textbook-clean.
- **Comparison figure coverage (−0.5+−0.5 on U):** Top-5 voxels are computed per (subject, story), but report figures only show 1 representative voxel per (subject, story) overlay. The Jaccard overlap table is computed in code but not surfaced in the report. Minor.
- All large files (model PKLs, embeddings) appear to be properly excluded; ~4GB repo size is Sequoia's concern, not Sean's.
