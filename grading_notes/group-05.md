# Group 05 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-05/` · **Leader row:** spreadsheet row 19 · **Leader handle/name:** ChenfeiP / Chenfei Peng
**Group members (rows 19-22):** Chenfei Peng, Sixian Liu, Ruoyang Li, Victoria Karai
**Report:** `report/lab3.pdf` (20 pages incl. Academic Honesty appendix; main body ~19 pages, under 20-page limit)

---

## Summary scores (Sean's columns)

| Col | Sub-category | Score | Max |
|---|---|---:|---:|
| Q | 3.2 BERT | **12** | 12 |
| S | 3.2 Modeling/Eval | **7.5** | 8 |
| U | 3.3 Interpretability | **8** | 12 |
| W | Stability Check | **5** | 5 |
| Y | Writing | **5** | 5 |
| | **Sean's subtotal** | **37.5** | **42** |

> **Final spreadsheet revision (+1.5 on S):** The "no CV α-search" deduction softened. The group explicitly framed single shared α=10 as a deliberate design choice with story-level held-out test as the validation signal — defensible. Small dock remains for not exercising the bootstrap_ridge CV machinery they had plumbed.

> **Post-audit revision (−4 on U, 12 → 8):** The adversarial audit caught a wrapper-pattern violation the original grader missed. Their SHAP method (`interpret_story_voxels.py:398-419`, `shap_tr_scores_for_voxel`) is exact linear decomposition of `x_z @ weights_v` — not a perturbation at all. Their LIME (`:463-505`, `lime_group_scores_for_voxel`) zero-masks TR-blocks of the already-encoded delayed embedding matrix. Neither re-encodes perturbed text through BERT per the reference guide's wrapper pattern. Same shortcoming as groups 02 and 11; calibrated at the same severity.

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6 since lab spec presents LoRA as the canonical fine-tune. (2) Stability check follows lab spec ("stability of one judgement call"), not the rubric's "1 pt per model" — depth-on-one-judgment-call earns 5/5.

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | Report §2.5 p.5 — `google-bert/bert-base-uncased`, hidden size 768, sliding windows of 128 words with 32-word overlap, subtoken vectors averaged to word level then accumulated across overlapping windows. Code: `build_pretrained_bert_X_by_story.py:57,301-302` (`AutoModel.from_pretrained`), `:162-176` (`sliding_windows`), `:215-230` uses `word_ids` for subtoken-to-word aggregation. |
| Fine-tuned (4) | 4/4 | BERT-LoRA serves as the fine-tuned model (per calibration). §2.6 p.5: MLM fine-tuning of `bert-base-uncased` with adapter rank r=16, α=16, dropout 0.1, target modules {query, key, value, dense}, batch 8, lr 1e-4, 10 epochs. Adapter then loaded onto base backbone and final hidden states extracted for both training and held-out test stories. Code: `train_lora_bert_mlm.py:289-298`. |
| LoRA (2) | 2/2 | Clean PEFT implementation. Code: `train_lora_bert_mlm.py:245` `from peft import LoraConfig, get_peft_model`, `:291-298` builds `LoraConfig` and wraps model. Adapter saved/reloaded via `PeftModel.from_pretrained` in `build_lora_bert_X_by_story.py:141,174`. SLURM script `run_lora_finetune.sh:38` runs with `--target-modules query key value dense`. |
| Performance comparison (2) | 2/2 | Table 1 (p.7) compares all 5 embeddings × 2 subjects on Mean / Median / Top 5% / Top 1% CC. Figure 1 (p.7) plots the same. §3.3 (p.10-11) explicitly walks through pre-trained-BERT vs LoRA-BERT and notes the contextual-vs-static gap dominates over fine-tuned-vs-pre-trained. Within-subject ordering BERT-LoRA > BERT > GloVe > Word2Vec stated and replicated across both subjects. |
| Description of each model (2) | 2/2 | §2.2-2.6 (p.3-5) describes BoW, Word2Vec, GloVe, pre-trained BERT, and LoRA-BERT each in dedicated subsections. Intro p.1-2 also orders them simplest-to-most-complex with motivation. |

**Column R comment (suggested):** Excellent — clear model descriptions, BERT-LoRA implemented cleanly via PEFT with adapters on Q/K/V/dense, comprehensive 5×2×4 performance table. Pre-trained BERT and LoRA pipelines share preprocessing stack, so head-to-head comparison is clean.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 7.5/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | §2.4 p.4 — voxel-wise ridge with explicit form Ŷ = XW and per-voxel CC = corr(Y_test,v, Ŷ_test,v). Standardization on training rows only, applied to test. Chunked fitting in 5,000-voxel blocks for memory. Code: `modeling.py:170-216` wraps `ridge_utils.ridge.bootstrap_ridge`. |
| Cross-validation analysis (2) | **0.5/2** | **No CV actually performed.** §2.4 p.4 states "ridge penalty α = 10 and a single shared alpha" — i.e. a fixed single α, not a CV-selected one. Confirmed in SLURM scripts: `run_modeling_lora.sh:65-67` and `run_modeling_bert_pretrain.sh:72-74` pass `--nboots 0 --alphas 10 --single-alpha`, so `bootstrap_ridge` is called with zero bootstrap folds and a single candidate α. The infrastructure for CV is present (`chunklen`, `nchunks`, `nboots` plumbed through `modeling.py:170-216`) but they did not exercise it. Story-level train/test split is correctly held out (`story_split_seed214.json`, 80 train / 21 test, §2.4 p.4), which avoids leakage at test time but doesn't replace CV-based α tuning. Partial credit for explicit story-level split design and discussion of "single shared alpha" as a deliberate choice, but no candidate-α grid and no held-out validation folds were run. |
| Best-performing model deep-dive (2) | 2/2 | BERT-LoRA identified as best; §3.2 p.7-9 distributional analysis with per-voxel CC histograms (Figs 2, 4) and threshold tables (Table 2). §3.3 p.10-11 deeply unpacks the BERT-vs-LoRA gain: mean/median trends, where the gain concentrates (top 1%/5% tail), and an explicit "What we should not conclude" paragraph framing what the small LoRA improvement does and does not imply. Strong PCS framing. |
| Performance over voxels (2) | 2/2 | §3.2 p.7-9 — full CC histograms across 5 embeddings × 2 subjects (Figs 2, 4), threshold-coverage table (Table 2) reporting % voxels with CC ≥ 0.05 and ≥ 0.10 plus Max CC. Right-skew explicitly characterized, mean-vs-median gap quantified per embedding/subject, and the result motivates restricting interpretation to top-tail voxels for SHAP/LIME — exactly the PCS handoff to §3.4-3.5. |

**Column T comment (suggested):** Ridge pipeline is clean with proper story-level held-out split and chunked fitting. Voxel-distribution analysis (Figs 2, 4 + Table 2) is excellent and ties cleanly to the interpretability threshold. Main deduction: no CV α-search was actually run — the team used a single fixed α=10 with `--nboots 0 --single-alpha`. The course `bootstrap_ridge` utility supports CV (chunks + nboots), so this would have been a small change.

---

## U. Lab 3.3 Interpretability (12 pts) — score: 8/12

> **Wrapper-pattern violation.** SHAP and LIME both run on already-encoded features rather than via the text-perturbation wrapper documented in `Lab3_Reference_Guide.html` §Interpretation. SHAP is literal exact linear decomposition (`shap_tr_scores_for_voxel` at `interpret_story_voxels.py:398-419` — no perturbation at all); LIME zero-masks the delayed embedding matrix rows (`lime_group_scores_for_voxel` at `:463-505`). −1 per (SHAP, LIME) per story = −4 on U.
>
> **Word-disambiguation limitation:** `aggregate_tr_to_words` explicitly divides each TR's SHAP value by the number of words mapped to that TR (`signed = tr_signed[t] / per_tr_counts[t]`), so words co-occurring at the same TR receive **identical (normalized) importance scores**. The wrapper pattern would have given each word a distinct attribution because re-encoding perturbed text produces a different BERT output and downsampled trajectory per word. The cross-method comparison (Tables 4, 6 + cross-voxel SHAP+LIME heatmap in Figs 6, 8) is genuine and earns full Comparison credit.

Two stories analyzed: `souls` (subject 2) and `wheretheressmoke` (subject 3). Top 5 voxels per story × full word-level analysis = 10 voxels total with paired SHAP+LIME. Cross-subject design (one story per subject) is well-motivated by the per-subject voxel-coverage difference (Table 2).

**`souls` (subject 2, 6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 1/2 | Figure 5 left panel (p.12) — SHAP signed-importance bar chart for top voxel 77640. Method (§2.7 p.6): "exact linear contribution implied by the fitted ridge model after z-scoring" — delayed-feature contributions aggregated back to word level. Code: `interpret_story_voxels.py:398-461` (`shap_tr_scores_for_voxel` + `aggregate_*`). **Wrapper-pattern violation**: this is literal exact linear decomposition, not a perturbation. The 5-voxel scope and figures are otherwise solid — but the methodology is off-spec. **−1 for method.** |
| LIME (2) | 1/2 | Figure 5 right panel (p.12) — LIME local-perturbation scores for voxel 77640. Setup §2.7 p.6: random binary masks over word/TR groups, local weighted ridge surrogate, kernel width 0.25, 800 samples, content-word version uses `--drop-stopwords`. Code: `interpret_story_voxels.py:463-528`. **Wrapper-pattern violation**: perturbations zero-mask the **already-encoded delayed embedding matrix** rather than re-encoding perturbed text through BERT. Engineering is clean but on the wrong input space. **−1 for method.** |
| Comparison (2) | 2/2 | Figure 5 (p.12) shows paired SHAP-vs-LIME for the same voxel; Table 4 (p.12) reports overlap counts and example overlapping words across all 5 voxels (overlap 8-10 words per voxel); Fig 6 (p.13) heatmap of combined SHAP+LIME importance across 5 voxels. §3.4 text discusses agreement on content words (mary, kay, thought, just, like, two) vs. method-specific words and frames the divergence as "SHAP captures stable global ridge direction, LIME highlights voxel-specific local perturbation effects". |

**`wheretheressmoke` (subject 3, 6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 1/2 | Figure 7 left panel (p.14) — SHAP for top voxel 90950 (story-level CC 0.417). Table 5 (p.14) lists 5 top voxels with story-level CC 0.39-0.42 — substantially more predictable than `souls`. Same wrapper-pattern violation as story 1. **−1 for method.** |
| LIME (2) | 1/2 | Figure 7 right panel (p.14) — LIME for the same voxel; content-word filtered. Same wrapper-pattern violation. **−1 for method.** |
| Comparison (2) | 2/2 | Table 6 (p.15) overlap counts (6-11 words) per voxel; Fig 8 (p.16) cross-voxel SHAP+LIME heatmap shows stronger shared structure than `souls`. §3.5 text identifies dialogue/scene words (says, said, out, home, cigarettes, beer, man) as cross-method agreements and ties the cleaner overlap to the higher voxel CCs. |

**Column V comment (suggested):** Strong cross-story replication on two subjects, paired SHAP+LIME with both signed-importance plots and overlap tables/heatmaps, content-word filtering applied cleanly. The decision to anchor interpretation in PCS predictability (top voxels with story-level CC > 0.30) is explicitly stated and exercised. Comparison narrative ("SHAP = global ridge direction, LIME = local perturbation sensitivity") is mechanistically grounded and shows up in both stories.

---

## W. Stability Check (5 pts) — score: 5/5

The stability check (§3.8 p.17-19) varies the **interpretation threshold** (the PCS predictability gate that decides which voxels are eligible for SHAP/LIME) across three alternatives: CC ≥ 0.10 (conservative), CC ≥ 0.05 (the value used in the main analysis), and "any positive CC" (lenient). The judgment call being tested is "should we restrict word-level interpretation to high-CC voxels, and if so where do we draw the line?" — a genuine PCS-style choice with a clear non-trivial scientific impact.

| Aspect | Evidence |
|---|---|
| What stays stable | §3.8 "What stays stable" p.18: (i) The contextual > static ordering is preserved at every threshold; LoRA > pre-trained survives at both CC ≥ 0.05 and CC ≥ 0.10 on both subjects. (ii) The 5 top voxels driving the actual SHAP/LIME analysis sit well above all three thresholds (story-level CC up to 0.31 and 0.42), so the qualitative interpretation conclusion is invariant. |
| What legitimately flexes | §3.8 "What legitimately flexes" p.18: the *scope* of interpretation changes — at CC ≥ 0.10 only ~0.05% of subject 2 voxels and ~1.37% of subject 3 voxels qualify; at CC ≥ 0.05 the eligible set is ~3% / ~9%; under the lenient rule >2/3 of voxels are eligible but many are statistically indistinguishable from zero. The threshold is "not a free hyperparameter" — it is determined by the predictability the model achieves. |
| Cross-subject framing | §3.7 (p.17) and Fig 9 (p.17) separately confirm cross-subject stability: BERT-LoRA > BERT > GloVe > Word2Vec ranking is preserved across subjects, only the magnitude differs. This is a complementary stability axis. |
| Final framing | §3.8 closing paragraph (p.18-19): "subject heterogeneity changes the absolute level of CC, while threshold heterogeneity changes the eligible voxel set, but neither changes the methodological ordering of embeddings nor the structure of the SHAP/LIME conclusions on the strongest voxels". This is precisely the PCS-style framing the lab spec is reaching for. Limitations paragraph acknowledges no multi-story bootstrap was run. |

→ Per calibration (depth-on-one-judgment-call done thoughtfully), **5/5**.

**Column X comment (suggested):** Exemplary stability analysis — the team identifies the interpretation threshold as a real PCS judgment call (not a free hyperparameter), perturbs it across three reasonable alternatives, and separates what flexes (the eligible voxel set's *scope*) from what stays stable (the embedding ordering and the qualitative top-voxel interpretation). The §3.7 cross-subject analysis is a clean companion piece. This is the lab spec's "stability of one judgement call" done very well.

---

## Y. Writing (5 pts) — score: 5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 3/3 | Cleanly organized: Intro (§1), Methods (§2.1-2.7 covering preprocessing, BoW, Word2Vec/GloVe, ridge, pre-trained BERT, LoRA, SHAP/LIME), Results (§3.1-3.8 covering embedding comparison, voxel distribution, pre-trained-vs-LoRA, two interpretability stories, scientific interpretation, cross-subject stability, judgment-call stability), Conclusion (§4), Bibliography (§5), Academic Honesty appendix (§A with explicit LLM-usage statement). Prose is dense but readable; "What we should not conclude" (§3.3 p.11) is exactly the kind of careful framing the lab is reaching for. 19 pages of main body — under 20-page limit. |
| Figure quality (0-2) | 2/2 | 9 numbered figures + 6 tables. Figs 1, 9 use consistent color conventions across subjects. Figs 2, 4 are clean 2×3 / 2×4 CC-histogram grids. Figs 5, 7 use the same SHAP-blue / LIME-orange convention for direct visual comparison. Figs 6, 8 are well-designed cross-voxel heatmaps. All captioned with substantive descriptions. Tables 1-6 are LaTeX booktabs style with consistent precision. Minor: histogram sub-titles in Figs 2, 4 use raw filename strings ("subject2: bag_of_words voxel CC distribution") which is a tiny visual nit, but readability is fine. |

**Column Z comment (suggested):** Polished, well-structured 19-page report. Clean modeling math, thorough captions, good color conventions, explicit academic honesty + LLM statement. Discussion paragraphs around what each result does and does not imply are unusually mature. No deductions.

---

## Calibration notes — applied

1. **Fine-tuned vs LoRA:** LoRA-only earns full 6/6 on Fine-tuned + LoRA (per locked rule).
2. **Stability check:** Lab spec ("one judgement call") wins over rubric. Depth-on-interpretation-threshold earns 5/5.
3. **CV deduction:** Sean should sanity-check the 0.5/2 score on CV. The team did set up the `bootstrap_ridge` plumbing for CV but ran with `--nboots 0 --alphas 10 --single-alpha`, so what was actually executed is fixed-α ridge, not CV-α-selected ridge. The story-level held-out split is correctly handled but is not a substitute for the alpha-search the rubric line item is asking about. 0.5/2 reflects "story-level split correctly held out + α explicitly defended as a single shared value" but no CV grid actually ran. If Sean wants to be more lenient (e.g., 1/2 because the held-out story split is itself a form of OOS validation), that would push the total to 40.5/42.
