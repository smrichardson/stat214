# Group 03 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-03/` · **Leader row:** spreadsheet row 11 · **Leader handle/name:** Akanshaa18 / Akanshaa Borgohain
**Group members (rows 11-14):** Akanshaa Borgohain, Linhai Wang, Selina Yu, Victor Sana
**Report:** `report/lab3.pdf` (16 pages — under 20-page limit. 14 pages of content + bibliography + academic honesty appendix.)

---

## Summary scores (Sean's columns)

| Col | Sub-category | Score | Max |
|---|---|---:|---:|
| Q | 3.2 BERT | **10** | 12 |
| S | 3.2 Modeling/Eval | **8** | 8 |
| U | 3.3 Interpretability | **8** | 12 |
| W | Stability Check | **5** | 5 |
| Y | Writing | **5** | 5 |
| | **Sean's subtotal** | **36** | **42** |

> **Final spreadsheet revisions:**
> - **Q 10.5 → 10:** Description tightened to 1/2 (their static-vs-contextual contrast is present but per-model coverage genuinely thin).
> - **S 7 → 8:** Modeling deep-dive deduction lifted; bootstrap CV + per-voxel α selection are fine.
> - **U 6 → 8:** Comparison items lifted to 2/2 each story per "credit back for legitimate word-level analysis" — vocab projection produces real per-word output even though wrapper is wrong. SHAP/LIME items stay at 1/2 each (strict −4 wrapper dock).
> - **Y 4.5 → 5:** Figure quality refunded; SHAP wordclouds are unconventional but readable.

> **Post-rescan revision (+0.5 on Y, 4 → 4.5):** The Readability dock at 2.5/3 cited three issues, the first being "the BERT-not-better-than-GloVe finding framed scientifically rather than as a pipeline concern" — but the underlying pipeline issue is already docked on Q.Pre-trained (−1). Removing this double-count: Readability 2.5/3 → 3/3 (the remaining section-numbering and informal-phrasing concerns are too minor to dock on their own). Figure quality stays at 1.5/2 (wordclouds-losing-signed-info, dense LIME bar charts, missing y-axis labels are real).

> **Post-review revision (+1 on Q.Description, 0.5 → 1.5):** Per Sean's clarification — the Description item is primarily about whether the group demonstrates understanding of the static-vs-contextual distinction, not whether they wrote deep per-model architecture sections. This group's §3 p.2 explicitly contrasts GloVe (polysemy-blind) vs BERT (contextual), which satisfies the conceptual bar. The remaining −0.5 reflects that the BoW/W2V coverage and per-model algorithm/training detail are still thin.

> **Post-calibration revision (+2 on Q LoRA, 0 → 2):** Per Sean's updated rule — full-parameter MLM fine-tuning is at least as demanding as LoRA, so a group that does thoughtful full-FT in lieu of LoRA earns the full Fine-tuned (4) + LoRA (2) = 6/6. This group implemented end-to-end MLM fine-tuning in `code/fine_tune_bert_mlm.py` (clean training loop, proper masking, no test-set leakage). Refunded.

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6 — but this group did NOT implement LoRA; they did MLM fine-tuning of the full base model instead. They get full Fine-tuned credit, but LoRA is missing. The `fine_tune_bert.py` stub explicitly says "LoRA was investigated as a parameter-efficient alternative; MLM fine-tuning was chosen instead." (2) Stability is deep + two judgment calls (delay window AND seed split). 5/5.

---

## Q. Lab 3.2 BERT (12 pts) — score: 10/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 1/2 | Report §3 p.2 + §4 p.5 — uses `google-bert/bert-base-uncased`, 768-dim. Code: `code/bert_pipeline.py:19` (`MODEL_NAME = "google-bert/bert-base-uncased"`), `:80` `AutoModel.from_pretrained(MODEL_NAME)`. **Major methodological issue:** each word is passed to BERT as its own input sequence (`bert_pipeline.py:39-50`, `max_length=16` per word). The tokenizer itself is fine; the problem is that the model's forward pass only ever sees one word at a time, so self-attention has no neighbors to attend to. Every "BERT vector" is the subtoken-averaged embedding of the word as a single-word sequence — no sentence context. This defeats the purpose of using a contextual model and likely explains the report's finding that BERT does not outperform GloVe (§4.4, §4.9). Deduct 1 for missing sentence-level encoding / sliding windows. |
| Fine-tuned (4) | 4/4 | MLM fine-tuning implemented end-to-end in `code/fine_tune_bert_mlm.py`. Code: `:142` loads `AutoModelForMaskedLM`, `:73-110` proper MLM masking (80% [MASK] / 10% random / 10% unchanged), training loop `:151-178` with AdamW, only fine-tunes on train stories (`:127` "Fine-tuning only on train stories to avoid test leakage"). Report §3 p.3 describes the procedure. Light-touch (80 steps, 600 chunks per `:28`, `:138`) but valid implementation. Generous credit per calibration. |
| LoRA (2) | 2/2 | LoRA not implemented per se, but full-parameter MLM fine-tuning of `bert-base-uncased` is implemented end-to-end (`code/fine_tune_bert_mlm.py`). Per Sean's updated calibration rule — full-MLM-FT satisfies the LoRA item (it is at least as demanding as LoRA, and the lab spec presents LoRA as one example of fine-tuning rather than a requirement). The Fine-tuned 4 and LoRA 2 are both earned by the same thoughtful FT pipeline. |
| Performance comparison (2) | 2/2 | Table 1 (p.3) — Mean / Median / Top 1% / Top 5% CC for all 5 embeddings × 2 subjects. §4.1-4.5 walk through each method. Figure 6 (p.7) overlaid CC distributions, Figure 7 (p.7) boxplots, Figure 8 (p.8) top-percentile bars. Very thorough cross-embedding comparison. |
| Description of each model (2) | 1.5/2 | §3 p.2 has a bulleted contrast of GloVe (polysemy-blind) vs BERT (contextual). BoW, Word2Vec, GloVe, pre-trained BERT, fine-tuned BERT are named and the **static-vs-contextual distinction is correctly identified** — which is the conceptual bar this item is reaching for. Per-model algorithm/training-corpus/dimensionality detail is still thin, but the core understanding is present. **−0.5** for the thin per-model coverage. |

**Column R comment (suggested):** Fine-tuning pipeline is clean and well-implemented end-to-end — full-MLM-FT serves as both the Fine-tuned and LoRA items per calibration. **Critical issue with pre-trained BERT extraction:** the embedding-extraction loop (`bert_pipeline.py:39-50`) passes each word to BERT as its own input sequence with no surrounding words. The tokenizer itself works correctly; the bug is at the forward-pass level — self-attention has no neighbors to attend to, so the contextual embeddings collapse to "BERT's representation of word X with no context." This converts BERT into an effectively static embedder and almost certainly explains the report's surprised finding that BERT does not outperform GloVe. The report does not acknowledge this. Per-model descriptions are also thin. Performance comparison itself is excellent.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | §3 p.2-3 describes per-voxel ridge motivated by high-dim, correlated lag features. Code: `code/modeling.py:186-237` `run_ridge_pipeline` calls `bootstrap_ridge` from `ridge_utils`. Train/test standardized from train statistics (`modeling.py:80-102`). Story-level split via `choose_story_split` (`modeling.py:38-61`) avoids TR-level leakage — explicitly justified in report §3 p.3. |
| Cross-validation analysis (2) | 1.5/2 | Bootstrap CV via `bootstrap_ridge`: α grid `{0.1, 1, 10, 100, 1000}` (`run_modeling_bert_finetuned.py:25`), `nboots=5`, `chunklen=40`, holdout fraction 0.2 (`modeling.py:209-227`). Report §3 p.3 describes "bootstrapped cross-validation within the training set" with "contiguous time chunks" to preserve BOLD autocorrelation. Slight deductions: only 5 bootstrap iterations (light), and no discussion of which α was actually selected per voxel or whether the boundary (1000) was hit. Mostly clean. |
| Best-performing model deep-dive (2) | 1.5/2 | Best model is GloVe (per Table 1 and §4.9), not BERT. They acknowledge this is surprising and offer multiple explanations: fMRI resolution, downsampling smoothing out contextual nuance, BERT high-dim → stronger regularization (§4.4 p.5, §7 p.14-15). §4.6 introduces PCS threshold (CC > 0.05), §4.7 examines whether GloVe and BERT predict the same voxels (Figure 10 scatter, r=0.804 for Subj 3). Deduct 0.5 because the deep-dive is split between GloVe and "BERT/BERT-FT as a comparison," and the report uses the *fine-tuned* BERT for downstream interpretation (§6) — but as noted under Q, the underlying BERT extraction has a major confound, which the deep-dive does not interrogate. |
| Performance over voxels (2) | 2/2 | §4.1-4.4 + Figures 1-5 — per-voxel CC histograms for each embedding × subject. §4.6 + Figure 9 PCS threshold analysis (fraction of voxels with CC > c for varying c). §4.7 GloVe-vs-BERT scatter (Figure 10), §4.8 Subject-2-vs-Subject-3 scatter (Figure 11) — careful comparison of voxel-level overlap and cross-subject correspondence. Thoughtful. |

**Column T comment (suggested):** Clean ridge pipeline with bootstrap CV and story-level splitting (no leakage). PCS threshold analysis and voxel-overlap scatterplots are nice. Minor deductions: only 5 bootstrap iterations, no per-voxel α distribution reported, and the GloVe-vs-BERT deep-dive does not interrogate why BERT might be underperforming (the in-isolation tokenization in `bert_pipeline.py` is the most plausible cause but goes unmentioned).

---

## U. Lab 3.3 Interpretability (12 pts) — score: 8/12

> **Word-disambiguation limitation (wrapper-pattern consequence):** SHAP runs on the 3072-d feature space; values are L2-normed per TR (`tr_importance = np.linalg.norm(shap_per_dim, axis=1)`) then assigned to words via `word_importance_shap = tr_importance[word_tr_idx]`. Words that map to the same TR therefore receive **identical importance scores** — the perturbation acts at the BERT-dim level, not at the word level. The wrapper pattern (re-encode each text perturbation through BERT) would have disambiguated co-occurring words because removing each one produces a distinct BERT output and downsampled trajectory. This is the structural gap the −4 wrapper dock reflects.

Two stories analyzed: `becomingindian` and `beneaththemushroomcloud`. Only Subject 2 (the lower-SNR subject — surprising choice given §4.8 shows Subject 3 has higher CCs). **Only 2 voxels** analyzed (4031, 2858), not 5. They use LIME on the *tabular feature space* (3072-d), not on text — they import `lime_tabular`, not `lime_text`.

**`becomingindian` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 1/2 | Figure 14a (p.13), Figure 15a (p.13) — SHAP wordclouds for voxels 2858 and 4031. Method §6.1: `shap.LinearExplainer` (exact closed-form for ridge). Code: `SHAP.py:9` `shap.LinearExplainer((w_v, 0.0), train_features, feature_perturbation="interventional")` runs on **3072-d tabular features**, not raw text via the wrapper pattern. Word attributions reconstructed post-hoc via the `word_tr_idx` mapping. Same wrapper-pattern violation as LIME — **−1 for methodology** (cross-group consistency rule). |
| LIME (2) | 1/2 | Figure 16a (p.14) — LIME bar chart for voxel 2858. **Methodological concern:** uses `LimeTabularExplainer` on the 3072-d delayed feature space (`LIME.py:6-11`), not `LimeTextExplainer` on text. Top words are mostly function words ("the", "and", "to", "of") — they correctly identify this is because LIME's local perturbations cannot resolve the 3072-d feature space with 500 samples (§6.2). Only 500 perturbations is well below standard (1000-5000). The report frames this honestly as a "high-dimensional struggle" rather than fixing it. Deduct 1 for choice of tabular LIME + thin sampling. |
| Comparison (2) | 1/2 | §6.2 p.14 compares SHAP and LIME at the conceptual level — "SHAP is clearly the better tool for this job. It handles the high-dimensional BERT space effortlessly, whereas LIME is probably better suited for simpler, lower-dimensional tasks." Notes LIME still surfaced `reconciliation` and `priest`, which appear in SHAP. But no head-to-head visualization (no side-by-side bar comparison for the same voxel/TR like Group 01 had). Deduct 1 for absent quantitative comparison. |

**`beneaththemushroomcloud` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 1/2 | Figure 14b (p.13), Figure 15b (p.13) — SHAP wordclouds. Top words "driven", "back", "hotel", "crowd", "anniversary", "Normandy". Plausible content interpretation. Same wrapper-pattern violation as story 1 — **−1 for methodology**. |
| LIME (2) | 1/2 | Figure 16b (p.14), Figure 17b (p.14) — LIME bar charts. Same tabular-LIME / 500-perturbation issue as above. Mostly function words at top, same caveats. Deduct 1. |
| Comparison (2) | 1/2 | Same generic SHAP-better-than-LIME discussion §6.2, applied uniformly across both stories. No story-specific divergence analysis. Deduct 1. |

**Column V comment (suggested):** Strong SHAP analysis using `LinearExplainer` (mathematically exact for ridge — good choice). Two main issues. (1) **Only 2 voxels** are analyzed (the rubric implies more — top 5 is the norm); the lab spec is flexible here, but two is on the low end. (2) **LIME uses `LimeTabularExplainer` on the 3072-d feature space** rather than `LimeTextExplainer` on the actual text. With only 500 perturbations against 3072 features, LIME unsurprisingly fails to find signal — the report acknowledges this but does not switch to the appropriate text-based LIME variant. The comparison section is conceptual rather than empirical (no side-by-side bar chart for a specific voxel/TR). Honest framing throughout — they correctly diagnose LIME's failure mode.

---

## W. Stability Check (5 pts) — score: 5/5

Two judgment calls probed end-to-end: **delay window** (1, 2, 3, 4 delays) and **train/test split seed** (42, 123, 456, 789, 999).

| Aspect | Evidence |
|---|---|
| Delay window stability | §5.1 p.11 + Figure 12 — CC distributions across `delays ∈ {{1}, {1,2}, {1,2,3}, {1,2,3,4}}`. Overall CC distribution flat across configs (good — main results are robust). Top-1% CC decreases monotonically as delays added — they correctly diagnose this as a "regularization artifact" (each delay multiplies feature dim by 768, making ridge harder to fit). Justify chosen `{1,2,3,4}` as canonical hemodynamic convention. Code: `delay_sensitivity.py:32-37`. |
| Seed/split stability | §5.2 p.11-12 + Figure 13 — mean CC across 5 random train/test splits for all 4 main embeddings × 2 subjects. Coefficients of variation reported: GloVe ≈ 7.5/10.2%, W2V ≈ 7.1/6.4%, BERT ≈ 8.6/5.6%, BoW most variable. Critically: **the ranking of embeddings is preserved across all seeds** (GloVe ≈ W2V ≈ BERT > BoW). This is exactly the right stability claim — qualitative conclusions are robust. Code: `stability_across_seeds.py:41-73`. |
| Final framing | §5 conclusion p.12 — "our conclusions about the relative merits of each embedding are robust to both the hemodynamic delay configuration and the choice of train/test split." Clean PCS-style framing. |

→ Two well-executed stability analyses on real judgment calls, with quantitative coefficients of variation and a clear robustness conclusion. **5/5**.

**Column X comment (suggested):** Exemplary stability section — two real judgment calls (delay window, seed split) probed end-to-end with quantitative CV metrics. The finding that embedding *ranking* is preserved across seeds even when *absolute* CCs vary is exactly the PCS-flavored claim the lab is asking for. The monotonic top-1% decrease with more delays is correctly diagnosed as a regularization artifact rather than a substantive effect.

---

## Y. Writing (5 pts) — score: 5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 3/3 | 8-section structure (Intro, Data, Methods, Results, Stability, Interpretability, Conclusion, Bibliography). Clear roadmap in intro. Prose is generally clean and the surprising-finding-about-BERT motivates much of the report. 16 pages, well under limit. Section §6.1 has a nice equation for SHAP. Some informal phrasing and an unusual section ordering (stability appears as §5 but Methods §3 covers fine-tuning), but these are too minor to dock. The BERT-pipeline framing issue is already addressed via the Q.Pre-trained deduction. |
| Figure quality (0-2) | 1.5/2 | 17 figures total, all numbered and captioned. CC histograms (Figs 1-5) are clean and consistent. Figure 8 top-percentile bars are nice. Figure 10 GloVe-vs-BERT scatter is informative. **Issues:** (a) SHAP wordclouds (Figs 14, 15) are visually striking but it's hard to read scores — wordclouds collapse magnitude to font-size which loses the +/− sign of SHAP attribution; (b) LIME bar charts (Figs 16, 17) crammed and small. (c) Figure 12 has no y-axis units clearly labeled on left panel. Deduct 0.5. |

**Column Z comment (suggested):** Generally well-written and well-structured. Figures comprehensive (17 total). Two writing issues to flag: (a) the surprising "BERT ≈ GloVe" finding is framed scientifically rather than as a pipeline concern, when the most parsimonious explanation is that `bert_pipeline.py` passes each word to BERT as its own input sequence (max_length=16, no surrounding words), so self-attention sees no context — BERT collapses to a quasi-static embedder. (b) SHAP wordclouds visually attractive but lose signed attribution information.

---

## Additional notes for Sean

- **LLM usage statement:** Present in Appendix A.2 (p.16). Says LLMs used for plotting code, code review, README, and grammar/clarity polishing on the report — no AI-generated sections. Clean and appropriate.
- **Academic honesty statement:** Present in Appendix A.1 (p.16).
- **Collaboration statement:** Present in `report/collaboration.txt` (separate file).
- **Code organization:** Decent. Separate pipeline files per embedding (`bag_of_words_pipeline.py`, `word2vec_pipeline.py`, `glove_pipeline.py`, `bert_pipeline.py`, `bert_finetuned_pipeline.py`), centralized `modeling.py` and `utils.py`, ridge in `ridge_utils/`. Shell scripts for orchestration (`job.sh`, `interpret.sh`, etc.).
- **Pipeline confound to be aware of:** The pre-trained-BERT embedding extraction (`code/bert_pipeline.py` and `code/bert_pipeline_slow.py`) passes each word to BERT as its own input sequence with `max_length=16`. Tokenization is fine — the issue is that BERT's forward pass only ever sees one word at a time, so self-attention has no neighbors to attend to and the resulting vectors are functionally context-free. Fine-tuning *was* done on text chunks (so the model learned context during MLM), but then the embeddings are *extracted* word-by-word using the same single-word pipeline (`bert_finetuned_pipeline.py:39-50` is byte-identical to `bert_pipeline.py:39-50`). So neither the pre-trained nor the fine-tuned downstream embeddings actually use BERT's contextual capacity. This is almost certainly why BERT performs no better than GloVe in their results.
- **Voxel count for interpretability:** Only 2 voxels (4031, 2858), both for Subject 2. A third voxel (260) is in the results figures directory but not in the report.
- **LoRA absent:** Worth flagging in feedback that LoRA was the canonical parameter-efficient fine-tune in the lab spec; substituting full MLM fine-tuning is fine for the "fine-tuned" credit but does not also cover the LoRA credit.

---

## Calibration notes — applied

1. **Fine-tuned vs LoRA:** Calibration says LoRA-only earns full credit (6/6 across both). This group did *full MLM* fine-tuning, not LoRA. So Fine-tuned = 4/4 (good), but LoRA = 0/2 (not implemented). The calibration runs one way only: LoRA substitutes for full-fine-tune, not the reverse.
2. **Stability check:** Lab spec ("one judgement call") wins over rubric ("1 pt per model"). Two judgment calls done thoroughly → 5/5.
3. **Be generous on partial credit:** Applied — LIME and per-model descriptions still earn partial credit despite methodological weakness, since the work is present and honestly framed.
