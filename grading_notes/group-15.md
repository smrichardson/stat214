# Group 15 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-15/` · **Leader row:** spreadsheet row 59 · **Leader handle/name:** DonjhaiH / Donjhai Holland
**Group members (rows 59-62):** Larissa, DJ (Donjhai Holland), Keyi, Lyla (per `report/collaboration.txt`)
**Report:** `report/lab3.pdf` (19 pages, just under 20-page limit)

---

## Summary scores (Sean's columns)

| Col | Sub-category | Score | Max |
|---|---|---:|---:|
| Q | 3.2 BERT | **12** | 12 |
| S | 3.2 Modeling/Eval | **8** | 8 |
| U | 3.3 Interpretability | **11** | 12 |
| W | Stability Check | **5** | 5 |
| Y | Writing | **5** | 5 |
| | **Sean's subtotal** | **41** | **42** |

> **Final spreadsheet revisions:**
> - **S 7 → 8:** α-grid-ceiling deduction lifted. Group flagged the issue honestly in their conclusion; this is a tuning observation rather than a methodological error.
> - **U 10 → 11:** Prose-only cross-method comparison dock softened (some quantitative observation present).
> - **Y 4.5 → 5:** Figure-not-embedded-in-PDF dock lifted; the figure exists on disk and the absence is administrative.

> **Above-and-beyond moderation (+0.5 on Y, 4 → 4.5):** Group implemented all three BERT variants (pretrained, full MLM fine-tuning, AND LoRA via PEFT) with proper Slurm-resilient training and clean adapter merging via `merge_and_unload()` — above the lab's bar. The Y −1 deduction was split between (a) Triangle Shirtwaist signed-driver figure missing from PDF body (existed on disk in `figs/shap/`) and (b) some other minor issues. The (a) component is borderline administrative — the figure exists, just wasn't embedded — so the deduction is moderated from −1 to −0.5.

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6 since lab spec presents LoRA as the canonical fine-tune. (2) Stability check follows lab spec ("stability of one judgement call"), not the rubric's "1 pt per model" — depth-on-one-model earns 5/5.

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

This group implemented **all three** BERT variants — pre-trained, full MLM fine-tuning, AND LoRA fine-tuning — which is more than required.

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | Report §2.2 p.3 — uses `bert-base-uncased` from Hugging Face hub with no further training. Word-level vectors recovered via `is_split_into_words=True` and `word_ids()` to average subword hidden states. Non-overlapping 250-word chunks (stays within BERT's 512-token limit), last hidden layer. Code: `lab3_2_extract_bert_features.py:154-201` (`_build_model`) and `:208-280` (`_story_word_embeddings`). |
| Fine-tuned (4) | 4/4 | Both full MLM fine-tuning AND LoRA fine-tuning are implemented (`lab3_2_train_bert_mlm.py`, `lab3_2_lora_bert.py`). MLM trained on training stories only (no test leakage), 128-token non-overlapping chunks, 15% mask probability (80/10/10 mask/random/keep), AdamW, LR 5e-5, 100 warmup steps, weight decay 0.01, 3 epochs, batch 16, fp16. Checkpoints every 500 steps to guard against Slurm wall-clock kill. Encoder reused for both train and held-out test embedding extraction (`extract_bert_features.py:178-180`, `variant=ft`). |
| LoRA (2) | 2/2 | Clean PEFT-based implementation. Code: `lab3_2_lora_bert.py:232-233` imports `LoraConfig, get_peft_model`; `:268-275` builds `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["query","value"], bias="none")` and wraps base `BertForMaskedLM`. Effective scale α/r = 2. Trains ~300K params vs ~110M base. MLM loss 2.79 → 2.45 in 200 steps reported in §2.2 p.4. Adapter merged via `merge_and_unload()` before encoder extraction (`extract_bert_features.py:191-194`). |
| Performance comparison (2) | 2/2 | Table 3 (p.7) compares all 5 embeddings on Subject 2 (Mean r, Median r, Frac r>0). Figure 4 (p.8) bar chart visualizes the same 5×2 metric grid. Three-tier hierarchy explicitly stated p.8: BoW ≈ random → GloVe/W2V (~50× improvement) → all three BERT variants (~10× more). The three BERT variants are reported as "nearly identical" with mean r ∈ [0.0217, 0.0221], and the small LoRA advantage is mentioned. |
| Description of each model (2) | 2/2 | §2.1 (p.3) describes BoW, Word2Vec, GloVe; §2.2 (p.3-4) describes pre-trained BERT, MLM-fine-tuned BERT, and LoRA BERT in three 1-paragraph blocks. LoRA paragraph (p.4) includes the explicit decomposition formula `ΔW = (α/r) BA` and motivates the rank-16 / α-32 / Q+V-target choice with reference to Hu et al. 2022. |

**Column R comment (suggested):** Excellent — three BERT variants (pre-trained, full MLM, LoRA) all implemented and compared. Clean PEFT-based LoRA on Q/V projections with explicit math, sensible hyperparameters, and proper checkpoint resumption logic for Slurm. Five-embedding × subject performance comparison reported as both table and bar chart.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | §3 (p.4) and §1.2 (p.2) describe per-voxel ridge with delayed-stimulus design matrix `X_story = [E'_{t-1}, E'_{t-2}, E'_{t-3}, E'_{t-4}] ∈ R^{T_s × 4p}`. 4 delays (1, 2, 3, 4 TRs ≈ 2-8s post-stimulus) motivated by HRF lag (peak 4-6s, baseline by 15s). Code: `lab3_2_ridge_bert.py:222-225` uses `sklearn.linear_model.RidgeCV`. Static-embedding pipeline uses `bootstrap_ridge` from `ridge_utils` with bootstrap CV (`lab3_1_fit_ridge.py:171-182`). |
| Cross-validation analysis (2) | 1.5/2 | §3.2 p.5 — leave-one-out cross-validation is used for α selection across `alphas=[1, 10, 100, 1000, 10000, 100000]` (default in `lab3_2_ridge_bert.py:139`). The chosen α was 10^5 uniformly across all voxels and all three BERT variants. Half-point dock: the conclusion (p.18) explicitly acknowledges that "the regularization grid used for ridge regression was bounded at α = 10^5, and since this was the grid maximum for the BERT models, it is possible that the optimal value lies beyond what we explored" — i.e. the CV may not have actually identified an interior optimum. They flag it transparently but did not iterate. Static-embedding ridge uses bootstrap CV (`lab3_1_fit_ridge.py:171-182`, 8 boots × 10 chunks × 10 TRs/chunk) which is a separate (and more thorough) CV protocol. |
| Best-performing model deep-dive (2) | 2/2 | BERT-LoRA identified as best (Table 1, p.6; Figure 2, p.6). §3.2 p.5-6 walks through the global mean/median r per BERT variant for Subject 2, plus per-story breakdown in Table 2 (p.7) and Figure 3 (p.6) over all 19 test stories. Story-level variation (~6× range from `lawsthatchokecreativity` 0.007 → `sweetaspie` 0.042) noted as far exceeding cross-variant variation (~1.02×). |
| Performance over voxels (2) | 1.5/2 | Figure 1 (p.5) shows per-voxel CC histograms for the three static embeddings × 2 subjects (6 panels). Figure 5 (p.9) shows the per-story positive-correlation voxel fraction for the three BERT variants. §3.4 (p.8-9) explicitly notes "vast majority of voxels show r ≈ 0, with only a small subset exhibiting meaningful correlations," ties this to known neurobiology (localized vs distributed language processing), and motivates restricting SHAP/LIME to top-CC voxels under PCS. Half-point dock: there is no histogram of BERT per-voxel CCs analogous to Figure 1 — the BERT-variant evidence is summary statistics + the positive-fraction boxplot. The 5×2 voxel-CC histogram grid that most groups produce is split across Fig 1 (static only) and Fig 13 (violin, all 6 embeddings × 2 subjects, in the stability section). |

**Column T comment (suggested):** Solid pipeline overall with clear preprocessing math (HRF delays, Lanczos downsample, story-level train/test split), per-voxel ridge, and PCS-aware framing for downstream interpretability. Two issues: (i) the RidgeCV grid topped out at α = 10^5 and that's exactly the alpha that got selected — the group flags this honestly in the conclusion but didn't extend the grid; (ii) no per-voxel CC histogram for the BERT variants in §3.2 (only the static three). The violin plot in Figure 13 partially covers this in the stability section.

---

## U. Lab 3.3 Interpretability (12 pts) — score: 11/12

Two stories analyzed: `Sweetaspie` and `The Triangle Shirtwaist Connection`. Top 5 voxels selected by held-out test-story CC under BERT-LoRA (the best model). Main interpretability analysis (SHAP + LIME) is on **Subject 2 only**; Subject 3 LIME is added in the stability section but no Subject 3 SHAP. PCS-style filtering: only voxels above the 95th-percentile CC threshold (≈0.041 for S2, ≈0.067 for S3) interpreted (§4 p.9).

**SHAP setup:** `shap.Explainer` with `shap.maskers.Text(r"\s+", mask_token="[MASK]")` — masks rather than deletes, preserving sequence structure (`lab3_3_shap_lime.py:135-138`).
**LIME setup:** `LimeTextExplainer` with `num_features=20`, `num_samples=500`, two-class wrapper around the signed predictor (`lab3_3_shap_lime.py:148-157`).
**Wrapper:** `fMRI_Predictor` class (`lab3_3_shap_lime.py:27-78`) replays the full BERT-embedding → Lanczos-downsample → trim → delay → ridge pipeline for each perturbed text input.

**`Sweetaspie` (6 pts → score 5/6):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Figure 6 (p.10) global word importance bar chart (top 25 words, mean \|SHAP\| across 5 voxels); Figure 7 (p.11) voxel × word heatmap (5 voxels × 35-word vocabulary); Figure 9 (p.12) per-voxel signed driver bar charts for all 5 top voxels. Figs available in `figs/shap/plot1_*.pdf`, `plot2_heatmap_Sweetaspie_*.pdf`, `plot4_signed_drivers_Sweetaspie_*.pdf`. |
| LIME (2) | 2/2 | Figure 11 (p.14) aggregate LIME word importance for Sweetaspie · S2 (top 15 words across 5 voxels). Per-voxel LIME plots in `figs/lime/sweetaspie_s2/lime_voxel_*.pdf` (5 voxels). |
| Comparison (2) | 1/2 | §4.4 (p.16) discusses LIME-vs-SHAP at the prose level: LIME emphasizes story-specific words (letter, public, president, epidemic); SHAP captures more conversational/narrative-structural tokens. "said" appears across multiple voxels in both methods, framed as a stable cross-method signal. No paired bar chart or quantitative overlap metric, and the comparison is uneven across the two stories. Half-point dock for prose-only comparison, half-point dock because the comparison narrative leans on Triangle Shirtwaist examples rather than Sweetaspie specifically. |

**`The Triangle Shirtwaist Connection` (6 pts → score 5/6):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Figure 6 (p.10, right panel) global word importance; Figure 8 (p.11) voxel × word heatmap. The per-voxel signed-driver figure for Triangle Shirtwaist is in `figs/shap/plot4_signed_drivers_The_Triangle_Shirtwaist_Connection_*.pdf` but is not embedded in the report (only the Sweetaspie signed-driver figure made it in). Discussion of "mother", "hospital", "me" as top SHAP drivers reflects narrative content. |
| LIME (2) | 2/2 | Figure 10 (p.13) aggregate LIME word importance for Triangle Shirtwaist · S2 (top 15 words; "said" 6.51, "course" 5.16, "asked" 5.01, "letter" 3.94, "seen" 3.87). Figure 12 (p.15) shows top-15 LIME words for 3 of the 5 voxels (6826, 52577, 52624). Per-voxel LIME plots saved in `figs/lime/triangle_s2/lime_voxel_*.pdf` (5 voxels). |
| Comparison (2) | 1/2 | §4.4 p.16 — "letter", "public", "president", "epidemic" highlighted as LIME-emphasized story-specific words that SHAP does not weight as highly. The text attributes this to LIME's deletion (vs SHAP's [MASK] replacement) and notes the deletion-based importance changes temporal structure of the sequence. "said" appears as a stable cross-method, cross-voxel signal. Same half-point docks as Sweetaspie — prose-only comparison, no head-to-head bar chart or overlap quantification per voxel. |

**Column V comment (suggested):** Solid SHAP+LIME implementation on top 5 voxels × 2 stories using the actual `shap` and `lime` Python libraries with sensible configuration (Text masker / `[MASK]`, deletion-based LIME with 500 samples, 20-feature ceiling). Three SHAP visualizations per story (global, heatmap, signed drivers) is thorough. Main deductions are around the cross-method comparison: it stays at the prose level (no overlap bar chart or quantitative comparison metric), and the Sweetaspie-specific comparison narrative is thinner than the Triangle Shirtwaist one. The per-voxel signed-driver figure for Triangle Shirtwaist exists in `figs/shap/` but did not get embedded in the report. Interpretability analysis is single-subject (Subject 2); Subject 3 LIME re-runs appear in the stability section but no Subject 3 SHAP.

---

## W. Stability Check (5 pts) — score: 5/5

The stability check is a **cross-subject comparison**: rerun the embedding-ranking analysis and the LIME interpretability for Subject 3, then compare against the Subject 2 results to assess whether (a) the embedding hierarchy and (b) the top LIME words are stable across subjects. The judgment call being tested is "do conclusions generalize across subjects, or are they Subject-2 specific?" — a real PCS-style appearance check.

| Aspect | Evidence |
|---|---|
| Embedding-hierarchy stability across subjects | Figure 13 (p.16) — violin plot of per-voxel CC for all 6 embeddings × 2 subjects. The BoW → W2V/GloVe → BERT/BERT-LoRA hierarchy reproduces on Subject 3, but Subject 3 has uniformly higher CCs (more signal) across all embeddings. Group attributes this to subject-level SNR rather than modeling inconsistency. |
| LIME stability (Sweetaspie) | Figures 14a/14b (p.17) — aggregate LIME word importance for Sweetaspie · S2 vs · S3. Top words overlap substantially ("said", "not", "his") but Subject 3 has several **negative** LIME contributions (bearing, meet, color, see) — group correctly explains this is sign asymmetry, not worse model performance. |
| LIME stability (Triangle Shirtwaist) | Figures 15a/15b (p.17) — "said" dominates in both subjects (very large LIME score). Beyond the top word, overlap is weaker — only "twenties" shared. Group acknowledges the tail is not shared. |
| Honest framing | §4.5 (p.16) and §5 (p.18) — explicit conclusion that "the dominant LIME words remain similar across both subjects, despite Subject 3 showing higher overall CC values and some differences in the sign of individual word contributions." Limitations of having only Subject 3 LIME (no Subject 3 SHAP) acknowledged in §5 future work. |

→ Per calibration (lab spec asks for stability of "one judgement call", done thoughtfully), **5/5**.

**Column X comment (suggested):** Cross-subject stability check executed across both the modeling (embedding-ranking) and interpretability (LIME word-importance) layers. Conclusion is honest — the BoW→W2V/GloVe→BERT/LoRA hierarchy reproduces on Subject 3 and the top LIME word ("said") is stable, while individual sign patterns shift. Acknowledged limitation (no Subject 3 SHAP) is appropriately flagged for future work. Thoughtful and well-framed.

---

## Y. Writing (5 pts) — score: 5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 2.5/3 | Clean structure (Introduction with Data and Feature Construction sub-sections → Embedding Methods → Modeling → Interpretation → Conclusion → Academic Honesty). Math notation correct (HRF delays, ridge objective, LoRA decomposition all stated cleanly). Prose is concise. 19 pages — just under the 20-page limit. Two small structural issues: (i) figures and tables are interleaved in a slightly cluttered way (Figure 6 split across both stories, Figure 7/8 separate but Figure 6 combined); (ii) the conclusion (§5) is honest about limitations but does not tie the SHAP/LIME findings back to neurobiology as crisply as the introduction promises. Half-point off for these. |
| Figure quality (0-2) | 1.5/2 | 15 figures + 3 tables, mostly well-captioned. Strong figures: Figure 1 (per-voxel CC histograms), Figure 4 (5-embedding bar chart), Figure 13 (cross-subject violin plot). Weaker: Figure 6 caption labels for the two SHAP global-importance panels are small; Figure 11 / Figure 14 axis text is also small at the embedded size. The per-voxel signed-driver SHAP figure exists for Triangle Shirtwaist (`figs/shap/plot4_signed_drivers_The_Triangle_Shirtwaist_Connection_·_S2.pdf`) but was not embedded in the report — only the Sweetaspie version (Fig 9) made it in. Half-point off for inconsistent SHAP-figure coverage across the two stories. |

**Column Z comment (suggested):** Well-organized 19-page report with clean math, consistent figure conventions, and an explicit LLM-usage statement (§6.2 p.19) covering both coding and writing assistance. Minor docks: the per-voxel SHAP signed-driver figure for Triangle Shirtwaist did not make it into the body (it exists in `figs/shap/`); axis text in several aggregate-importance bar charts is small at the embedded size; conclusion could tie SHAP/LIME findings back to neurobiology more crisply.

---

## Calibration notes — resolved

Both rubric/spec tensions discussed with Sean before grading the rest of the cohort:

1. **Fine-tuned vs LoRA:** LoRA-only earns full credit (6/6). Locked. (This group did all three variants — pre-trained, full MLM, AND LoRA — so full credit is straightforward.)
2. **Stability check:** Lab spec ("one judgement call") wins over rubric ("1 pt per model"). A thoughtful depth-on-one-model check earns 5/5. Locked.

See `memory/feedback_stat214_lab3_calibration.md` for the durable rule.

---

## Notes for Sean — items worth a personal look

1. **Ridge α grid hit the ceiling.** `lab3_2_ridge_bert.py` used RidgeCV with alphas up to 10^5, and 10^5 was selected uniformly for every BERT variant. The group flags this transparently in the conclusion (§5 p.18) but did not rerun with a wider grid. I docked half a point on the CV item. If Sean wants to be stricter (since the CV essentially never converged to an interior optimum) he could dock more. If Sean considers the transparent acknowledgement sufficient, +0.5 is defensible.
2. **3.3 cross-method comparison stays at the prose level.** No paired bar charts, no overlap metric, and the comparison text leans more on Triangle Shirtwaist than Sweetaspie. I docked 1 pt per story (2 pts total). If Sean considers the §4.4 discussion adequate, +1 to +2 is defensible.
3. **Triangle Shirtwaist per-voxel SHAP figure not embedded.** The figure exists in `figs/shap/plot4_signed_drivers_The_Triangle_Shirtwaist_Connection_·_S2.pdf` but was not put in the PDF. The Sweetaspie equivalent (Fig 9, p.12) is embedded. I factored this into the writing dock; could also be argued under U. Triangle Shirtwaist SHAP item.
4. **Subject coverage for 3.3.** Main analysis is Subject 2 only; Subject 3 LIME appears in the stability section but no Subject 3 SHAP. Lab spec doesn't strictly require both subjects so I did not dock for this in U (only noted as a limitation the group themselves flag).
5. **All three BERT variants implemented.** This group went beyond the lab requirement and trained both full MLM fine-tuning AND LoRA. Worth recognizing in qualitative feedback even though it doesn't add to the rubric score.
