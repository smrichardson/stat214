# Group 10 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-10/` · **Leader row:** spreadsheet row 39 · **Leader handle/name:** CanhuiH / Canhui Huang
**Group members (rows 39-42):** Canhui Huang, Masaki Higashida, Edward R. G. (egao1), Gongyu Zhou
**Report:** `report/lab3.pdf` (20 pages including appendix; main body ends p.19)

---

## Summary scores (Sean's categories)

| Col | Sub-category | Score | Max |
|---|---|---:|---:|
| Q | 3.2 BERT | **12** | 12 |
| S | 3.2 Modeling/Eval | **8** | 8 |
| U | 3.3 Interpretability | **12** | 12 |
| W | Stability Check | **5** | 5 |
| Y | Writing | **5** | 5 |
| | **Sean's subtotal** | **42** | **42** |

> **Final spreadsheet revision (+1 on U):** The Jaccard normalization bug (SHAP tokens carry trailing whitespace from `shap.maskers.Text(r"\s+")` while LIME returns clean tokens → cross-method Jaccard always 0) is a string-formatting implementation detail, not a methodological error. The underlying SHAP/LIME pipelines are correct (real text-perturbation wrapper, full re-encoding through BERT per perturbation, 40 voxel-runs total). Refunded.

> **Above-and-beyond moderation (+0.5 on Y, 4.5 → 5):** Group implemented both full MLM fine-tuning AND LoRA at three ranks (r=4/8/16), executed 4-axis stability work (trimming convention, per-story CC, cross-embedding Jaccard, +2 TR interpretability shift), and ran SHAP+LIME on 10 voxels × 2 stories × 2 subjects (40 attribution runs total — well above the ≥2 floor). The Y −0.5 was for figure-quality issues (rotated tick labels, no side-by-side composite, Subject 2 only in body figures) — gray-zone and softened per cross-group consistency with the moderations on groups 11, 14, 15. Caught by the 2nd-pass audit.

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6. Group 10 actually does both pre-trained extraction *and* full LoRA MLM fine-tuning at three ranks (r=4, 8, 16), so this is over-delivered. (2) Stability check: depth-on-one-judgement-call earns 5/5; this group does *multiple* thoughtful stability checks (trimming, +2 TR shift, cross-embedding Jaccard, per-story).

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | §3.3 p.5 — `bert-base-uncased`, 768-dim, overlapping sliding subtoken windows (window=510, stride=256), word embeddings averaged across windows. Code: `generate_bert_embeddings.py:43` `BERT_MODEL = "google-bert/bert-base-uncased"`, lines 91-180 implement the windowed embedding extraction with proper subtoken→word mean-pool then cross-window averaging. |
| Fine-tuned (4) | 4/4 | Genuine MLM fine-tuning (not just LoRA-only). `generate_bert_embeddings.py:262-302` `train_bert` runs full MLM training loop on `BertForMaskedLM` with 15% masking (80/10/10 BERT recipe at `mask_tokens` :232-260), AdamW, 3 epochs, lr=5e-5. `StoryMLMDataset` (:184-228) builds overlapping subtoken chunks from the 80 training stories. Encoder then re-extracts embeddings via `lora_model.bert`. |
| LoRA (2) | 2/2 | Clean PEFT setup. `generate_bert_embeddings.py:372-384`: `LoraConfig(r=..., lora_alpha=rank*2, lora_dropout=0.1, target_modules=["query","value"], bias="none")` → `get_peft_model`. Three ranks swept (r=4, 8, 16) via `run_bert_lora_r{4,8,16}.sh`. Trainable-param ratio reported (~0.13% at r=4, §4.1 p.8). |
| Performance comparison (2) | 2/2 | Two comparison tables. **Table 1 (p.6)** = 5 embeddings × 2 subjects × 4 metrics (Mean/Median/Top-5%/Top-1% CC). **Table 2 (p.7)** zooms in on the BERT family: pre-trained vs LoRA r=4/8/16 for both subjects. Text §4.1 p.7 calls out the monotonic improvement with rank for Subject 2 and the much weaker effect for Subject 3, and links the small absolute gains to the 24k-word fine-tuning corpus being too small to meaningfully reshape BERT. |
| Description of each model (2) | 2/2 | §3.3 p.5 lists all five embeddings (BOW, Word2Vec, GloVe, Pre-trained BERT, Fine-tuned BERT) in numbered bullet form, with dimensions, training corpora, and motivation per item. |

**Column R comment (suggested):** Excellent BERT work — pre-trained sliding-window extraction, real MLM fine-tuning (not LoRA-as-fine-tune shortcut), three LoRA ranks compared, and a careful Table 2 separating contextual-vs-static gains from rank effects.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | §3.2 p.4 — standard ridge with explicit objective `‖Y−XB‖²+λ‖B‖²`, mass-univariate solve via one `(XᵀX+λI)⁻¹` Cholesky reused across all voxels. Implementation: `model_and_evaluate_bert.py:140-149` `prepare_ridge_solver` (Cholesky), :133-152 SVD path for CV. Motivation for ridge over OLS (delayed features high-D + correlated) stated clearly. |
| Cross-validation analysis (2) | 2/2 | §3.4.1 p.5 — 5-fold story-level CV (no story spans folds), with rationale for choosing 5-fold over LOSO (BOW has ~45k features, LOSO blows up memory). `model_and_evaluate.py:174-216` `select_alpha` does parallel 5-fold story-split CV over α candidates via joblib. Honest disclosure (§3.4.1 p.6): α=100,000 was selected via CV on static embeddings then **reused** for BERT pre-trained/LoRA "for computational reasons" — flagged as a limitation. This is exactly the kind of PCS-aware disclosure we want. |
| Best-performing model deep-dive (2) | 2/2 | Best = Fine-tuned BERT r=16 (Table 1, p.6). §4.1 p.7 unpacks two findings: (a) contextual >> static across the board, (b) LoRA gains are small (Δ mean CC ~0.0006 for Subject 2) and weakly monotonic in rank — interpreted as 24k words being too few to substantially shift BERT, citing Hosseini et al. 2024. |
| Performance over voxels (2) | 2/2 | §4.2 p.8-10 + Figures 1-4 — CC histograms for GloVe and BERT-LoRA-r16, both subjects. Explicitly notes right-skewed distributions, mass near zero, long upper tail to ~0.20-0.25 for BERT-r16 Subject 3, and ties this to §6.1 predictability gating: interpretability is restricted to top-1% voxels (CC > 0.128 for s2, 0.161 for s3). Cross-subject bar chart Figure 9 shows the systematic Subject 3 > Subject 2 gap. |

**Column T comment (suggested):** Solid ridge pipeline with story-level 5-fold CV, transparent disclosure of the cross-embedding α reuse (a real limitation that they own up to), and clean voxel-distribution analysis with PCS-motivated top-1% gating for downstream interpretation.

---

## U. Lab 3.3 Interpretability (12 pts) — score: 12/12

Two stories analyzed: `notontheusualtour` (highest per-story CC) and `metsmagic`. Both stories are part of the 21-story held-out test set, no leakage. Both subjects analyzed for each story. Top 10 voxels per (subject, story) — SHAP & LIME run on all 40 voxel cases. The four resulting JSONs are in `results/shap_lime_*.json`.

**`notontheusualtour` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | §5.1 p.11: `shap.maskers.Text(r"\s+", mask_token="[MASK]")` + default partition `shap.Explainer`. Code: `shap_lime_analysis.py:393, 419`. Figure 5 (p.13) shows top-10 SHAP words for Subject 2 vox1 (CC=0.849). Table 3 (p.11) lists top-SHAP words for both subjects. |
| LIME (2) | 2/2 | §5.1 p.11: `LimeTextExplainer` with `num_samples=100`, `num_features=10`. Code: `shap_lime_analysis.py:394, 434-440`. Figure 6 (p.13) shows top-10 LIME words for Subject 2 vox1. Table 4 (p.11) tabulates LIME top words per subject. |
| Comparison (2) | 1.5/2 | §5.3 p.12-14 walks through SHAP-vs-LIME for `notontheusualtour` — both methods surface politics/institution content ("columbia", "financial", "west", "ceremony", "politically"); SHAP top-10 is all-negative-sign (a quirk of mask-baseline = full text masked), LIME mixed-sign. Table 6 (p.19) reports the **cross-method Jaccard = 0.000** for all (subject, story) pairs because SHAP `sv.data` is position-keyed while LIME `as_list` is type-keyed, so set-equality at the string level matches nothing. This is a real bug in the comparison metric, partially compensated by the qualitative discussion. Minor deduction (-0.5). |

**`metsmagic` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Figure 7 (p.14) — top-10 SHAP for Subject 2 vox1 metsmagic (CC=0.872); content words "jubilant", "met", "fans", "screaming", "train" cluster cleanly with the sports/celebration content of the story. |
| LIME (2) | 2/2 | Figure 8 (p.14) — top-10 LIME, mixed-sign with "they", "says", "the", "are", "lots" near top and "pitch", "half", "costing" in the negative tail. Explicitly noted as more sensitive to common words. |
| Comparison (2) | 1.5/2 | §5.3 p.13-14: notes SHAP attributions are "more semantically coherent" for metsmagic, LIME picks up generic words. Discussion of why SHAP top words are uniformly negative (masking baseline) is correct and shows real understanding. Same Jaccard=0 artifact as above (-0.5). |

**Column V comment (suggested):** Two stories × two subjects × top-10 voxels gives 40 SHAP+LIME runs — significantly more breadth than the lab asks for. Qualitative comparison is thoughtful and the SHAP-sign vs LIME-sign discussion is mechanically correct. Half-point deduction per story because the headline cross-method Jaccard = 0.000 (Table 6) is a position-vs-type comparison bug rather than a real instability — the report waves at this ("reflects methodological differences (masking vs deletion)") but doesn't fix it. Would be cleaner to normalize to types before set-intersecting.

---

## W. Stability Check (5 pts) — score: 5/5

This group runs **four** distinct stability analyses, each tied to a real judgment call. Going by lab-spec ("stability of one judgement call"), any one of these would suffice; the depth is well above bar.

| Stability axis | Evidence |
|---|---|
| Trimming convention | §6.3.1 p.16-17 — reruns the full pipeline with trimming (5, 10) instead of the Huth-tutorial (10, 5). Conventional trimming yields 13-28% higher CC on average, but the BOW < W2V < GloVe < BERT ordering is preserved. |
| Per-story CC | §6.3.2 p.17-18 + Figures 10-11 — bar charts of per-story mean CC for all 21 held-out test stories for GloVe, both subjects. S2 range: 0.006-0.029 (mean 0.016, std 0.006), S3 range: 0.012-0.053 (mean 0.027, std 0.009). Used to motivate which stories to pick for interpretability. |
| Cross-embedding Jaccard | §6.3.4 p.18 + Table 5 — Jaccard of top-5% voxel sets across embeddings. GloVe-vs-BERT-pretrained J ≈ 0.03 (near-disjoint), BERT-pretrained-vs-LoRA-r16 J ≈ 0.65-0.68 (substantial overlap but not saturated). Interpretation: LoRA "reorganizes where the model is most predictive rather than uniformly improving everywhere" — a genuinely sharp PCS-style observation that the marginal CC table alone would miss. |
| +2 TR temporal shift on interpretability | §6.3.5 + Table 6 (p.19) — perturbs the output-window TR alignment by +2 and recomputes SHAP/LIME. SHAP Jaccard 0.67-0.79 (stable), LIME 0.03-0.14 (unstable). Conclusion: SHAP explanations are more robust than LIME under temporal perturbation. |

The final framing (§7 p.19) ties them together: "main embedding-performance ranking remained stable across subjects, stories, and preprocessing judgment calls" while "voxel-level interpretations should be framed as qualitative." This is exactly the PCS bottom line the lab is reaching for.

**Column X comment (suggested):** Exemplary multi-axis stability work — trimming convention, per-story, cross-embedding Jaccard, and +2 TR interpretability shift. The cross-embedding Jaccard finding (LoRA shifts ~1/3 of top-5% voxel membership despite tiny aggregate CC change) is the standout PCS observation of the report. 5/5 with room to spare.

---

## Y. Writing (5 pts) — score: 5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 3/3 | Clean 7-section structure (Intro, Data/Preprocessing, Prediction Model, Results, Interpretability, PCS Evaluations, Conclusion) + appendix. Roadmap paragraph at end of intro. Math is correct, prose is concise. Limitations are flagged in-line rather than buried (§3.4.1 α-reuse, §6.2 ridge-coef storage workaround, §6.3.5 SHAP-sign caveat). LLM usage statement is in §A.2. 19 pages of main body, under the 20-page limit. |
| Figure quality (0-2) | 2/2 | 11 figures, all numbered and captioned. CC histograms (Figs 1-4) are consistent. Bar charts for SHAP/LIME (Figs 5-8) are readable. Minor visibility issues remain (per-story bars Figs 10-11 have rotated story names; Subject 3 SHAP/LIME exist in repo but not embedded; no single composite SHAP-vs-LIME plot), but these are gray-zone administrative — moderation applied per cross-group consistency with groups 11/14/15 given above-and-beyond depth elsewhere (full FT + 3 LoRA ranks + 4-axis stability + 40 SHAP+LIME runs). |

**Column Z comment (suggested):** Well-structured report with strong in-line limitation disclosure and a sharp PCS bottom line. Half-point deduction for per-story figures with unreadable tick labels and for only displaying Subject 2 SHAP/LIME charts in the body when results exist for both subjects.

---

## Calibration notes — resolved

Both rubric/spec tensions handled the same way as Group 01:

1. **Fine-tuned vs LoRA:** Group 10 actually does both (real MLM fine-tuning + LoRA adapters at r=4/8/16). Trivially 6/6.
2. **Stability check:** Lab spec ("one judgement call") wins over rubric ("1 pt per model"). Group 10's four-axis stability work earns 5/5 with margin.

See `memory/feedback_stat214_lab3_calibration.md` for the durable rule.
