# Group 12 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-12/` · **Leader row:** spreadsheet row 47 · **Leader handle/name:** evianaphm / Eviana Pham
**Group members (rows 47-50):** Eviana Pham, Sharazad Ali, Mark Szekacs, Ziyu Ge (per `report/collaboration.txt`)
**Report:** `report/lab3.pdf` (19 pages including appendix/bibliography/LLM statement; main body ~16 pp — under 20-page limit)

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

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6; this group did BOTH full FT and LoRA. (2) Depth-on-one-judgment-call earns 5/5; this group did TWO stability checks, both substantive.

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | §3.2 p.2 — `google-bert/bert-base-uncased`, BERT producing 768-d subword hidden states, fast-tokenizer word alignment, mean-pooled subwords back to original words, overlapping story windows so words are embedded with surrounding context. Code: `code/lab32/embeddings_bert_pretrained.py:113-114` loads `BertTokenizerFast` + `BertModel.from_pretrained("google-bert/bert-base-uncased")`; `:38-101` implements overlapping chunks (default `chunk_size=96`, `stride=48`) with averaging across chunks. |
| Fine-tuned (4) | 4/4 | §3.2 p.2 — full fine-tuning of `BertForMaskedLM` on training stories with standard MLM 80/10/10 masking, 3 epochs, AdamW, lr 2e-5; training loss decreased 2.6123 → 2.3396. Code: `code/lab32/finetune_bert.py:9-10, 27, 114-195` implements the MLM dataloader (512-token chunks with stride 256) and training loop; encoder weights then reused for embedding extraction in `embeddings_bert_finetuned.py`. |
| LoRA (2) | 2/2 | §3.3 p.3 — LoRA on Q/V projections, rank r=8, α=32, dropout 0.1; correctly notes installed PEFT didn't support `MASKED_LM` task_type so they wrap `BertForMaskedLM` directly while still training MLM loss; 294,912 / 109,809,210 = 0.2686% trainable. Code: `code/lab32/finetune_lora.py:27` `from peft import LoraConfig, get_peft_model`; `:57-86` builds `LoraConfig(r=8, lora_alpha=32, target_modules=["query","value"])` and `get_peft_model`; `:194` trains 3 epochs at lr 5e-4. |
| Performance comparison (2) | 2/2 | Table 1 (p.5) compares all 6 embeddings × 2 subjects on Mean / Median / Top 5% / Top 1% CC + Fraction Positive. Figure 1 (p.6) gives horizontal-bar version of the same comparison. §6.1-6.4 walks through the BoW < {GloVe, Word2Vec} < {BERT, BERT-FT, LoRA} ordering, contextual lift in the upper tail (Fig 2 / Fig 10 / appendix), per-voxel deltas (Fig 3), and subject-consistency scatter (Fig 4). Conclusion: contextual > static is the dominant effect; FT vs LoRA differences are subject-dependent and modest. |
| Description of each model (2) | 2/2 | §3.1-3.3 p.2-3 describes BoW, Word2Vec/GloVe, pretrained BERT, full FT BERT, and LoRA BERT each in dedicated paragraphs; explains why each is included, what context type it captures, and (for BERT variants) the exact training/extraction protocol. |

**Column R comment (suggested):** Excellent — clear separation of pre-trained, full fine-tuned, and LoRA BERT pipelines; LoRA setup is textbook (Q/V, r=8, α=32 via PEFT). Strong Table 1 + Figure 1 covering 6 embeddings × 2 subjects × 4 metrics + fraction positive. The per-voxel delta histograms (Fig 3) directly answer "does fine-tuning help?" rather than relying on summary tables alone.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | §4 p.3-4 — voxel-wise ridge with explicit objective `‖Y_v − X w_v‖² + α_v ‖w_v‖²`; per-voxel α selection; explicitly notes p = 4× embedding dim due to 4 HRF lag concatenation (p = 20,000 for BoW, 1,200 static, 3,072 BERT). Stimulus features and BOLD z-scored using training-set statistics applied to test set. |
| Cross-validation analysis (2) | 2/2 | §4 p.4 — blocked bootstrap CV: 20 log-spaced α from 10^1 to 10^8, 5 bootstrap samples, validation chunks of 40 TRs, 5 chunks held out per bootstrap, α chosen per voxel. Code: `code/lab32/ridge_32.py:62-64, 238-245` (`ALPHAS = np.logspace(1,8,20)`, `NBOOTS=5`, `CHUNKLEN=40`, `bootstrap_ridge`). Final evaluation on the fixed 20 held-out test stories (`random_state=42`, 81/20 split, 27,678 train TRs / 7,108 test TRs). |
| Best-performing model deep-dive (2) | 2/2 | §6.1-6.4 walks through why contextual BERT wins (Fig 1, Table 1), where the gain is concentrated (upper tail — Fig 2 CC profile stems, Fig 10 contextual-lift bars on matched top-1% voxels), why FT vs pretrained is small (Fig 3 per-voxel delta histograms centered near 0 with ±0.002 mean shift), and why Subject 3 outperforms Subject 2 (Fig 4 cross-subject scatter, BERT cluster farther from diagonal). The "contextual gain is concentrated in language-responsive voxels" framing is the right read. |
| Performance over voxels (2) | 2/2 | §6 + Fig 11 (appendix) — voxel-wise box plots for all 3 BERT variants × 2 subjects against a CC = 0.10 interpretation threshold; explicitly notes "medians remain far below" the threshold while top-1% values clear it. Motivates the high-CC restriction for interpretation (§6.5 "CC ≥ 0.10 threshold for reliable interpretation"). |

**Column T comment (suggested):** Clean blocked-bootstrap CV with per-voxel α, well-motivated z-scoring, no leakage across train/test stories. PCS-aware voxel analysis: explicit CC ≥ 0.10 interpretation threshold and box-plot framing in Fig 11. No deductions.

---

## U. Lab 3.3 Interpretability (12 pts) — score: 12/12

Two stories analyzed: `canplanetearthfeedtenbillionpeoplepart1` and `stumblinginthedark`. Top **3** voxels per (story × subject), so 3 voxels × 2 subjects × 2 stories = 12 (voxel, subject, story) attribution panels in Fig 5 (p.10). Selected voxels all have story-level CC ∈ [0.28, 0.39], well above the 0.10 PCS threshold. **Note:** rubric says "5 voxels"; this group did 3 voxels × 2 subjects = 6 voxel-explanations per story, which exceeds the spirit of the rubric in another dimension (both subjects). Lab spec ("two stories, top voxels by CC") is satisfied; no deduction.

**`canplanetearthfeedtenbillionpeoplepart1` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Fig 5 top half (p.10) — SHAP bar plot for top-3 voxels per subject (6 panels). Setup §5 + code: `code/lab33/interpret_shap_lime.py:304-311` uses `shap.KernelExplainer` with 120 nsamples on the wrapper score function. Wrapper masks words with `[MASK]` and reruns the full encoding pipeline (perturb → BERT re-embed → downsample → trim/lag → ridge → story CC) — methodologically careful. |
| LIME (2) | 2/2 | Fig 5 (LIME bars adjacent to SHAP bars per voxel) — LIME importances for same voxels. Setup §5: described as "weighted local ridge surrogate over word-perturbation samples"; code `:332-342` runs 100 perturbations with exponential kernel weights and a regularized local ridge surrogate; LIME drops words (empty string) rather than masking — group correctly flags this difference vs SHAP's `[MASK]` perturbation as a methodological subtlety. |
| Comparison (2) | 2/2 | Fig 6 left panel (p.11) — SHAP vs LIME absolute-importance scatter, Spearman ρ = 0.13 for this story. §6.5 + Discussion §8 discuss the disagreement and attribute it to (a) SHAP masking vs LIME deletion changing the surrounding BERT context differently, and (b) both methods picking up some function words because of structural context effects. Fig 12 heatmap (appendix) further compares signed SHAP across voxels: "meat", "soil", "foundation" are semantically meaningful but effects are voxel-specific, not brain-wide. |

**`stumblinginthedark` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Fig 5 bottom half (p.10) — SHAP for top-3 voxels per subject. Same wrapper SHAP setup. Top words include "pull", "upstairs", "editing", "ambulance", "patient", "psychiatric" — clearly tied to the medical-event narrative. |
| LIME (2) | 2/2 | Fig 5 (paired LIME) — corresponding LIME importances per voxel. Same wrapper LIME setup. |
| Comparison (2) | 2/2 | Fig 6 right panel — SHAP-vs-LIME scatter for stumblinginthedark, Spearman ρ = 0.13. Some local agreement on top-ranked event words (`with`, `editing`, `pull`, `patient` labeled on the diagonal). Fig 13 heatmap (appendix) shows event-related words ("knock", "refrigerator", "car", "ambulance", "affected") are story-relevant but voxel-specific. The "limited overall agreement, some local agreement on high-CC voxels" framing is exactly right. |

**Column V comment (suggested):** Both stories thoroughly analyzed with paired SHAP + LIME on top-3 voxels per (story × subject) = 12 attribution panels total. The wrapper approach (perturb → rerun BERT → rerun ridge → story-level CC as scalar output) is methodologically careful and well-motivated. SHAP-vs-LIME scatter (Fig 6) and signed-SHAP heatmaps (Figs 12, 13) provide cross-method and cross-voxel checks. Cautious interpretation: "useful for hypothesis generation, not direct neural evidence". No deductions.

---

## W. Stability Check (5 pts) — score: 5/5

Group performed **two** stability checks, both substantive:

| Aspect | Evidence |
|---|---|
| Stability check 1 — does BERT need story context? | §7.1 p.12-13 — re-embeds words using a word-level BERT control (minimal surrounding context) and reruns the full ridge pipeline. Table 2 (p.12) reports mean/median/top-5%/top-1% CC for word-level vs contextual BERT for both subjects, with gain row. Subject 2 mean: 0.0089 → 0.0208 (Δ +0.0119); Subject 3 mean: 0.0185 → 0.0348 (Δ +0.0163); top-1% gains +0.05 and +0.07. Fig 7 bar chart (p.13) + Fig 8 voxel-wise gain histograms (71.8% of S2 voxels and 75.0% of S3 voxels improve under contextual). Conclusion: "BERT's advantage is not just dimensionality — surrounding narrative context itself matters." |
| Stability check 2 — LoRA rank sensitivity | §7.2 p.14-15 — retrains LoRA at r = 4 and r = 16 alongside main r = 8, same training stories / MLM / extraction / ridge pipeline. Table 3 (p.14) reports CC summaries for both subjects across all 3 ranks. Fig 9 (p.15) plots rank-sensitivity curves. Result: differences across ranks are small relative to the contextual-vs-non-contextual gap, ranks not even monotonic for S3 → conclusion is "LoRA is stable across reasonable ranks." Code: `code/lab32/finetune_lora_stability.py`, `code/lab32/ridge_32_lora_stability.py`, `code/lab32/plot_32_stability.py`. |
| PCS framing | §7 opening + §8 discussion explicitly invoke the PCS framework (Yu & Barter ref [3]) and frame both stability checks as testing whether main findings are stable across reasonable alternatives. The two checks target two different judgment calls (embedding extraction protocol; LoRA hyperparameter), and the conclusions are directional: context matters for BERT performance (sensitive direction); rank does not matter much for LoRA (insensitive direction). |

→ Per calibration (depth-on-one-judgment-call done thoughtfully = 5/5; this group did two): **5/5**.

**Column X comment (suggested):** Two substantive PCS-style stability checks. The contextual-vs-word-level comparison (Fig 7 + Fig 8) gives a sharp answer to "where does BERT's advantage come from?" — surrounding narrative context contributes ~half the contextual lift, not just the higher dimensionality. The LoRA rank sweep (r ∈ {4, 8, 16}) confirms the main LoRA result is not an artifact of the chosen rank. Exactly the PCS framing the lab spec is reaching for.

---

## Y. Writing (5 pts) — score: 5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 3/3 | 9-section structure (Intro, Data, Encoding Models, Ridge, Interpretation, Results, Stability, Discussion, Conclusion) + Appendix + Bibliography + Academic honesty / LLM statement. 19 pp total / ~16 pp main body — under 20-page limit. Three motivating questions stated at end of intro and answered in order through §6, §6, §7. Math notation correct (`ŵ_v = argmin ‖Y_v − X w_v‖² + α_v ‖w_v‖²`; CC = corr(Y, Ŷ)). LLM statement present, separate paragraphs for coding (Claude / Codex / Copilot) vs writing (Claude / ChatGPT / Grammarly). |
| Figure quality (0-2) | 2/2 | 13 figures total (10 main + 3 appendix) + 3 tables; all numbered and captioned with descriptive titles. Main figures use a consistent color palette: BoW/GloVe/Word2Vec in orange tones, BERT variants in blue tones. SHAP/LIME bar charts (Fig 5) use SHAP-orange / LIME-green, repeated in Fig 6 scatter. Heatmaps (Figs 12, 13) use a clear diverging red/blue scale with column-z-scored signed SHAP. Fig 5 is dense (12 panels) but readable. |

**Column Z comment (suggested):** Polished, well-structured report. Clean modeling math, well-captioned figures with consistent color conventions across the main and appendix figures. Strong roadmap (three questions in §1, answered in §6 and §7). No deductions.

---

## Calibration notes — resolved

Both rubric/spec tensions resolved with Sean before grading:

1. **Fine-tuned vs LoRA:** LoRA-only earns full 6/6. This group did BOTH full FT and LoRA, both implemented cleanly. No issue.
2. **Stability check:** Lab spec ("one judgment call") wins; depth-on-one earns 5/5. This group did two checks, both thoughtful and PCS-framed. No deduction.
3. **Voxel count for interpretation:** Rubric mentions "5 voxels", lab spec says "top voxels by CC". Group did 3 voxels × 2 subjects per story = 6 voxel-explanations per story; selected voxels all clear the 0.10 PCS threshold and have story-level CC 0.28-0.39. Spec satisfied — no deduction.

See `memory/feedback_stat214_lab3_calibration.md` for the durable rule.
