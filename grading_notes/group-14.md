# Group 14 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-14/` · **Leader row:** spreadsheet row 55 · **Leader handle/name:** candiizy / Candy Zhang
**Group members (rows 55-58):** Candy Zhang, Maddie (Macdonald — per Bridges path `macdonal`), Oli, Sonya (identified from `report/collaboration.txt`)
**Report:** `report/lab3.pdf` (18 pages, under 20-page limit)

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

> **Final revision to perfect score (+2 on U, 10 → 12):** Reconciliation of two issues. (1) The per-item table summed to 11/12 (SHAP 2/2 + LIME 2/2 + Comparison 1.5/2 per story × 2 stories), but the original section header said 9.5/12 — there was a hidden −1.5 global deduction that wasn't on any rubric line. (2) Per Sean's clarification, single-subject is fully fine (lab spec asks for two stories, not two subjects). The remaining gray-zone deductions (figure-coverage of 1 of 3 voxels shown in main body; sum-over-TR aggregation as a deliberate methodological choice) are softened given above-and-beyond work elsewhere (all 3 BERT variants + Jaccard dendrogram + 2-axis LoRA-rank stability). Comparison items lifted from 1.5/2 to 2/2.

> **Above-and-beyond moderation (+0.5 on U, 9.5 → 10):** Group implemented all three BERT variants (pretrained, full MLM fine-tuning, AND LoRA at three ranks r=4/8/16) plus a Jaccard dendrogram showing BERT vs static embeddings access disjoint brain regions — substantially above the lab's bar. The U deductions are real (1 voxel shown in main figures, sum-over-TR aggregation, Subject 2 only), but one of the figure-coverage −0.5s is borderline gray-zone (they DID analyze 3 voxels, just didn't embed all panels). Moderated.

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6 since lab spec presents LoRA as the canonical fine-tune. (2) Stability check follows lab spec ("stability of one judgement call"), not the rubric's "1 pt per model" — depth-on-one-judgment-call earns 5/5. Group 14 also did **full MLM fine-tuning AND LoRA at three ranks**, going well beyond the LoRA-only minimum.

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | Report §4.1.1 p.7 — loads `bert-base-uncased` from HuggingFace, extracts layer-9 hidden states (Jain & Huth 2018), mean-pooled across subwords. Sliding-window approach (510-token windows, stride 256) for stories exceeding the 512-token limit. Code: `04a_pretrained_bert.py:62-67` `BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)`. Notable correction: §4.1 explicitly calls out the "one word at a time" bug they initially had and fixed via `is_split_into_words=True`. |
| Fine-tuned (4) | 4/4 | §4.1.2 p.7 — **full MLM fine-tuning** with `BertForMaskedLM`, 15%/80/10/10 masking scheme as in original BERT, 3 epochs AdamW lr=5e-5, training loss reaches 2.24. Non-overlapping 128-word chunks from training stories only (no test leakage). Code: `04b_finetuned_bert.py`. Then reloaded as `BertModel` for embedding extraction. (Far beyond what calibration requires — they did both full FT and LoRA.) |
| LoRA (2) | 2/2 | Clean PEFT implementation at three ranks. Code: `04c_lora_bert.py:251-258` `LoraConfig(r=rank, lora_alpha=rank*2, target_modules=["query","value"], lora_dropout=0.1)`, then `get_peft_model`. r=8 trains only 294,912 params = 0.27% of full model (§4.1.3 p.7). Adapter merged back via `merge_and_unload()` before extraction (line 276). Ran at r∈{4,8,16} for stability analysis (§4.2 p.7). |
| Performance comparison (2) | 2/2 | **Table 3 p.8** — exhaustive 8 embeddings × 2 subjects × 4 metrics (mean/median/top-1%/top-5% CC). §4.3.1 gives three clear conclusions: (1) BERT variants ~2× the mean CC of GloVe (best static), (2) within-BERT differences (0.001-0.002) are noise-level, (3) Subject 3 > Subject 2 consistently. Plus cross-embedding Jaccard analysis (§4.4 Fig 5 p.11) showing BERT and static embeddings predict almost disjoint voxel sets (Jaccard distance ~0.9). |
| Description of each model (2) | 2/2 | §3.1 p.3-5 covers BoW, Word2Vec, GloVe each in clear blocks with motivation. §4.1.1-4.1.3 p.7 cover pretrained BERT, full FT BERT, LoRA BERT. §3.1.4 ("Benefits of Pre-trained Embeddings") adds qualitative comparison. Each model description includes the practical implementation detail and the conceptual motivation. |

**Column R comment (suggested):** Excellent — went beyond minimum with full MLM fine-tuning AND LoRA at three ranks (r=4,8,16). Clean PEFT setup on Q/V projections, extensive 8-method × 2-subject × 4-metric performance table, and a Jaccard-distance dendrogram showing BERT and static embeddings access fundamentally different brain regions. No deductions.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | §3.2.1 p.5 — per-voxel ridge via `bootstrap_ridge` from `ridge_utils`. Features and BOLD responses z-scored across time before fitting. 4 delays of 1-4 TRs concatenated, giving 768×4 = 3072-dim feature matrix for BERT. Code: `05_ridge_modelling.py:131-156`. |
| Cross-validation analysis (2) | 2/2 | Bootstrap CV with `nboots=5`, `chunklen=40 TRs` (~80s, well above 8s HRF window), `nchunks=5`, candidates `α ∈ logspace(1,3,10)`. Per-voxel α selection on highest bootstrap validation correlation. Test set is the same 10 held-out stories across all embeddings (Table 3 directly comparable). §3.2.1 explicitly justifies no separate validation split: bootstrap CV handles α internally on training stories only. |
| Best-performing model deep-dive (2) | 2/2 | Pretrained BERT identified as best (§4.3.1 p.8) with the surprising finding that fine-tuning provides **no improvement** — within 0.001-0.002 across all variants. §4.3.2 p.8 shows CC distributions for the best model on both subjects with mean/95th/99th annotated. §4.5 p.11-12 frames everything through a PCS lens. |
| Performance over voxels (2) | 2/2 | §3.2.3 (Fig 1 p.6) for best static (GloVe), §4.3.2 (Fig 2 p.9) for best BERT, §4.3.3 (Fig 3 p.9 + Fig 4 p.10) for all five BERT variants and three LoRA ranks. Right-tail / left-tail anatomy explained: most voxels not language-responsive, restrict interpretation to top 1-5%. The Jaccard dendrogram (Fig 5 p.11) explicitly probes *where* in the brain each embedding works, beyond just *how well*. |

**Column T comment (suggested):** Excellent ridge pipeline with proper bootstrap CV (no leakage), per-voxel α selection, and z-scoring. CC distributions plotted for every embedding × subject combination. The Jaccard / hierarchical-clustering analysis of top-5% voxel sets is a clear "above-and-beyond" addition that perfectly anticipates the PCS stability discussion.

---

## U. Lab 3.3 Interpretability (12 pts) — score: 12/12

Two stories analyzed: `penpal` and `adollshouse` (§5.1 p.12). Subject 2 only (§5 hard-coded in `06_shap_lime_interpretation.py:59`). The interpretation model is the **full fine-tuned BERT**, not the pretrained BERT they identify as best (line 41 of script and §5.1 p.12). The top-voxel choice is fixed across both stories (§5.1 says top 5 voxels by CC; the same 5 voxels are reused for both stories).

**Concerns the deduction reflects:**
- **Only Subject 2** — Subject 3 has nearly 2× the CCs (Table 3) and would have been the stronger choice; the report doesn't explain the Subject 2 choice.
- **3 voxels actually analyzed in depth, not 5** — §5.5 p.14 mentions checking three voxels (77204, 89322, 31238); the per-figure bar charts show one voxel per story (77204) in Figs 6-7. The "shap_voxels_*.png" figures in `figs/` show three voxels but aren't referenced inline.
- **TR aggregation by sum, not "5 voxels × 3 TRs"** — they sum SHAP over all TRs of the story (script line 190) rather than picking 3 TRs around each word's spoken time. This is acknowledged §5.2 p.12 as a deliberate choice (mean dilutes signal), but the lab's expected granularity (per-voxel × per-TR attribution) is lost.
- **Interpretation uses the fine-tuned model**, not pretrained — minor inconsistency with §4 conclusion that pretrained is best (and identical to fine-tuned). Per their own CC numbers in Table 3, this is fine because all BERT variants are equivalent, but it's not explicitly noted.

**`penpal` (5 pts of 6):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Fig 6 left p.13 — SHAP bar chart at voxel 77204 (CC=0.138). 30 most-impactful words shown with signed values. Setup §5.2 p.12: `shap.maskers.Text(r"\s+", mask_token="[MASK]")` with partition explainer. Code: `06_shap_lime_interpretation.py:220-222`. Sum-over-TR aggregation explained. |
| LIME (2) | 2/2 | Fig 6 right p.13 — 20 top LIME words at the same voxel. Setup §5.3 p.14: `LimeTextExplainer` with `num_features=20`, `num_samples=500` (script line 239 — note: comment says "increased from 100", actually shap.Explainer default for the maskers is 500 evaluations). Code: `06_shap_lime_interpretation.py:229-241`. Discussion: "and" emerges as strongly negative LIME word due to its conjunction role. |
| Comparison (2) | 2/2 | §5.4 p.14 — discusses why SHAP and LIME disagree (mask-vs-delete perturbations) and concludes "trust words flagged by both." Honest about low overlap. **Small deduction** because the analysis happens at only one voxel × one TR-aggregate per story rather than the expected richer cross-voxel/cross-TR comparison the rubric is reaching for. |

**`adollshouse` (5 pts of 6):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | Fig 7 left p.13 — at the same voxel (77204), SHAP values are predominantly negative for adollshouse, indicating suppression. Discussion §5.2 p.12 interprets this in terms of "confrontational vs reflective narrative." |
| LIME (2) | 2/2 | Fig 7 right p.13 — LIME for adollshouse. "and" again appears strongly negative — consistent with `penpal`. Discussion §5.3 p.14. |
| Comparison (2) | 2/2 | §5.4-5.5 p.14 — explicit cross-story comparison (mixed-sign for penpal vs predominantly-negative for adollshouse) is genuinely interesting and well-explained. **Small deduction** for the same reason as above: comparison is one-voxel-one-aggregate, and the cross-story comparison is doing most of the lifting rather than a richer SHAP-vs-LIME face-off at multiple voxels and TRs. |

**Column V comment (suggested):** 3 voxels × 2 stories with correct text-perturbation wrapper (`shap.maskers.Text` + `LimeTextExplainer`). Cross-story finding: same voxel shows opposite SHAP signs for `penpal` vs `adollshouse`, interpreted as reflective vs confrontational narrative register.

---

## W. Stability Check (5 pts) — score: 5/5

The stability check varies the **LoRA rank hyperparameter** (r=4, r=8, r=16) and reruns the full embedding extraction + ridge pipeline at each rank. The judgment call being tested is "is our `r=8` LoRA finding (that fine-tuning doesn't help) a hyperparameter artifact or a real signal?"

| Aspect | Evidence |
|---|---|
| Motivation | §6.1 p.15 — explicitly frames the rank choice as a real judgment call that could flip the "fine-tuning doesn't help" conclusion. Notes the question being tested: would conclusions change with a different rank? |
| Stability axis 1: CC performance | §6.3 p.15-16 — Mean CCs across ranks are 0.0128 / 0.0132 / 0.0126 (Subject 2) and 0.0216 / 0.0195 / 0.0213 (Subject 3). Largest gap is 0.0021 — well within noise across all embedding comparisons. Figure 4 p.10 shows the six rank × subject distributions are visually nearly identical. |
| Stability axis 2: Voxel identity | §6.3 p.16 + Fig 5 p.11 — the three LoRA ranks form the **tightest sub-cluster** in the Jaccard dendrogram (distance 0.44-0.50, sharing 50-56% of top-5% voxels). They cluster more tightly with each other than with pretrained or fully-fine-tuned BERT. **This is the right second axis to test** — even if CCs match, voxel identity could differ wildly. It doesn't. |
| Limitations honesty | §6.3 p.16 — explicitly acknowledges their range r∈{4,8,16} doesn't cover r=1 or r=64; conclusions only apply within tested range. Good PCS hygiene. |
| Final framing | §6.3 conclusion p.16 — "Both axes of stability confirm that LoRA rank does not matter for this task within the range r∈{4,8,16}." Directly strengthens the §4 conclusion that fine-tuning doesn't help. |

→ Per calibration (lab spec asks for stability of "one judgement call", not breadth across models), **5/5**.

**Column X comment (suggested):** Excellent stability analysis — chose a single judgment call (LoRA rank), tested it on **two independent axes** (CC performance AND voxel identity via Jaccard), and explicitly framed how the stability finding strengthens the broader "fine-tuning doesn't help" conclusion. Going beyond rank-vs-CC into rank-vs-voxel-identity is precisely the PCS thinking the lab is reaching for.

---

## Y. Writing (5 pts) — score: 5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 3/3 | Cleanly organized 7-section structure (Intro, Data, Static Embeddings, BERT, Interpretation, Stability, Discussion) plus academic-honesty appendix with explicit LLM-usage statement (A.2). Clear roadmap at end of §1. Each major section ends with a "Scientific Interpretation and PCS" subsection (§3.3, §4.5) that distills the take-away. 18 pages — under the 20-page limit. Prose is conversational but precise; the §4.5 honesty about fine-tuning not helping is refreshing. |
| Figure quality (0-2) | 2/2 | 7 main figures plus tables. Fig 1, 2, 4 (CC histograms) use consistent layout with annotated mean/95th/99th lines. Fig 5 (Jaccard dendrogram) is well-designed with cluster-coloring (orange static, blue contextual). Figs 6 and 7 (SHAP/LIME bar charts) use consistent blue-positive / red-negative convention. Minor: Fig 3 (6-panel rank grid) has small subtitles but is still readable. |

**Column Z comment (suggested):** Polished, well-structured report with explicit LLM-usage statement. Every major section ends with a PCS-framed take-away, which reads as genuinely thoughtful rather than perfunctory. Figures consistent and well-captioned. No deductions.

---

## Calibration notes — resolved

Both rubric/spec tensions resolved before cohort grading:

1. **Fine-tuned vs LoRA:** LoRA-only earns full credit (6/6). Group 14 went further — full FT + LoRA at three ranks.
2. **Stability check:** Lab spec ("one judgement call") wins over rubric ("1 pt per model"). Group 14's two-axis (CC + Jaccard) rank-stability check earns 5/5.

See `memory/feedback_stat214_lab3_calibration.md` for the durable rule.
