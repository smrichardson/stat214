# Group 07 — Grading Notes (Sean's categories: 3.2 + 3.3)

**Repo:** `student_repos/group-07/` · **Leader row:** spreadsheet row 27 · **Leader handle/name:** AlexVolkov0404 / Oleksandr Volkov
**Group members (rows 27-30):** Oleksandr Volkov, Evalynna Ong, Yijiao Zhou, Joshua Ren
**Report:** `report/lab3.pdf` (21 pages — 1 page over the 20-page limit; main body ends p.19, refs p.20, appendix p.21)

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

> **Post-audit refund (+1 on Y, 4 → 5):** The original grader docked 1 pt on Readability for the 21-page report. The audit caught a cross-group consistency issue: group 08 also submitted 21 pages with body within 20 + admin overflow and received 0 deduction; group 04 submitted 22 pages with body exactly 20 + refs/honesty/LLM on p.21-22 and received 5/5. Group 07's structure (body p.19, refs p.20, LLM/honesty p.21) is the same pattern. The Spring 2026 Rubric v2 explicitly says "body within 20 + admin overflow = compliant." Refund applied for consistency.

> **Calibration applied** (per Sean): (1) LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6 since lab spec presents LoRA as the canonical fine-tune. (2) Stability check follows lab spec ("stability of one judgement call"), not the rubric's "1 pt per model" — depth-on-one-model earns 5/5.

---

## Q. Lab 3.2 BERT (12 pts) — score: 12/12

| Item | Pts | Evidence |
|---|---:|---|
| Pre-trained (2) | 2/2 | Report §3.3 p.4 — uses `bert-base-uncased`, 768-dim word-level contextual embeddings. Code: `get_embedding_bert.py:63-68` `BertModel.from_pretrained('bert-base-uncased')`; processes story in chunks of 512 tokens, subtoken aggregation by word_ids alignment (`:122-127`). Final feature matrix dimension reported as `R^{N×3072}` after lag stacking. |
| Fine-tuned (4) | 4/4 | LoRA-MLM fine-tuning on story corpus serves as the fine-tuned model (per calibration: LoRA = canonical fine-tune). MLM objective with 15% masking, sliding-window chunking with stride 448, then encoder reused to extract per-word contextual embeddings for both training and test stories. Code: `fine_tune_bert.py:131-138` builds `BertForMaskedLM`, wraps with `get_peft_model`. |
| LoRA (2) | 2/2 | Clean PEFT implementation. `fine_tune_bert.py:9` `from peft import LoraConfig, PeftModel, get_peft_model`; `:132-138` constructs `LoraConfig(r, lora_alpha, lora_dropout, target_modules)` with all config-driven hyperparameters. `configs/config_bert_lora_exp020.yaml` shows the selected final settings (r=8, α=16, dropout=0.1, target_modules: [query, value]). |
| Performance comparison (2) | 2/2 | Report Table 10 (p.8) compares all 5 embeddings × 2 subjects on Mean / Median / Top-1% / Top-5% CC, plus Var(CC) and selected α. §5.2.1 text gives explicit ordering for Subject 2 (Fine-tuned BERT > Word2Vec > GloVe > BoW > Pre-trained BERT) with the noteworthy finding that pre-trained BERT underperforms BoW. ~32% relative gain from fine-tuning at top-5%. |
| Description of each model (2) | 2/2 | §3.1-3.3 + §4 each describe one model in dedicated paragraphs (BoW p.3, Word2Vec/GloVe §3.2 p.3, Pre-trained BERT §3.3 p.4, Fine-tuned BERT with LoRA §4 p.4-7). Table 1 (p.4) summarizes static embedding comparison; §4.1-4.3 describe LoRA setup in depth with seven tuning sub-experiments. |

**Column R comment (suggested):** Excellent — LoRA fine-tuning executed with careful hyperparameter exploration (28 experiments tabulated across rank, α, LR, epochs, stride, target modules) and clear final-model selection rationale. All five model families described in dedicated subsections. Pre-trained BERT-vs-BoW inversion noted and discussed.

---

## S. Lab 3.2 Modeling & Evaluation (8 pts) — score: 8/8

| Item | Pts | Evidence |
|---|---:|---|
| Ridge Regression (2) | 2/2 | §2.2.4 p.2 — per-voxel ridge with explicit objective `‖Y − Xβ‖² + α‖β‖²`. Motivation given: high-dimensionality + correlated lagged copies. Code: `run_ridge_pipeline.py:135-203` implements alpha-selection via block-wise voxel processing with `bootstrap_ridge`. |
| Cross-validation analysis (2) | 2/2 | §5.1 + code `run_ridge_pipeline.py:136-203` describe two-stage workflow: (1) global α selected on validation split by maximizing mean validation CC across voxels over candidate grid `np.logspace(1, 4, num=8)`; (2) refit on train+val using α*, evaluate on held-out test stories. Block-wise voxel fitting (5,000 voxels/block) for memory. Best-α values reported per embedding in Table 10. |
| Best-performing model deep-dive (2) | 2/2 | Selected model is BERT-LoRA `exp020` (q,v target modules, stride 448, 5 epochs, LR 5e-4). §4 walks through 28 experiments via 5 tables (Tables 2-7), §4.3 reports complete final settings (Table 8) + final performance (Table 9). Also includes seed stability check in §4.2.5 (Table 7) before declaring the model final. §5.2.1 discusses why fine-tuned BERT outperforms pre-trained BERT and §5.2.2 frames the right-skewed CC distribution. |
| Performance over voxels (2) | 2/2 | §5.2.2 p.9 + Figure 1 — voxel-wise CC distributions for fine-tuned BERT on both subjects, strongly right-skewed with mean ~0.07-0.08, top-5% ~0.21-0.22, top-1% ~0.29-0.31. Mean / top-5% / top-1% lines marked on histogram. Motivates restricting interpretation to top voxels for SHAP/LIME under PCS framing (§6, p.9). |

**Column T comment (suggested):** Methodologically careful ridge pipeline. Global α-selection over `logspace(1, 4, 8)` is reasonable given the block-wise memory constraint. 28-experiment LoRA hyperparameter sweep with seed-stability check on the winning config is thorough. Voxel CC distributions plotted and discussed.

---

## U. Lab 3.3 Interpretability (12 pts) — score: 12/12

Two stories analyzed: *A Father's Cover* and *Birth of a Nation*. Top 2 voxels (49760 and 49800) for Subject 2 only. Both SHAP and LIME computed for each (story, voxel) pair plus a part-of-speech aggregation layer.

**`A Father's Cover` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | §6.3.2 p.11 + Table 11 + Table 12 (p.12) — mean-absolute SHAP rankings on voxel 49760 and 49800. Code: `run_shap.py:211-216` `shap.maskers.Text(r"\s+", mask_token="[MASK]")` + `shap.Explainer(predict_fn, masker)`, `max_evals=300`. §6.3.3 + Figures 2-3 (p.13) add a POS-aggregated SHAP analysis (PART/DET/CCONJ/VERB/NOUN bars). |
| LIME (2) | 2/2 | §6.3.1 p.11 — top-15 LIME words per voxel discussed in prose (no figure for story 1, but full word lists + magnitudes given). Code: `run_lime.py:213-219` `LimeTextExplainer(class_names=...)` with `num_samples=100`, signed importances saved per word. Identifies "in", "to", "the" as dominating function words — explicitly flagged as a LIME limitation. |
| Comparison (2) | 2/2 | §6.5 p.15-16 — head-to-head SHAP-vs-LIME discussion. Finding: SHAP highlights content words (freeze, pin, moving, hear, drop) while LIME ranks function words (in, the) at top due to deletion-based perturbation amplifying high-frequency tokens. Authors explicitly attribute the divergence to LIME's perturbation mechanism and caution interpretation accordingly. |

**`Birth of a Nation` (6 pts):**
| Item | Pts | Evidence |
|---|---:|---|
| SHAP (2) | 2/2 | §6.4.2 p.15 + Table 13 — top-15 SHAP words for both voxels 49760 and 49800. Voxel 49760 emphasizes action/character words (boy, hot, read, throws, alfred, belt); voxel 49800 emphasizes scene/objects (bell, bottles, cream, milk, milkman, silver). |
| LIME (2) | 2/2 | §6.4.1 p.14 + Figure 4 — top-15 LIME word bar chart per voxel, with signed (positive/negative) importance values. Voxel 49800 shows interesting negative attributions ("might", "wasw"). Tokenization artifact ("d" appearing as a top word) noted by authors. |
| Comparison (2) | 2/2 | §6.5 p.15-16 cross-method comparison applies to both stories. Specifically calls out: SHAP "more stable and semantically coherent"; LIME's local-deletion sampling causes function-word inflation. Same content themes (social/identity vs. environmental) recover under both methods even though specific word rankings differ. |

**Column V comment (suggested):** Strong interpretability section. Two stories × two voxels × {SHAP + LIME + POS-aggregated SHAP} is more depth than required. The POS-level SHAP analysis (Figs 2-3) is a particularly nice addition that addresses the "many words have nearly-identical SHAP scores" issue authors identified. Cross-method comparison is mechanistically explained, not just descriptive. Minor: only Subject 2 analyzed (not Subject 3); only 2 voxels per story (lab suggests ~5). Doesn't change score given calibration generosity, but worth a note.

---

## W. Stability Check (5 pts) — score: 5/5

The stability check is multi-pronged, with three distinct judgment calls probed (§7.3 p.16-18):

| Aspect | Evidence |
|---|---|
| Voxel-subset stability of top-5% CC | §7.3.1 + Figure 5 (p.17) — randomly partition voxels into 5 subsets, recompute top-5% CC for all 5 embeddings × 2 subjects. Ordering Fine-tuned BERT > {Word2Vec, GloVe} > BoW > Pre-trained BERT remains intact across all 5 subsets and both subjects. Conclusion: embedding ranking is **stable** under voxel subsetting. |
| Cross-subject stability of top-voxel identity | §7.3.2 + Table 14a (p.18) — Jaccard similarity of top-5% voxel sets across Subject 2 and Subject 3 is only 0.0414 (≈8% overlap). Conclusion: voxel-level identity is **unstable** across subjects, suggesting interpretations don't generalize across individuals. Cautious interpretation given different voxel counts per subject. |
| Cross-embedding stability of top-voxel identity | §7.3.3 + Table 14b (p.18) — Jaccard similarities between top-5% voxels of Fine-tuned BERT vs Word2Vec vs GloVe are 0.76-0.87 in both subjects. Conclusion: top voxels identified by the three best embeddings are **stable** (the same brain regions are language-responsive regardless of the embedding choice). |
| Seed stability of LoRA fine-tune | §4.2.5 + Table 7 (p.7) — rerun selected `exp020` config with two additional random seeds (213, 215). CC values vary by <2% on top-5%/top-1%, MLM loss varies more but the embedding ordering is unchanged. |
| Final framing | §7 is explicitly framed as PCS Evaluation (Predictability §7.1, Computability §7.2, Stability §7.3). The stability subsection synthesizes a clear three-tier finding: top-voxel **identity** does NOT generalize across subjects, but top-voxel **identity** IS robust across the three best embeddings, and embedding-**ordering** is robust to voxel subsetting. This is exactly the PCS framing the lab is reaching for. |

→ Per calibration (lab spec asks for stability of "one judgement call"), this group probes four judgment calls (voxel subsetting, cross-subject, cross-embedding, seed), each with quantitative evidence. **5/5**.

**Column X comment (suggested):** Exemplary PCS-style stability analysis — explicitly labeled "PCS Evaluation" with Predictability/Computability/Stability subsections. Four distinct stability probes (voxel subsetting, cross-subject Jaccard, cross-embedding Jaccard, seed reruns). The cross-subject Jaccard finding (0.04 ≈ 8% overlap) leading to a cautious "voxel-level interpretations don't generalize across individuals" conclusion is the PCS framing the lab is reaching for.

---

## Y. Writing (5 pts) — score: 5/5

| Item | Pts | Notes |
|---|---:|---|
| Report readability (0-3) | 3/3 | 21 pages total, but main body ends p.19 — refs p.20 and appendix p.21 are admin overflow, which the Spring 2026 Rubric v2 treats as compliant (matching how group 04's 22-page and group 08's 21-page reports were handled). Well-organized 8-section structure (Intro, Data/Methodology, Language Representations, Fine-tuned BERT, Ridge, Model Interpretation, PCS Evaluation, Discussion). The LoRA section §4 is very table-heavy (8 tables in 4 pages); could be condensed but does not affect score. Prose is generally clear, math notation is correct, contributions clearly stated. LLM-usage statement present (§A.2, p.21). Academic-honesty statement clean (§A.1, p.21). |
| Figure quality (0-2) | 2/2 | 5 numbered figures with descriptive captions. Figure 1 (CC histogram) and Figure 5 (stability line plot across voxel subsets) are clean and informative. Figure 4 (LIME bar chart for story 2) is well-laid-out with positive/negative color coding. Figures 2-3 (POS SHAP) use clear bar+box-plot pairing. One minor issue: Figure 4 has small axis text on the dense bar chart, but still readable. Tables are extensive (14 tables) and well-formatted with bold for best rows. |

**Column Z comment (suggested):** Solid technical writing with clear LoRA experiment narrative. One-page deduction for exceeding the 20-page limit (main body to p.19, refs p.20, appendix p.21). The LoRA tuning section §4 is very table-heavy — 8 tables in 4 pages could be compressed into 2-3 summary tables to recover the page. Otherwise no concerns.

---

## Calibration notes — resolved

Both rubric/spec tensions discussed with Sean before grading the rest of the cohort:

1. **Fine-tuned vs LoRA:** LoRA-only earns full credit (6/6). Locked.
2. **Stability check:** Lab spec ("one judgement call") wins over rubric ("1 pt per model"). A thoughtful depth-on-one-model check earns 5/5. Locked.

**Group-07-specific notes:**
- Page-limit deduction: -1 on readability for 21 pages > 20. The body itself ends at p.19, so this is a soft call — flag for Sean if they want a stricter or more lenient interpretation.
- Interpretability scope: Subject 2 only and only 2 voxels per story (rather than the suggested ~5). Per the generous-partial-credit calibration, not deducted, but Sean may want to confirm.
- POS-aggregated SHAP (Figures 2-3) is a nice bonus interpretability layer beyond what the spec requires.
- Seed stability check on the LoRA config (Table 7) is in addition to the three "main" stability probes in §7.3.

See `memory/feedback_stat214_lab3_calibration.md` for the durable rule.
