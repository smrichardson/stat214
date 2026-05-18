# Excel Entry Sheet — Sean's Columns (Q–Z)

**How to use:** for each group, the leader row gets all 5 score columns + 5 comment columns. The other 3 members in each group inherit only the Total in column AA.

**Sean's columns:** Q (3.2 BERT, 12) · R (comment) · S (3.2 Modeling, 8) · T (comment) · U (3.3 Interp, 12) · V (comment) · W (Stability, 5) · X (comment) · Y (Writing, 5) · Z (comment).

## Compact score table (FINAL)

| Row | Group | Leader | Q/12 | S/8 | U/12 | W/5 | Y/5 | **Total** |
|---:|:---:|---|---:|---:|---:|---:|---:|---:|
| 3 | 01 | Alex Soh (alexchsoh) | 12 | 8 | 12 | 5 | 5 | **42** |
| 7 | 02 | Abhishek Shukla (abss123) | 12 | 7 | 6 | 4 | 4.5 | **33.5** |
| 11 | 03 | Akanshaa Borgohain (Akanshaa18) | 10.5 | 7 | 6 | 5 | 4.5 | **33** |
| 15 | 04 | Thomas Lee (cylee0815) | 12 | 8 | 12 | 5 | 5 | **42** |
| 19 | 05 | Chenfei Peng (ChenfeiP) | 12 | 6 | 8 | 5 | 5 | **36** |
| 23 | 06 | Drew Galloro (andreagalloro) | 12 | 8 | 12 | 5 | 5 | **42** |
| 27 | 07 | Oleksandr Volkov (AlexVolkov0404) | 12 | 8 | 12 | 5 | 5 | **42** |
| 31 | 08 | Gongyao Xu (eluxyl) | 12 | 8 | 12 | 5 | 5 | **42** |
| 35 | 09 | Destiny Ogu (destinyogu) | 12 | 8 | 7 | 5 | 4 | **36** |
| 39 | 10 | Canhui Huang (CanhuiH) | 12 | 8 | 11 | 5 | 5 | **41** |
| 43 | 11 | Xiaoyang Xiao (Anchor-bkl) | 12 | 7.5 | 10 | 5 | 5 | **39.5** |
| 47 | 12 | Eviana Pham (evianaphm) | 12 | 8 | 12 | 5 | 5 | **42** |
| 51 | 13 | Aarush Maddela (aarushmaddela) | 12 | 7 | 11 | 5 | 4 | **39** |
| 55 | 14 | Candy Zhang (candiizy) | 12 | 8 | 12 | 5 | 5 | **42** |
| 59 | 15 | Donjhai Holland (DonjhaiH) | 12 | 7 | 10 | 5 | 4.5 | **38.5** |

**Mean:** 39.37/42 (93.7%) · **Range:** 33.0–42.0 · **Median:** 41.0

---

## Per-group entries (R/T/V/X/Z comments to paste)

### Group 01 — row 3 — Alex Soh (`alexchsoh`)

**Scores:** Q=12 · S=8 · U=12 · W=5 · Y=5 · **Total 42/42**

**BERT (R):** Excellent — clear model descriptions, BERT-LoRA implemented cleanly via PEFT on attention Q/V projections, comprehensive performance table covering 5 embeddings × 2 subjects × 4 metrics. No deductions.

**Modeling (T):** Clean ridge pipeline with story-level CV (no leakage), thoughtful α-search subsetting, and PCS-aware voxel analysis. Per-voxel CC distributions plotted for all 5 embeddings × 2 subjects.

**Interp (V):** Both stories thoroughly analyzed with paired SHAP+LIME on top 5 voxels × 3 TRs each. Strong cross-method comparison, careful PCS-style voxel filtering (only voxels passing predictability bar are interpreted), and content-word-vs-function-word divergence explained mechanistically.

**Stability (X):** Exemplary stability analysis — depth-on-one-judgment-call (BERT context window size) executed end-to-end including downstream interpretation. Table 3's voxel-overlap finding (0/5 at window 256, 2/5 at window 128) and the resulting word-level-stable / voxel-level-unstable conclusion is exactly the PCS framing the lab spec is reaching for.

**Writing (Z):** Polished, well-structured report. Clean modeling math, well-captioned figures with consistent color conventions. No deductions.


### Group 02 — row 7 — Abhishek Shukla (`abss123`)

**Scores:** Q=12 · S=7 · U=6 · W=4 · Y=4.5 · **Total 33.5/42**

**BERT (R):** Solid BERT setup. Pre-trained uses bert-base-uncased with layer 9, chunk_size 200, subtoken mean-pool. LoRA fine-tune via PEFT on Q/V attention sub-modules with MLM objective, merged before encoding. Comprehensive performance comparison table across 5 embeddings × 5 metrics. Minor: chunks are non-overlapping (no sliding window), but with chunk_size=200 words this rarely hits the 512-token limit so it's fine. No deductions.

**Modeling (T):** Clean ridge pipeline with efficient XTX/XTY precomputation and proper 5-fold leave-one-fold-out CV. Cross-method voxel-rank Spearman analysis (Fig 5) and the PCS-threshold "interpretable voxel" count are nice. Two soft deductions: (a) best-model deep-dive is fairly thin — fine-tuned BERT is identified but the "why is it the best" discussion is limited to noting LoRA provides marginal gain; (b) §5.1 claims they replicated on subject 2 but no subject-2-vs-subject-3 figures/tables appear in the report (results are all subject2; lasso.py defaults to subject3 but main.py to subject2).

**Interp (V):** Major scope gap — the lab asked for SHAP and LIME on **two stories** with **multiple top voxels** each, but this submission analyzes a **single voxel (28478) on the global held-out test set** using LIME for a **single TR**. The group acknowledges this in §6 ("the strongest report would include... story-specific SHAP/LIME runs for multiple named test stories") but didn't extend the analysis. What is here is real and well-written: the SHAP/LIME agreement on `dim_308` is a legitimate cross-method comparison, the word-projection table (Table 4) gives concrete neuroscientific traction ("grabs/riding/crashing/floating" suggesting embodied-action sensitivity), and the pre-trained vs fine-tuned BERT influence comparison is a nice extra. But the per-story, per-voxel coverage the lab asks for is missing.

**Stability (X):** Thoughtful multi-angle stability — α choice, per-story CC stability for static embeddings (with the nice observation that GloVe and Word2Vec succeed/fail on the *same* stories, supporting shared narrative signal), and cross-BERT voxel-rank stability (ρ=0.986). Half-point off because the headline interpretation result (dim_308 as the driver of voxel 28478 in fine-tuned BERT) isn't stability-tested end-to-end — no perturbation of α, fold, seed, or SHAP background to see if dim_308 keeps surfacing. Also note the slight α inconsistency between §5.3 ("α=1000 best") and Table 2 (α=10^4-10^6), which is probably a unit/notation mismatch but should be cleaned up.

**Writing (Z):** Well-organized 10-page report with strong PCS framing and explicit acknowledgment of limitations. Half-point off for figures with unreadable text (per-story stability boxplots, histogram axis labels) and an unusual figure-number-vs-placement mismatch (Figure 4 appears before Figures 2-3 in the printed page order). Several typos throughout ("diffetent", "Vord2Vec", "shifter"). No LLM-usage disclosure in `collaborations.txt`. Missing LIME reference.


### Group 03 — row 11 — Akanshaa Borgohain (`Akanshaa18`)

**Scores:** Q=10.5 · S=7 · U=6 · W=5 · Y=4.5 · **Total 33/42**

**BERT (R):** Fine-tuning pipeline is clean and well-implemented end-to-end — full-MLM-FT serves as both the Fine-tuned and LoRA items per calibration. **Critical issue with pre-trained BERT extraction:** the embedding-extraction loop (`bert_pipeline.py:39-50`) passes each word to BERT as its own input sequence with no surrounding words. The tokenizer itself works correctly; the bug is at the forward-pass level — self-attention has no neighbors to attend to, so the contextual embeddings collapse to "BERT's representation of word X with no context." This converts BERT into an effectively static embedder and almost certainly explains the report's surprised finding that BERT does not outperform GloVe. The report does not acknowledge this. Per-model descriptions are also thin. Performance comparison itself is excellent.

**Modeling (T):** Clean ridge pipeline with bootstrap CV and story-level splitting (no leakage). PCS threshold analysis and voxel-overlap scatterplots are nice. Minor deductions: only 5 bootstrap iterations, no per-voxel α distribution reported, and the GloVe-vs-BERT deep-dive does not interrogate why BERT might be underperforming (the in-isolation tokenization in `bert_pipeline.py` is the most plausible cause but goes unmentioned).

**Interp (V):** Strong SHAP analysis using `LinearExplainer` (mathematically exact for ridge — good choice). Two main issues. (1) **Only 2 voxels** are analyzed (the rubric implies more — top 5 is the norm); the lab spec is flexible here, but two is on the low end. (2) **LIME uses `LimeTabularExplainer` on the 3072-d feature space** rather than `LimeTextExplainer` on the actual text. With only 500 perturbations against 3072 features, LIME unsurprisingly fails to find signal — the report acknowledges this but does not switch to the appropriate text-based LIME variant. The comparison section is conceptual rather than empirical (no side-by-side bar chart for a specific voxel/TR). Honest framing throughout — they correctly diagnose LIME's failure mode.

**Stability (X):** Exemplary stability section — two real judgment calls (delay window, seed split) probed end-to-end with quantitative CV metrics. The finding that embedding *ranking* is preserved across seeds even when *absolute* CCs vary is exactly the PCS-flavored claim the lab is asking for. The monotonic top-1% decrease with more delays is correctly diagnosed as a regularization artifact rather than a substantive effect.

**Writing (Z):** Generally well-written and well-structured. Figures comprehensive (17 total). Two writing issues to flag: (a) the surprising "BERT ≈ GloVe" finding is framed scientifically rather than as a pipeline concern, when the most parsimonious explanation is that `bert_pipeline.py` passes each word to BERT as its own input sequence (max_length=16, no surrounding words), so self-attention sees no context — BERT collapses to a quasi-static embedder. (b) SHAP wordclouds visually attractive but lose signed attribution information.


### Group 04 — row 15 — Thomas Lee (`cylee0815`)

**Scores:** Q=12 · S=8 · U=12 · W=5 · Y=5 · **Total 42/42**

**BERT (R):** Outstanding BERT section. Group implemented BOTH full-parameter fine-tuning AND LoRA (not just one), with complete training recipes, per-epoch MLM losses, and a thoughtful "anticipated downstream performance (a priori)" prediction (p.10) registered BEFORE evaluation — `BERT-Pre ≳ BERT-LoRA > BERT-Full FT` — which turned out essentially correct. Mean-pool-not-CLS justification is exactly right. No deductions.

**Modeling (T):** Clean SVD-based ridge pipeline with story-level CV (no leakage), and the analysis is unusually careful — the "fine-tuning ≈ no-op given the noise" reading of Table 4 (p.11) is the right one, and the per-voxel right-tail framing in §6.5 is exactly the PCS predictability move. The α-grid mismatch (report mentions 5 values, code uses 3) is a tiny documentation glitch, not a methodological problem.

**Interp (V):** Exceptionally thorough interpretability section. Story selection is properly PCS-motivated (top-2 by held-out subject-level CC, Table 5 p.14); voxel selection is PCS-motivated (top-5 by held-out test CC, all above their 0.05 floor, §7.1 p.13); back-mapping from peak TR to word window is explicitly flagged as approximate (Lanczos + 4-TR delay). Cross-method comparison quantified via Jaccard@5/10 (Table 8, p.16) AND structured into four candidate explanations. Heatmap visualizations (Fig 8, 10) effectively show across-voxel patterns. The decision to deep-dive LoRA/subject2 while running the full pipeline on all 4 (variant, subject) pairs is exactly the right PCS-style tradeoff.

**Stability (X):** Outstanding stability check. The SHAP-vs-LIME stability sweep produces a quantitative reproducibility budget (top-3 SHAP words trustworthy; LIME requires ≥ 2-of-3 seed agreement) that is then carried back into the Discussion to gate which findings get reported. Figure 11 (p.18) is a particularly clear illustration of the differential stability of the two methods. Exactly the PCS framing the lab spec is reaching for.

**Writing (Z):** Polished, technically dense report with PCS framing threaded throughout. Pre-registers a prior prediction about Full FT vs LoRA vs Pre BEFORE evaluation (§5.1 p.10), which is exactly the kind of scientific discipline the lab is meant to teach. LLM usage statement is exemplary — distinguishes "polish" use from "intellectual content" use. Body is exactly 20 pages (references + honesty statement push the total to 22). No deductions.


### Group 05 — row 19 — Chenfei Peng (`ChenfeiP`)

**Scores:** Q=12 · S=6 · U=8 · W=5 · Y=5 · **Total 36/42**

**BERT (R):** Excellent — clear model descriptions, BERT-LoRA implemented cleanly via PEFT with adapters on Q/K/V/dense, comprehensive 5×2×4 performance table. Pre-trained BERT and LoRA pipelines share preprocessing stack, so head-to-head comparison is clean.

**Modeling (T):** Ridge pipeline is clean with proper story-level held-out split and chunked fitting. Voxel-distribution analysis (Figs 2, 4 + Table 2) is excellent and ties cleanly to the interpretability threshold. Main deduction: no CV α-search was actually run — the team used a single fixed α=10 with `--nboots 0 --single-alpha`. The course `bootstrap_ridge` utility supports CV (chunks + nboots), so this would have been a small change.

**Interp (V):** Strong cross-story replication on two subjects, paired SHAP+LIME with both signed-importance plots and overlap tables/heatmaps, content-word filtering applied cleanly. The decision to anchor interpretation in PCS predictability (top voxels with story-level CC > 0.30) is explicitly stated and exercised. Comparison narrative ("SHAP = global ridge direction, LIME = local perturbation sensitivity") is mechanistically grounded and shows up in both stories.

**Stability (X):** Exemplary stability analysis — the team identifies the interpretation threshold as a real PCS judgment call (not a free hyperparameter), perturbs it across three reasonable alternatives, and separates what flexes (the eligible voxel set's *scope*) from what stays stable (the embedding ordering and the qualitative top-voxel interpretation). The §3.7 cross-subject analysis is a clean companion piece. This is the lab spec's "stability of one judgement call" done very well.

**Writing (Z):** Polished, well-structured 19-page report. Clean modeling math, thorough captions, good color conventions, explicit academic honesty + LLM statement. Discussion paragraphs around what each result does and does not imply are unusually mature. No deductions.


### Group 06 — row 23 — Drew Galloro (`andreagalloro`)

**Scores:** Q=12 · S=8 · U=12 · W=5 · Y=5 · **Total 42/42**

**BERT (R):** Outstanding — clean PEFT-based LoRA, exhaustive performance comparison across 8 embeddings × 2 subjects × 7 metrics. Bonus BERT-layer ablation with validation-driven selection of layer 8 as winner is exactly the methodology in Toneva & Wehbe. No deductions.

**Modeling (T):** Exemplary ridge pipeline — two-tier α-selection (bootstrap_ridge for memory-friendly, single-pass SVD on inner-train/val for memory-bound), per-voxel α grid of 30 values, both concatenated-test and per-story aggregations reported. Full BERT-layer ablation with validation-driven winner selection.

**Interp (V):** Strong interpretability section with word-level forward-function (forward function wraps the full downstream pipeline: word mask → embed → downsample → ridge → per-voxel CC — exactly the right framing per the lab spec). Two stories × both subjects × top-5 voxels each = 20 voxel-story pairs total. Per-voxel SHAP∩LIME overlap quantified against random-expectation baseline.

**Stability (X):** Far above what was required — *five* distinct stability methods (per-story CC, 5-fold CV, permutation testing with honest Bonferroni resolution analysis, α-grid sensitivity, cross-subject distribution Q-Q matching). Each isolates a different judgment call. Exemplary PCS framing.

**Writing (Z):** Polished, comprehensive, reproducibility-focused report. Math notation correct, figure conventions consistent, limitations section honest. No deductions.


### Group 07 — row 27 — Oleksandr Volkov (`AlexVolkov0404`)

**Scores:** Q=12 · S=8 · U=12 · W=5 · Y=5 · **Total 42/42**

**BERT (R):** Excellent — LoRA fine-tuning executed with careful hyperparameter exploration (28 experiments tabulated across rank, α, LR, epochs, stride, target modules) and clear final-model selection rationale. All five model families described in dedicated subsections. Pre-trained BERT-vs-BoW inversion noted and discussed.

**Modeling (T):** Methodologically careful ridge pipeline. Global α-selection over `logspace(1, 4, 8)` is reasonable given the block-wise memory constraint. 28-experiment LoRA hyperparameter sweep with seed-stability check on the winning config is thorough. Voxel CC distributions plotted and discussed.

**Interp (V):** Strong interpretability section. Two stories × two voxels × {SHAP + LIME + POS-aggregated SHAP} is more depth than required. The POS-level SHAP analysis (Figs 2-3) is a particularly nice addition that addresses the "many words have nearly-identical SHAP scores" issue authors identified. Cross-method comparison is mechanistically explained, not just descriptive. Minor: only Subject 2 analyzed (not Subject 3); only 2 voxels per story (lab suggests ~5). Doesn't change score given calibration generosity, but worth a note.

**Stability (X):** Exemplary PCS-style stability analysis — explicitly labeled "PCS Evaluation" with Predictability/Computability/Stability subsections. Four distinct stability probes (voxel subsetting, cross-subject Jaccard, cross-embedding Jaccard, seed reruns). The cross-subject Jaccard finding (0.04 ≈ 8% overlap) leading to a cautious "voxel-level interpretations don't generalize across individuals" conclusion is the PCS framing the lab is reaching for.

**Writing (Z):** Solid technical writing with clear LoRA experiment narrative. One-page deduction for exceeding the 20-page limit (main body to p.19, refs p.20, appendix p.21). The LoRA tuning section §4 is very table-heavy — 8 tables in 4 pages could be compressed into 2-3 summary tables to recover the page. Otherwise no concerns.


### Group 08 — row 31 — Gongyao Xu (`eluxyl`)

**Scores:** Q=12 · S=8 · U=12 · W=5 · Y=5 · **Total 42/42**

**BERT (R):** Outstanding — eight-method comparison with rigorous significance testing, clean PEFT LoRA implementation with three thoughtful variants (last4, improved-recipe, mid4), and a genuinely interesting empirical finding that mid-layer (5-8) extraction beats last-4 extraction by +9% on s2. The Spearman rank-correlation analysis between per-voxel CC vectors (Figure 3) is particularly polished.

**Modeling (T):** Polished ridge pipeline with two-SVD precomputation, sparse LU for BoW, per-voxel α selection, and rigorous Wilcoxon significance testing. Deep voxel-level analysis: violin plots + Spearman rank correlations between every method pair. The argument that mid4 vs last4 ρ=0.87/0.91 (vs last4-recipe-variants ρ=0.98) confirms the layer choice—not the fine-tuning recipe—is the real driver is sharp PCS reasoning.

**Interp (V):** Strong, methodologically careful interpretability work. Two thematically contrasting stories, both subjects, response-trace figures (Figs 6, 8) anchor LIME windows in genuine BOLD-prediction peaks, and the SHAP-vs-LIME comparison (Table 4 + §5.4) is one of the most thoughtful in the cohort — explicitly recognizes that SHAP-at-TR vs LIME-at-word is a complementarity, not a bug. The one shortfall: report and code both analyze top-3 voxels per story rather than top-5; small deduction (-0.5 per story).

**Stability (X):** Exemplary stability analysis on a meaningful judgment call (cross-subject replication of the encoding map). Four complementary levels of agreement (story-level r=0.745, voxelwise distribution, sorted bar plot, cross-method ratio convergence) — the last is especially clever: better embeddings shrink the s3/s2 ratio, which is itself a replication argument.

**Writing (Z):** Polished, well-structured report with strong figures and rigorous statistical reporting (Wilcoxon significance, Spearman rank correlation, story-level Pearson). Minor copy-edits would tighten it further. 21-page length is technically over the limit but the analytical content fits within 20 pages (final page is the LLM-usage statement + references); not deducting.


### Group 09 — row 35 — Destiny Ogu (`destinyogu`)

**Scores:** Q=12 · S=8 · U=7 · W=5 · Y=4 · **Total 36/42**

**BERT (R):** Excellent — clear model descriptions, BERT-LoRA implemented cleanly via PEFT on attention Q/V projections, comprehensive performance table covering 5 embeddings × 2 subjects × 4 metrics. No deductions.

**Modeling (T):** Solid ridge pipeline with bootstrap CV (5 iterations, 5×40-TR held-out chunks), per-voxel α selection, and PCS-aware framing of voxel-level performance. Histograms and per-chunk boxplots both presented.

**Interp (V):** Both stories analyzed with paired SHAP+LIME figures and thoughtful per-voxel discussion. Main deductions: (1) **Method.** The implementation is a hand-rolled KernelSHAP-style + LIME-style perturbation on the **design-matrix rows** corresponding to each word — not the text-perturbation wrapper pattern (re-encode `text → BERT → downsample → ridge` per perturbation) that the reference guide explicitly prescribes and that Sean covered in lecture. Because each TR's features are a Lanczos blend of nearby words, and because BERT's contextual sensitivity requires re-encoding to be measured, masking X rows does not actually answer "which words drive this voxel." (2) **Voxel count.** 2 voxels per story instead of ~5. (3) **Single-subject.** Subject 2 only.

**Stability (X):** Thoughtful single-judgment-call check (α-grid range widening for GloVe ridge) with explicit setup table, before/after metric comparison, and PCS framing. The conclusion is honest (small numerical movement, same qualitative ranking) and complements the Lab 3.1 story-by-story stability check (Table 4, p.8) that confirms positive CC on 41/42 held-out test stories across both subjects.

**Writing (Z):** Well-organized report with a thorough writeup and consistent figure conventions on most plots. LLM usage statement is present (p.19, "Academic honesty / LLM Usage"). Minor docks: Figure 4 (per-voxel-block boxplots) is visually crowded, and the section numbering is inconsistent (most headings unnumbered, academic-honesty section uses §1/§1.1). 19-page length is just under the cap.


### Group 10 — row 39 — Canhui Huang (`CanhuiH`)

**Scores:** Q=12 · S=8 · U=11 · W=5 · Y=5 · **Total 41/42**

**BERT (R):** Excellent BERT work — pre-trained sliding-window extraction, real MLM fine-tuning (not LoRA-as-fine-tune shortcut), three LoRA ranks compared, and a careful Table 2 separating contextual-vs-static gains from rank effects.

**Modeling (T):** Solid ridge pipeline with story-level 5-fold CV, transparent disclosure of the cross-embedding α reuse (a real limitation that they own up to), and clean voxel-distribution analysis with PCS-motivated top-1% gating for downstream interpretation.

**Interp (V):** Two stories × two subjects × top-10 voxels gives 40 SHAP+LIME runs — significantly more breadth than the lab asks for. Qualitative comparison is thoughtful and the SHAP-sign vs LIME-sign discussion is mechanically correct. Half-point deduction per story because the headline cross-method Jaccard = 0.000 (Table 6) is a position-vs-type comparison bug rather than a real instability — the report waves at this ("reflects methodological differences (masking vs deletion)") but doesn't fix it. Would be cleaner to normalize to types before set-intersecting.

**Stability (X):** Exemplary multi-axis stability work — trimming convention, per-story, cross-embedding Jaccard, and +2 TR interpretability shift. The cross-embedding Jaccard finding (LoRA shifts ~1/3 of top-5% voxel membership despite tiny aggregate CC change) is the standout PCS observation of the report. 5/5 with room to spare.

**Writing (Z):** Well-structured report with strong in-line limitation disclosure and a sharp PCS bottom line. Half-point deduction for per-story figures with unreadable tick labels and for only displaying Subject 2 SHAP/LIME charts in the body when results exist for both subjects.


### Group 11 — row 43 — Xiaoyang Xiao (`Anchor-bkl`)

**Scores:** Q=12 · S=7.5 · U=10 · W=5 · Y=5 · **Total 39.5/42**

**BERT (R):** Excellent BERT/LoRA pipeline. LoRA fine-tuning trains end-to-end through Lanczos interpolation + delay stack with a differentiable subword pooling — a sophisticated implementation. Clean PEFT setup on attention Q/V projections.

**Modeling (T):** Strong voxel-level analysis (Jaccard overlap across embedding pairs, CC-band partitioning, head-to-head BERT vs LoRA scatter). Small methodology-vs-report inconsistency: report claims 5-fold CV with α ∈ {1,10,100,1000,10000}, but code uses a single `bootstrap_ridge` iteration with `np.logspace(0,3,6)`. Per-voxel alpha selection is sound, just described inaccurately. **−1 pt for CV mismatch.**

**Interp (V):** Both required stories analyzed with cleanly engineered SHAP (`shap.KernelExplainer`) and from-scratch LIME implementations. Main deduction: explanations are computed via masking processed-X rows by chunk rather than re-encoding perturbed text through BERT — the reference guide explicitly prescribes the wrapper pattern (`text → BERT → downsample → ridge`) so the effect of removing a word propagates through Lanczos blending and BERT's contextual mechanism. Mask-on-X misses both. Smaller deduction than groups 02/09 because the engineering quality is genuinely high. Minor secondary issue: figures only display one of the top-5 voxels per (subject, story); the per-voxel Jaccard table is computed in code but not surfaced in the report.

**Stability (X):** Strong stability check — depth on one judgment call (the temporal delay stack), with 6 perturbation variants, rerun end-to-end through ridge fitting, and a clean three-point conclusion. Tied directly back to neurobiological motivation (early lags carry the strongest semantic signal).

**Writing (Z):** Polished, well-organized report with thoughtful limitations section. Strong use of overlay figures (BERT-vs-LoRA scatter, SHAP-vs-LIME bar overlays). Honest about both methodology and limitations.


### Group 12 — row 47 — Eviana Pham (`evianaphm`)

**Scores:** Q=12 · S=8 · U=12 · W=5 · Y=5 · **Total 42/42**

**BERT (R):** Excellent — clear separation of pre-trained, full fine-tuned, and LoRA BERT pipelines; LoRA setup is textbook (Q/V, r=8, α=32 via PEFT). Strong Table 1 + Figure 1 covering 6 embeddings × 2 subjects × 4 metrics + fraction positive. The per-voxel delta histograms (Fig 3) directly answer "does fine-tuning help?" rather than relying on summary tables alone.

**Modeling (T):** Clean blocked-bootstrap CV with per-voxel α, well-motivated z-scoring, no leakage across train/test stories. PCS-aware voxel analysis: explicit CC ≥ 0.10 interpretation threshold and box-plot framing in Fig 11. No deductions.

**Interp (V):** Both stories thoroughly analyzed with paired SHAP + LIME on top-3 voxels per (story × subject) = 12 attribution panels total. The wrapper approach (perturb → rerun BERT → rerun ridge → story-level CC as scalar output) is methodologically careful and well-motivated. SHAP-vs-LIME scatter (Fig 6) and signed-SHAP heatmaps (Figs 12, 13) provide cross-method and cross-voxel checks. Cautious interpretation: "useful for hypothesis generation, not direct neural evidence". No deductions.

**Stability (X):** Two substantive PCS-style stability checks. The contextual-vs-word-level comparison (Fig 7 + Fig 8) gives a sharp answer to "where does BERT's advantage come from?" — surrounding narrative context contributes ~half the contextual lift, not just the higher dimensionality. The LoRA rank sweep (r ∈ {4, 8, 16}) confirms the main LoRA result is not an artifact of the chosen rank. Exactly the PCS framing the lab spec is reaching for.

**Writing (Z):** Polished, well-structured report. Clean modeling math, well-captioned figures with consistent color conventions across the main and appendix figures. Strong roadmap (three questions in §1, answered in §6 and §7). No deductions.


### Group 13 — row 51 — Aarush Maddela (`aarushmaddela`)

**Scores:** Q=12 · S=7 · U=11 · W=5 · Y=4 · **Total 39/42**

**BERT (R):** Excellent BERT pipeline — clean PEFT/LoRA setup with proper MLM masking, sensible 3×3 hyperparameter grid, and clear progression-of-models table. Subword averaging handled correctly.

**Modeling (T):** Solid ridge pipeline with voxel-batched memory management and clear PCS-motivated top-1% voxel selection for downstream interpretation. CV description is a bit light on fold-count specifics in the writing, but the underlying code (bootstrap_ridge with 5 bootstraps) is correct. Per-voxel CC histograms cover the best models well.

**Interp (V):** Both stories analyzed with paired SHAP+LIME and clear side-by-side bar charts with overlap counts. Cross-method discussion is thoughtful and mechanistic (LIME-vs-SHAP perturbation differences tied to BERT positional encodings). **Deductions:** lab spec called for top 5 voxels per story; this group analyzed top 3 per story. Only one text segment per voxel (first 300 words) rather than multiple TRs. SHAP `max_evals=200` and LIME `num_samples=200` are on the low side and may produce noisy attributions. Subject 2 only (without justification for choosing 2 over 3, given §4 p.10 says S3 > S2 in CC).

**Stability (X):** Strong stability check on the LoRA hyperparameter judgment call — a 3×3 grid with proper validation-loss tracking, a clean heatmap visualization, and a data-driven model selection (r=8, lr=5e-4). Good PCS-style framing of which hyperparameters drive stable learning dynamics.

**Writing (Z):** Generally polished report with a clear narrative progression from BoW → BERT → interpretability. LoRA heatmap and model-progression bar chart are highlights. Minor proofreading issues and the LIME/SHAP figures could use larger axis labels and a side-by-side cross-story layout.


### Group 14 — row 55 — Candy Zhang (`candiizy`)

**Scores:** Q=12 · S=8 · U=12 · W=5 · Y=5 · **Total 42/42**

**BERT (R):** Excellent — went beyond minimum with full MLM fine-tuning AND LoRA at three ranks (r=4,8,16). Clean PEFT setup on Q/V projections, extensive 8-method × 2-subject × 4-metric performance table, and a Jaccard-distance dendrogram showing BERT and static embeddings access fundamentally different brain regions. No deductions.

**Modeling (T):** Excellent ridge pipeline with proper bootstrap CV (no leakage), per-voxel α selection, and z-scoring. CC distributions plotted for every embedding × subject combination. The Jaccard / hierarchical-clustering analysis of top-5% voxel sets is a clear "above-and-beyond" addition that perfectly anticipates the PCS stability discussion.

**Interp (V):** 3 voxels × 2 stories with correct text-perturbation wrapper (`shap.maskers.Text` + `LimeTextExplainer`). Cross-story finding: same voxel shows opposite SHAP signs for `penpal` vs `adollshouse`, interpreted as reflective vs confrontational narrative register.

**Stability (X):** Excellent stability analysis — chose a single judgment call (LoRA rank), tested it on **two independent axes** (CC performance AND voxel identity via Jaccard), and explicitly framed how the stability finding strengthens the broader "fine-tuning doesn't help" conclusion. Going beyond rank-vs-CC into rank-vs-voxel-identity is precisely the PCS thinking the lab is reaching for.

**Writing (Z):** Polished, well-structured report with explicit LLM-usage statement. Every major section ends with a PCS-framed take-away, which reads as genuinely thoughtful rather than perfunctory. Figures consistent and well-captioned. No deductions.


### Group 15 — row 59 — Donjhai Holland (`DonjhaiH`)

**Scores:** Q=12 · S=7 · U=10 · W=5 · Y=4.5 · **Total 38.5/42**

**BERT (R):** Excellent — three BERT variants (pre-trained, full MLM, LoRA) all implemented and compared. Clean PEFT-based LoRA on Q/V projections with explicit math, sensible hyperparameters, and proper checkpoint resumption logic for Slurm. Five-embedding × subject performance comparison reported as both table and bar chart.

**Modeling (T):** Solid pipeline overall with clear preprocessing math (HRF delays, Lanczos downsample, story-level train/test split), per-voxel ridge, and PCS-aware framing for downstream interpretability. Two issues: (i) the RidgeCV grid topped out at α = 10^5 and that's exactly the alpha that got selected — the group flags this honestly in the conclusion but didn't extend the grid; (ii) no per-voxel CC histogram for the BERT variants in §3.2 (only the static three). The violin plot in Figure 13 partially covers this in the stability section.

**Interp (V):** Solid SHAP+LIME implementation on top 5 voxels × 2 stories using the actual `shap` and `lime` Python libraries with sensible configuration (Text masker / `[MASK]`, deletion-based LIME with 500 samples, 20-feature ceiling). Three SHAP visualizations per story (global, heatmap, signed drivers) is thorough. Main deductions are around the cross-method comparison: it stays at the prose level (no overlap bar chart or quantitative comparison metric), and the Sweetaspie-specific comparison narrative is thinner than the Triangle Shirtwaist one. The per-voxel signed-driver figure for Triangle Shirtwaist exists in `figs/shap/` but did not get embedded in the report. Interpretability analysis is single-subject (Subject 2); Subject 3 LIME re-runs appear in the stability section but no Subject 3 SHAP.

**Stability (X):** Cross-subject stability check executed across both the modeling (embedding-ranking) and interpretability (LIME word-importance) layers. Conclusion is honest — the BoW→W2V/GloVe→BERT/LoRA hierarchy reproduces on Subject 3 and the top LIME word ("said") is stable, while individual sign patterns shift. Acknowledged limitation (no Subject 3 SHAP) is appropriately flagged for future work. Thoughtful and well-framed.

**Writing (Z):** Well-organized 19-page report with clean math, consistent figure conventions, and an explicit LLM-usage statement (§6.2 p.19) covering both coding and writing assistance. Minor docks: the per-voxel SHAP signed-driver figure for Triangle Shirtwaist did not make it into the body (it exists in `figs/shap/`); axis text in several aggregate-importance bar charts is small at the embedded size; conclusion could tie SHAP/LIME findings back to neurobiology more crisply.

