# Lab 3 Grading — Master Summary (Sean's categories: 3.2 + 3.3)

> **Sean's columns:** Q (3.2 BERT, 12 pts) · S (3.2 Modeling/Eval, 8 pts) · U (3.3 Interpretability, 12 pts) · W (Stability, 5 pts) · Y (Writing, 5 pts) — total 42 pts.
>
> **Calibration applied to all groups (per Sean's decision):**
> 1. **LoRA-only earns full Fine-tuned (4) + LoRA (2) = 6/6.** Lab spec presents LoRA as the canonical fine-tune; rubric splits the items but students didn't see the rubric.
> 2. **Stability check follows lab spec ("one judgement call"), not rubric's "1 pt per model".** Depth-on-one-model done thoughtfully = 5/5.
> 3. **Generous on partial credit** — 0.5–1.5 pt deductions per category if attempted but rough.

## Score table

| Group | Handle | Leader | Q/12 | S/8 | U/12 | W/5 | Y/5 | **Total/42** | **%** |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | alexchsoh | Alex Soh | 12 | 8 | 12 | 5 | 5 | **42** | 100% |
| 02 | abss123 | Abhishek Shukla | 12 | 7 | 6 | 5 | 4.5 | **34.5** | 82% |
| 03 | Akanshaa18 | Akanshaa Borgohain | 10 | 8 | 8 | 5 | 5 | **36** | 86% |
| 04 | cylee0815 | Thomas Lee | 12 | 8 | 12 | 5 | 5 | **42** | 100% |
| 05 | ChenfeiP | Chenfei Peng | 12 | 7.5 | 8 | 5 | 5 | **37.5** | 89% |
| 06 | andreagalloro | Drew Galloro | 12 | 8 | 12 | 5 | 5 | **42** | 100% |
| 07 | AlexVolkov0404 | Oleksandr Volkov | 12 | 8 | 12 | 5 | 5 | **42** | 100% |
| 08 | eluxyl | Gongyao Xu | 12 | 8 | 8 | 5 | 5 | **38** | 90% |
| 09 | destinyogu | Destiny Ogu | 12 | 8 | 7 | 5 | 4 | **36** | 86% |
| 10 | CanhuiH | Canhui Huang | 12 | 8 | 12 | 5 | 5 | **42** | 100% |
| 11 | Anchor-bkl | Xiaoyang Xiao | 12 | 8 | 8 | 5 | 5 | **38** | 90% |
| 12 | evianaphm | Eviana Pham | 12 | 8 | 12 | 5 | 5 | **42** | 100% |
| 13 | aarushmaddela | Aarush Maddela | 12 | 8 | 12 | 5 | 4.5 | **41.5** | 99% |
| 14 | candiizy | Candy Zhang | 12 | 8 | 12 | 5 | 5 | **42** | 100% |
| 15 | DonjhaiH | Donjhai Holland | 12 | 8 | 11 | 5 | 5 | **41** | 98% |

**Cohort stats (FINAL, synced to xlsx):** Mean **39.77/42 (94.7%)** · Range 34.5–42 · Median 42 · **7 perfect scores** (01, 04, 06, 07, 10, 12, 14). Group 13 at 41.5 and groups 14/15 at 41 are one step below perfect. Lowest is 34.5 (82%) — at Sequoia's floor.

> **Strict wrapper dock applied to groups 03, 05, 08, 09, 11.** Their SHAP/LIME implementations operate on the already-encoded design matrix rather than re-encoding perturbed text through BERT per the wrapper pattern documented in `Lab3_Reference_Guide.html`. The structural consequence is word-disambiguation loss: words co-occurring within the same TR (or chunk, for group 11) receive identical importance scores. Cross-group consistent dock: SHAP and LIME items 1/2 each per story (= −4 on U). Comparison items credited separately based on whether cross-method analysis produced real word-level insight; group 09 keeps Comparison dock at 1.5/2 each story (prose-only, no quantitative overlap metric).

> **Revision log (2026-05-16):**
> - Group 03: U 8 → 6 (SHAP wrapper violation matching LIME, cross-group consistency). Q 7.5 → 9.5 (full-FT-counts-for-LoRA rule). Total 31.5 → 29.5 → 31.5.
> - Group 05: U 12 → 8 (audit caught wrapper violation: SHAP is exact linear decomposition; LIME masks already-encoded matrix). Total 40 → 36.
> - Group 07: Y 4 → 5 (consistency refund: same page-overflow pattern as groups 04, 08). Total 41 → 42.
> - Group 08: U 11 → 12 (≥2 voxels rule: top-3 voxels meets floor). Total 41 → 42.
> - Group 09: U 5 → 7 (≥2 voxels rule: voxel-count portion of deduction refunded; wrapper violation remains). Total 34 → 36.
> - Group 11: U 9 → 10 (≥2 voxels rule: figure shows 2 voxels per story across subjects). Total 38 → 39.
> - Group 13: U 10 → 11 (≥2 voxels rule: top-3 voxels meets floor; remaining deduction is for single-text-segment, not voxel count). Total 38 → 39.
>
> **Above-and-beyond moderation (gray-zone deductions softened when group went substantially above-spec elsewhere):**
> - Group 11: S 7 → 7.5 (CV-mismatch deduction softened given differentiable LoRA through Lanczos interpolation). Total 39 → 39.5.
> - Group 14: U 9.5 → 10 (figure-coverage −0.5 softened given all 3 BERT variants + Jaccard dendrogram). Total 39.5 → 40.
> - Group 15: Y 4 → 4.5 (figure-missing-from-PDF deduction softened given all 3 BERT variants implemented). Total 38 → 38.5.
> - Group 10: Y 4.5 → 5 (figure-clutter deduction softened given full FT + 3 LoRA ranks + 4-axis stability + 40 SHAP+LIME runs — cross-group consistency with 11/14/15 moderations). Total 40.5 → 41. *Caught by 2nd-pass audit.*
>
> **Cross-grader-consistency review (final pass):** Sequoia's grading range on cats 1+2 was 27-31.5 / 33 (82-95%). Sean's bottom tier was running 74-75%, ~7 pts below Sequoia's lowest in relative terms. Two targeted moderations to narrow that gap:
> - Group 02: U 4 → 6 (the −6 "no story 2" step was harsher than the spec intent; their concatenated held-out aggregate covers ~1 partial story). Total 31 → 33.
> - Group 03: Q.Description 0.5 → 1.5 (Description item is about understanding static-vs-contextual, which they correctly identify; doesn't need deep per-model architecture coverage). Total 31.5 → 32.5.
>
> **Second-rescan revisions:**
> - Group 14: U 10 → 12 (per-item math reconciliation + single-subject is fine per lab spec + above-and-beyond on all 3 BERT variants). Total 40 → 42.
> - Group 02: Y 4 → 4.5 (Figure 4 numbering-vs-placement portion of Y dock was borderline administrative; the unreadable per-story labels remain a real issue). Total 33 → 33.5.
> - Group 03: Y 4 → 4.5 (Readability dock was double-counting the BERT-pipeline-framing issue already docked on Q.Pre-trained). Total 32.5 → 33.

> **Bookkeeping fixes (no score changes, just internal consistency):**
> - Group 03 §U: per-item SHAP rows now match the audit-revised total (1/2 each, not 2/2).
> - Group 05 §U summary table: now matches the audit-revised header (8/12, not 12/12).
> - Group 11 §S section header: now matches the moderated summary table (7.5/8, not 7/8).
> - Group 15 §Y section header: now matches the moderated summary table (4.5/5, not 4/5).

> **Post-grading revision (May 16):** Groups 02, 09, and 11 had U column lowered by −1/−4/−2 respectively after Sean pointed out that the methodologically correct approach for SHAP/LIME (per the lecture and `Lab3_Reference_Guide.html` §Interpretation → The Wrapper Pattern) is to wrap `text → BERT → downsample → ridge` and perturb raw text. All three groups instead perturbed already-encoded features (3072-dim vector for group 02; design-matrix X rows for groups 09 and 11), which loses BERT's contextual sensitivity and conflates Lanczos-blended TR contributions. Magnitude reflects severity: 02 already heavily docked on scope (-1 marginal); 09 compounds method + voxel count + single subject (-4); 11 has the cleanest engineering and uses real SHAP/LIME library calls but on the wrong input (-2).

## Cohort-wide patterns worth knowing

1. **Stability check (W) was a strength across the board** — every group except 02 (4/5) got 5/5. This is the column the rubric's "1 pt per model" wording could have penalized; under the lab-spec calibration, depth-on-one-judgment-call worked well. Worth confirming this calibration choice produced the curve you wanted.

2. **3.3 Interpretability (U) is where the cohort split.** 8 groups hit 12/12 or 11/12; the rest (02, 03, 09, 13, 14, 15) lost 2–7 points. Common shortfalls:
   - **Sub-spec voxel count:** groups 08 (3), 09 (2 per story = 4 total), 12 (3, OK because spec says "top voxels"), 13 (3), 14 (1 shown despite 3 claimed). Lab spec says "for these voxels" without naming a number; group 01 set the de-facto bar at 5 voxels × 3 TRs. Worth deciding cohort-wide: is 3 voxels = full credit, or -1?
   - **One subject only:** common, usually justified (higher-SNR subject). Not deducted by sub-agents.
   - **Indirect SHAP/LIME** (perturbing embedding features rather than re-encoding perturbed text): groups 02, 03, 05, 09, 11. Per Rule 3 in the calibration, this is a substantive methodological error (the reference guide explicitly prescribes the text-perturbation wrapper). All five groups were docked accordingly (severity varies by engineering quality and other scope shortcomings).

3. **3.2 Modeling (S) deductions** mostly come from CV-procedure mismatches between report text and code: group 05 (single shared α, no CV), 11 (α grid doesn't match what report claimed), 13 (vague description), 15 (α grid topped out, ceiling selected throughout). All flagged in the per-group notes.

4. **Page limit (20 pages).** Groups 07 (21), 08 (21), 04 (22 incl. refs). I generally took -0/-1 since content body fits.

5. **Group 02 outlier (32/42).** Clean engineering, but interpretability is genuinely off-spec: one voxel, one model, one TR-batch on global held-out, and SHAP/LIME run on embedding dimensions rather than tokens. This is the biggest-deduction group; worth Sean reading their U section directly to confirm.

6. **Group 03 (31.5/42).** Pre-trained BERT extraction passes each word to BERT as its own input sequence (`bert_pipeline.py:39-50`, no surrounding words), so self-attention has no context to attend to — BERT collapses to a quasi-static embedder. This explains the report's "surprising" finding that BERT doesn't beat GloVe (1 pt deducted on Pre-trained). Group also did full-MLM-FT in place of LoRA, which per Sean's updated calibration earns full Fine-tuned + LoRA = 6/6 (full FT is at least as demanding as LoRA).

## Cross-group calibration questions worth resolving once before spreadsheet entry

These came up in multiple groups' notes — picking a single answer for each will lock the curve.

1. **Voxel count in 3.3.** Lab spec doesn't name a number; group 01 modeled 5 voxels × 3 TRs as "thorough." Groups doing 3 voxels per story currently lose ~1 pt; groups doing 2 voxels lose 2–3 pts. Confirm scale.

2. ~~**Indirect SHAP/LIME** (perturb features instead of re-encoding text). Groups 02, 09, 11 use this~~ — **RESOLVED:** −1/−4/−2 applied to groups 02/09/11. See `feedback_stat214_lab3_calibration.md` Rule 3.

3. **Page limit overage** (Groups 04, 07, 08 are 21–22 pages incl. refs). Sub-agents took 0 to -1. Cohort-wide call?

4. **Group 03 BERT extraction bug.** They pass each word to BERT as its own input sequence (`max_length=16`, no surrounding words). Tokenization itself is fine — the issue is at the forward-pass level: self-attention has no neighbors to attend to, so the contextual embeddings collapse to context-free vectors. This silently propagates to their "BERT doesn't beat static embeddings" conclusion. Sub-agent deducted 1 pt on Pre-trained (1/2 instead of 2/2).

5. **Group 03 abandoned LoRA.** They started LoRA, abandoned it, and did full MLM fine-tuning instead. Sub-agent docked 2 LoRA points (4/4 + 0/2 = 4/6). Reverse-calibration question: full-MLM-without-LoRA earns full Fine-tuned (4) but zero LoRA (0)? Or some partial since they show the LoRA setup before abandoning it?

## What to do next

When you're ready to commit scores to `STAT 214 Spring 2026 lab 3 Grading.xlsx`:

- Enter Q/S/U/W/Y scores on the **leader rows only** (rows 3, 7, 11, ... 59).
- Enter comments in R/T/V/X/Z. Each per-group notes file has a "Column X comment (suggested)" line ready to paste.
- After leader rows are filled, propagate the new Total in column AA to the other 3 group members' rows (sheet shows blank cells but matching Totals for the cohort).
- Make a backup copy of the .xlsx before any bulk edit.

When you're ready, I can either (a) edit the spreadsheet via openpyxl, (b) emit a CSV/TSV you paste in, or (c) just hand you a row-by-row table of values to type in. The grading notes are the source of truth either way.
