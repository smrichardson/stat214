# Concerns Ledger — Items Worth Sean's Personal Eyes

Aggregated from each group's grading notes file. Use this as a single ranked list of judgment calls and items the grader (or auditor) flagged for your personal review before final sign-off.

## Tier A — Bottom-tier groups (≤36/42, biggest grade impact)

These are the grades most likely to generate student appeals; the deductions are big enough that being wrong is consequential.

- **Group 02** (31.0/42) — see `group-02.md` end-of-file 'Notes for Sean' or 'Items for Sean' section.
- **Group 03** (29.5/42) — see `group-03.md` end-of-file 'Notes for Sean' or 'Items for Sean' section.
- **Group 05** (36.0/42) — see `group-05.md` end-of-file 'Notes for Sean' or 'Items for Sean' section.
- **Group 09** (34.0/42) — see `group-09.md` end-of-file 'Notes for Sean' or 'Items for Sean' section.

## Tier B — Groups with post-audit revisions (calibration changes)

These groups had scores adjusted after the audit pass. Worth re-reading the revision blockquotes to confirm you agree with the magnitude.

- **Group 05** (36.0/42) — revision blockquote at top of `group-05.md`.
- **Group 07** (42.0/42) — revision blockquote at top of `group-07.md`.
- **Group 09** (34.0/42) — revision blockquote at top of `group-09.md`.
- **Group 11** (38.0/42) — revision blockquote at top of `group-11.md`.

## Tier C — Per-group flags raised by graders or auditors

Items the graders and auditors explicitly called out as worth your eyes. Most have already been weighted into the score; these are visibility notes.

### Group 02 (31.0/42)

1. **U-column score (5/12) is the biggest call.** The group ran SHAP+LIME on only one voxel of one model with no per-story split. This is well below the spec (two stories, multiple voxels). However, the cross-method agreement on dim_308 and the projection-based word table are real interpretive content. Sean — confirm 5/12 feels right; the alternative argument for 6-7/12 would lean on the fact that pre-trained BERT SHAP also exists (a third method comparison) and the dim_308 finding is a substantive result. The argument for going lower (3-4/12) would emphasize that LIME explains only one TR and SHAP/LIME both run on dimension features rather than text.
2. **No LLM-usage statement** — Sean may want to flag this for the writing column or surface it to the course coordinator. Currently no deduction applied beyond the existing 1pt writing deduction.
3. **Subject2 vs subject3 mismatch** in code defaults (`main.py` → subject2, `lasso.py` → subject3) and the §5.1 claim of "we applied the same approach to a second subject" without any visible per-subject results in the report — worth a quick visual check.
4. **§5.3 best-α inconsistency** with Table 2 (text says α=1000, table says α=10^4-10^6) — could be a unit-vs-log-axis confusion, but worth confirming the right number actually got used in the final regressions.

### Group 09 (34.0/42)

1. **3.3 voxel count.** The lab spec asks for ~5 voxels per story; this group did 2 voxels per story × 2 stories = 4 voxels total. I deducted 0.5 per item (1.5/2 × 6 items = 9/12). If Sean wants to be more lenient given that the discussion per voxel is genuinely thoughtful, +1 to +2 would be defensible. If Sean wants to apply the rubric more strictly, leaving this at 9/12 (or even lower) is also defensible.
2. **3.3 SHAP/LIME implementation.** Group did NOT use the `shap` / `lime` Python libraries. Instead they hand-rolled KernelSHAP-style coalition weighting (`07_interpret_bert.py:263-283`) and LIME-style random-keep masks with exponential-distance kernel weighting (`:286-315`), with the "perturbation" being zeroing the design-matrix rows for that word's TRs. Mathematically reasonable but Sean may want to verify this matches his expectation for the task — the report calls them "SHAP-style" and "LIME-style" so this is at least transparent.
3. **3.3 subject coverage.** Only Subject 2 was interpreted. Lab spec doesn't strictly require both subjects, but most groups do both.
4. **Repo size.** This is the ~4GB repo Sequoia flagged — committed data/models. Not Sean's deduction.

### Group 11 (38.0/42)

- **CV mismatch (−1 pt on S):** Report claims 5-fold CV with α ∈ {1, 10, 100, 1000, 10000}, but code is `bootstrap_ridge(nboots=1, chunklen=5, nchunks=2)` with `ALPHAS=np.logspace(0,3,6)`. The alpha selection is per-voxel and uses a held-out fold from training, so the core idea is correct — but the description is not faithful to the code. Worth flagging in feedback even if the deduction is small.
- **Mask-on-X interpretability (−2 on U, post-calibration):** Their LIME/SHAP perturb the processed feature matrix `X_story_z` by zero-ing chunk-aligned rows, rather than re-encoding perturbed text through BERT. Per the reference guide, the wrapper pattern is the canonical way; smaller dock than groups 02/09 because the SHAP/LIME engineering itself is textbook-clean.
- **Comparison figure coverage (−0.5+−0.5 on U):** Top-5 voxels are computed per (subject, story), but report figures only show 1 representative voxel per (subject, story) overlay. The Jaccard overlap table is computed in code but not surfaced in the report. Minor.
- All large files (model PKLs, embeddings) appear to be properly excluded; ~4GB repo size is Sequoia's concern, not Sean's.

### Group 15 (38.0/42)

1. **Ridge α grid hit the ceiling.** `lab3_2_ridge_bert.py` used RidgeCV with alphas up to 10^5, and 10^5 was selected uniformly for every BERT variant. The group flags this transparently in the conclusion (§5 p.18) but did not rerun with a wider grid. I docked half a point on the CV item. If Sean wants to be stricter (since the CV essentially never converged to an interior optimum) he could dock more. If Sean considers the transparent acknowledgement sufficient, +0.5 is defensible.
2. **3.3 cross-method comparison stays at the prose level.** No paired bar charts, no overlap metric, and the comparison text leans more on Triangle Shirtwaist than Sweetaspie. I docked 1 pt per story (2 pts total). If Sean considers the §4.4 discussion adequate, +1 to +2 is defensible.
3. **Triangle Shirtwaist per-voxel SHAP figure not embedded.** The figure exists in `figs/shap/plot4_signed_drivers_The_Triangle_Shirtwaist_Connection_·_S2.pdf` but was not put in the PDF. The Sweetaspie equivalent (Fig 9, p.12) is embedded. I factored this into the writing dock; could also be argued under U. Triangle Shirtwaist SHAP item.
4. **Subject coverage for 3.3.** Main analysis is Subject 2 only; Subject 3 LIME appears in the stability section but no Subject 3 SHAP. Lab spec doesn't strictly require both subjects so I did not dock for this in U (only noted as a limitation the group themselves flag).
5. **All three BERT variants implemented.** This group went beyond the lab requirement and trained both full MLM fine-tuning AND LoRA. Worth recognizing in qualitative feedback even though it doesn't add to the rubric score.
