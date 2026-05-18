**Verdict:** Mixed — final summary score consistent but per-item table NOT updated; small bookkeeping inconsistency.

**Specific issues:**
1. **Summary table (U=6/12) correctly applies previous audit's −2 on SHAP wrapper violation,** but the per-item table under §U still shows "SHAP 2/2" for both stories, summing to 8/12. The header note acknowledges the calibration revisions but doesn't reflect the SHAP dock. Cosmetic/bookkeeping issue only — final 31.5/42 is correct. Suggest updating per-item rows to "SHAP 1/2" each.
2. **Verified Q=9.5/12 correctly applies Rule 1** — full-MLM-FT-only earns Fine-tuned 4 + LoRA 2 (per Rule 1's symmetric reading per the section header note). Pre-trained −1 for word-by-word forward pass (`bert_pipeline.py:39-50`, max_length=16) is correct — not a calibration item, a real methodological bug. Description −1.5 reasonable.
3. **No above-and-beyond moderation triggered.** Stability section (delay window + seed × 5) is exemplary and already at 5/5; cannot soften other deductions since the BERT-extraction bug and SHAP wrapper violation are methodological errors, not gray-zone.
4. **Q header (7.5/12) is also inconsistent** with item sum (1+4+2+2+0.5=9.5) — header was not updated post-LoRA revision. Same cosmetic issue.

Final 31.5/42 stands; recommend syncing per-item tables.
