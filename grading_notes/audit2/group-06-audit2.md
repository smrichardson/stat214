**Verdict:** OK (42/42 stands)

**Specific issues:**
1. SHAP/LIME wrapper: Group 06 uses a forward function that replaces masked-out word *embeddings* with the per-story embedding mean rather than re-tokenizing/re-encoding through BERT per perturbation (`interpret_shap_lime.py:128-167`, prior audit flag). Under Rule 3 a strict reading is "perturbing already-encoded features = substantive deduction." However, this group's wrapper is at the *word level* (not feature/3072-d level) — it preserves the rest of the BERT-derived embedding matrix and reruns Lanczos/delay/ridge/CC per word-mask, which is closer to the wrapper spirit than the X-row masking that Rule 3 explicitly targets. Combined with five distinct stability checks, full BERT-layer ablation, exhaustive 8-method × 2-subject × 7-metric comparison, and 5 voxels × 2 stories × 2 subjects coverage, this group is a textbook "above-and-beyond moderation" candidate — gray-zone wrapper deviation softened. No deduction.
2. Page count 16, LLM statement present, 5/5 stability passes Rule 2. LoRA-only path satisfies Rule 1. ≥2 voxels satisfies Rule 4. Page limit Rule 5 satisfied (16 << 20).

42/42 internally consistent. No change recommended.
