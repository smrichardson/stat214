**Verdict:** OK (42/42 stands)

**Specific issues:**
1. U revised from 11 to 12 per Rule 4 (top-3 voxels per story ≥ 2-voxel floor). Refund justified. Verified Figs 7/9/10 show rank-1/rank-2 voxels (top-3 in code, top-2 displayed in figures), which still exceeds Rule 4 minimum.
2. SHAP/LIME wrapper: closed-form linear SHAP `ϕ_j = w_j(x_j - E[x_j])` on the 3072-d feature space; LIME via `LimeTabularExplainer` with `predict_fn` that masks via zero-out of word-TR rows in the pre-built X. This is technically "feature-space" perturbation (Rule 3 gray zone). However, the LIME `predict_fn` does respect word→TR mapping (binary mask over unique word types → zero word-TR blocks → re-run ridge), and the closed-form SHAP is mathematically exact for the linear model. Combined with above-and-beyond depth (8-method comparison with Wilcoxon significance, three LoRA recipe variants, mid-layer ablation showing ρ=0.87/0.91 vs ρ=0.98 within recipe → layer-not-recipe-is-driver, BOLD-response-trace anchored LIME windows, four-level cross-subject stability), this earns "above-and-beyond moderation" — gray-zone wrapper softened. 12/12 defensible.
3. Page limit: 21 pages, but p.21 = LLM statement + 4-item references. Rule 5 compliant.
4. Rules 1, 2 cleanly satisfied.

42/42 internally consistent. No change recommended.
