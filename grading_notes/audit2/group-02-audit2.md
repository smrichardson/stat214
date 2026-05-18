**Verdict:** OK (internally consistent post-revision)

**Specific issues:**
1. **U=4/12 correctly reflects Rule 3 + Rule 4.** Verified `influence.py:18` uses `LimeTabularExplainer` (not text), and `:46` uses `shap.LinearExplainer` on 3072-d feature space with `shap.maskers.Independent` — both violate raw-text wrapper rule. Single voxel (Rule 4 <2 = scope deficiency). Previous audit's −1 on Comparison and +0.5 on SHAP roughly cancel — final 4/12 is defensible.
2. **No above-and-beyond moderation triggered.** Group has cross-method dim_308 agreement and pre-trained vs fine-tuned SHAP comparison, but these don't constitute "substantially above spec elsewhere" given the substantive methodological violation on the headline interpretability item. Moderation rule explicitly does not apply to methodological errors.
3. **W=4/5 is fair** — multi-angle stability but no end-to-end perturbation of the headline interpretation result. Under Rule 2, depth-on-one-judgment-call should earn 5/5; here the analysis is broad-but-shallow rather than deep-on-one-call, so −1 is justified.
4. **Y=4/5 fair** — figure issues, typos, no LLM disclosure already docked. LLM-disclosure absence flagged but not double-deducted.

Final 31/42 stands.
