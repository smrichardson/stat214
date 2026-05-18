**Verdict:** OK with one minor flag — 42 stands (no deduction recommended).

**Specific issues:**
1. **SHAP/LIME wrapper deviates from spec's "re-encode through BERT per perturbation" pattern.** `interpret_shap_lime.py:128-167` encodes per-word BERT embeddings ONCE upfront with ±10 context window, then `WordMaskPredictor.__call__` (lines 208-235) masks at the *embedding level* by swapping in the story-mean embedding, not by re-running BERT on perturbed text. The Reference Guide §"Wrapper Pattern" (line 576, 610) explicitly says "every perturbation requires a full forward pass through BERT + the preprocessing pipeline." Group 06 only re-runs the *downstream* pipeline (sincmat → delays → ridge → CC) per perturbation. This is a real but defensible deviation: BERT context is fixed, contextual word embeddings are preserved, and downsample/delay/ridge propagation is correct. Given the calibration "generous on partial credit" and that the wrapper correctly handles word-level attribution through the blending step, this is judgment-call territory — not a clear deduction. Worth noting in case of cross-group consistency challenge.

**Verified clean:**
- LoRA via PEFT confirmed (`train_bert_lora.py:22, 301-309`), real MLM fine-tune 3 epochs × 2×2 grid.
- 5 voxels × 2 stories × 2 subjects × SHAP+LIME all present.
- Five stability probes all substantive.
- 16 pages, LLM statement present p.16.
