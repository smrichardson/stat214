**Verdict:** Possibly over-credited (Δ ~2 pts on U)

**Specific issues:**

1. **SHAP also violates wrapper rule (U over-credit ~2 pts).** Calibration: "SHAP/LIME MUST perturb raw text via the wrapper pattern." Reference guide §"Wrapper Pattern" is explicit: explainers must operate on raw text so perturbations propagate through downsampling. `SHAP.py:9` uses `shap.LinearExplainer((w_v, 0.0), train_features, ...)` on the 3072-d tabular feature space — identical architectural flaw to the LIME tabular issue the grader penalized. Word attributions are reconstructed post-hoc via `word_tr_idx` mapping (`SHAP.py:20-22`), not via text perturbation. Grader gave SHAP 2/2 in both stories (4 pts total) while penalizing LIME for the same sin. Consistency requires deducting ~1 pt per SHAP (→ 4→2), bringing U from 8 to ~6.

2. **Verified correctly:** `fine_tune_bert.py` (lines 1-30) is a stub with `pass` bodies and an explicit comment that LoRA was abandoned for MLM; no `LoraConfig`/`peft` import. 0/2 LoRA is correct.

3. **Verified correctly:** `bert_pipeline.py:39-50` tokenizes each word in isolation with `max_length=16`, no context. 1/2 magnitude is defensible (generous-partial calibration); not under-credited.

4. **Verified:** Only 2 voxels (4031, 2858), Subject 2 only.
