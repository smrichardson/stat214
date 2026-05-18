**Verdict:** OK, score holds at 42.

**Verifications performed:**
1. **Full FT + LoRA both implemented.** `bert_finetune_pipeline.py:208-237` `train_full_finetune` instantiates `BertForMaskedLM` and updates all 109.8M params; `:240-290` `train_lora_finetune` uses `LoraConfig(r=16, alpha=32, target_modules=["query","value"])` via PEFT, then `merge_and_unload()` (line 284). Both have per-epoch losses reported in Table 3 (p.9).
2. **SHAP/LIME via text-perturbation wrapper.** `interpret_utils.py:338-385` `run_shap` uses `shap.PartitionExplainer` with `shap.maskers.Text(r"\s")`; `run_lime` uses `LimeTextExplainer(bow=False, split_expression=r"\s+")`. The wrapper at `:280-330` re-encodes the perturbed word window through BERT -> Lanczos -> `make_delayed` -> ridge before returning the scalar prediction. This is the correct text-perturbation pattern.
3. **5 voxels x 2 stories x SHAP+LIME confirmed.** Tables 6, 7 (p.14) list 5 voxels each for `gangstersandcookies` and `fromboyhoodtofatherhood`; heatmaps Figs 8, 10 show all 5 voxels x both methods.
4. **Stability check substantive.** SHAP `max_evals` x3, LIME `num_samples` x3 plus seed x3, with J@5/J@10/Spearman/cosine (Tables 9, 10 p.18) and an interpretive "what we can claim" filter (p.19) that gates downstream claims.
5. **Page limit:** 22 pages total; body ends p.20, references p.21, honesty/LLM statement p.22. Within spirit.
6. **LLM statement:** Present and detailed (§11.2 p.22).
7. **Code/report consistencies:** alpha-grid mismatch (report says {1,10,100,1000,10000}; `train.py` default {1,10,100}) already flagged by grader as non-credit-affecting, and best alpha=100 makes the wider grid irrelevant.

No over-credit found. Calibration rules correctly applied.
