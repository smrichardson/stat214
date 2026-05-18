**Verdict:** OK (U=7 stands; total 36/42)

**Specific issues:**
1. Rule 3 (wrapper-pattern violation) correctly applied. Code (`07_interpret_bert.py:234-241`, `:263-283`, `:286-315`) hand-rolls KernelSHAP-style and LIME-style perturbations by zeroing word-TR rows in the *already-computed* design matrix X. No re-tokenization/re-BERT-encoding per perturbation. Per Rule 3 deduction scale: "Hand-rolled feature-perturbation + sub-spec voxel count: ~−4 on U." Applied: 12 - 4 = 8 (method) − but report frames as "SHAP-style"/"LIME-style," then refunded by +2 for voxel-count compliance under Rule 4 (2 voxels ≥ floor), ending at 7. Consistent with Rule 3's deduction-scale tier 2.
2. Rule 4 refund (+2) correctly applied: 2 voxels per story meets the minimum. Verified Table 6 p.13: 2 voxels × 2 stories.
3. Subject 2 only (not both subjects) is implicitly accepted under the calibration-generous principle; lab spec doesn't require both. Not double-counted.
4. Above-and-beyond moderation: This group is NOT a candidate — modest depth elsewhere; the wrapper-pattern violation is a clear methodological error (not gray-zone), so moderation does not apply per the new rule. No softening warranted.
5. Rules 1, 2, 5 cleanly satisfied (LoRA via PEFT, single judgment-call stability with PCS framing, 19 pages).

36/42 internally consistent. No change recommended.
