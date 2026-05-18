# Lab 3 Grading Rubric — Spring 2026

**Status:** v2 (post-grading clarifications). The clarifications in this document resolve specific tensions between the original rubric and the lab spec / reference guide that were uncovered during grading. They reflect what was actually applied to Spring 2026 submissions.

**Sources of truth:**
- Lab spec (assigned to students): `stat-214-gsi/lab3/instructions/main.tex` (`214_lab_3.pdf`).
- Reference guide (uploaded to students): `Lab3_Reference_Guide.html`.
- This rubric (used by graders, NOT distributed to students).

When the rubric and the lab spec or reference guide are in tension, **the version the students saw wins** — penalizing students for not matching wording they couldn't see is unfair.

---

## Total: 75 points across 4 categories

1. Code style, structure, reproducibility, and data processing — **12 pts** (Sequoia)
2. Lab 3.1 embeddings and modeling (BoW, Word2Vec, GloVe) — **21 pts** (Sequoia)
3. Lab 3.2 embeddings and modeling (BERT pretrained + fine tuned) — **20 pts** (Sean)
4. Lab 3.3 interpretability and report — **22 pts** (Sean)

---

## 1. Code style, structure, reproducibility, and data processing (12 pts)

### Skeleton/Structure (5 pts)
One point each for:
- `run.sh`
- `environment.yaml`
- `lab3.tex` or `lab3.ipynb`
- `lab3.pdf`
- Note about where pretrained models are stored (HuggingFace, Bridges2, SCF)

Subtract one point (cannot go below 0) for any of:
- Report goes over the page limit (20 pages)
- Code is included in the report
- Data files are committed to the Git repository
- Not clear where pretrained models are stored

### Code Style (3 pts)
- Sufficient comments (1)
- Sufficient docstrings (1)
- Consistent style used (1)

### Reproducibility (2 pts)
- No hard-coded paths outside the `lab3` folder (1)
- Clear instructions on how to reproduce results (1)

### Data Preprocessing (2 pts)
- Trimming (1)
- `make_delayed` (1)

---

## 2. Lab 3.1 embeddings and modeling (BoW, Word2Vec, GloVe) — 21 pts

### Generate Embeddings (9 pts)
- Bag-of-Words (2)
- Word2Vec (2)
- GloVe (2)
- Description of each model (3)

### Modeling and Evaluation (12 pts)
- Ridge Regression (3)
- Cross-validation analysis (3)
- Detailed analysis of best performing model (3)
- Detailed analysis of performance over voxels (3)

---

## 3. Lab 3.2 embeddings and modeling (BERT pretrained + fine tuned) — 20 pts

### 3.2 BERT (12 pts)
- **Pre-trained (2)** — Loads `bert-base-uncased` (or equivalent), extracts contextual embeddings from a sensible layer with proper subtoken aggregation and sliding/overlapping windows for long stories.
- **Fine-tuned (4)** — Fine-tunes the pretrained BERT on the lab's text corpus, *or* uses LoRA as the fine-tuning method (see Clarification below). Re-encodes both training and held-out stories with the adapted model.
- **LoRA (2)** — Implements parameter-efficient fine-tuning via LoRA (PEFT library or equivalent), with adapters on attention sub-modules (typically Q, V projections).
- **Performance comparison (2)** — Quantitative comparison of pretrained vs fine-tuned BERT (and ideally vs the 3.1 embeddings).
- **Description of each model (2)** — Clear narrative description of what each model is and how it's set up.

> **Clarification (Spring 2026):** The lab spec presents LoRA as the canonical example of fine-tuning ("Investigate a parameter-efficient approach to fine-tuning... such as LoRA"). Students were not given this rubric. Therefore:
> - **LoRA-only counts for both items:** a group that implemented a thoughtful LoRA pipeline and produced LoRA-based embeddings earns **6/6** on Fine-tuned (4) + LoRA (2), even if no separate full-parameter fine-tune exists.
> - **Full-FT-only also counts for both items (6/6):** since full-parameter fine-tuning is at least as demanding as LoRA, a group that did a thoughtful full-MLM fine-tune in lieu of LoRA earns the full 6/6. The LoRA item is satisfied by any thoughtful fine-tuning effort.
> - **Both implemented separately** is the gold standard but is not required for full credit.

### 3.2 Modeling and Evaluation for BERT (8 pts)
- Ridge Regression (2)
- Cross-validation analysis (2) — including reported mean/median/top-1%/top-5% CC.
- Detailed analysis of best performing model (2)
- Detailed analysis of performance over voxels (2) — distribution plot + PCS interpretation criterion.

---

## 4. Lab 3.3 interpretability and report — 22 pts

### 3.3 Interpretability (12 pts)
Two stories analyzed, each scored out of 6 pts:
- SHAP (2)
- LIME (2)
- Comparison of SHAP vs LIME (2)

> **Clarification (Spring 2026):** SHAP and LIME must perturb **raw text** via the wrapper pattern documented in the reference guide (`Lab3_Reference_Guide.html` §Interpretation → "The Wrapper Pattern"). The pattern is:
> 1. Wrap the entire pipeline as a callable `f(text_list) → predictions`.
> 2. Inside the wrapper: re-embed perturbed text via BERT → downsample (`downsample_word_vectors`) → stack delays (`make_delayed`) → multiply by ridge weights.
> 3. SHAP/LIME perturb the *input text*, and the effect propagates naturally through Lanczos blending and BERT's contextual mechanism.
>
> Methods that perturb already-encoded features (e.g., `shap.LinearExplainer` or `LimeTabularExplainer` operating on the 3072-d design matrix; zero-masking word-aligned rows of `X_story_z`) bypass BERT's contextual sensitivity — removing word X cannot change the contextual embedding of word Y when re-encoding is skipped. This is a substantive methodological error.
>
> **Deduction scale on U for skipping the wrapper pattern:**
> - Heavy method violation + scope shortcomings (e.g., 1 voxel, 1 story, embedding-feature SHAP): docked into the 3–5/12 range.
> - Hand-rolled feature-perturbation + sub-spec voxel count: ~−4 on U.
> - Clean engineering (real `shap.KernelExplainer` / textbook LIME) but on processed-X rows rather than text: ~−2 on U.
> - **Consistency rule:** if a group's SHAP and LIME share the same wrong input space, they must be docked equivalently on both line items. Don't ding LIME and let SHAP off the hook (or vice versa).
>
> **Voxel count expectation:** the lab spec uses the vague phrase "top voxels" without pinning a number. **At least 2 voxels per story is sufficient for full credit.** Any voxel count ≥2 satisfies the rubric. Fewer than 2 voxels per story (i.e., a single voxel or a global aggregate) is a scope deficiency worth ~−1 to −2 on U. Showing more than 2 (5 voxels × 3 TRs is the gold standard set by group 01) does not earn bonus credit but signals depth.

### Stability Check (5 pts)
> **Original rubric wording:** "Relevance of stability check (0–5) 1 pt for each model."
>
> **Clarification (Spring 2026 — overrides the original wording):** the lab spec asks for stability of *one of your choices or judgement calls* — depth on one judgment call, not breadth across all 5 models. Score 0–5 as:
> - **5/5** — Thoughtful stability check on a real judgment call (window size, α grid, train/test split, LoRA rank, delay stack, voxel selection threshold, etc.) with end-to-end pipeline reruns and a clean PCS-style conclusion separating what's stable from what's not.
> - **4/5** — Solid stability check, but somewhat narrow (e.g., only summary stats, no downstream interpretation rerun) or limited in scope.
> - **3/5** — Stability section present but shallow (one before/after comparison, no quantitative framing).
> - **0–2/5** — Stability check missing, off-topic, or doesn't probe a real judgment call.
>
> The original "1 pt per model" wording over-rewards groups who run a perfunctory check on all 5 embeddings and under-rewards groups who go deep on one. This calibration aligns the rubric with what the lab spec actually asked.

### Writing (5 pts)
- Report readability and structure (0–3)
- Overall quality of figures (0–2)

> **Notes (Spring 2026):**
> - Page limit is 20 pages. Reports that fit content body within 20 but have references/honesty/LLM-usage on pages 21–22 are treated as compliant (no deduction); a body-text overage triggers the Skeleton/Structure deduction in Category 1.
> - LLM-usage statement required in `collaboration.txt` or the report. Missing statement is a Category 1 issue, not a Category 4 issue.

---

## Cross-cutting calibration principles (Spring 2026)

These three principles apply throughout grading:

1. **The lab spec and reference guide win over the rubric** when they conflict, because students were never shown the rubric. Penalizing students for not matching wording they couldn't see is unfair.

2. **Be generous on partial credit.** If a group attempted a thing and it's basically there (even with rough edges), give most of the points and note the issue. Typical deductions: 0.5–1.5 per category for attempted-but-rough work.

3. **Cross-group consistency on the same shortcoming.** If feature-perturbation costs group N a point on SHAP, it must cost group M the same point on the same SHAP item. The single biggest grading risk is the same shortcoming being penalized at different magnitudes across groups. The grading_notes/COLUMN_AUDIT.md (cross-group view) is the tool for checking this.

---

## Changelog

- **v2 (2026-05-16):** Added Clarifications on Fine-tuned/LoRA, Stability Check, and SHAP/LIME wrapper pattern after Sean's calibration pass. Added cross-cutting principles section.
- **v1 (original):** PDF version distributed to grading team.
