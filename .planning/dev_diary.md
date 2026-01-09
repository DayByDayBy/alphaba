# Development Diary - Alphabet Generation Pipeline

This diary tracks all implementation decisions, changes, and rationale.
Each entry is automatically generated during implementation.

---

### 2026-01-09 - Step P0.01: Initialize Dev Diary

**Changed**: Created `.planning/dev_diary.md` to track all implementation decisions and changes throughout Phase 1 and Phase 2.

**Why**: This diary provides traceability for architectural decisions and debugging context. Each step appends exactly one entry explaining what changed and why.

**Limitation**: None. This is pure infrastructure.

---

### 2026-01-09 - Step P0.02: Phase Plan Review

**Changed**: None (review step). Created `.planning/architecture_decisions.md` to record Phase 2 output representation decision.

**Reviewed**:
- `.planning/plan_phase1.md` (pipeline fixes, alphabet-relative normalization, tensor output, data loader)
- `.planning/plan_phase2.md` (encoder/decoder, training loop, constraints, generation)

**Key Understanding**:
- **Alphabet-relative normalization**: Preserves inter-glyph scale relationships (M vs i) critical for alphabet-level style learning
- **DeepSets**: Enable permutation invariance via sum/mean pooling over per-element transforms (φ per glyph → aggregate → ρ)
- **Point clouds for Phase 2**: Match pipeline output, fixed dimension, differentiable; strokes deferred to Phase 3

**Files to touch in Phase 1**: `src/alphabet_pipeline.py`, `src/alphabet_data_loader.py`, `tests/test_alphabet_pipeline.py`, `notebooks/00_pipeline_validation.ipynb`

**Files to touch in Phase 2**: `src/alphabet_encoder.py`, `src/glyph_decoder.py`, `src/alphabet_vae.py`, `src/generative_training.py`, `src/generative_eval.py`, notebooks

**Limitation**: None identified.
