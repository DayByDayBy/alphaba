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

---

### 2026-01-09 - Step P0.03: Phase 0 Exit Checklist

**Changed**: Created `.planning/phase0_exit_checklist.md` and recorded PASS/FAIL/N/A for Phase 0 invariants and artifacts.

**Why**: Phase 1 modifies pipeline behavior. Capturing Phase 0 status first provides a baseline and prevents regressions from being mistaken for Phase 0 failures.

**Key Findings**:
- Pipeline produces valid artifacts for 28/52 glyphs (54% coverage)
- qCurveTo bug causes remaining failures (planned fix: P1.02–P1.04)
- All "Disallowed States" items PASS
- Phase 0 status: **PASS** — architecture sound, parsing needs fix

**Limitation**: This checklist reflects only GoogleSans font; additional fonts may reveal new edge cases.

---

### 2026-01-09 - Step P1.01: Add Missing Pipeline Dependencies

**Changed**: Verified required dependencies in `pyproject.toml` — fonttools, svgpathtools, cairosvg, pillow, scikit-image, scipy, pytest.

**Why**: The pipeline imports these libraries at runtime; declaring them in `pyproject.toml` makes execution deterministic under `uv`.

**Limitation**: None. All dependencies already present and validated.

---

### 2026-01-09 - Step P1.02: Resolve Glyph Characters via cmap

**Changed**: Updated `src/alphabet_pipeline.py:glyph_to_path()` to resolve Unicode characters through the font cmap before extracting outlines.

**Why**: Coverage validation and corner-case glyph checks use characters (e.g. `@`) that often do not match font-internal glyph names.

**Limitation**: Fonts with incomplete cmap tables may still be missing expected characters.

---

### 2026-01-09 - Step P1.03: Extract Pen→SVG Path Conversion Helper

**Changed**: Refactored `src/alphabet_pipeline.py:glyph_to_path()` by extracting `RecordingPen` conversion into `pen_value_to_svg_path_string()` helper.

**Why**: The qCurveTo fix needs clean, localized logic that can be unit-tested without reworking font loading.

**Limitation**: None identified (no intended behavior change).

---

### 2026-01-09 - Step P1.04: Fix TrueType qCurveTo Handling

**Changed**: Updated `src/alphabet_pipeline.py:pen_value_to_svg_path_string()` to expand TrueType multi-point `qCurveTo` sequences into standard SVG quadratic curves with implicit on-curve midpoints.

**Why**: TrueType fonts use a compact format where consecutive off-curve points have an implied on-curve midpoint between them. The previous implementation failed to expand these, causing parse errors on most glyphs.

**Validated**: `g`, `Q`, `y`, `S`, `@`, `&` now all parse successfully (previously failed).

**Limitation**: Extremely unusual contour structures (e.g. entirely off-curve) are untested.

---

### 2026-01-09 - Step P1.05: Add Font Coverage Validation Helper

**Changed**: Added `src/alphabet_pipeline.py:validate_font_coverage()` to check character coverage before processing.

**Why**: Prevents silent failures by identifying missing characters up front.

**Limitation**: None identified.
