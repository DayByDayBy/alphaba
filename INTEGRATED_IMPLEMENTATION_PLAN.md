# Alphabet Generation Pipeline: Integrated Implementation Plan

## Overview

This plan integrates Phase 1 (Alphabet-Level Representation) and Phase 2 (Generative Architecture) into a single, strictly sequential, git-traceable implementation.

Each step is designed for autonomous or semi-autonomous execution with:
- atomic scope (≤3 files, 1 conceptual change)
- explicit validation
- a required dev diary entry

---

## Global Invariants

The following rules apply to all phases and all steps:

- Each step introduces exactly ONE conceptual change.
- Each step touches ≤3 files (or creates 1 new file + tests).
- Every step must have concrete, checkable validation.
- Every step must append exactly one date-timed dev diary entry.
- Failures and blockers must be logged, never silently patched.
- Progression past a phase boundary requires a completed exit checklist.

---

## Phase 0: Pipeline Validation

### Step P0.01: Initialize Dev Diary

**Goal**: Create development tracking infrastructure.

**Files**:
- `.planning/dev_diary.md` (create)

**Implementation**:
- Create `.planning/` directory if it doesn't exist
- Create `.planning/dev_diary.md` with header:
  ```
  # Development Diary - Alphabet Generation Pipeline
  
  This diary tracks all implementation decisions, changes, and rationale.
  Each entry is automatically generated during implementation.
  
  ---
  ```
- Append the Step P0.01 diary entry below the header

**Validation**:
- [ ] File exists at `.planning/dev_diary.md`
- [ ] File is readable and markdown-formatted

**Commit Message**:
```
chore: initialize development diary
```

**Dev Diary Entry**:
```
### 2026-01-08 - Step P0.01: Initialize Dev Diary

**Changed**: Created `.planning/dev_diary.md` to track all implementation decisions and changes throughout Phase 1 and Phase 2.

**Why**: This diary provides traceability for architectural decisions and debugging context. Each step appends exactly one entry explaining what changed and why.

**Limitation**: None. This is pure infrastructure.
```

---

### Step P0.02: Review Phase Plans

**Goal**: Ensure understanding of Phase 1 + Phase 2 requirements before implementation.

**Files**: None (review only)

**Implementation**:
- Read `.planning/integration_prompt.md` completely
- Read `.planning/plan_phase1.md` completely
- Read `.planning/plan_phase2.md` completely
- List all files to be created/modified in Phase 1
- List all files to be created/modified in Phase 2
- Record the Phase 2 output representation decision in .planning/architecture_decisions.md (create if missing), including rationale and rejected alternatives.

**Validation**:
- [ ] Can explain: Why alphabet-relative normalization matters
- [ ] Can explain: How DeepSets enable permutation invariance
- [ ] Can explain: Why Phase 2 should start with point clouds
- [ ] List created: All files to be touched in Phase 1
- [ ] List created: All files to be touched in Phase 2

**Commit Message**: N/A (no code changes)

**Dev Diary Entry**:
```
### 2026-01-08 - Step P0.02: Phase Plan Review

**Changed**: None (review step).

**Reviewed**:
- `.planning/plan_phase1.md` (pipeline fixes, alphabet-relative normalization, tensor output, data loader)
- `.planning/plan_phase2.md` (encoder/decoder, training loop, constraints, generation)

**Why**: This prevents accidental scope creep and ensures all steps can be executed with the required ≤3-files/1-concept granularity.

**Limitation**: None identified.
```

---

### Step P0.03: Instantiate Phase 0 Exit Checklist

**Goal**: Record Phase 0 checklist results (PASS/FAIL/N/A) before entering Phase 1.

**Files**:
- `.planning/phase0_exit_checklist.md` (create)

**Implementation**:
- Copy the contents of `.planning/phase_00_checklist.md` into `.planning/phase0_exit_checklist.md`
- For every checkbox item, replace `[ ]` with one of:
  - `[PASS]`
  - `[FAIL]`
  - `[N/A]`
- For any item marked `[PASS]`, add an evidence line referencing:
  - a file path under `output/<font_name>/...`, or
  - a Phase 0 command run, or
  - an inspection step ID
- Set `PHASE STATUS` to `PASS` only if all “Disallowed States (Hard Stop)” items are `[PASS]`

**Validation**:
- [ ] File exists at `.planning/phase0_exit_checklist.md`
- [ ] Every checklist item is marked `[PASS]`, `[FAIL]`, or `[N/A]`
- [ ] If any “Disallowed States” item is `[FAIL]`, execution stops (do not proceed to Phase 1)

**Commit Message**:
```
chore: instantiate phase 0 exit checklist
```

**Dev Diary Entry**:
```
### 2026-01-08 - Step P0.03: Phase 0 Exit Checklist

**Changed**: Created `.planning/phase0_exit_checklist.md` and recorded PASS/FAIL/N/A for Phase 0 invariants and artifacts.

**Why**: Phase 1 modifies pipeline behavior. Capturing Phase 0 status first provides a baseline and prevents regressions from being mistaken for Phase 0 failures.

**Limitation**: This checklist reflects only the fonts inspected during Phase 0; additional fonts may reveal new edge cases.
```

---

## Phase 0 Complete: Validation Checkpoint

> ⚠️ **Phase Boundary Warning**
>
> Changes made in Phase 1 may invalidate assumptions verified in Phase 0.
> If Phase 1 modifies glyph parsing, normalization semantics, or pipeline
> invariants, the Phase 0 exit checklist MUST be re-instantiated and reviewed.


At this point:
- ✅ Dev diary initialized
- ✅ Phase 1 and Phase 2 plans reviewed
- ✅ Phase 0 exit checklist instantiated

**PAUSE HERE**: Review the plans. Ensure you understand:
- The qCurveTo bug and its fix
- Why alphabet-relative normalization matters
- How DeepSets enable permutation invariance
- The difference between reconstruction and generation

**Continue only when**: You are confident in the approach.

---

## Phase 1: Alphabet-Level Representation

### Step P1.01: Add Missing Pipeline Dependencies

**Goal**: Ensure the geometry pipeline and Phase 1 validations run in a clean `uv` environment.

**Files**:
- `pyproject.toml` (modify)
- `uv.lock` (modify)

**Implementation**:
- Run: `uv add fonttools svgpathtools cairosvg pillow scikit-image scipy pytest`

**Validation**:
- [ ] `uv run python -c "from fontTools.ttLib import TTFont; import svgpathtools, cairosvg; from PIL import Image; from skimage.morphology import skeletonize; from scipy import ndimage; import pytest"` exits 0

**Commit Message**:
```
chore: add pipeline and test dependencies
```

**Dev Diary Entry**:
```
### 2026-01-08 - Step P1.01: Add Missing Pipeline Dependencies

**Changed**: Added required dependencies for the geometry pipeline and pytest-based validation.

**Why**: The pipeline imports these libraries at runtime; declaring them in `pyproject.toml` makes execution deterministic under `uv`.

**Limitation**: None.
```

---

(TODO: Steps P1.02+)

---

## Phase 2: Generative Architecture

(TODO: Steps P2.01+)

---

## Review Checkpoints

(TODO: Add explicit checkpoint summaries and phase-exit checklist steps for Phase 0/1/2)
