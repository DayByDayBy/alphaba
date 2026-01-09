# Execution Prompt: Alphabet Generation Pipeline - Phases 1 & 2

## Your Task

You are an implementation agent working in a git repository. Your goal is to integrate two planning documents (`plan_phase1.md` and `plan_phase2.md`) into a single, executable, step-by-step implementation plan.

This plan will be used for autonomous or semi-autonomous execution by an LLM (possibly yourself, possibly another agent). Every step must be concrete, testable, and traceable.

---

## Input Documents

You have been provided:
- `plan_phase1.md`: Alphabet-Level Representation (pipeline fixes, normalization, tensor output, data loader)
- `plan_phase2.md`: Generative Architecture (encoder, decoder, VAE, training)

---


## Refusal Conditions

If any of the following are true, you MUST refuse to produce an output and explain why:

- The input phase plans contradict each other
- A required artifact or file is missing
- A step cannot be made concrete or testable
- The requested granularity would violate the ≤3 file / 1 concept rule


## Output Requirements

### 1. STRUCTURE

Produce a **single markdown document** with the following structure:

```
# Alphabet Generation Pipeline: Integrated Implementation Plan

## Phase 0: Pipeline Validation
[Review/validation steps only, no new implementation]

## Phase 1: Alphabet-Level Representation
[Steps P1.01 through P1.XX]

## Phase 2: Generative Architecture
[Steps P2.01 through P2.XX]

## Review Checkpoints
[Explicit pause/review steps]
```

The plan must be **strictly sequential** - each step builds on the previous.

### 2. STEP SPECIFICATION FORMAT

Each step must follow this exact template:

```markdown
### Step P1.03: [Descriptive Title]

**Goal**: [1-2 sentence outcome statement]

**Files**:
- `src/alphabet_pipeline.py` (modify lines 91-94)
- `tests/test_qcurve.py` (create)

**Implementation**:
- [Bullet point: what to add/change]
- [Bullet point: what to add/change]
- [No "consider" or "think about" - only concrete actions]

**Validation**:
- [ ] Test passes: `pytest tests/test_qcurve.py`
- [ ] No errors on corner-case glyphs: g, Q, y, S, @, &
- [ ] Visual inspection: glyphs render correctly in output/test/rasters/

**Commit Message**:

fix: handle TrueType implicit on-curve points in qCurveTo
```

**Dev Diary Entry**:


**Dev Diary Entry Requirements**:
- Must reference at least one concrete artifact (file, function, metric, or output)
- Must state one explicit reason the change was necessary
- Must state at least one limitation, risk, or open question (or explicitly state “None identified”)


```
### 2025-01-09 - Step P1.03: qCurveTo Handling

**Changed**: Modified `glyph_to_path()` to correctly parse TrueType quadratic 
curves with implicit on-curve points. Previously failed on glyphs like 'g' and 
'S' with complex curves.

**Why**: TrueType fonts store multiple control points with implicit midpoints. 
The original implementation only handled single control point case. This fix 
ensures all glyphs parse correctly before normalization.

**Limitation**: Assumes all fonts follow TrueType spec. OpenType CFF fonts may 
need separate handling in future.
```


### 3. GRANULARITY REQUIREMENTS

**Each step must**:
- Touch ≤3 files (or create 1 new file + tests)
- Introduce exactly ONE conceptual change
- Be small enough for a single focused commit
- Be independently testable

**Examples of CORRECT granularity**:
- ✅ "Add qCurveTo fix" (one bug, one file section)
- ✅ "Create AlphabetDataLoader class skeleton" (one new abstraction)
- ✅ "Add font metrics extraction" (one function, one purpose)

**Examples of INCORRECT granularity**:
- ❌ "Fix all pipeline bugs" (too broad)
- ❌ "Implement encoder and decoder" (two separate concerns)
- ❌ "Add normalization and statistics" (two phases mixed)

### 4. DEV DIARY INTEGRATION

**First step must be**:
```markdown
### Step P0.01: Initialize Dev Diary

**Goal**: Create development tracking infrastructure.

**Files**:
- `.planning/dev_diary.md` (create)

**Implementation**:
- Create `.planning/` directory if it doesn't exist
- Create `dev_diary.md` with header:
  ```
  # Development Diary - Alphabet Generation Pipeline
  
  This diary tracks all implementation decisions, changes, and rationale.
  Each entry is automatically generated during implementation.
  
  ---
  ```

**Validation**:
- [ ] File exists at `.planning/dev_diary.md`
- [ ] File is readable and markdown-formatted

**Commit Message**:
```
chore: initialize development diary
```

**Dev Diary Entry**:
```
### 2025-01-09 - Step P0.01: Initialize Dev Diary

**Changed**: Created `.planning/dev_diary.md` to track all implementation 
decisions and changes throughout Phase 1 and Phase 2.

**Why**: This diary provides traceability for architectural decisions and 
debugging context. Each step appends one entry explaining what changed and why.

**Limitation**: None. This is pure infrastructure.
```
```

**Every subsequent step** must append its diary entry to this file.

### 5. PHASE BOUNDARIES

The plan must preserve these explicit boundaries:

```markdown
## Phase 0 Complete: Validation Checkpoint

At this point:
- ✅ Dev diary initialized
- ✅ Phase 1 and Phase 2 plans reviewed

**PAUSE HERE**: Review the plans. Ensure you understand:
- The qCurveTo bug and its fix
- Why alphabet-relative normalization matters
- How DeepSets work for set encoding
- The difference between reconstruction and generation

**Continue only when**: You are confident in the approach.

---

## Phase 1: Alphabet-Level Representation

[Steps P1.01 through P1.XX]

---

## Phase 1 Complete: Inspection Checkpoint

At this point:
- ✅ Pipeline runs without errors on 10+ fonts
- ✅ Alphabet tensors loadable
- ✅ Data loader functional
- ✅ Visual validation performed

**PAUSE HERE**: Manually inspect outputs before Phase 2:
1. Open `output/[font_name]/rasters/` - do glyphs look correct?
2. Load `alphabet_samples.npy` - is shape (52, 256, 2)?
3. Run data loader - do triplets make sense?

**Continue only when**: Phase 1 outputs are validated.

---

## Phase 2: Generative Architecture

[Steps P2.01 through P2.XX]

---

## Phase 2 Complete: Final Review

[Validation steps]
```

### 6. FAILURE & UNCERTAINTY HANDLING

When a step involves uncertain behavior, add an **explicit inspection step** immediately after:

```markdown
### Step P1.15: Skeletonize All Fonts

[...implementation...]

---

### Step P1.16: Inspect Skeletonization Outputs

**Goal**: Manually verify skeleton quality before proceeding.

**Files**: None (inspection only)

**Implementation**:
- Open `output/*/skeletons/*.png` for 3-5 fonts
- Check for:
  - Single-pixel medial axes (not thick lines)
  - Connected components where expected
  - No fragmentation in serif terminals
- Review `metadata.json` for any fonts with >10% failed glyphs

**Validation**:
- [ ] Skeletons visually correct for at least 80% of glyphs
- [ ] Failed glyphs documented in dev diary
- [ ] Decision made: continue or adjust skeletonization params

**Commit Message**: N/A (no code changes)

**Dev Diary Entry**:
```
### 2025-01-09 - Step P1.16: Skeleton Inspection

**Changed**: None (inspection step).

**Observed**: [Agent fills this in after inspection]
- Font X: all skeletons good
- Font Y: serif glyphs show slight fragmentation but acceptable
- Font Z: 'i' and 'l' skeletons too sparse (logged as known issue)

**Decision**: Proceeding with current skeletonization params. Fragmentation 
in ultra-thin glyphs is acceptable for Phase 1. Will revisit if Phase 2 
topology metrics show issues.
```
```

### 7. VALIDATION CRITERIA

Each step's validation must be **concrete and checkable**:

✅ **Good validation**:
- `pytest tests/test_normalization.py` exits 0
- File `alphabet_samples.npy` has shape (52, 256, 2)
- Running `python -c "from src.data_loader import AlphabetDataLoader; ..."` prints "Loaded 15 alphabets"
- Visual: Opening `output/Helvetica/rasters/A.png` shows filled letter A

❌ **Bad validation**:
- "Code looks correct" (not checkable)
- "Should work" (not tested)
- "Verify normalization is good" (too vague)

### 8. GIT COMMIT MESSAGES

Follow conventional commits format:

```
<type>: <description>

[optional body]
```

**Types**:
- `feat`: New feature (new function, class, capability)
- `fix`: Bug fix
- `refactor`: Code restructure without behavior change
- `test`: Add or modify tests
- `docs`: Documentation only
- `chore`: Build, dependencies, tooling

**Examples**:
- `feat: add alphabet-relative normalization`
- `fix: handle TrueType implicit on-curve points`
- `test: add corner-case glyph coverage`
- `refactor: extract font metrics to separate function`

### 9. NON-GOALS (MUST NOT DO)

❌ **Do NOT**:
- Invent architecture not in the phase plans
- Merge steps for "efficiency" (keep granular)
- Assume ML training will succeed (include validation)
- Skip visualization/inspection steps
- Add "bonus features" beyond the plans
- Use phrases like "consider adding" or "might want to" (be concrete)

✅ **DO**:
- Follow the plans exactly
- Break large steps into smaller ones if needed
- Add inspection steps where uncertainty exists
- Make every step independently executable

---

## Execution Instructions

1. **Read both phase plans completely**
2. **Identify all distinct implementation tasks** from both plans
3. **Order them sequentially** (Phase 0 → Phase 1 → Phase 2)
4. **Break each task into commit-sized steps** (following template above)
5. **Add inspection/pause steps** after uncertain operations
6. **Number steps monotonically** (P0.01, P0.02, P1.01, P1.02, ...)
7. **Ensure every step appends to dev diary**


### Phase Exit Enforcement

Before declaring any phase complete, you MUST:

1. Instantiate the corresponding checklist file:
   - Phase 0 → `.planning/phase0_exit_checklist.md`
   - Phase 1 → `.planning/phase1_exit_checklist.md`
   - Phase 2 → `.planning/phase2_exit_checklist.md`

2. Explicitly mark every item as PASS / FAIL / N/A.

3. Append a summary of the checklist outcome to `dev_diary.md`.

4. If ANY item in “Disallowed States” is FAIL:
   - Do NOT proceed
   - Add a blocking diary entry
   - Stop execution

5. Checklist items marked PASS must reference:
    - a test run,
    - a file path,
    - or an inspection step ID.

---

## Output

Produce a single markdown file named `INTEGRATED_IMPLEMENTATION_PLAN.md` containing:

1. Header section with plan overview
2. Phase 0 (validation/setup)
3. Phase 1 (all steps from plan_phase1.md)
4. Phase 2 (all steps from plan_phase2.md)
5. Review checkpoints at each phase boundary
6. Final validation section

The document should be **directly executable** by an LLM agent working in the repository.

---

## Quality Check

Before finalizing, verify:
- [ ] Every step follows the exact template
- [ ] Step IDs are monotonic and gap-free
- [ ] No step is too large (>3 files or >1 concept)
- [ ] Every step has concrete validation criteria
- [ ] Dev diary entries explain "what" AND "why"
- [ ] Phase boundaries are explicit with pause instructions
- [ ] No invented architecture beyond the plans
- [ ] Inspection steps exist after uncertain operations

---

## Example of First Few Steps

```markdown
# Alphabet Generation Pipeline: Integrated Implementation Plan

## Overview

This plan integrates Phase 1 (Alphabet-Level Representation) and Phase 2 
(Generative Architecture) into a sequential, git-traceable implementation.

Each step is designed for autonomous execution with explicit validation and 
diary tracking.

---

## Phase 0: Initialization & Review

### Step P0.01: Initialize Dev Diary

**Goal**: Create development tracking infrastructure.

**Files**:
- `.planning/dev_diary.md` (create)

**Implementation**:
- Create `.planning/` directory
- Create `dev_diary.md` with header and initial entry

**Validation**:
- [ ] File exists at `.planning/dev_diary.md`
- [ ] File is markdown-formatted and readable

**Commit Message**:
```
chore: initialize development diary
```

**Dev Diary Entry**:
```
### 2025-01-09 - Step P0.01: Initialize Dev Diary

**Changed**: Created `.planning/dev_diary.md`.

**Why**: Provides traceability for all implementation decisions throughout 
Phase 1 and Phase 2. Each step appends exactly one entry.

**Limitation**: None.
```

---

### Step P0.02: Review Phase Plans

**Goal**: Ensure understanding of architecture before implementation.

**Files**: None (review only)

**Implementation**:
- Read `plan_phase1.md` completely
- Read `plan_phase2.md` completely
- Identify all files to be created/modified
- Note any unclear specifications

**Validation**:
- [ ] Can explain: Why alphabet-relative normalization matters
- [ ] Can explain: How DeepSets enable permutation invariance
- [ ] Can explain: Why point clouds vs raster for Phase 2
- [ ] List created: All files to be touched in Phase 1
- [ ] List created: All files to be touched in Phase 2

**Commit Message**: N/A (no code changes)

**Dev Diary Entry**:
```
### 2025-01-09 - Step P0.02: Phase Plan Review

**Changed**: None (review step).

**Reviewed**:
- Phase 1: Pipeline fixes, alphabet normalization, tensor output, data loader
- Phase 2: Encoder/decoder architecture, VAE, training loop, generation

**Clarifications needed**: [Agent lists any unclear items here, or "None"]

**Ready to proceed**: [Yes/No - if No, list blockers]
```

---

## Phase 0 Complete: Validation Checkpoint

At this point:
- ✅ Dev diary initialized  
- ✅ Phase plans reviewed and understood

**PAUSE HERE**: Do not proceed until you are confident in:
1. The qCurveTo bug fix approach
2. The difference between glyph-local and alphabet-relative normalization
3. The DeepSets architecture for encoding
4. The point cloud decoder approach

**Continue only when**: You can explain each architectural choice.

---

## Phase 1: Alphabet-Level Representation

### Step P1.01: Fix qCurveTo Handling - Part 1

[Continue with actual implementation steps from phase plans...]

```

This example shows the required structure and detail level. Continue this 
pattern for all remaining steps from both phase plans.

---

## Final Notes

- **Be exhaustive**: Don't skip steps from the phase plans
- **Be granular**: If a plan step seems large, break it into 2-3 sub-steps
- **Be concrete**: No speculation, only actionable instructions
- **Be traceable**: Every change must appear in git log and dev diary

Your output will be used for direct execution. Make it deterministic and complete.