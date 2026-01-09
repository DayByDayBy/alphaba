# 001-PLAN: Generate Integrated Implementation Plan (Phases 1 & 2)

<objective>
Create `INTEGRATED_IMPLEMENTATION_PLAN.md` by integrating `.planning/plan_phase1.md` and `.planning/plan_phase2.md`, following the exact formatting + gating requirements in `.planning/integration_prompt.md` (strict step template, monotonic `P0.xx`/`P1.xx`/`P2.xx`, explicit phase boundaries, dev diary entries, exit checklists).
</objective>

## Plan Analysis
- Complexity: Moderate (single large, structured plan document)
- Dependencies: Linear (Phase 0 → Phase 1 → Phase 2)
- User interaction:
  - Decision required: Phase 2 output representation
  - Human verification: phase boundary checkpoints

<strategy>
Decision-Dependent (one explicit decision checkpoint) + segmented human-verify checkpoints at phase boundaries.
</strategy>

<domain_context>
- Phase 0 code exists in `src/alphabet_pipeline.py`.
- Phase 1 adds: pipeline bug fixes, alphabet-relative normalization, alphabet tensors, alphabet-level data loader, tests + validation notebook.
- Phase 2 adds: DeepSets alphabet encoder, conditional glyph decoder, training loop, evaluation + generation demos.
</domain_context>

<tasks>
<task id="01-01" type="auto" estimated_time="30 min">
  <title>Preflight + skeleton doc</title>
  <description>Confirm required inputs exist and create `INTEGRATED_IMPLEMENTATION_PLAN.md` with the required top-level structure and placeholders.</description>
  <requirements>
    - Inputs present: `.planning/plan_phase1.md`, `.planning/plan_phase2.md`, `.planning/integration_prompt.md`
    - Refusal conditions checked (missing artifacts / contradictions)
  </requirements>
  <files>
    - `INTEGRATED_IMPLEMENTATION_PLAN.md` (create)
  </files>
  <verification>
    - [ ] File exists and includes headings: Overview, Phase 0, Phase 1, Phase 2, Review Checkpoints
    - [ ] Refusal conditions not triggered
  </verification>
</task>

<task id="01-02" type="auto" estimated_time="45 min">
  <title>Write Phase 0 (P0.xx) + Phase 0 exit gate</title>
  <description>Add Phase 0 steps including required Step P0.01 (dev diary init) and the Phase 0 checkpoint block, plus the Phase 0 exit checklist creation step referencing `.planning/phase0_exit_checklist.md`.</description>
  <requirements>
    - Every step uses the exact template from `integration_prompt.md`
    - Includes: Goal, Files, Implementation, Validation, Commit Message, Dev Diary Entry
  </requirements>
  <files>
    - `INTEGRATED_IMPLEMENTATION_PLAN.md` (modify)
  </files>
  <verification>
    - [ ] Steps are monotonic and gap-free: P0.01, P0.02, ...
    - [ ] Phase 0 checkpoint block included verbatim (PAUSE HERE semantics)
  </verification>
</task>

<task id="01-03" type="auto" estimated_time="90 min">
  <title>Translate Phase 1 Step 1 into atomic P1.xx steps</title>
  <description>Convert Phase 1 “Fix Critical Pipeline Bugs” into granular steps (qCurveTo fix, coverage validation, corner-case glyph tests, verify with test fonts) while keeping ≤3 files per step.</description>
  <requirements>
    - Validation commands prefer `uv run ...`
    - Include an explicit inspection step if behavior is uncertain
  </requirements>
  <files>
    - `INTEGRATED_IMPLEMENTATION_PLAN.md` (modify)
  </files>
  <verification>
    - [ ] Every P1.xx step has concrete validation (commands + expected artifact)
    - [ ] No step touches >3 files
  </verification>
</task>

<task id="01-04" type="auto" estimated_time="90 min">
  <title>Translate Phase 1 Steps 2–3 into atomic P1.xx steps</title>
  <description>Convert Phase 1 “Alphabet-Relative Normalization” and “Alphabet Tensor Output” into granular, commit-sized steps (metrics extraction, normalization switch, tensor aggregation, saving `alphabet_*.npy` + `glyph_order.json`).</description>
  <requirements>
    - Preserve Phase 1 boundaries and add the Phase 1 inspection checkpoint block
  </requirements>
  <files>
    - `INTEGRATED_IMPLEMENTATION_PLAN.md` (modify)
  </files>
  <verification>
    - [ ] All deliverables from Phase 1 Steps 2–3 are represented
    - [ ] Includes an inspection step for relative scaling (e.g., 'M' > 'i')
  </verification>
</task>

<task id="01-05" type="auto" estimated_time="90 min">
  <title>Translate Phase 1 Steps 4–5 + Phase 1 exit gate</title>
  <description>Convert the Phase 1 data loader + tests/notebook items into granular steps, and add the Phase 1 exit checklist creation step (`.planning/phase1_exit_checklist.md`) per the prompt’s “Phase Exit Enforcement”.</description>
  <requirements>
    - Mention that `tests/` directory does not currently exist and will be created by the execution plan
  </requirements>
  <files>
    - `INTEGRATED_IMPLEMENTATION_PLAN.md` (modify)
  </files>
  <verification>
    - [ ] Phase 1 includes an explicit “PAUSE HERE” inspection checkpoint block
    - [ ] Exit checklist step exists and references evidence requirements
  </verification>
</task>

<task id="02-01" type="checkpoint:decision" estimated_time="15 min">
  <title>Decide Phase 2 output representation</title>
  <description>Select Phase 2 output representation (A=point cloud, B=stroke sequence, C=raster). Recommended default: A (point cloud) to match Phase 1 `samples` outputs.</description>
  <requirements>
    - Decision recorded in Step P2.01
  </requirements>
  <files>
    - `INTEGRATED_IMPLEMENTATION_PLAN.md` (modify)
  </files>
  <verification>
    - [ ] Decision is explicit and downstream steps align
  </verification>
</task>

<task id="02-02" type="auto" estimated_time="120 min">
  <title>Translate Phase 2 Steps 2–3 (encoder + decoder) into atomic P2.xx steps</title>
  <description>Convert the Phase 2 encoder/decoder sections into granular steps (new files, core classes, reconstruction loss) with concrete smoke validations.</description>
  <requirements>
    - Keep ≤3 files per step
    - Include minimal “import + forward pass” validations using `uv run python -c ...`
  </requirements>
  <files>
    - `INTEGRATED_IMPLEMENTATION_PLAN.md` (modify)
  </files>
  <verification>
    - [ ] All Phase 2 Step 2–3 deliverables are represented
  </verification>
</task>

<task id="02-03" type="auto" estimated_time="90 min">
  <title>Translate Phase 2 Step 4 (training loop) into atomic P2.xx steps</title>
  <description>Convert the training loop into granular steps (combined model wrapper, train step, training protocol, basic validation generation).</description>
  <files>
    - `INTEGRATED_IMPLEMENTATION_PLAN.md` (modify)
  </files>
  <verification>
    - [ ] Includes concrete loss thresholds / expected outputs per the Phase 2 Validation Criteria
  </verification>
</task>

<task id="02-04" type="auto" estimated_time="120 min">
  <title>Translate Phase 2 Steps 5–6 + eval/notebooks + Phase 2 exit gate</title>
  <description>Convert style constraints, optional VAE regularization, novel generation, evaluation metrics, and notebooks into granular steps. Add Phase 2 exit checklist creation (`.planning/phase2_exit_checklist.md`) and the final review block.</description>
  <requirements>
    - Optional items (KL/EMD) must be gated behind explicit inspection/decision steps
  </requirements>
  <files>
    - `INTEGRATED_IMPLEMENTATION_PLAN.md` (modify)
  </files>
  <verification>
    - [ ] Phase 2 complete section includes checkable metrics + saved artifacts
    - [ ] Exit checklist step references evidence (tests/files/inspection step IDs)
  </verification>
</task>

<task id="03-01" type="checkpoint:human-verify" estimated_time="30 min">
  <title>Final quality + compliance review</title>
  <description>Verify the finished `INTEGRATED_IMPLEMENTATION_PLAN.md` satisfies every item in the “Quality Check” section of `integration_prompt.md` (template compliance, monotonic IDs, no vague language, phase boundaries, checkpoints, and refusal conditions).</description>
  <files>
    - `INTEGRATED_IMPLEMENTATION_PLAN.md` (verify)
  </files>
  <verification>
    - [ ] Explicit sign-off recorded (PASS/FAIL) in your review notes
  </verification>
</task>
</tasks>

<success_criteria>
- `INTEGRATED_IMPLEMENTATION_PLAN.md` exists and matches the exact structure required by `.planning/integration_prompt.md`.
- Every Phase 1 + Phase 2 item is represented as one or more atomic steps (≤3 files, 1 concept) with concrete validation.
- Explicit phase boundary checkpoints exist, and exit checklist creation steps exist for Phase 0/1/2 using:
  - `.planning/phase0_exit_checklist.md`
  - `.planning/phase1_exit_checklist.md`
  - `.planning/phase2_exit_checklist.md`
</success_criteria>

## Deviation Rules
- Minor: formatting/wording corrections that do not change step semantics.
- Major: adding/removing steps, changing file targets, or deviating from Phase 1/2 plans → create a decision checkpoint and document rationale in the dev diary during execution.

## Metadata
- Estimated task count: 10
- Estimated total time: ~11 hours
- Recommended execution strategy: Decision-Dependent
