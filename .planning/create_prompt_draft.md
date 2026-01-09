/create-prompt

Create a robust, execution-oriented prompt that instructs an LLM to integrate
@plan_phase1.md and @plan_phase2.md into a single, granular, step-by-step
implementation plan.

The resulting plan MUST be suitable for autonomous or semi-autonomous execution
by an LLM working in a git repository.

REQUIREMENTS:

1. STRUCTURE
- The plan must be strictly sequential.
- Each step must be small enough to reasonably fit in a single git commit.
- No step may introduce more than one conceptual change.
- Phase 1 and Phase 2 boundaries must be explicit and preserved.

2. PER-STEP SPECIFICATION
For EACH step, include:
- Step ID (monotonic, e.g. P1.03, P2.11)
- Goal (1–2 sentences, outcome-focused)
- Files touched (explicit list)
- Implementation notes (what to do, not how to think)
- Validation criteria (how to know this step is complete)
- Git commit message (imperative, concise)
- Dev diary entry (what was done AND why it was done)

3. DEV DIARY INTEGRATION
- A file `.planning/dev_diary.md` must be created if it does not exist.
- Each step must append exactly one date-timed entry to the diary.
- Diary entries must explain:
  - What changed
  - Why this change exists at this point in the pipeline
  - Any assumptions or known limitations introduced

4. FAILURE & PAUSE SEMANTICS
- If a step depends on uncertain behavior (e.g. skeletonization stability),
  the plan must include an explicit "inspection / validation pause" step.
- Failures and blockersmust be logged, never silently patched.
- The plan should include explicit review checkpoints at the end of Phase 0,
  Phase 1, and Phase 2.

5. NON-GOALS (MUST NOT DO)
- Do NOT invent new architecture beyond what exists in the plans.
- Do NOT merge steps for brevity.
- Do NOT assume ML training success.
- Do NOT skip validation or visualization steps.

6. OUTPUT FORMAT
- The final output must be a single markdown document.
- Use consistent headings and bullet structure.
- The document must be directly usable as an execution script by an LLM.

GOAL:
Produce a plan that could be handed to a different engineer or agent and
executed deterministically, with full traceability via git history and
dev_diary.md.
