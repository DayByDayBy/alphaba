# Phase Exit Checklist — Phase {{PHASE_NUMBER}}

> This checklist is a **hard gate**.  
> All items must be explicitly marked **PASS / FAIL / N/A**.  
> Any FAIL in a **Disallowed State** blocks progression.

---

## A. Required Artifacts

_All artifacts must exist on disk and be referenced explicitly._

- [ ] Source code committed  
  - Commit hashes:
    - 
- [ ] `dev_diary.md` updated  
  - One **date-timed entry per step**
  - Entries correspond 1:1 with commits
- [ ] Planned outputs saved to disk  
  - Paths:
    - 
- [ ] Configuration / constants frozen  
  - Or justification documented

---

## B. Invariants

_These must hold without exception._

- [ ] Phase-specific invariants validated
- [ ] No concepts from later phases introduced
- [ ] No undocumented heuristics
- [ ] No “temporary” logic left unresolved

Evidence / notes:

---

## C. Metrics & Observations

_All metrics defined in the phase plan must be computed and persisted._

- [ ] Metrics computed as specified
- [ ] Metrics saved to disk (JSON / CSV)
- [ ] Summary statistics recorded in diary

Files:

---

## D. Failures & Blockers

> **Failures and blockers must be logged, never silently patched.**

- [ ] All failures documented with:
  - Trigger condition
  - Scope (glyphs / fonts / steps)
  - Hypothesized cause
- [ ] No silent fixes or unlogged workarounds
- [ ] Known limitations explicitly listed

Failures:

---

## E. Validation Evidence

_Claims must be backed by artifacts._

- [ ] Visual artifacts saved (images / plots)
- [ ] Validation method stated
- [ ] At least one negative / edge-case example included

Evidence:

---

## F. Disallowed States (Hard Stop)

If **any** item below is true → **STOP**.

- [ ] Blocking TODOs remain
- [ ] Unreviewed failures exist
- [ ] Phase boundary violated
- [ ] Results justified only qualitatively (“looks good”)

---

## PHASE STATUS

**Status:** PASS / FAIL

**Rationale:**
