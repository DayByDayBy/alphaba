# Phase 0 Exit Checklist — Geometry & Topology Extraction

---

## A. Required Artifacts

- [ ] Normalized SVG path files for all attempted glyphs
- [ ] Arc-length sample arrays saved (`*.npy`)
- [ ] Raster images saved (≥512×512)
- [ ] Skeleton images saved where successful
- [ ] `metadata.json` containing:
  - Per-glyph statistics
  - Alphabet-level aggregates

Paths:

---

## B. Invariants

- [ ] Normalization is **glyph-local**, not alphabet-relative
- [ ] Sampling is **arc-length based**, not parameter-space
- [ ] Skeletonization operates on **filled glyphs**
- [ ] No machine learning or learned parameters
- [ ] No glyph reuse

Evidence:

---

## C. Metrics & Observations

### Per-Glyph
- [ ] Bounding box
- [ ] Aspect ratio
- [ ] Arc length
- [ ] Skeleton pixel count
- [ ] Number of components
- [ ] Endpoints
- [ ] Junctions
- [ ] Junction degree histogram

### Per-Alphabet
- [ ] Mean / std aspect ratio
- [ ] Mean / std arc length
- [ ] Junction degree histogram
- [ ] List of failed glyphs

Files:

---

## D. Failures & Blockers

- [ ] Glyphs with missing paths listed
- [ ] Zero-length or degenerate paths documented
- [ ] Skeletonization failures listed
- [ ] Corner cases documented:
  - g
  - Q
  - y
  - S
- [ ] Hypothesized causes recorded (not guesses)

Failures:

---

## E. Validation Evidence

- [ ] Side-by-side vector / raster / skeleton images saved
- [ ] At least one **clean** skeleton example
- [ ] At least one **broken** skeleton example
- [ ] One clearly incorrect topology example identified

Evidence:

---

## F. Disallowed States (Hard Stop)

- [ ] Skeleton fixes applied without logging → STOP
- [ ] Font-wide normalization applied → STOP
- [ ] Raster-driven geometry decisions → STOP

---

## PHASE STATUS

**Status:** PASS / FAIL

**Rationale:**
