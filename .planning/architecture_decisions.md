# Architecture Decisions

This document records significant architectural decisions made during implementation.

---

## AD-001: Phase 2 Output Representation — Point Clouds

**Date**: 2026-01-09  
**Status**: Accepted  
**Context**: Phase 2 requires selecting an output representation for the glyph decoder.

### Decision

Start with **point cloud** representation: `(256, 2)` — fixed number of 2D points (arc-length samples).

### Rationale

1. **Matches pipeline output**: Phase 1 produces arc-length sampled points; direct compatibility
2. **Fixed dimension**: Easy to decode with standard MLP; no variable-length handling needed
3. **Differentiable**: Chamfer distance provides smooth reconstruction loss
4. **Proven**: Point cloud generation is well-understood (PointNet, etc.)

### Rejected Alternatives

| Alternative | Reason for Rejection |
|-------------|---------------------|
| **Stroke sequence** `(T, 5)` | Variable length requires RNN/Transformer; more complex training; defer to Phase 3 |
| **Raster** `(64, 64, 1)` | Loses vector precision; harder to extract usable glyphs for downstream use |

### Consequences

- Reconstruction loss requires correspondence-free metric (Chamfer or EMD)
- Generated points lack explicit stroke ordering (acceptable for Phase 2)
- Future Phase 3 may migrate to stroke sequences for better structure

### References

- `.planning/plan_phase2.md` §Step 1: Define Output Representation
