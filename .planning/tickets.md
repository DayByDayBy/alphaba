# Improvement Tickets

## Pipeline Fixes (Priority: High)

---

### TICK-001: Fix TrueType Quadratic Bézier Handling

**Component**: `glyph_to_path()`  
**Severity**: Bug  
**Effort**: Medium

**Problem**: TrueType fonts use implicit on-curve points in quadratic Bézier sequences. The current implementation emits all `qCurveTo` points as a single `Q` command, which is invalid SVG for multi-point sequences.

**Current code**:
```python
elif op == 'qCurveTo':
    points = ' '.join(f"{x} {y}" for x, y in args)
    path_data.append(f"Q {points}")
```

**Fix**: Split multi-point qCurveTo into individual quadratic segments with implied on-curve midpoints.

```python
elif op == 'qCurveTo':
    # TrueType qCurveTo can have multiple off-curve points
    # Implicit on-curve points are midpoints between consecutive off-curve points
    if len(args) == 1:
        # Simple case: single control point + endpoint
        cx, cy = args[0]
        path_data.append(f"Q {cx} {cy}")
    else:
        # Multiple off-curve points: generate implicit on-curve midpoints
        for i in range(len(args) - 1):
            cx, cy = args[i]
            nx, ny = args[i + 1]
            # Midpoint becomes implicit on-curve point
            mx, my = (cx + nx) / 2, (cy + ny) / 2
            path_data.append(f"Q {cx} {cy} {mx} {my}")
        # Final segment to last point
        cx, cy = args[-1]
        path_data.append(f"Q {cx} {cy}")
```

**Acceptance**: Test with fonts containing complex TrueType outlines (e.g., Arial, Times New Roman).

---

### TICK-002: Implement Alphabet-Relative Normalization

**Component**: `normalize_path()`, `process_font()`  
**Severity**: Design Flaw  
**Effort**: Medium

**Problem**: Per-glyph normalization destroys inter-glyph scale relationships. An 'i' and 'M' become same-height, losing critical style information.

**Solution**: Extract font metrics and normalize relative to them.

```python
def get_font_metrics(ttf_path: str) -> Dict[str, float]:
    """Extract font-level normalization metrics."""
    font = TTFont(ttf_path)
    os2 = font.get('OS/2')
    head = font.get('head')
    
    return {
        'units_per_em': head.unitsPerEm,
        'x_height': getattr(os2, 'sxHeight', None),
        'cap_height': getattr(os2, 'sCapHeight', None),
        'ascender': os2.sTypoAscender,
        'descender': os2.sTypoDescender,
    }

def normalize_path_alphabet_relative(path, samples, font_metrics):
    """Normalize path relative to font metrics, not glyph bounds."""
    upm = font_metrics['units_per_em']
    # Scale by UPM, preserving relative glyph sizes
    return path.scaled(1 / upm)
```

**Acceptance**: Post-normalization, 'M' should be visibly larger than 'i'.

---

### TICK-003: Add Font Coverage Validation

**Component**: `process_font()`  
**Severity**: Data Quality  
**Effort**: Low

**Problem**: Fonts with missing glyphs silently produce incomplete alphabets.

**Solution**: Add coverage validation with configurable strictness.

```python
def validate_font_coverage(
    ttf_path: str, 
    required_glyphs: List[str],
    min_coverage: float = 1.0
) -> Tuple[bool, List[str]]:
    """Validate font has required glyph coverage."""
    font = TTFont(ttf_path)
    cmap = font.getBestCmap()
    
    missing = []
    for glyph in required_glyphs:
        if ord(glyph) not in cmap:
            missing.append(glyph)
    
    coverage = 1 - len(missing) / len(required_glyphs)
    return coverage >= min_coverage, missing
```

**Acceptance**: Fonts missing >10% of glyphs logged as warnings; optionally skipped.

---

## Architecture Improvements (Priority: Medium)

---

### TICK-004: Produce Alphabet-Level Output Structure

**Component**: `process_font()`, new function  
**Severity**: Design Gap  
**Effort**: Medium

**Problem**: Pipeline outputs 52 independent files per font. No unified alphabet representation exists.

**Solution**: Add alphabet aggregation step.

```python
def create_alphabet_tensor(glyph_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack per-glyph samples into alphabet tensor.
    
    Returns:
        Array of shape (n_glyphs, n_samples, 2) for point samples
        Or (n_glyphs, height, width) for rasters
    """
    glyph_order = sorted(glyph_data.keys())  # Canonical ordering
    return np.stack([glyph_data[g] for g in glyph_order], axis=0)
```

Output structure:
```
font_name/
├── alphabet_samples.npy      # (52, 256, 2) - all glyphs, all samples
├── alphabet_skeletons.npy    # (52, 512, 512) - all skeletons
├── glyph_order.json          # ["A", "B", ..., "z"]
└── metadata.json
```

**Acceptance**: Single file load provides full alphabet.

---

### TICK-005: Add Stroke Graph Extraction

**Component**: New module `stroke_graph.py`  
**Severity**: Feature Gap  
**Effort**: High

**Problem**: Skeletons are raster-derived and unordered. No explicit stroke representation.

**Solution**: Extract stroke graphs from skeletons or directly from vector paths.

```python
@dataclass
class StrokeGraph:
    nodes: np.ndarray          # (n_nodes, 2) positions
    edges: List[Tuple[int, int]]
    node_types: List[str]      # 'endpoint', 'junction', 'curve'
    
def skeleton_to_stroke_graph(skeleton: np.ndarray) -> StrokeGraph:
    """Convert skeleton image to graph representation."""
    # Find endpoints and junctions
    # Trace paths between them
    # Return structured graph
    ...
```

**Acceptance**: Graph accurately represents stroke topology; can reconstruct skeleton from graph.

---

### TICK-006: Support Non-Latin Alphabets

**Component**: `GLYPH_SET` constant, `process_font()`  
**Severity**: Generalization  
**Effort**: Low

**Problem**: `GLYPH_SET` hardcodes A-Za-z. Pipeline cannot process Cyrillic, Greek, etc.

**Solution**: Make glyph set configurable; add Unicode block definitions.

```python
UNICODE_BLOCKS = {
    'latin_upper': (0x0041, 0x005A),
    'latin_lower': (0x0061, 0x007A),
    'greek_upper': (0x0391, 0x03A9),
    'cyrillic_upper': (0x0410, 0x042F),
    # etc.
}

def get_glyph_set(block_names: List[str]) -> List[str]:
    """Build glyph set from Unicode blocks."""
    glyphs = []
    for name in block_names:
        start, end = UNICODE_BLOCKS[name]
        glyphs.extend(chr(i) for i in range(start, end + 1))
    return glyphs
```

**Note**: `unicode_alphabet_loader.py` already has this pattern—unify.

---

## Testing & Validation (Priority: Medium)

---

### TICK-007: Add Pipeline Unit Tests

**Component**: New `tests/test_pipeline.py`  
**Severity**: Quality  
**Effort**: Medium

**Tests needed**:
1. `test_glyph_to_path_extracts_valid_svg` — known font, verify path parses
2. `test_arc_length_sampling_uniform` — verify spacing is geometric, not parametric
3. `test_normalization_preserves_topology` — same number of segments pre/post
4. `test_skeletonization_minimum_size` — verify rejection of degenerate glyphs
5. `test_topology_analysis_counts` — known skeleton, verify endpoint/junction counts

```python
def test_arc_length_sampling_uniform():
    # Create a known path (e.g., unit circle)
    circle = parse_path("M 1,0 A 1,1 0 1,1 -1,0 A 1,1 0 1,1 1,0 Z")
    samples = arc_length_sample(circle, n_samples=100)
    
    # Verify roughly uniform spacing
    distances = [abs(samples[i+1] - samples[i]) for i in range(len(samples)-1)]
    assert np.std(distances) < 0.1 * np.mean(distances)
```

---

### TICK-008: Add Visual Validation Notebook

**Component**: New `notebooks/00_pipeline_validation.ipynb`  
**Severity**: Quality  
**Effort**: Low

**Purpose**: Visual sanity checks for pipeline outputs.

Cells:
1. Load a processed font
2. Display original glyph vs normalized path vs skeleton side-by-side
3. Overlay arc-length samples on path
4. Show topology markers (endpoints=red, junctions=blue)
5. Compare multiple fonts' 'A' glyphs to verify style preservation

---

## Future Architecture (Priority: Low — Phase 1+)

---

### TICK-009: Define Generative Target Format

**Component**: Design doc / new module  
**Severity**: Architectural  
**Effort**: Planning

**Question**: What will the generative model output?

Options:
- **Raster**: 64×64 or 128×128 binary images
- **Vector**: SVG path strings (variable length, hard to decode)
- **Stroke sequence**: Ordered (x, y, pen_state) tuples (à la Sketch-RNN)
- **Point cloud**: Fixed-size (x, y) arrays (current samples)

**Recommendation**: Stroke sequences or point clouds. Both are fixed-dimension and decoder-friendly.

**Deliverable**: Write `GENERATIVE_TARGET.md` specifying format, dimensionality, and encoding scheme.

---

### TICK-010: Design Alphabet Latent Space

**Component**: Design doc  
**Severity**: Architectural  
**Effort**: Planning

**Core question**: How will "alphabet style" be encoded?

Options:
- **Set encoder**: Encode all 52 glyphs → aggregate → single style vector
- **Hierarchical VAE**: Alphabet-level prior, glyph-level posterior
- **Neural Process**: Condition on subset of glyphs, predict rest

**Deliverable**: Write `LATENT_SPACE_DESIGN.md` with architecture diagrams.

---

## Summary Table

| Ticket | Priority | Effort | Type |
|--------|----------|--------|------|
| TICK-001 | High | Medium | Bug |
| TICK-002 | High | Medium | Design |
| TICK-003 | High | Low | Quality |
| TICK-004 | Medium | Medium | Feature |
| TICK-005 | Medium | High | Feature |
| TICK-006 | Medium | Low | Feature |
| TICK-007 | Medium | Medium | Testing |
| TICK-008 | Medium | Low | Testing |
| TICK-009 | Low | Planning | Design |
| TICK-010 | Low | Planning | Design |
