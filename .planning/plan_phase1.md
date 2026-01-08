# Phase 1 Plan: Alphabet-Level Representation

**Goal**: Transform per-glyph geometric features into alphabet-level representations suitable for generative modeling.

**Prerequisites**: Phase 0 pipeline (`alphabet_pipeline.py`) producing valid outputs.

---

## Step 1: Fix Critical Pipeline Bugs

**Duration**: 1-2 days  
**Tickets**: TICK-001, TICK-003

### 1.1 Fix qCurveTo handling

Edit `@/Users/gboa/alphaba/src/alphabet_pipeline.py:91-94`:

```python
elif op == 'qCurveTo':
    # TrueType implicit on-curve handling
    points = args
    if len(points) == 1:
        x, y = points[0]
        path_data.append(f"Q {x} {y}")
    else:
        for i in range(len(points) - 1):
            cx, cy = points[i]
            nx, ny = points[i + 1]
            mx, my = (cx + nx) / 2, (cy + ny) / 2
            path_data.append(f"Q {cx} {cy} {mx} {my}")
        x, y = points[-1]
        path_data.append(f"Q {x} {y}")
```

### 1.2 Add font coverage check

Add to `process_font()` before glyph loop:

```python
# Validate coverage
font = TTFont(ttf_path)
cmap = font.getBestCmap()
missing = [g for g in glyph_set if ord(g) not in cmap]
if len(missing) > len(glyph_set) * 0.1:
    logger.warning(f"Font {font_name} missing {len(missing)} glyphs: {missing[:5]}...")
```

### 1.3 Add corner-case glyph tests

Glyphs with complex curves that stress the qCurveTo fix:

```python
CORNER_CASE_GLYPHS = ['g', 'Q', 'y', 'S', '@', '&']  # Descenders, complex curves

def test_corner_case_glyphs(ttf_path: str):
    """Verify no curve failures on problematic glyphs."""
    for glyph in CORNER_CASE_GLYPHS:
        path = glyph_to_path(ttf_path, glyph)
        if path is None:
            logger.warning(f"Corner case glyph '{glyph}' not in font")
            continue
        
        samples = arc_length_sample(path)
        assert len(samples) > 0, f"Failed to sample {glyph}"
        
        # Verify no NaN/Inf in samples
        for s in samples:
            assert np.isfinite(s.real) and np.isfinite(s.imag), f"Invalid sample in {glyph}"
```

### 1.4 Verify with test fonts

```bash
uv run python -c "
from src.alphabet_pipeline import process_font
process_font('path/to/test_font.ttf', 'output/test')
"
```

---

## Step 2: Implement Alphabet-Relative Normalization

**Duration**: 2-3 days  
**Ticket**: TICK-002

### 2.1 Extract font metrics

Add new function:

```python
def get_font_metrics(ttf_path: str) -> Dict[str, Any]:
    """Extract font-level metrics for alphabet-relative normalization."""
    font = TTFont(ttf_path)
    head = font['head']
    
    metrics = {
        'units_per_em': head.unitsPerEm,
    }
    
    if 'OS/2' in font:
        os2 = font['OS/2']
        metrics.update({
            'x_height': getattr(os2, 'sxHeight', None),
            'cap_height': getattr(os2, 'sCapHeight', None),
            'ascender': os2.sTypoAscender,
            'descender': os2.sTypoDescender,
        })
    
    if 'hhea' in font:
        hhea = font['hhea']
        metrics['line_gap'] = hhea.lineGap
    
    return metrics
```

### 2.2 Modify normalization

Add option for alphabet-relative scaling:

```python
def normalize_path_v2(
    path: SVGPath, 
    samples: List[complex],
    glyph_name: str,
    font_metrics: Optional[Dict] = None
) -> Optional[SVGPath]:
    """Normalize path with optional font-relative scaling.
    
    Uses x_height for lowercase, cap_height for uppercase when available.
    Falls back to units_per_em if specific heights unavailable.
    """
    if font_metrics:
        upm = font_metrics['units_per_em']
        
        # Use case-specific normalization if available
        if glyph_name.islower() and font_metrics.get('x_height'):
            scale_ref = font_metrics['x_height']
        elif glyph_name.isupper() and font_metrics.get('cap_height'):
            scale_ref = font_metrics['cap_height']
        else:
            scale_ref = upm
        
        # Scale relative to reference, then normalize to unit space
        return path.scaled(1 / scale_ref)
    else:
        # Fall back to glyph-local normalization
        return normalize_path(path, samples)
```

**Note**: This preserves lowercase/uppercase height relationships — critical for alphabet-level embeddings where 'a' should be shorter than 'A'.

### 2.3 Update process_font

```python
def process_font(ttf_path, output_dir, glyph_set=GLYPH_SET, alphabet_relative=True):
    font_metrics = get_font_metrics(ttf_path) if alphabet_relative else None
    # ... pass font_metrics to normalize_path_v2
```

---

## Step 3: Create Alphabet Tensor Output

**Duration**: 2 days  
**Ticket**: TICK-004

### 3.1 Define canonical glyph ordering

```python
CANONICAL_ORDER = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

def glyph_to_index(glyph: str) -> int:
    return CANONICAL_ORDER.index(glyph)
```

### 3.2 Create aggregation function

```python
def create_alphabet_tensor(
    base_path: Path,
    representation: str = 'samples'  # 'samples', 'skeletons', 'rasters'
) -> np.ndarray:
    """Load and stack all glyphs into single tensor."""
    
    if representation == 'samples':
        subdir = 'samples'
        loader = lambda p: np.load(p)
        shape_suffix = (DEFAULT_SAMPLE_COUNT, 2)
    elif representation == 'skeletons':
        subdir = 'skeletons'
        loader = lambda p: np.array(Image.open(p)) / 255.0
        shape_suffix = (DEFAULT_RASTER_SIZE, DEFAULT_RASTER_SIZE)
    
    tensors = []
    for glyph in CANONICAL_ORDER:
        path = base_path / subdir / f'{glyph}_samples.npy'
        if path.exists():
            tensors.append(loader(path))
        else:
            # Placeholder for missing glyphs
            tensors.append(np.zeros(shape_suffix))
    
    return np.stack(tensors, axis=0)
```

### 3.3 Save unified outputs

Add to end of `process_font()`:

```python
# Create unified alphabet tensors
alphabet_samples = create_alphabet_tensor(base_path, 'samples')
np.save(base_path / 'alphabet_samples.npy', alphabet_samples)

alphabet_skeletons = create_alphabet_tensor(base_path, 'skeletons')
np.save(base_path / 'alphabet_skeletons.npy', alphabet_skeletons)

# Save glyph order for reference
with open(base_path / 'glyph_order.json', 'w') as f:
    json.dump(CANONICAL_ORDER, f)
```

### 3.4 Optional: Combined tensor with topology channels

For models that benefit from skeleton topology alongside point clouds:

```python
def create_combined_tensor(base_path: Path) -> np.ndarray:
    """Create (52, 256, 4) tensor: [x, y, skeleton_x, skeleton_y].
    
    Skeleton points are sampled to match point cloud count.
    """
    samples = create_alphabet_tensor(base_path, 'samples')  # (52, 256, 2)
    
    # Sample skeleton points to match
    skeleton_samples = []
    for glyph in CANONICAL_ORDER:
        skel_path = base_path / 'skeletons' / f'{glyph}.png'
        if skel_path.exists():
            skel = np.array(Image.open(skel_path))
            points = np.argwhere(skel > 0)  # (N, 2)
            if len(points) >= DEFAULT_SAMPLE_COUNT:
                idx = np.linspace(0, len(points)-1, DEFAULT_SAMPLE_COUNT, dtype=int)
                skeleton_samples.append(points[idx] / skel.shape[0])  # Normalize
            else:
                # Pad with zeros if skeleton too sparse
                padded = np.zeros((DEFAULT_SAMPLE_COUNT, 2))
                padded[:len(points)] = points / skel.shape[0]
                skeleton_samples.append(padded)
        else:
            skeleton_samples.append(np.zeros((DEFAULT_SAMPLE_COUNT, 2)))
    
    skeleton_tensor = np.stack(skeleton_samples, axis=0)  # (52, 256, 2)
    return np.concatenate([samples, skeleton_tensor], axis=-1)  # (52, 256, 4)
```

**Use case**: Feed topology features directly to encoder without separate skeleton branch.

---

## Step 4: Build Alphabet Data Loader

**Duration**: 2-3 days  
**New file**: `src/alphabet_data_loader.py`

### 4.1 Create loader class

```python
class AlphabetDataLoader:
    """Load processed alphabets as sets for training."""
    
    def __init__(self, processed_dir: str, representation: str = 'samples'):
        self.processed_dir = Path(processed_dir)
        self.representation = representation
        self.alphabets = {}
        self._load_alphabets()
    
    def _load_alphabets(self):
        for font_dir in self.processed_dir.iterdir():
            if not font_dir.is_dir():
                continue
            
            tensor_path = font_dir / f'alphabet_{self.representation}.npy'
            if tensor_path.exists():
                self.alphabets[font_dir.name] = np.load(tensor_path)
        
        print(f"Loaded {len(self.alphabets)} alphabets")
    
    def get_alphabet(self, font_name: str) -> np.ndarray:
        """Get full alphabet tensor (52, ...) for a font."""
        return self.alphabets[font_name]
    
    def sample_alphabet_pair(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Sample two alphabets; return (alph1, alph2, same_style)."""
        names = list(self.alphabets.keys())
        
        if np.random.random() < 0.5:
            # Same alphabet (positive pair)
            name = np.random.choice(names)
            return self.alphabets[name], self.alphabets[name], True
        else:
            # Different alphabets (negative pair)
            name1, name2 = np.random.choice(names, 2, replace=False)
            return self.alphabets[name1], self.alphabets[name2], False
```

### 4.2 Triplet sampling at alphabet level

```python
def sample_alphabet_triplet(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (anchor, positive, negative) at alphabet level.
    
    Anchor and positive are from same font (different augmentation).
    Negative is from different font.
    """
    names = list(self.alphabets.keys())
    
    anchor_name = np.random.choice(names)
    negative_name = np.random.choice([n for n in names if n != anchor_name])
    
    anchor = self.alphabets[anchor_name]
    positive = self._augment(anchor)  # Same alphabet, augmented
    negative = self.alphabets[negative_name]
    
    return anchor, positive, negative

def _augment(self, alphabet: np.ndarray) -> np.ndarray:
    """Apply consistent augmentation across all glyphs.
    
    IMPORTANT: Augmentation applied AFTER normalization to preserve
    alphabet-level relationships. Same transform for all 52 glyphs.
    """
    # Small rotation, scale, translation — same transform for all glyphs
    angle = np.random.uniform(-5, 5) * np.pi / 180
    scale = np.random.uniform(0.95, 1.05)
    tx, ty = np.random.uniform(-0.05, 0.05, size=2)
    
    # Build 2D affine matrix
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    transform = np.array([
        [scale * cos_a, -scale * sin_a, tx],
        [scale * sin_a,  scale * cos_a, ty]
    ])
    
    # Apply to all glyphs (alphabet shape: 52, 256, 2)
    augmented = np.zeros_like(alphabet)
    for i in range(alphabet.shape[0]):
        points = alphabet[i]  # (256, 2)
        # Homogeneous coords
        ones = np.ones((points.shape[0], 1))
        homogeneous = np.concatenate([points, ones], axis=1)  # (256, 3)
        augmented[i] = (transform @ homogeneous.T).T  # (256, 2)
    
    return augmented
```

---

## Step 5: Validation & Testing

**Duration**: 1-2 days  
**Tickets**: TICK-007, TICK-008

### 5.1 Create test file

`tests/test_alphabet_pipeline.py`:

```python
import pytest
import numpy as np
from src.alphabet_pipeline import (
    glyph_to_path, arc_length_sample, normalize_path,
    skeletonize_glyph, analyze_skeleton_topology
)

TEST_FONT = "path/to/known/test.ttf"

def test_glyph_extraction():
    path = glyph_to_path(TEST_FONT, 'A')
    assert path is not None
    assert path.length() > 0

def test_arc_length_uniformity():
    path = glyph_to_path(TEST_FONT, 'O')
    samples = arc_length_sample(path, n_samples=100)
    
    distances = [abs(samples[i+1] - samples[i]) for i in range(len(samples)-1)]
    cv = np.std(distances) / np.mean(distances)  # Coefficient of variation
    assert cv < 0.3  # Reasonably uniform

def test_alphabet_tensor_shape():
    from src.alphabet_pipeline import create_alphabet_tensor
    tensor = create_alphabet_tensor(Path("output/TestFont"), 'samples')
    assert tensor.shape == (52, 256, 2)
```

### 5.2 Create validation notebook

`notebooks/00_pipeline_validation.ipynb`:

- Cell 1: Load processed font
- Cell 2: Display original vs normalized for A, i, M (verify relative scaling)
- Cell 3: Overlay samples on path
- Cell 4: Show skeleton with topology markers
- Cell 5: Compare same letter across fonts

---

## Deliverables

| Item | Location | Format |
|------|----------|--------|
| Fixed pipeline | `src/alphabet_pipeline.py` | Python |
| Font metrics extraction | `src/alphabet_pipeline.py` | Python |
| Alphabet tensor output | `output/*/alphabet_samples.npy` | NumPy |
| Alphabet data loader | `src/alphabet_data_loader.py` | Python |
| Unit tests | `tests/test_alphabet_pipeline.py` | pytest |
| Validation notebook | `notebooks/00_pipeline_validation.ipynb` | Jupyter |

---

## Success Criteria

1. **Pipeline runs without errors** on 10+ diverse fonts
2. **Relative glyph scaling preserved** — visual inspection shows 'M' > 'i'
3. **Alphabet tensors loadable** — single `np.load()` gives full alphabet
4. **Data loader functional** — can generate triplets for training
5. **Tests pass** — `pytest tests/` green

---

## Next Phase Preview

Phase 2 will implement the **alphabet encoder** — a set-based neural network that takes alphabet tensors and produces style embeddings. This will replace/augment the current triplet network which operates on individual glyphs.
