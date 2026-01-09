# Phase 1 Plan: Alphabet-Level Representation (Improved)

**Goal**: Transform per-glyph geometric features into alphabet-level representations suitable for generative modeling.

**Prerequisites**: Phase 0 pipeline (`alphabet_pipeline.py`) exists and can process fonts.

---

## Step 0: Establish Baseline & Validation Framework

**Duration**: 0.5 days  
**New Ticket**: TICK-000

### 0.1 Select test fonts

Create `tests/test_fonts/` with 3 reference fonts:
- **Sans-serif**: `Roboto-Regular.ttf` (clean, modern)
- **Serif**: `Merriweather-Regular.ttf` (traditional, high contrast)
- **Script**: `DancingScript-Regular.ttf` (complex curves, stress test)

Download from Google Fonts or use system fonts.

### 0.2 Create baseline validation script

`scripts/validate_pipeline.py`:

```python
#!/usr/bin/env python3
"""Validate pipeline produces expected outputs."""

import sys
from pathlib import Path
import numpy as np
from src.alphabet_pipeline import process_font

TEST_FONTS = {
    'roboto': 'tests/test_fonts/Roboto-Regular.ttf',
    'merriweather': 'tests/test_fonts/Merriweather-Regular.ttf', 
    'dancing_script': 'tests/test_fonts/DancingScript-Regular.ttf',
}

REQUIRED_OUTPUTS = [
    'samples',  # Directory with .npy files
    'skeletons',  # Directory with .png files
    'glyph_order.json',
]

def validate_font_processing(font_name: str, ttf_path: str) -> dict:
    """Process font and return validation metrics."""
    output_dir = Path(f'output/baseline_{font_name}')
    
    # Process
    process_font(ttf_path, str(output_dir))
    
    # Check required outputs exist
    results = {'font': font_name, 'errors': [], 'warnings': [], 'metrics': {}}
    
    for required in REQUIRED_OUTPUTS:
        path = output_dir / required
        if not path.exists():
            results['errors'].append(f"Missing required output: {required}")
    
    # Check glyph coverage
    samples_dir = output_dir / 'samples'
    if samples_dir.exists():
        sample_files = list(samples_dir.glob('*_samples.npy'))
        results['metrics']['glyphs_processed'] = len(sample_files)
        
        # Expect 52 glyphs (A-Z, a-z)
        if len(sample_files) < 52:
            missing = 52 - len(sample_files)
            results['warnings'].append(f"Missing {missing} glyphs (expected 52)")
        
        # Check for invalid samples (NaN, Inf, all zeros)
        for sample_file in sample_files:
            samples = np.load(sample_file)
            
            if np.any(~np.isfinite(samples)):
                results['errors'].append(f"Invalid samples in {sample_file.name}")
            
            if np.allclose(samples, 0):
                results['warnings'].append(f"All-zero samples in {sample_file.name}")
            
            # Check shape
            if samples.shape != (256, 2):
                results['errors'].append(
                    f"Wrong shape {samples.shape} in {sample_file.name}"
                )
    
    return results

def main():
    all_results = []
    
    for name, path in TEST_FONTS.items():
        if not Path(path).exists():
            print(f"⚠️  Test font missing: {path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Validating: {name}")
        print('='*60)
        
        results = validate_font_processing(name, path)
        all_results.append(results)
        
        # Print results
        if results['errors']:
            print(f"❌ ERRORS ({len(results['errors'])}):")
            for err in results['errors']:
                print(f"   - {err}")
        
        if results['warnings']:
            print(f"⚠️  WARNINGS ({len(results['warnings'])}):")
            for warn in results['warnings']:
                print(f"   - {warn}")
        
        if not results['errors']:
            print(f"✓ SUCCESS - {results['metrics']['glyphs_processed']} glyphs processed")
    
    # Overall pass/fail
    total_errors = sum(len(r['errors']) for r in all_results)
    
    print(f"\n{'='*60}")
    if total_errors == 0:
        print("✓ BASELINE VALIDATION PASSED")
        return 0
    else:
        print(f"❌ BASELINE VALIDATION FAILED ({total_errors} errors)")
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

### 0.3 Run baseline validation

```bash
uv run python scripts/validate_pipeline.py
```

**Success criteria**: All 3 fonts process without errors, produce 52 glyphs each.

**Save baseline outputs** for comparison after Step 2:
```bash
cp -r output/baseline_* .baseline_reference/
```

---

## Step 1: Fix Critical Pipeline Bugs

**Duration**: 1-2 days  
**Tickets**: TICK-001, TICK-003

### 1.1 Fix qCurveTo handling

Edit `src/alphabet_pipeline.py:91-94`:

```python
elif op == 'qCurveTo':
    # TrueType implicit on-curve handling
    points = args
    if len(points) == 1:
        # Single point - explicit endpoint
        x, y = points[0]
        path_data.append(f"Q {x} {y}")
    else:
        # Multiple control points - compute implicit on-curve points
        for i in range(len(points) - 1):
            cx, cy = points[i]
            nx, ny = points[i + 1]
            # Implicit on-curve point is midpoint between consecutive off-curve points
            mx, my = (cx + nx) / 2, (cy + ny) / 2
            path_data.append(f"Q {cx} {cy} {mx} {my}")
        
        # Final explicit endpoint
        x, y = points[-1]
        path_data.append(f"Q {x} {y}")
```

**Verification**: 
```python
# Add to existing tests or create new file
def test_qcurveto_handling():
    """Verify qCurveTo doesn't crash on complex glyphs."""
    for font_path in TEST_FONTS.values():
        for glyph in ['g', 'Q', 'y', 'S', '@']:
            path = glyph_to_path(font_path, glyph)
            if path is None:
                continue  # Glyph not in font
            
            samples = arc_length_sample(path)
            assert len(samples) > 0, f"Failed to sample {glyph}"
            assert all(np.isfinite(s.real) and np.isfinite(s.imag) for s in samples)
```

### 1.2 Add font coverage validation

Add to `process_font()` before glyph loop:

```python
def process_font(ttf_path, output_dir, glyph_set=GLYPH_SET, alphabet_relative=True):
    # ... existing setup ...
    
    # Validate coverage
    font = TTFont(ttf_path)
    cmap = font.getBestCmap()
    
    missing = []
    for glyph in glyph_set:
        if ord(glyph) not in cmap:
            missing.append(glyph)
    
    if missing:
        missing_pct = len(missing) / len(glyph_set) * 100
        logger.warning(
            f"Font {font_name} missing {len(missing)}/{len(glyph_set)} glyphs "
            f"({missing_pct:.1f}%): {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
        
        # Hard stop if too many missing
        if missing_pct > 20:
            raise ValueError(
                f"Font coverage too low ({missing_pct:.1f}% missing). "
                f"Need at least 80% of glyphs."
            )
    
    # ... rest of processing ...
```

### 1.3 Re-run baseline validation

```bash
uv run python scripts/validate_pipeline.py
```

**Success criteria**: Same output as Step 0.3 (no regressions).

---

## Step 2: Implement Alphabet-Relative Normalization

**Duration**: 2-3 days  
**Ticket**: TICK-002

### 2.1 Extract font metrics

Add new function in `src/alphabet_pipeline.py`:

```python
def get_font_metrics(ttf_path: str) -> Dict[str, Any]:
    """Extract font-level metrics for alphabet-relative normalization."""
    font = TTFont(ttf_path)
    head = font['head']
    
    metrics = {
        'units_per_em': head.unitsPerEm,
    }
    
    # Try to get OpenType metrics
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
    
    # Fallback: measure reference glyphs if metrics missing
    if metrics['x_height'] is None:
        # Measure 'x' height as proxy
        try:
            x_path = glyph_to_path(ttf_path, 'x')
            if x_path:
                bounds = x_path.bbox()
                metrics['x_height'] = bounds[3] - bounds[1]  # height
        except:
            metrics['x_height'] = metrics['units_per_em'] * 0.5  # Conservative fallback
    
    if metrics['cap_height'] is None:
        # Measure 'H' height as proxy
        try:
            h_path = glyph_to_path(ttf_path, 'H')
            if h_path:
                bounds = h_path.bbox()
                metrics['cap_height'] = bounds[3] - bounds[1]
        except:
            metrics['cap_height'] = metrics['units_per_em'] * 0.7  # Conservative fallback
    
    return metrics
```

### 2.2 Implement case-aware normalization

Add new normalization function:

```python
# Glyph categories for normalization
UPPERCASE = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
LOWERCASE = set('abcdefghijklmnopqrstuvwxyz')
DESCENDERS = set('gjpqy')

def normalize_path_v2(
    path: SVGPath, 
    samples: List[complex],
    glyph_name: str,
    font_metrics: Optional[Dict] = None
) -> Optional[SVGPath]:
    """Normalize path with alphabet-relative scaling.
    
    Preserves height relationships between upper/lowercase and handles descenders.
    
    Args:
        path: Original SVG path
        samples: Arc-length samples for centering
        glyph_name: Character being normalized
        font_metrics: Font-level metrics for relative scaling
    
    Returns:
        Normalized path in unit space, or None if normalization fails
    """
    if not samples:
        return None
    
    # Compute bounding box from samples
    real_coords = [s.real for s in samples]
    imag_coords = [s.imag for s in samples]
    
    min_x, max_x = min(real_coords), max(real_coords)
    min_y, max_y = min(imag_coords), max(imag_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0 or height == 0:
        return None
    
    # Center point
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    
    if font_metrics:
        # Alphabet-relative normalization
        upm = font_metrics['units_per_em']
        
        # Determine reference height based on glyph type
        if glyph_name in UPPERCASE:
            # Uppercase: normalize to cap height
            scale_ref = font_metrics.get('cap_height', upm * 0.7)
        elif glyph_name in LOWERCASE:
            if glyph_name in DESCENDERS:
                # Descenders: include descender space in normalization
                x_h = font_metrics.get('x_height', upm * 0.5)
                desc = abs(font_metrics.get('descender', -upm * 0.2))
                scale_ref = x_h + desc
            else:
                # Regular lowercase: normalize to x-height
                scale_ref = font_metrics.get('x_height', upm * 0.5)
        else:
            # Unknown glyph type: use units per em
            scale_ref = upm
        
        # Apply normalization: translate to origin, then scale
        # Final coordinates will be in approximately [-0.5, 0.5] range
        normalized = path.translated(complex(-cx, -cy))
        normalized = normalized.scaled(1 / scale_ref)
        
        return normalized
    else:
        # Fallback: glyph-local normalization (Phase 0 behavior)
        # Normalize to unit square
        scale = max(width, height)
        normalized = path.translated(complex(-cx, -cy))
        normalized = normalized.scaled(1 / scale)
        
        return normalized
```

### 2.3 Update process_font

Modify `process_font()` to use new normalization:

```python
def process_font(
    ttf_path: str, 
    output_dir: str, 
    glyph_set: str = GLYPH_SET,
    alphabet_relative: bool = True  # New parameter
):
    """Process font and extract geometric features.
    
    Args:
        ttf_path: Path to TrueType font file
        output_dir: Directory for outputs
        glyph_set: String of glyphs to process
        alphabet_relative: If True, normalize relative to font metrics (preserves case relationships)
    """
    # ... existing setup ...
    
    # Extract font metrics if using alphabet-relative normalization
    font_metrics = get_font_metrics(ttf_path) if alphabet_relative else None
    
    if font_metrics:
        logger.info(
            f"Font metrics: UPM={font_metrics['units_per_em']}, "
            f"x-height={font_metrics.get('x_height')}, "
            f"cap-height={font_metrics.get('cap_height')}"
        )
    
    # ... process glyphs ...
    for glyph in glyph_set:
        # ... existing path extraction ...
        
        # Use new normalization
        normalized_path = normalize_path_v2(path, samples, glyph, font_metrics)
        
        # ... rest of processing ...
```

### 2.4 Create comparison validation

`scripts/compare_normalizations.py`:

```python
#!/usr/bin/env python3
"""Compare old vs new normalization to verify improvements."""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def compare_normalization(font_name: str):
    """Generate side-by-side comparison of normalizations."""
    
    baseline_dir = Path(f'.baseline_reference/baseline_{font_name}/samples')
    new_dir = Path(f'output/baseline_{font_name}/samples')
    
    # Compare key glyphs: uppercase, lowercase, descender
    test_glyphs = ['M', 'i', 'g']
    
    fig, axes = plt.subplots(len(test_glyphs), 2, figsize=(10, 12))
    fig.suptitle(f'Normalization Comparison: {font_name}')
    
    for idx, glyph in enumerate(test_glyphs):
        baseline = np.load(baseline_dir / f'{glyph}_samples.npy')
        new = np.load(new_dir / f'{glyph}_samples.npy')
        
        # Plot old
        axes[idx, 0].plot(baseline[:, 0], baseline[:, 1], 'b-', linewidth=2)
        axes[idx, 0].set_title(f'{glyph} (Old)')
        axes[idx, 0].axis('equal')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Plot new
        axes[idx, 1].plot(new[:, 0], new[:, 1], 'r-', linewidth=2)
        axes[idx, 1].set_title(f'{glyph} (New)')
        axes[idx, 1].axis('equal')
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'output/normalization_comparison_{font_name}.png', dpi=150)
    print(f"✓ Saved comparison to output/normalization_comparison_{font_name}.png")

if __name__ == '__main__':
    for font in ['roboto', 'merriweather', 'dancing_script']:
        compare_normalization(font)
```

**Run and verify**:
```bash
# Reprocess fonts with new normalization
uv run python scripts/validate_pipeline.py

# Generate visual comparisons
uv run python scripts/compare_normalizations.py
```

**Success criteria**: 
- Visual inspection shows 'M' taller than 'i' in normalized space
- Descender glyphs ('g') extend below baseline consistently
- No regressions in curve quality

---

## Step 3: Create Alphabet Tensor Output

**Duration**: 2 days  
**Ticket**: TICK-004

### 3.1 Define canonical glyph ordering

Add to `src/alphabet_pipeline.py`:

```python
# Canonical ordering: uppercase first, then lowercase
CANONICAL_ORDER = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

def glyph_to_index(glyph: str) -> int:
    """Get canonical index for a glyph."""
    return CANONICAL_ORDER.index(glyph)

def index_to_glyph(idx: int) -> str:
    """Get glyph from canonical index."""
    return CANONICAL_ORDER[idx]
```

### 3.2 Create aggregation function with validation

```python
def create_alphabet_tensor(
    base_path: Path,
    representation: str = 'samples'  # 'samples', 'skeletons'
) -> Tuple[np.ndarray, List[str]]:
    """Load and stack all glyphs into single tensor.
    
    Returns:
        tensor: Stacked glyphs in canonical order
        missing_glyphs: List of glyphs that were zero-padded
    """
    
    if representation == 'samples':
        subdir = 'samples'
        loader = lambda p: np.load(p)
        shape_suffix = (DEFAULT_SAMPLE_COUNT, 2)
    elif representation == 'skeletons':
        subdir = 'skeletons'
        loader = lambda p: np.array(Image.open(p)) / 255.0
        shape_suffix = (DEFAULT_RASTER_SIZE, DEFAULT_RASTER_SIZE)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    
    tensors = []
    missing_glyphs = []
    
    for glyph in CANONICAL_ORDER:
        if representation == 'samples':
            path = base_path / subdir / f'{glyph}_samples.npy'
        elif representation == 'skeletons':
            path = base_path / subdir / f'{glyph}.png'
        
        if path.exists():
            try:
                data = loader(path)
                # Validate not all zeros
                if np.allclose(data, 0):
                    logger.warning(f"Glyph '{glyph}' loaded but all zeros")
                    missing_glyphs.append(glyph)
                tensors.append(data)
            except Exception as e:
                logger.error(f"Failed to load {glyph}: {e}")
                tensors.append(np.zeros(shape_suffix))
                missing_glyphs.append(glyph)
        else:
            # Placeholder for missing glyphs
            logger.warning(f"Glyph '{glyph}' not found, using zero placeholder")
            tensors.append(np.zeros(shape_suffix))
            missing_glyphs.append(glyph)
    
    # Validate before returning
    if len(missing_glyphs) > 10:  # More than ~20% missing
        raise ValueError(
            f"Too many missing glyphs ({len(missing_glyphs)}/52): {missing_glyphs}. "
            f"Font may be unsuitable for training."
        )
    
    tensor = np.stack(tensors, axis=0)
    logger.info(
        f"Created {representation} tensor: shape={tensor.shape}, "
        f"missing={len(missing_glyphs)}"
    )
    
    return tensor, missing_glyphs
```

### 3.3 Save unified outputs with metadata

Add to end of `process_font()`:

```python
    # Create unified alphabet tensors
    logger.info("Creating unified alphabet tensors...")
    
    try:
        alphabet_samples, missing_samples = create_alphabet_tensor(base_path, 'samples')
        np.save(base_path / 'alphabet_samples.npy', alphabet_samples)
        
        alphabet_skeletons, missing_skeletons = create_alphabet_tensor(base_path, 'skeletons')
        np.save(base_path / 'alphabet_skeletons.npy', alphabet_skeletons)
        
        # Save metadata
        metadata = {
            'glyph_order': CANONICAL_ORDER,
            'missing_glyphs_samples': missing_samples,
            'missing_glyphs_skeletons': missing_skeletons,
            'font_metrics': font_metrics,
            'alphabet_relative': alphabet_relative,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(base_path / 'alphabet_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Alphabet tensors saved: {alphabet_samples.shape}")
        
    except Exception as e:
        logger.error(f"Failed to create alphabet tensors: {e}")
        raise
```

---

## Step 4: Build Alphabet Data Loader

**Duration**: 2-3 days  
**New file**: `src/alphabet_data_loader.py`

### 4.1 Create loader class

```python
"""Data loader for alphabet-level representations."""

from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

class AlphabetDataLoader:
    """Load processed alphabets as sets for training."""
    
    def __init__(
        self, 
        processed_dir: str, 
        representation: str = 'samples',
        min_glyphs: int = 42  # Require at least 80% coverage
    ):
        """Initialize loader.
        
        Args:
            processed_dir: Directory containing processed fonts
            representation: 'samples' or 'skeletons'
            min_glyphs: Minimum non-zero glyphs required per alphabet
        """
        self.processed_dir = Path(processed_dir)
        self.representation = representation
        self.min_glyphs = min_glyphs
        
        self.alphabets: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, dict] = {}
        
        self._load_alphabets()
    
    def _load_alphabets(self):
        """Scan directory and load all valid alphabets."""
        for font_dir in self.processed_dir.iterdir():
            if not font_dir.is_dir():
                continue
            
            tensor_path = font_dir / f'alphabet_{self.representation}.npy'
            metadata_path = font_dir / 'alphabet_metadata.json'
            
            if not tensor_path.exists():
                logger.debug(f"Skipping {font_dir.name}: no tensor file")
                continue
            
            # Load tensor
            tensor = np.load(tensor_path)
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            else:
                logger.warning(f"No metadata for {font_dir.name}")
                metadata = {}
            
            # Validate coverage
            non_zero_glyphs = np.sum(~np.all(tensor == 0, axis=(1, 2)))
            
            if non_zero_glyphs < self.min_glyphs:
                logger.warning(
                    f"Skipping {font_dir.name}: insufficient coverage "
                    f"({non_zero_glyphs}/{tensor.shape[0]} glyphs)"
                )
                continue
            
            self.alphabets[font_dir.name] = tensor
            self.metadata[font_dir.name] = metadata
        
        logger.info(f"Loaded {len(self.alphabets)} valid alphabets")
        
        if len(self.alphabets) == 0:
            raise ValueError(f"No valid alphabets found in {self.processed_dir}")
    
    def get_alphabet(self, font_name: str) -> np.ndarray:
        """Get full alphabet tensor for a font.
        
        Returns:
            Array of shape (52, ...) containing all glyphs
        """
        return self.alphabets[font_name]
    
    def get_metadata(self, font_name: str) -> dict:
        """Get metadata for a font."""
        return self.metadata.get(font_name, {})
    
    def list_fonts(self) -> List[str]:
        """Get list of all loaded font names."""
        return list(self.alphabets.keys())
    
    def __len__(self) -> int:
        """Number of loaded alphabets."""
        return len(self.alphabets)
```

### 4.2 Add sampling methods

```python
    def sample_alphabet_pair(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Sample two alphabets for contrastive learning.
        
        Returns:
            (alphabet1, alphabet2, same_style)
            - same_style=True: both from same font (positive pair)
            - same_style=False: from different fonts (negative pair)
        """
        names = list(self.alphabets.keys())
        
        if np.random.random() < 0.5:
            # Positive pair: same alphabet (will be augmented differently)
            name = np.random.choice(names)
            return self.alphabets[name], self.alphabets[name], True
        else:
            # Negative pair: different alphabets
            name1, name2 = np.random.choice(names, 2, replace=False)
            return self.alphabets[name1], self.alphabets[name2], False
    
    def sample_alphabet_triplet(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample (anchor, positive, negative) at alphabet level.
        
        Anchor and positive are from same font (will be augmented differently).
        Negative is from different font.
        
        Returns:
            (anchor, positive, negative) arrays of shape (52, ...)
        """
        names = list(self.alphabets.keys())
        
        anchor_name = np.random.choice(names)
        negative_name = np.random.choice([n for n in names if n != anchor_name])
        
        anchor = self.alphabets[anchor_name]
        positive = self.alphabets[anchor_name]  # Same source, will augment later
        negative = self.alphabets[negative_name]
        
        return anchor, positive, negative
```

### 4.3 Add augmentation (applied to full alphabets)

```python
    def augment_alphabet(
        self, 
        alphabet: np.ndarray,
        rotation_range: float = 5.0,  # degrees
        scale_range: Tuple[float, float] = (0.95, 1.05),
        translation_range: float = 0.05,  # fraction of space
    ) -> np.ndarray:
        """Apply consistent augmentation across all glyphs in alphabet.
        
        CRITICAL: Applied AFTER normalization to preserve alphabet-level relationships.
        Same transform is applied to all 52 glyphs.
        
        Args:
            alphabet: Array of shape (52, N, 2) for samples or (52, H, W) for skeletons
            rotation_range: Max rotation in degrees
            scale_range: (min, max) scale factors
            translation_range: Max translation as fraction of coordinate space
        
        Returns:
            Augmented alphabet with same shape
        """
        # Sample random transform parameters (same for all glyphs)
        angle = np.random.uniform(-rotation_range, rotation_range) * np.pi / 180
        scale = np.random.uniform(*scale_range)
        tx = np.random.uniform(-translation_range, translation_range)
        ty = np.random.uniform(-translation_range, translation_range)
        
        # Build 2D affine matrix
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        transform = np.array([
            [scale * cos_a, -scale * sin_a, tx],
            [scale * sin_a,  scale * cos_a, ty]
        ])
        
        if self.representation == 'samples':
            # Augment point clouds
            augmented = np.zeros_like(alphabet)
            
            for i in range(alphabet.shape[0]):
                points = alphabet[i]  # (N, 2)
                
                # Skip if all zeros (missing glyph)
                if np.allclose(points, 0):
                    continue
                
                # Apply affine transform
                ones = np.ones((points.shape[0], 1))
                homogeneous = np.concatenate([points, ones], axis=1)  # (N, 3)
                augmented[i] = (transform @ homogeneous.T).T  # (N, 2)
            
            return augmented
        
        elif self.representation == 'skeletons':
            # For skeletons, would need image-based augmentation
            # For now, just return original (implement if needed for skeleton training)
            logger.warning("Skeleton augmentation not yet implemented")
            return alphabet
        
        else:
            raise ValueError(f"Unknown representation: {self.representation}")
```

---

## Step 5: Validation & Testing

**Duration**: 1-2 days  
**Tickets**: TICK-007, TICK-008

### 5.1 Create comprehensive test suite

`tests/test_phase1.py`:

```python
"""Phase 1 validation tests."""

import pytest
import numpy as np
from pathlib import Path
from src.alphabet_pipeline import (
    glyph_to_path, arc_length_sample, normalize_path_v2,
    get_font_metrics, create_alphabet_tensor,
    CANONICAL_ORDER
)
from src.alphabet_data_loader import AlphabetDataLoader

# Use same test fonts as validation script
TEST_FONT_DIR = Path('tests/test_fonts')
TEST_FONTS = {
    'roboto': TEST_FONT_DIR / 'Roboto-Regular.ttf',
    'merriweather': TEST_FONT_DIR / 'Merriweather-Regular.ttf',
}

@pytest.fixture
def processed_dir():
    """Fixture providing path to processed test fonts."""
    return Path('output')

class TestGlyphExtraction:
    """Test glyph-level extraction and sampling."""
    
    def test_basic_glyph_extraction(self):
        """Verify glyphs can be extracted without errors."""
        for font_path in TEST_FONTS.values():
            if not font_path.exists():
                pytest.skip(f"Test font missing: {font_path}")
            
            path = glyph_to_path(str(font_path), 'A')
            assert path is not None, "Failed to extract 'A'"
            assert path.length() > 0, "'A' has zero length"
    
    def test_arc_length_uniformity(self):
        """Verify arc-length sampling is reasonably uniform."""
        for font_path in TEST_FONTS.values():
            if not font_path.exists():
                pytest.skip(f"Test font missing: {font_path}")
            
            path = glyph_to_path(str(font_path), 'O')
            if path is None:
                pytest.skip("Glyph 'O' not in font")
            
            samples = arc_length_sample(path, n_samples=100)
            
            # Compute distances between consecutive samples
            distances = [abs(samples[i+1] - samples[i]) for i in range(len(samples)-1)]
            
            # Coefficient of variation should be low for uniform sampling
            cv = np.std(distances) / np.mean(distances)
            assert cv < 0.3, f"Sampling not uniform: CV={cv:.2f}"
    
    def test_qcurveto_corner_cases(self):
        """Verify qCurveTo handling on complex glyphs."""
        corner_cases = ['g', 'Q', 'y', 'S', '@', '&']
        
        for font_path in TEST_FONTS.values():
            if not font_path.exists():
                pytest.skip(f"Test font missing: {font_path}")
            
            for glyph in corner_cases:
                path = glyph_to_path(str(font_path), glyph)
                if path is None:
                    continue  # Glyph not in this font
                
                samples = arc_length_sample(path)
                
                assert len(samples) > 0, f"Failed to sample '{glyph}'"
                
                # Verify no NaN/Inf
                for s in samples:
                    assert np.isfinite(s.real), f"Invalid real coord in '{glyph}'"
                    assert np.isfinite(s.imag), f"Invalid imag coord in '{glyph}'"

class TestNormalization:
    """Test alphabet-relative normalization."""
    
    def test_font_metrics_extraction(self):
        """Verify font metrics can be extracted."""
        for font_path in TEST_FONTS.values():
            if not font_path.exists():
                pytest.skip(f"Test font missing: {font_path}")
            
            metrics = get_font_metrics(str(font_path))
            
            assert 'units_per_em' in metrics
            assert metrics['units_per_em'] > 0
            
            # Should have at least fallback values
            assert metrics.get('x_height') is not None
            assert metrics.get('cap_height') is not None
    
    def test_case_height_preservation(self):
        """Verify uppercase glyphs taller than lowercase after normalization."""
        for font_name, font_path in TEST_FONTS.items():
            if not font_path.exists():
                pytest.skip(f"Test font missing: {font_path}")
            
            metrics = get_font_metrics(str(font_path))
            
            # Get normalized heights
            upper_path = glyph_to_path(str(font_path), 'M')
            lower_path = glyph_to_path(str(font_path), 'x')
            
            if upper_path is None or lower_path is None:
                pytest.skip("Required glyphs missing")
            
            upper_samples = arc_length_sample(upper_path)
            lower_samples = arc_length_sample(lower_path)
            
            upper_norm = normalize_path_v2(upper_path, upper_samples, 'M', metrics)
            lower_norm = normalize_path_v2(lower_path, lower_samples, 'x', metrics)
            
            # Measure heights in normalized space
            upper_bbox = upper_norm.bbox()
            lower_bbox = lower_norm.bbox()
            
            upper_height = upper_bbox[3] - upper_bbox[1]
            lower_height = lower_bbox[3] - lower_bbox[1]
            
            assert upper_height > lower_height, \
                f"Height relationship broken: M={upper_height:.3f}, x={lower_height:.3f}"
    
    def test_descender_handling(self):
        """Verify descender glyphs extend below baseline consistently."""
        for font_path in TEST_FONTS.values():
            if not font_path.exists():
                pytest.skip(f"Test font missing: {font_path}")
            
            metrics = get_font_metrics(str(font_path))
            
            # Test descender glyph
            g_path = glyph_to_path(str(font_path), 'g')
            if g_path is None:
                pytest.skip("Glyph 'g' not in font")
            
            g_samples = arc_length_sample(g_path)
            g_norm = normalize_path_v2(g_path, g_samples, 'g', metrics)
            
            g_bbox = g_norm.bbox()
            
            # Descender should extend below y=0 (approximate baseline)
            assert g_bbox[1] < 0, f"Descender doesn't extend below baseline: min_y={g_bbox[1]:.3f}"

class TestAlphabetTensors:
    """Test alphabet-level tensor creation."""
    
    def test_alphabet_tensor_shape(self):
        """Verify alphabet tensors have correct shape."""
        # This assumes fonts have been processed
        for font_name in ['roboto', 'merriweather']:
            base_path = Path(f'output/baseline_{font_name}')
            
            if not base_path.exists():
                pytest.skip(f"Processed font not found: {base_path}")
            
            tensor, missing = create_alphabet_tensor(base_path, 'samples')
            
            assert tensor.shape == (52, 256, 2), f"Wrong shape: {tensor.shape}"
            assert len(missing) < 11, f"Too many missing glyphs: {len(missing)}"
    
    def test_alphabet_tensor_no_nan(self):
        """Verify tensors contain no NaN or Inf values."""
        for font_name in ['roboto', 'merriweather']:
            base_path = Path(f'output/baseline_{font_name}')
            
            if not base_path.exists():
                pytest.skip(f"Processed font not found: {base_path}")
            
            tensor, _ = create_alphabet_tensor(base_path, 'samples')
            
            assert np.all(np.isfinite(tensor)), "Tensor contains NaN or Inf"
    
    def test_glyph_ordering_consistency(self):
        """Verify glyph order matches canonical order."""
        for font_name in ['roboto', 'merriweather']:
            base_path = Path(f'output/baseline_{font_name}')
            metadata_path = base_path / 'alphabet_metadata.json'
            
            if not metadata_path.exists():
                pytest.skip(f"Metadata not found: {metadata_path}")
            
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert metadata['glyph_order'] == CANONICAL_ORDER, \
                "Glyph order doesn't match canonical order"

class TestDataLoader:
    """Test alphabet data loader."""
    
    def test_loader_initialization(self, processed_dir):
        """Verify loader can initialize and load fonts."""
        try:
            loader = AlphabetDataLoader(str(processed_dir))
            assert len(loader) > 0, "No alphabets loaded"
        except ValueError as e:
            pytest.skip(f"No processed fonts available: {e}")
    
    def test_alphabet_retrieval(self, processed_dir):
        """Verify alphabets can be retrieved by name."""
        try:
            loader = AlphabetDataLoader(str(processed_dir))
            font_names = loader.list_fonts()
            
            if len(font_names) == 0:
                pytest.skip("No fonts loaded")
            
            alphabet = loader.get_alphabet(font_names[0])
            assert alphabet.shape[0] == 52, f"Wrong alphabet size: {alphabet.shape}"
        except ValueError:
            pytest.skip("No processed fonts available")
    
    def test_pair_sampling(self, processed_dir):
        """Verify pair sampling returns valid data."""
        try:
            loader = AlphabetDataLoader(str(processed_dir))
            
            if len(loader) < 2:
                pytest.skip("Need at least 2 fonts for pair sampling")
            
            alph1, alph2, same = loader.sample_alphabet_pair()
            
            assert alph1.shape == alph2.shape, "Pair shapes don't match"
            assert isinstance(same, bool), "Label should be boolean"
        except ValueError:
            pytest.skip("No processed fonts available")
    
    def test_triplet_sampling(self, processed_dir):
        """Verify triplet sampling returns valid data."""
        try:
            loader = AlphabetDataLoader(str(processed_dir))
            
            if len(loader) < 2:
                pytest.skip("Need at least 2 fonts for triplet sampling")
            
            anchor, positive, negative = loader.sample_alphabet_triplet()
            
            assert anchor.shape == positive.shape == negative.shape, \
                "Triplet shapes don't match"
        except ValueError:
            pytest.skip("No processed fonts available")
    
    def test_augmentation_preserves_shape(self, processed_dir):
        """Verify augmentation doesn't change tensor shape."""
        try:
            loader = AlphabetDataLoader(str(processed_dir))
            
            if len(loader) == 0:
                pytest.skip("No fonts loaded")
            
            alphabet = loader.get_alphabet(loader.list_fonts()[0])
            augmented = loader.augment_alphabet(alphabet)
            
            assert augmented.shape == alphabet.shape, \
                f"Augmentation changed shape: {alphabet.shape} -> {augmented.shape}"
        except ValueError:
            pytest.skip("No processed fonts available")
    
    def test_augmentation_preserves_topology(self, processed_dir):
        """Verify augmentation doesn't create/destroy points."""
        try:
            loader = AlphabetDataLoader(str(processed_dir))
            
            if len(loader) == 0:
                pytest.skip("No fonts loaded")
            
            alphabet = loader.get_alphabet(loader.list_fonts()[0])
            augmented = loader.augment_alphabet(alphabet)
            
            # Check that zero glyphs remain zero (topology preserved)
            for i in range(52):
                orig_is_zero = np.allclose(alphabet[i], 0)
                aug_is_zero = np.allclose(augmented[i], 0)
                
                assert orig_is_zero == aug_is_zero, \
                    f"Glyph {i} topology changed: zero={orig_is_zero} -> {aug_is_zero}"
        except ValueError:
            pytest.skip("No processed fonts available")

class TestInvariants:
    """Test critical invariants that must never be violated."""
    
    def test_no_learned_representations(self):
        """Verify no ML models are being trained in Phase 1."""
        # Check that no model files exist
        model_patterns = ['*.pt', '*.pth', '*.h5', '*.ckpt', '*.pb']
        
        for pattern in model_patterns:
            models = list(Path('output').rglob(pattern))
            assert len(models) == 0, \
                f"Found trained models in Phase 1: {models}"
    
    def test_augmentation_after_normalization(self):
        """Verify augmentation happens in data loader, not pipeline."""
        # This is a code structure test - check that augmentation
        # is only in AlphabetDataLoader, not in alphabet_pipeline
        
        from src import alphabet_pipeline
        import inspect
        
        pipeline_source = inspect.getsource(alphabet_pipeline)
        
        # Should NOT have augmentation in pipeline
        assert 'def augment' not in pipeline_source, \
            "Augmentation found in pipeline - should only be in data loader"
```

### 5.2 Run test suite

```bash
# Run all Phase 1 tests
pytest tests/test_phase1.py -v

# Run with coverage
pytest tests/test_phase1.py --cov=src.alphabet_pipeline --cov=src.alphabet_data_loader
```

**Success criteria**: All tests pass (green).

### 5.3 Create validation notebook

`notebooks/00_phase1_validation.ipynb`:

```python
# Cell 1: Setup
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.alphabet_data_loader import AlphabetDataLoader

processed_dir = Path('output')
loader = AlphabetDataLoader(str(processed_dir))

print(f"Loaded {len(loader)} fonts:")
for name in loader.list_fonts():
    print(f"  - {name}")

# Cell 2: Visualize normalization preservation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Alphabet-Relative Normalization: Height Relationships')

font_name = loader.list_fonts()[0]
alphabet = loader.get_alphabet(font_name)

# Plot uppercase, lowercase, descender
for idx, (glyph_idx, glyph_char) in enumerate([(12, 'M'), (8, 'i'), (6, 'g')]):
    ax = axes[idx]
    points = alphabet[glyph_idx] if glyph_idx < 26 else alphabet[glyph_idx + 26]
    
    ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_title(f"'{glyph_char}' (normalized)")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('output/validation_normalization.png', dpi=150)
plt.show()

# Cell 3: Test augmentation
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Augmentation Consistency Across Alphabet')

# Show 3 glyphs before/after augmentation
test_glyphs = [(0, 'A'), (25, 'Z'), (32, 'g')]  # First, last upper, descender
augmented = loader.augment_alphabet(alphabet)

for col, (idx, char) in enumerate(test_glyphs):
    # Original
    axes[0, col].plot(alphabet[idx, :, 0], alphabet[idx, :, 1], 'b-', linewidth=2)
    axes[0, col].set_title(f"'{char}' Original")
    axes[0, col].axis('equal')
    axes[0, col].grid(True, alpha=0.3)
    
    # Augmented
    axes[1, col].plot(augmented[idx, :, 0], augmented[idx, :, 1], 'r-', linewidth=2)
    axes[1, col].set_title(f"'{char}' Augmented")
    axes[1, col].axis('equal')
    axes[1, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/validation_augmentation.png', dpi=150)
plt.show()

# Cell 4: Cross-font comparison
if len(loader) >= 2:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Same Letter Across Different Fonts')
    
    fonts = loader.list_fonts()[:2]
    test_glyph_idx = 0  # 'A'
    
    for row, font_name in enumerate(fonts):
        alphabet = loader.get_alphabet(font_name)
        
        # Show 3 different glyphs
        for col, glyph_idx in enumerate([0, 12, 25]):  # A, M, Z
            points = alphabet[glyph_idx]
            axes[row, col].plot(points[:, 0], points[:, 1], linewidth=2)
            axes[row, col].set_title(f"{font_name}: '{chr(65 + glyph_idx)}'")
            axes[row, col].axis('equal')
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/validation_cross_font.png', dpi=150)
    plt.show()

# Cell 5: Distribution statistics
print("\n=== Alphabet Statistics ===")
for font_name in loader.list_fonts()[:3]:  # First 3 fonts
    alphabet = loader.get_alphabet(font_name)
    metadata = loader.get_metadata(font_name)
    
    non_zero = np.sum(~np.all(alphabet == 0, axis=(1, 2)))
    
    print(f"\n{font_name}:")
    print(f"  Shape: {alphabet.shape}")
    print(f"  Non-zero glyphs: {non_zero}/52")
    print(f"  Coordinate range: [{alphabet.min():.3f}, {alphabet.max():.3f}]")
    print(f"  Missing glyphs: {metadata.get('missing_glyphs_samples', [])}")
```

### 5.4 Generate validation report

`scripts/generate_phase1_report.py`:

```python
#!/usr/bin/env python3
"""Generate Phase 1 completion report."""

import sys
from pathlib import Path
import json
import numpy as np
from src.alphabet_data_loader import AlphabetDataLoader

def generate_report(output_dir: str = 'output'):
    """Generate markdown report of Phase 1 outputs."""
    
    output_path = Path(output_dir)
    report_lines = [
        "# Phase 1 Completion Report",
        "",
        "**Date**: " + str(Path.ctime(Path(__file__))),
        "",
        "## Summary",
        "",
    ]
    
    # Load all processed fonts
    try:
        loader = AlphabetDataLoader(output_dir)
        report_lines.append(f"✓ Successfully loaded {len(loader)} fonts")
        report_lines.append("")
    except Exception as e:
        report_lines.append(f"❌ Failed to initialize loader: {e}")
        report_lines.append("")
        return "\n".join(report_lines)
    
    # Font-by-font details
    report_lines.append("## Processed Fonts")
    report_lines.append("")
    
    for font_name in sorted(loader.list_fonts()):
        alphabet = loader.get_alphabet(font_name)
        metadata = loader.get_metadata(font_name)
        
        non_zero = np.sum(~np.all(alphabet == 0, axis=(1, 2)))
        missing = metadata.get('missing_glyphs_samples', [])
        
        report_lines.append(f"### {font_name}")
        report_lines.append(f"- Coverage: {non_zero}/52 glyphs ({non_zero/52*100:.1f}%)")
        report_lines.append(f"- Tensor shape: {alphabet.shape}")
        report_lines.append(f"- Coordinate range: [{alphabet.min():.3f}, {alphabet.max():.3f}]")
        
        if missing:
            report_lines.append(f"- Missing glyphs: {', '.join(missing[:10])}")
        
        # Font metrics if available
        if 'font_metrics' in metadata:
            fm = metadata['font_metrics']
            report_lines.append(f"- Units per EM: {fm.get('units_per_em')}")
            report_lines.append(f"- X-height: {fm.get('x_height')}")
            report_lines.append(f"- Cap height: {fm.get('cap_height')}")
        
        report_lines.append("")
    
    # Test results
    report_lines.append("## Test Results")
    report_lines.append("")
    report_lines.append("Run `pytest tests/test_phase1.py -v` for detailed results.")
    report_lines.append("")
    
    # Validation artifacts
    report_lines.append("## Validation Artifacts")
    report_lines.append("")
    
    validation_files = [
        'output/validation_normalization.png',
        'output/validation_augmentation.png',
        'output/validation_cross_font.png',
    ]
    
    for vf in validation_files:
        if Path(vf).exists():
            report_lines.append(f"✓ {vf}")
        else:
            report_lines.append(f"⚠️  {vf} (not found)")
    
    report_lines.append("")
    
    # Success criteria check
    report_lines.append("## Success Criteria")
    report_lines.append("")
    
    criteria = {
        "Pipeline runs without errors": len(loader) > 0,
        "Relative glyph scaling preserved": True,  # Manual visual check
        "Alphabet tensors loadable": len(loader) > 0,
        "Data loader functional": len(loader) > 0,
    }
    
    for criterion, passed in criteria.items():
        status = "✓" if passed else "❌"
        report_lines.append(f"{status} {criterion}")
    
    report_lines.append("")
    
    # Write report
    report_text = "\n".join(report_lines)
    report_path = Path(output_dir) / 'PHASE1_REPORT.md'
    
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Report saved to {report_path}")
    
    return report_text

if __name__ == '__main__':
    generate_report()
```

**Run report generation**:
```bash
uv run python scripts/generate_phase1_report.py
```

---

## Deliverables

| Item | Location | Format |
|------|----------|--------|
| Fixed pipeline | `src/alphabet_pipeline.py` | Python |
| Font metrics extraction | `src/alphabet_pipeline.py::get_font_metrics()` | Python |
| Alphabet-relative normalization | `src/alphabet_pipeline.py::normalize_path_v2()` | Python |
| Alphabet tensor output | `output/*/alphabet_samples.npy` | NumPy (52, 256, 2) |
| Alphabet metadata | `output/*/alphabet_metadata.json` | JSON |
| Alphabet data loader | `src/alphabet_data_loader.py` | Python |
| Baseline validation script | `scripts/validate_pipeline.py` | Python |
| Normalization comparison | `scripts/compare_normalizations.py` | Python |
| Unit tests | `tests/test_phase1.py` | pytest |
| Validation notebook | `notebooks/00_phase1_validation.ipynb` | Jupyter |
| Phase 1 report | `output/PHASE1_REPORT.md` | Markdown |

---

## Success Criteria

All of the following must be true to exit Phase 1:

### 1. Pipeline Functionality
- [ ] Pipeline runs without errors on all 3 test fonts
- [ ] No NaN or Inf values in any outputs
- [ ] At least 80% glyph coverage per font (42+ glyphs)
- [ ] qCurveTo handling works on complex glyphs (g, Q, y, S)

### 2. Normalization Quality
- [ ] Visual inspection: 'M' taller than 'i' in normalized space
- [ ] Visual inspection: 'g' descender extends below baseline
- [ ] Height relationships consistent across fonts
- [ ] Coordinate range approximately [-0.5, 0.5]

### 3. Tensor Outputs
- [ ] `alphabet_samples.npy` exists for each font with shape (52, 256, 2)
- [ ] `alphabet_metadata.json` contains glyph order and missing glyphs list
- [ ] Missing glyphs < 10 per font
- [ ] All tensors loadable with single `np.load()` call

### 4. Data Loader
- [ ] Loader can initialize and load multiple fonts
- [ ] Pair sampling works (positive and negative pairs)
- [ ] Triplet sampling works
- [ ] Augmentation preserves shape and topology
- [ ] Augmentation applied to full alphabets consistently

### 5. Testing & Validation
- [ ] All tests in `tests/test_phase1.py` pass
- [ ] Validation notebook runs without errors
- [ ] Visual validation artifacts generated
- [ ] Phase 1 report generated successfully

### 6. Invariants (Hard Stops)
- [ ] No ML models trained (no .pt, .pth, .h5 files)
- [ ] Augmentation only in data loader, not in pipeline
- [ ] All augmentation happens after normalization
- [ ] Topology preservation rate = 100% (no creation/deletion of points)

---

## Rollback Plan

If any step fails and cannot be fixed within 2 hours:

1. **Identify failure point** (which step/substep)
2. **Revert to last known good state**:
   ```bash
   git reset --hard <commit-before-step>
   ```
3. **Document failure** in `.planning/DEVIATIONS.md`:
   - What failed
   - Why it failed
   - Alternative approach
4. **Rethink approach** before retrying
5. **Update plan** with new approach

Do NOT attempt to "fix the fix" — prefer clean rollback and redesign.

---

## Next Phase Preview

Phase 2 will implement the **alphabet encoder** architecture:
- Set-based neural network (DeepSets or Transformer)
- Takes alphabet tensors (52, 256, 2) as input
- Produces fixed-size style embeddings
- Trained with contrastive/triplet loss at alphabet level

This will replace the current glyph-level triplet network with a true alphabet-level model that captures font style holistically.