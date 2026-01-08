"""
Alphabet Geometry Pipeline — Phase 0
Goal: Convert font files (.ttf) into explicit, normalized geometric 
and topological representations suitable for learning alphabet-level structure.

Core Principles:
- No machine learning
- No glyph reuse
- Failures are signals, not bugs
- Geometric invariance over pixel convenience
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import numpy as np
from collections import Counter

# Font processing
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen

# Vector geometry
from svgpathtools import parse_path, Path as SVGPath, Line, CubicBezier, QuadraticBezier, Arc

# Rasterization
import cairosvg
from io import BytesIO
from PIL import Image

# Skeletonization
import cv2
from skimage.morphology import skeletonize
from scipy import ndimage

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SAMPLE_COUNT = 256
DEFAULT_RASTER_SIZE = 512
MIN_SKELETON_PIXELS = 10
GLYPH_SET = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + \
            [chr(i) for i in range(ord('a'), ord('z') + 1)]


# ============================================================================
# 1. FONT PARSING
# ============================================================================

def glyph_to_path(ttf_path: str, glyph_name: str) -> Optional[SVGPath]:
    """
    Extract vector path from TTF glyph.
    
    Composite glyphs are fully resolved into concrete contours.
    Returns a single Path object that may contain multiple subpaths.
    
    Args:
        ttf_path: Path to .ttf file
        glyph_name: Glyph name (e.g. 'A', 'a')
    
    Returns:
        svgpathtools.Path or None if glyph doesn't exist
    """
    try:
        font = TTFont(ttf_path)
        glyph_set = font.getGlyphSet()
        
        if glyph_name not in glyph_set:
            logger.warning(f"Glyph '{glyph_name}' not found in font")
            return None
        
        # Use RecordingPen to capture path operations
        pen = RecordingPen()
        glyph_set[glyph_name].draw(pen)
        
        # Convert to SVG path data
        path_data = []
        for op, args in pen.value:
            if op == 'moveTo':
                x, y = args[0]
                path_data.append(f"M {x} {y}")
            elif op == 'lineTo':
                x, y = args[0]
                path_data.append(f"L {x} {y}")
            elif op == 'qCurveTo':
                # QuadraticBezier
                points = ' '.join(f"{x} {y}" for x, y in args)
                path_data.append(f"Q {points}")
            elif op == 'curveTo':
                # CubicBezier
                points = ' '.join(f"{x} {y}" for x, y in args)
                path_data.append(f"C {points}")
            elif op == 'closePath':
                path_data.append("Z")
        
        if not path_data:
            logger.warning(f"Glyph '{glyph_name}' has no path data")
            return None
        
        svg_path_string = ' '.join(path_data)
        path = parse_path(svg_path_string)
        
        return path
        
    except Exception as e:
        logger.error(f"Failed to extract path for '{glyph_name}': {e}")
        return None


# ============================================================================
# 2. ARC-LENGTH SAMPLING
# ============================================================================

def arc_length_sample(path: SVGPath, n_samples: int = DEFAULT_SAMPLE_COUNT) -> List[complex]:
    """
    Sample path uniformly by arc length (not parameter space).
    
    Provides geometric invariance - samples are evenly spaced along
    the actual curve, not in Bezier parameter space.
    
    Args:
        path: svgpathtools.Path object
        n_samples: Number of sample points
    
    Returns:
        List of complex points
    """
    try:
        total_length = path.length()
        
        if total_length == 0:
            logger.warning("Path has zero length")
            return [path.point(0)]
        
        target_lengths = np.linspace(0, total_length, n_samples)
        samples = []
        
        for target_length in target_lengths:
            t = path.ilength(target_length)
            samples.append(path.point(t))
        
        return samples
        
    except Exception as e:
        logger.error(f"Arc-length sampling failed: {e}")
        return []


# ============================================================================
# 3. GEOMETRIC NORMALIZATION (CANONICAL)
# ============================================================================

def normalize_path(path: SVGPath, samples: List[complex]) -> Optional[SVGPath]:
    """
    Normalize path to unit box with aspect ratio preservation.
    
    Normalization is glyph-local, not alphabet-relative.
    
    Process:
    1. Translate to origin
    2. Scale to unit box (preserving aspect ratio)
    
    Args:
        path: svgpathtools.Path object
        samples: Arc-length samples for bounds computation
    
    Returns:
        Normalized Path or None on failure
    """
    try:
        if not samples:
            logger.warning("Cannot normalize with empty samples")
            return None
        
        xs = [p.real for p in samples]
        ys = [p.imag for p in samples]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = max_x - min_x if max_x > min_x else 1
        height = max_y - min_y if max_y > min_y else 1
        
        scale = max(width, height)
        
        # Translate to origin → scale to unit box
        offset = complex(-min_x, -min_y)
        translated = path.translated(offset)
        normalized = translated.scaled(1/scale)
        
        return normalized
        
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        return None


# ============================================================================
# 4. RASTERIZATION (ANALYSIS ONLY)
# ============================================================================

def path_to_bitmap(
    path: SVGPath,
    size: int = DEFAULT_RASTER_SIZE,
    fill: bool = True,
    stroke_width: Optional[float] = None
) -> Optional[np.ndarray]:
    """
    Render path to binary bitmap for skeletonization.
    
    Glyphs are rendered as filled regions. A minimal stroke may 
    optionally be added to stabilize extremely thin features, but 
    filled rendering remains primary.
    
    Stroke is additive, never substitutive.
    Stroke width is specified in output pixel space, not font units.
    
    Args:
        path: Normalized svgpathtools.Path
        size: Output image size (square)
        fill: Whether to fill the path
        stroke_width: Optional stroke width in pixels
    
    Returns:
        Binary numpy array or None on failure
    """
    try:
        # Convert path to SVG path data string
        path_d = path.d()
        
        # Build SVG
        fill_attr = 'fill="black"' if fill else 'fill="none"'
        stroke_attr = f'stroke="black" stroke-width="{stroke_width}"' if stroke_width else ''
        
        svg = f'''
        <svg xmlns="http://www.w3.org/2000/svg" 
             width="{size}" height="{size}" 
             viewBox="0 0 1 1">
          <rect width="1" height="1" fill="white"/>
          <path d="{path_d}" {fill_attr} {stroke_attr}/>
        </svg>
        '''
        
        # Render to PNG
        png_data = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
        
        # Load as grayscale
        img = Image.open(BytesIO(png_data)).convert('L')
        img_array = np.array(img)
        
        # Binarize (invert: black glyph on white background → white glyph on black)
        _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)
        
        return binary
        
    except Exception as e:
        logger.error(f"Rasterization failed: {e}")
        return None


# ============================================================================
# 5. SKELETONIZATION
# ============================================================================

def skeletonize_glyph(glyph_name: str, binary_img: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract medial axis skeleton from binary glyph image.
    
    Skeletonization operates on connected components independently.
    Topology is preserved only insofar as the raster permits.
    
    Args:
        glyph_name: Glyph identifier for logging
        binary_img: Binary image (white=foreground)
    
    Returns:
        Skeleton as uint8 array or None on failure
    """
    try:
        # Ensure boolean input
        binary = binary_img > 0
        
        # Skeletonize
        skeleton = skeletonize(binary)
        skeleton_uint8 = skeleton.astype(np.uint8) * 255
        
        # Structural validation
        skeleton_pixels = skeleton.sum()
        if skeleton_pixels < MIN_SKELETON_PIXELS:
            logger.warning(
                f"{glyph_name}: skeleton too small ({skeleton_pixels} pixels)"
            )
            return None
        
        return skeleton_uint8
        
    except Exception as e:
        logger.error(f"Skeletonization failed for {glyph_name}: {e}")
        return None


# ============================================================================
# 6. TOPOLOGY ANALYSIS
# ============================================================================

def analyze_skeleton_topology(skeleton: np.ndarray) -> Dict[str, Any]:
    """
    Extract topological features from skeleton.
    
    Returns:
        Dictionary with:
        - skeleton_pixels: total skeleton pixels
        - n_components: number of disconnected components
        - endpoints: number of endpoint pixels
        - junctions: number of junction pixels
        - junction_degrees: histogram of junction connectivity
    """
    # Count pixels
    skeleton_pixels = int(skeleton.sum() / 255)
    
    # Count connected components
    labeled, n_components = ndimage.label(skeleton > 0)
    
    # Compute local neighborhood sums
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(
        (skeleton > 0).astype(np.uint8), 
        -1, 
        kernel
    )
    
    # Mask to only skeleton pixels
    neighbor_count = neighbor_count * (skeleton > 0)
    
    # Endpoints: 1 neighbor
    endpoints = int((neighbor_count == 1).sum())
    
    # Junctions: 3+ neighbors
    junctions = int((neighbor_count >= 3).sum())
    
    # Junction degree histogram
    junction_pixels = neighbor_count[neighbor_count >= 3]
    if len(junction_pixels) > 0:
        degree_counts = Counter(junction_pixels.tolist())
        junction_degrees = {
            str(k): int(v) for k, v in sorted(degree_counts.items())
        }
    else:
        junction_degrees = {}
    
    return {
        'skeleton_pixels': skeleton_pixels,
        'n_components': int(n_components),
        'endpoints': endpoints,
        'junctions': junctions,
        'junction_degrees': junction_degrees
    }


# ============================================================================
# 7. STATISTICS COMPUTATION
# ============================================================================

def compute_glyph_stats(
    glyph_name: str,
    path: SVGPath,
    samples: List[complex],
    skeleton: Optional[np.ndarray]
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a single glyph.
    """
    xs = [p.real for p in samples]
    ys = [p.imag for p in samples]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    
    stats = {
        'glyph': glyph_name,
        'bounding_box': [min_x, min_y, max_x, max_y],
        'aspect_ratio': width / height if height > 0 else 0,
        'arc_length': path.length(),
        'n_samples': len(samples)
    }
    
    if skeleton is not None:
        topology = analyze_skeleton_topology(skeleton)
        stats.update(topology)
    else:
        stats['skeleton_failed'] = True
    
    return stats


def compute_alphabet_stats(glyph_stats: List[Dict[str, Any]], font_name: str) -> Dict[str, Any]:
    """
    Compute alphabet-level statistics from per-glyph data.
    """
    successful = [s for s in glyph_stats if not s.get('skeleton_failed', False)]
    failed = [s for s in glyph_stats if s.get('skeleton_failed', False)]
    
    if not successful:
        return {
            'font_name': font_name,
            'pipeline_version': 'phase0_v1',
            'n_glyphs': 0,
            'failed_glyphs': [s['glyph'] for s in failed],
            'error': 'All glyphs failed'
        }
    
    # Aggregate junction degrees
    all_junction_degrees = Counter()
    for s in successful:
        if 'junction_degrees' in s:
            for degree, count in s['junction_degrees'].items():
                all_junction_degrees[degree] += count
    
    return {
        'font_name': font_name,
        'pipeline_version': 'phase0_v1',
        'n_glyphs': len(successful),
        'mean_aspect_ratio': float(np.mean([s['aspect_ratio'] for s in successful])),
        'std_aspect_ratio': float(np.std([s['aspect_ratio'] for s in successful])),
        'mean_arc_length': float(np.mean([s['arc_length'] for s in successful])),
        'std_arc_length': float(np.std([s['arc_length'] for s in successful])),
        'mean_skeleton_pixels': float(np.mean([s['skeleton_pixels'] for s in successful])),
        'junction_degree_histogram': dict(all_junction_degrees),
        'failed_glyphs': [s['glyph'] for s in failed]
    }


# ============================================================================
# 8. BATCH PROCESSING
# ============================================================================

def process_font(
    ttf_path: str,
    output_dir: str,
    glyph_set: List[str] = GLYPH_SET
) -> Dict[str, Any]:
    """
    Process entire font through pipeline.
    
    Creates output directory structure and saves all artifacts.
    
    Args:
        ttf_path: Path to .ttf file
        output_dir: Base output directory
        glyph_set: List of glyph names to process
    
    Returns:
        Alphabet-level statistics dictionary
    """
    font_name = Path(ttf_path).stem
    logger.info(f"Processing font: {font_name}")
    
    # Create output directories
    base_path = Path(output_dir) / font_name
    (base_path / 'vectors').mkdir(parents=True, exist_ok=True)
    (base_path / 'samples').mkdir(parents=True, exist_ok=True)
    (base_path / 'rasters').mkdir(parents=True, exist_ok=True)
    (base_path / 'skeletons').mkdir(parents=True, exist_ok=True)
    
    glyph_stats_list = []
    
    for glyph_name in glyph_set:
        logger.info(f"  Processing glyph: {glyph_name}")
        
        # 1. Extract path
        path = glyph_to_path(ttf_path, glyph_name)
        if path is None:
            continue
        
        # 2. Arc-length sampling
        samples = arc_length_sample(path)
        if not samples:
            continue
        
        # 3. Normalize
        normalized_path = normalize_path(path, samples)
        if normalized_path is None:
            continue
        
        # Save vector path
        with open(base_path / 'vectors' / f'{glyph_name}.svgpath.txt', 'w') as f:
            f.write(normalized_path.d())
        
        # Save samples
        samples_array = np.array([[p.real, p.imag] for p in samples])
        np.save(base_path / 'samples' / f'{glyph_name}_samples.npy', samples_array)
        
        # 4. Rasterize
        bitmap = path_to_bitmap(normalized_path)
        if bitmap is None:
            continue
        
        # Save raster
        Image.fromarray(bitmap).save(base_path / 'rasters' / f'{glyph_name}.png')
        
        # 5. Skeletonize
        skeleton = skeletonize_glyph(glyph_name, bitmap)
        
        if skeleton is not None:
            # Save skeleton
            Image.fromarray(skeleton).save(base_path / 'skeletons' / f'{glyph_name}.png')
        
        # 6. Compute stats
        stats = compute_glyph_stats(glyph_name, normalized_path, samples, skeleton)
        glyph_stats_list.append(stats)
    
    # Compute alphabet-level statistics
    alphabet_stats = compute_alphabet_stats(glyph_stats_list, font_name)
    
    # Save metadata
    with open(base_path / 'metadata.json', 'w') as f:
        json.dump({
            'alphabet_stats': alphabet_stats,
            'glyph_stats': glyph_stats_list
        }, f, indent=2)
    
    logger.info(f"Completed font: {font_name}")
    logger.info(f"  Successful: {alphabet_stats['n_glyphs']}")
    logger.info(f"  Failed: {len(alphabet_stats.get('failed_glyphs', []))}")
    
    return alphabet_stats


def process_font_directory(
    fonts_dir: str,
    output_dir: str,
    glyph_set: List[str] = GLYPH_SET
) -> List[Dict[str, Any]]:
    """
    Process all .ttf files in a directory.
    
    Args:
        fonts_dir: Directory containing .ttf files
        output_dir: Base output directory
        glyph_set: List of glyph names to process
    
    Returns:
        List of alphabet-level statistics for all fonts
    """
    fonts_path = Path(fonts_dir)
    ttf_files = list(fonts_path.glob('*.ttf'))
    
    logger.info(f"Found {len(ttf_files)} font files")
    
    results = []
    for ttf_path in ttf_files:
        try:
            stats = process_font(str(ttf_path), output_dir, glyph_set)
            results.append(stats)
        except Exception as e:
            logger.error(f"Failed to process {ttf_path.name}: {e}")
    
    return results


# ============================================================================
# 9. MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <fonts_dir> <output_dir>")
        sys.exit(1)
    
    fonts_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    logger.info("=" * 80)
    logger.info("Alphabet Geometry Pipeline - Phase 0")
    logger.info("=" * 80)
    
    results = process_font_directory(fonts_dir, output_dir)
    
    logger.info("=" * 80)
    logger.info(f"Pipeline complete. Processed {len(results)} fonts.")
    logger.info("=" * 80)