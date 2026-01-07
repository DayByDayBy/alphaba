# ttfont, but it should work with otf too 

from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
import numpy as np
from pathlib import Path


class SamplingPen(BasePen):
    def __init__(self, glyphSet, samples_per_curve=20):
        super().__init__(glyphSet)
        self.coords = []
        self.samples = samples_per_curve
        self.last = (0, 0)

    def _moveTo(self, p):
        self.last = p
        self.coords.append(p)

    def _lineTo(self, p):
        self.coords.append(p)
        self.last = p

    def _curveToOne(self, p1, p2, p3):
        p0 = np.array(self.last)
        p1, p2, p3 = map(np.array, (p1, p2, p3))
        t = np.linspace(0, 1, self.samples)
        pts = (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3
        self.coords.extend(map(tuple, pts))
        self.last = tuple(p3)

    def _closePath(self):
        pass


def extract_font_coords(font_path, chars, samples_per_curve=20, normalize=True):
    """extracts normalized coordinate arrays for given characters in a font."""
    font = TTFont(font_path)
    cmap = font.getBestCmap()
    glyph_set = font.getGlyphSet()

    char_arrays = {}
    for ch in chars:
        glyph_name = cmap.get(ord(ch))
        if glyph_name is None:
            continue
        glyph = glyph_set[glyph_name]
        pen = SamplingPen(glyph_set, samples_per_curve)
        glyph.draw(pen)
        coords = np.array(pen.coords)
        if len(coords) == 0:
            continue

        if normalize:
            coords -= coords.mean(axis=0)
            max_abs = np.abs(coords).max() or 1
            coords /= max_abs

        char_arrays[ch] = coords

    return char_arrays
