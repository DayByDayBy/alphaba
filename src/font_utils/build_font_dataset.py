import json
import numpy as np
from pathlib import Path
from font_extractor import extract_font_coords

DATASET_DIR = Path("alphabet_dataset")
FONTS_DIR = Path("fonts")

# Example: collect alphabets (you can expand these)
ALPHABETS = {
    "latin": [chr(c) for c in range(0x41, 0x5A+1)] + [chr(c) for c in range(0x61, 0x7A+1)],
    "georgian": [chr(c) for c in range(0x10D0, 0x10FF)],
    # add others as needed
}

def save_glyph_data(font_name, alphabet_name, glyph_dict):
    out_dir = DATASET_DIR / alphabet_name / font_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for ch, coords in glyph_dict.items():
        ch_code = f"U+{ord(ch):04X}"
        np.save(out_dir / f"{ch_code}.npy", coords)
    # Optional: summary metadata
    meta = {ch: len(coords) for ch, coords in glyph_dict.items()}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def main():
    for font_path in FONTS_DIR.glob("*.ttf"):
        font_name = font_path.stem
        for alphabet_name, chars in ALPHABETS.items():
            glyphs = extract_font_coords(font_path, chars)
            if glyphs:
                save_glyph_data(font_name, alphabet_name, glyphs)
                print(f"Saved {len(glyphs)} glyphs from {font_name} ({alphabet_name})")

if __name__ == "__main__":
    main()
