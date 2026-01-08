#!/usr/bin/env python3
"""
Register the new font files from font_samples directory
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.extensible_alphabet_loader import ExtensibleAlphabetLoader
from pathlib import Path

def register_font_directory(font_dir_path, script_type, alphabet_name):
    """Register all TTF files from a directory"""
    loader = ExtensibleAlphabetLoader()
    font_dir = Path(font_dir_path)
    
    if not font_dir.exists():
        print(f"Font directory not found: {font_dir_path}")
        return
    
    # Find TTF files
    ttf_files = list(font_dir.glob("*.ttf"))
    
    if not ttf_files:
        print(f"No TTF files found in {font_dir_path}")
        return
    
    print(f"Found {len(ttf_files)} TTF files in {font_dir.name}")
    
    # Register each TTF file
    for i, ttf_file in enumerate(ttf_files):
        font_name = f"{alphabet_name}_{i+1}"
        
        try:
            loader.register_custom_font(font_name, str(ttf_file), script_type)
            print(f"  ✓ Registered: {font_name}")
        except Exception as e:
            print(f"  ✗ Failed to register {ttf_file.name}: {e}")

def update_character_mappings():
    """Add character mappings for new scripts"""
    loader = ExtensibleAlphabetLoader()
    
    # Add mappings for new scripts
    new_mappings = {
        'georgian_modern': {
            'Ⴀ': 'A', 'Ⴁ': 'B', 'Ⴂ': 'G', 'Ⴃ': 'D', 'Ⴄ': 'E',
            'Ⴅ': 'V', 'Ⴆ': 'Z', 'Ⴇ': 'T', 'Ⴈ': 'I', 'Ⴉ': 'K',
            'Ⴊ': 'L', 'Ⴋ': 'M', 'Ⴌ': 'N', 'Ⴍ': 'H', 'Ⴎ': 'P',
            'Ⴏ': 'ZH', 'Ⴐ': 'R', 'Ⴑ': 'S', 'Ⴒ': 'T', 'Ⴓ': 'U',
            'Ⴔ': 'PH', 'Ⴕ': 'KH', 'Ⴖ': 'GH', 'Ⴗ': 'Q', 'Ⴘ': 'Y',
            'Ⴙ': 'C', 'Ⴚ': 'CH', 'Ⴛ': 'J', 'Ⴜ': 'X', 'Ⴝ': 'JH',
            'Ⴞ': 'H', 'Ⴟ': 'E', 'Ⴠ': 'W', 'Ⴡ': 'H', 'Ⴢ': 'Y'
        },
        'avestan': {
            '𐎠': 'A', '𐎡': 'A', '𐎢': 'A', '𐎣': 'A', '𐎤': 'A',
            '𐎥': 'K', '𐎦': 'G', '𐎧': 'G', '𐎨': 'CH', '𐎩': 'CH',
            '𐎪': 'T', '𐎫': 'T', '𐎬': 'D', '𐎭': 'D', '𐎮': 'D',
            '𐎯': 'TH', '𐎰': 'P', '𐎱': 'P', '𐎲': 'F', '𐎳': 'B',
            '𐎴': 'B', '𐎵': 'N', '𐎶': 'N', '𐎷': 'M', '𐎸': 'M',
            '𐎹': 'Y', '𐎺': 'V', '𐎻': 'R', '𐎼': 'L', '𐎽': 'S',
            '𐎾': 'SH', '𐎿': 'Z', '𐏀': 'SH', '𐏁': 'S', '𐏂': 'ZH',
            '𐏃': 'H', '𐏄': 'H'
        },
        'deseret': {
            '𐐀': 'A', '𐐁': 'B', '𐐂': 'C', '𐐃': 'D', '𐐄': 'E',
            '𐐅': 'F', '𐐆': 'G', '𐐇': 'H', '𐐈': 'I', '𐐉': 'J',
            '𐐊': 'K', '𐐋': 'L', '𐐌': 'M', '𐐍': 'N', '𐐎': 'O',
            '𐐏': 'P', '𐐐': 'Q', '𐐑': 'R', '𐐒': 'S', '𐐓': 'T',
            '𐐔': 'U', '𐐕': 'V', '𐐖': 'W', '𐐗': 'X', '𐐘': 'Y',
            '𐐙': 'Z', '𐐚': 'AW', '𐐛': 'AY', '𐐜': 'EE', '𐐝': 'IE',
            '𐐞': 'OE', '𐐟': 'OO', '𐐠': 'U', '𐐡': 'OI', '𐐢': 'IY',
            '𐐣': 'E', '𐐤': 'A', '𐐥': 'O', '𐐦': 'W', '𐐧': 'Y',
            '𐐨': 'H', '𐐩': 'P', '𐐪': 'I', '𐐫': 'K', '𐐬': 'NG',
            '𐐭': 'L', '𐐮': 'M', '𐐯': 'N', '𐐰': 'G', '𐐱': 'R',
            '𐐲': 'S', '𐐳': 'T', '𐐴': 'D', '𐐵': 'SH', '𐐶': 'TH',
            '𐐷': 'TS', '𐐸': 'Z', '𐐹': 'CH', '𐐺': 'J', '𐐻': 'F'
        },
        'osage': {
            '𐒰': 'A', '𐒱': 'B', '𐒲': 'CH', '𐒳': 'D', '𐒴': 'E',
            '𐒵': 'F', '𐒶': 'G', '𐒷': 'H', '𐒸': 'I', '𐒹': 'K',
            '𐒺': 'L', '𐒻': 'M', '𐒼': 'N', '𐒽': 'O', '𐒾': 'P',
            '𐒿': 'R', '𐓀': 'S', '𐓁': 'SH', '𐓂': 'T', '𐓃': 'TH',
            '𐓄': 'U', '𐓅': 'V', '𐓆': 'W', '𐓇': 'X', '𐓈': 'Y',
            '𐓉': 'Z', '𐓊': 'ZH', '𐓋': 'BR', '𐓌': 'ST', '𐓍': 'SK'
        }
    }
    
    # Update character mappings
    loader.character_mappings.update(new_mappings)
    loader.save_character_mappings()
    
    print("Updated character mappings for new scripts")

def main():
    font_samples_dir = Path("font_samples")
    
    # Register each font directory
    font_configs = [
        ("Google_Sans", "latin", "Google Sans"),
        ("Noto_Sans_Armenian", "armenian", "Noto Sans Armenian"),
        ("Noto_Sans_Avestan", "avestan", "Noto Sans Avestan"),
        ("Noto_Sans_Deseret", "deseret", "Noto Sans Deseret"),
        ("Noto_Sans_Georgian", "georgian_modern", "Noto Sans Georgian"),
        ("Noto_Sans_Osage", "osage", "Noto Sans Osage"),
        ("Noto_Serif_Georgian", "georgian_modern", "Noto Serif Georgian")
    ]
    
    for font_dir, script_type, alphabet_name in font_configs:
        font_path = font_samples_dir / font_dir
        register_font_directory(font_path, script_type, alphabet_name)
    
    # Update character mappings
    update_character_mappings()
    
    # Show summary
    loader = ExtensibleAlphabetLoader()
    print(f"\nTotal registered alphabets: {len(loader.get_enabled_alphabets())}")
    loader.list_registered_alphabets()

if __name__ == "__main__":
    main()
