#!/usr/bin/env python3
"""
Add more diverse fonts to the alphabet registry
Focus on fonts with distinct visual characteristics
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.extensible_alphabet_loader import ExtensibleAlphabetLoader
import matplotlib.font_manager as fm
from PIL import ImageFont

def find_diverse_fonts():
    """Find fonts with distinct visual characteristics"""
    loader = ExtensibleAlphabetLoader()
    
    # Test characters for key scripts
    test_chars = {
        'greek': ['Α', 'Β', 'Γ'],
        'armenian': ['Ա', 'Բ', 'Գ'],
        'latin': ['A', 'B', 'C']
    }
    
    diverse_fonts = {
        'serif': [],
        'sans_serif': [],
        'decorative': [],
        'monospace': []
    }
    
    font_paths = fm.findSystemFonts()
    
    for font_path in font_paths[:100]:  # Check first 100 fonts
        try:
            font_name = fm.FontProperties(fname=font_path).get_name()
            pil_font = ImageFont.truetype(font_path, 20)
            
            # Test if it renders our characters
            works = True
            for script, chars in test_chars.items():
                for char in chars:
                    try:
                        bbox = pil_font.getbbox(char)
                        if bbox == (0, 0, 0, 0):
                            works = False
                            break
                    except:
                        works = False
                        break
                if not works:
                    break
            
            if works:
                # Categorize by font characteristics
                font_lower = font_name.lower()
                if any(keyword in font_lower for keyword in ['serif', 'times', 'georgia', 'garamond']):
                    diverse_fonts['serif'].append((font_path, font_name))
                elif any(keyword in font_lower for keyword in ['mono', 'courier', 'consolas']):
                    diverse_fonts['monospace'].append((font_path, font_name))
                elif any(keyword in font_lower for keyword in ['decorative', 'display', 'brush', 'script']):
                    diverse_fonts['decorative'].append((font_path, font_name))
                else:
                    diverse_fonts['sans_serif'].append((font_path, font_name))
                    
        except:
            continue
    
    return diverse_fonts

def register_diverse_fonts():
    """Register diverse fonts for each script"""
    loader = ExtensibleAlphabetLoader()
    diverse_fonts = find_diverse_fonts()
    
    print("Found diverse fonts:")
    for category, fonts in diverse_fonts.items():
        print(f"  {category}: {len(fonts)} fonts")
    
    # Register a few from each category
    font_counter = 1
    
    for script in ['greek', 'armenian', 'latin']:
        print(f"\nRegistering {script} fonts:")
        
        # Try to register one from each category
        for category, fonts in diverse_fonts.items():
            if fonts and font_counter <= 10:  # Limit to avoid too many
                font_path, font_name = fonts[0]  # Take first from category
                
                # Clean font name for registry
                clean_name = f"{script.title()} {category.title()} {font_counter}"
                clean_name = clean_name.replace(' ', '_')
                
                try:
                    loader.register_system_font(clean_name, font_path, script)
                    print(f"  ✓ {clean_name}: {font_name}")
                    font_counter += 1
                except Exception as e:
                    print(f"  ✗ Failed to register {font_name}: {e}")
    
    print(f"\nTotal registered alphabets: {len(loader.get_enabled_alphabets())}")

if __name__ == "__main__":
    register_diverse_fonts()
