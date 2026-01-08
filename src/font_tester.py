#!/usr/bin/env python3
"""
Test font availability and character rendering for alphabet training
"""

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class FontTester:
    def __init__(self):
        self.available_fonts = {}
        self.test_characters = {
            'Latin': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'Greek': 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ',
            'Armenian': 'ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖ',
            'Georgian': 'ႠႡႢႣႤႥႦႧႨႩႪႫႬႭႮႯႰႱႲႳႴႵႶႷႸႹႺႻႼႽႾႿ',
        }
    
    def find_fonts_for_script(self, script_name):
        """Find fonts that can render characters for a specific script"""
        characters = self.test_characters.get(script_name, '')
        suitable_fonts = []
        
        # Get all system fonts
        font_paths = fm.findSystemFonts()
        
        for font_path in font_paths[:50]:  # Limit for testing
            try:
                font = ImageFont.truetype(font_path, 20)
                can_render = True
                
                # Test if font can render key characters
                for char in characters[:5]:  # Test first 5 chars
                    try:
                        bbox = font.getbbox(char)
                        if bbox == (0, 0, 0, 0):
                            can_render = False
                            break
                    except:
                        can_render = False
                        break
                
                if can_render:
                    suitable_fonts.append(font_path)
            except:
                continue
        
        return suitable_fonts
    
    def test_all_scripts(self):
        """Test font availability for all scripts"""
        for script_name in self.test_characters.keys():
            fonts = self.find_fonts_for_script(script_name)
            self.available_fonts[script_name] = fonts
            print(f"{script_name}: {len(fonts)} suitable fonts")
    
    def render_sample(self, font_path, characters, output_path):
        """Render sample characters from a font"""
        img = Image.new('RGB', (800, 200), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype(font_path, 40)
            x, y = 20, 50
            
            for i, char in enumerate(characters[:20]):  # First 20 chars
                draw.text((x, y), char, fill='black', font=font)
                x += 35
                
                if (i + 1) % 10 == 0:  # New line after 10 chars
                    x = 20
                    y += 60
            
            img.save(output_path)
            return True
        except Exception as e:
            print(f"Error rendering {font_path}: {e}")
            return False
    
    def generate_font_samples(self):
        """Generate sample images for available fonts"""
        os.makedirs('font_samples', exist_ok=True)
        
        for script_name, font_paths in self.available_fonts.items():
            characters = self.test_characters[script_name]
            
            for i, font_path in enumerate(font_paths[:3]):  # Top 3 fonts
                font_name = os.path.basename(font_path).split('.')[0]
                output_path = f'font_samples/{script_name}_{font_name}.png'
                
                if self.render_sample(font_path, characters, output_path):
                    print(f"Generated: {output_path}")

if __name__ == "__main__":
    tester = FontTester()
    tester.test_all_scripts()
    tester.generate_font_samples()
