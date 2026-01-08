"""
Unicode-based alphabet loader for Noto fonts
Uses Unicode ranges to auto-enumerate characters - no manual mappings needed
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from collections import defaultdict

# Unicode block definitions for alphabets
UNICODE_BLOCKS = {
    'latin': {
        'name': 'Latin',
        'range': (0x0041, 0x005A),  # A-Z uppercase
        'description': 'Basic Latin uppercase'
    },
    'greek': {
        'name': 'Greek',
        'range': (0x0391, 0x03A9),  # Α-Ω
        'description': 'Greek uppercase',
        'skip': [0x03A2]  # Skip reserved
    },
    'coptic': {
        'name': 'Coptic',
        'range': (0x2C80, 0x2CFF),
        'description': 'Coptic alphabet'
    },
    'georgian': {
        'name': 'Georgian',
        'range': (0x10A0, 0x10FF),
        'description': 'Georgian Asomtavruli and Mkhedruli'
    },
    'armenian': {
        'name': 'Armenian',
        'range': (0x0531, 0x0556),  # Uppercase Armenian
        'description': 'Armenian uppercase'
    },
    'caucasian_albanian': {
        'name': 'Caucasian Albanian',
        'range': (0x10530, 0x1056F),
        'description': 'Ancient Caucasian Albanian script'
    },
    'old_permic': {
        'name': 'Old Permic',
        'range': (0x10350, 0x1037F),
        'description': 'Old Permic (Abur) script'
    },
    'lycian': {
        'name': 'Lycian',
        'range': (0x10280, 0x1029F),
        'description': 'Ancient Lycian script'
    },
    'elbasan': {
        'name': 'Elbasan',
        'range': (0x10500, 0x1052F),
        'description': 'Elbasan Albanian script'
    },
    'osage': {
        'name': 'Osage',
        'range': (0x104B0, 0x104FF),
        'description': 'Osage script'
    },
    'deseret': {
        'name': 'Deseret',
        'range': (0x10400, 0x1044F),
        'description': 'Deseret alphabet'
    },
    'avestan': {
        'name': 'Avestan',
        'range': (0x10B00, 0x10B3F),
        'description': 'Avestan script'
    },
    'old_hungarian': {
        'name': 'Old Hungarian',
        'range': (0x10C80, 0x10CFF),
        'description': 'Old Hungarian (Székely-Hungarian Rovás)'
    },
    'tifinagh': {
        'name': 'Tifinagh',
        'range': (0x2D30, 0x2D7F),
        'description': 'Tifinagh (Berber) script'
    },
    'cyrillic': {
        'name': 'Cyrillic',
        'range': (0x0410, 0x042F),  # А-Я uppercase
        'description': 'Cyrillic uppercase'
    }
}

class UnicodeAlphabetLoader:
    """Load alphabets using Unicode ranges and Noto fonts"""
    
    def __init__(self, font_samples_dir="font_samples", data_dir="alphabet_data"):
        self.font_samples_dir = Path(font_samples_dir)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.registered_alphabets = {}
        self.character_cache = {}
        
        # Auto-discover fonts
        self.available_fonts = self._discover_fonts()
    
    def _discover_fonts(self):
        """Auto-discover Noto fonts in font_samples directory"""
        fonts = {}
        
        if not self.font_samples_dir.exists():
            print(f"Font samples directory not found: {self.font_samples_dir}")
            return fonts
        
        for font_dir in self.font_samples_dir.iterdir():
            if not font_dir.is_dir():
                continue
            
            # Find TTF files
            ttf_files = list(font_dir.glob("*.ttf"))
            if not ttf_files:
                # Check static subdirectory
                static_dir = font_dir / "static"
                if static_dir.exists():
                    ttf_files = list(static_dir.glob("*.ttf"))
            
            if ttf_files:
                # Use the first TTF (or variable font if available)
                variable_fonts = [f for f in ttf_files if 'Variable' in f.name]
                font_file = variable_fonts[0] if variable_fonts else ttf_files[0]
                
                # Infer script type from directory name
                dir_name = font_dir.name.lower()
                script_type = self._infer_script_type(dir_name)
                
                fonts[font_dir.name] = {
                    'path': str(font_file),
                    'script_type': script_type,
                    'dir_name': font_dir.name
                }
                
        print(f"Discovered {len(fonts)} font directories")
        return fonts
    
    def _infer_script_type(self, dir_name):
        """Infer Unicode block from directory name"""
        dir_lower = dir_name.lower().replace('_', ' ').replace('-', ' ')
        
        mappings = {
            'armenian': 'armenian',
            'avestan': 'avestan',
            'caucasian albanian': 'caucasian_albanian',
            'coptic': 'coptic',
            'deseret': 'deseret',
            'elbasan': 'elbasan',
            'georgian': 'georgian',
            'greek': 'greek',
            'lycian': 'lycian',
            'old hungarian': 'old_hungarian',
            'old permic': 'old_permic',
            'osage': 'osage',
            'tifinagh': 'tifinagh',
            'cyrillic': 'cyrillic',
            'latin': 'latin',
            'google sans': 'latin',
            'padauk': 'myanmar',  # Myanmar script
            'tirra': 'latin',  # Assume Latin if unknown
        }
        
        for key, script in mappings.items():
            if key in dir_lower:
                return script
        
        return 'latin'  # Default fallback
    
    def get_unicode_characters(self, script_type):
        """Get list of Unicode characters for a script"""
        if script_type not in UNICODE_BLOCKS:
            print(f"Unknown script type: {script_type}")
            return []
        
        block = UNICODE_BLOCKS[script_type]
        start, end = block['range']
        skip = block.get('skip', [])
        
        characters = []
        for code_point in range(start, end + 1):
            if code_point in skip:
                continue
            try:
                char = chr(code_point)
                characters.append({
                    'char': char,
                    'code_point': code_point,
                    'hex': f'U+{code_point:04X}'
                })
            except:
                continue
        
        return characters
    
    def render_character(self, font_path, char, size=(64, 64), font_size=48):
        """Render a single character from a font"""
        try:
            font = ImageFont.truetype(font_path, font_size)
            img = Image.new('L', size, 255)  # White background
            draw = ImageDraw.Draw(img)
            
            # Get character bounds
            bbox = font.getbbox(char)
            if bbox is None or bbox == (0, 0, 0, 0):
                return None
            
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            
            # Skip if character is too small (likely not in font)
            if char_width < 2 or char_height < 2:
                return None
            
            # Center character
            x = (size[0] - char_width) // 2 - bbox[0]
            y = (size[1] - char_height) // 2 - bbox[1]
            
            draw.text((x, y), char, fill=0, font=font)
            
            # Convert to numpy
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=-1)
            
            return img_array
            
        except Exception as e:
            return None
    
    def load_alphabet(self, font_name):
        """Load all characters for a font using its Unicode range"""
        if font_name not in self.available_fonts:
            print(f"Font not found: {font_name}")
            return []
        
        font_info = self.available_fonts[font_name]
        font_path = font_info['path']
        script_type = font_info['script_type']
        
        # Get Unicode characters for this script
        unicode_chars = self.get_unicode_characters(script_type)
        
        if not unicode_chars:
            print(f"No Unicode range defined for {script_type}")
            return []
        
        # Render each character
        characters = []
        for char_info in unicode_chars:
            img = self.render_character(font_path, char_info['char'])
            if img is not None:
                characters.append({
                    'image': img,
                    'char': char_info['char'],
                    'code_point': char_info['code_point'],
                    'hex': char_info['hex'],
                    'font': font_name,
                    'script': script_type
                })
        
        return characters
    
    def load_all_alphabets(self):
        """Load all available alphabets"""
        all_alphabets = {}
        
        for font_name in self.available_fonts:
            print(f"Loading {font_name}...")
            characters = self.load_alphabet(font_name)
            
            if characters:
                all_alphabets[font_name] = characters
                print(f"  ✓ {len(characters)} characters")
            else:
                print(f"  ✗ No characters rendered")
        
        self.registered_alphabets = all_alphabets
        return all_alphabets
    
    def get_alphabet_stats(self):
        """Get statistics about loaded alphabets"""
        stats = {}
        for font_name, characters in self.registered_alphabets.items():
            if characters:
                scripts = set(c['script'] for c in characters)
                stats[font_name] = {
                    'character_count': len(characters),
                    'scripts': list(scripts),
                    'font_path': self.available_fonts[font_name]['path']
                }
        return stats
    
    def sample_triplet(self):
        """Sample a triplet for training"""
        # Get alphabets with characters
        valid_alphabets = [k for k, v in self.registered_alphabets.items() if v]
        
        if len(valid_alphabets) < 2:
            raise ValueError("Need at least 2 alphabets for triplet sampling")
        
        # Pick anchor alphabet
        anchor_alphabet = np.random.choice(valid_alphabets)
        anchor_chars = self.registered_alphabets[anchor_alphabet]
        anchor_data = np.random.choice(anchor_chars)
        
        # Pick positive (different character from same alphabet)
        positive_candidates = [c for c in anchor_chars if c['char'] != anchor_data['char']]
        if not positive_candidates:
            positive_candidates = anchor_chars
        positive_data = np.random.choice(positive_candidates)
        
        # Pick negative (character from different alphabet)
        negative_alphabet = np.random.choice([a for a in valid_alphabets if a != anchor_alphabet])
        negative_chars = self.registered_alphabets[negative_alphabet]
        negative_data = np.random.choice(negative_chars)
        
        return anchor_data['image'], positive_data['image'], negative_data['image']
    
    def generate_batch(self, batch_size=32):
        """Generate a training batch"""
        anchors, positives, negatives = [], [], []
        
        for _ in range(batch_size):
            a, p, n = self.sample_triplet()
            anchors.append(a)
            positives.append(p)
            negatives.append(n)
        
        return np.array(anchors), np.array(positives), np.array(negatives)


if __name__ == "__main__":
    # Test the loader
    loader = UnicodeAlphabetLoader()
    
    print("\nAvailable fonts:")
    for font_name, info in loader.available_fonts.items():
        print(f"  {font_name}: {info['script_type']}")
    
    print("\nLoading alphabets...")
    loader.load_all_alphabets()
    
    print("\nAlphabet statistics:")
    for font_name, stats in loader.get_alphabet_stats().items():
        print(f"  {font_name}: {stats['character_count']} chars ({stats['scripts']})")
    
    # Test batch generation
    if loader.registered_alphabets:
        print("\nTesting batch generation...")
        anchors, positives, negatives = loader.generate_batch(4)
        print(f"Batch shape: {anchors.shape}")
