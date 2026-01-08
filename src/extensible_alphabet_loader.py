"""
Extensible alphabet data loader
Supports: system fonts, custom TTFs, user-designed alphabets, generated alphabets
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from collections import defaultdict
import cv2

class ExtensibleAlphabetLoader:
    def __init__(self, data_root="alphabet_data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        
        # Subdirectories
        self.fonts_dir = self.data_root / "fonts"
        self.authentic_dir = self.data_root / "authentic"
        self.generated_dir = self.data_root / "generated"
        self.user_dir = self.data_root / "user"
        
        for dir_path in [self.fonts_dir, self.authentic_dir, self.generated_dir, self.user_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.alphabet_registry = {}
        self.character_mappings = {}
        self.load_registry()
        self.load_character_mappings()
    
    def load_registry(self):
        """Load alphabet registry from JSON file"""
        registry_file = self.data_root / "alphabet_registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                self.alphabet_registry = json.load(f)
    
    def save_registry(self):
        """Save alphabet registry to JSON file"""
        registry_file = self.data_root / "alphabet_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(self.alphabet_registry, f, indent=2)
    
    def load_character_mappings(self):
        """Load character-to-A-Z mappings"""
        mapping_file = self.data_root / "character_mappings.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                self.character_mappings = json.load(f)
        else:
            # Default mappings
            self.character_mappings = {
                'greek': {
                    'Α': 'A', 'Β': 'B', 'Γ': 'G', 'Δ': 'D', 'Ε': 'E',
                    'Ζ': 'Z', 'Η': 'H', 'Θ': 'TH', 'Ι': 'I', 'Κ': 'K',
                    'Λ': 'L', 'Μ': 'M', 'Ν': 'N', 'Ξ': 'X', 'Ο': 'O',
                    'Π': 'P', 'Ρ': 'R', 'Σ': 'S', 'Τ': 'T', 'Υ': 'Y',
                    'Φ': 'F', 'Χ': 'CH', 'Ψ': 'PS', 'Ω': 'W'
                },
                'armenian': {
                    'Ա': 'A', 'Բ': 'B', 'Գ': 'G', 'Դ': 'D', 'Ե': 'E',
                    'Զ': 'Z', 'Է': 'E', 'Ը': 'Y', 'Թ': 'T', 'Ժ': 'ZH',
                    'Ի': 'I', 'Լ': 'L', 'Խ': 'KH', 'Ծ': 'TS', 'Կ': 'K',
                    'Հ': 'H', 'Ձ': 'D', 'Ղ': 'GH', 'Ճ': 'CH', 'Մ': 'M',
                    'Յ': 'Y', 'Ն': 'N', 'Շ': 'SH', 'Ո': 'V', 'Չ': 'CH',
                    'Պ': 'P', 'Ջ': 'J', 'Ռ': 'R', 'Ս': 'S', 'Վ': 'V',
                    'Տ': 'T', 'Ր': 'R', 'Ց': 'TS', 'Ւ': 'W', 'Փ': 'P',
                    'Ք': 'K', 'Օ': 'O', 'Ֆ': 'F'
                },
                'georgian': {
                    'Ⴀ': 'A', 'Ⴁ': 'B', 'Ⴂ': 'G', 'Ⴃ': 'D', 'Ⴄ': 'E',
                    'Ⴅ': 'V', 'Ⴆ': 'Z', 'Ⴇ': 'T', 'Ⴈ': 'I', 'Ⴉ': 'K',
                    'Ⴊ': 'L', 'Ⴋ': 'M', 'Ⴌ': 'N', 'Ⴍ': 'H', 'Ⴎ': 'P',
                    'Ⴏ': 'ZH', 'Ⴐ': 'R', 'Ⴑ': 'S', 'Ⴒ': 'T', 'Ⴓ': 'U',
                    'Ⴔ': 'PH', 'Ⴕ': 'KH', 'Ⴖ': 'GH', 'Ⴗ': 'Q', 'Ⴘ': 'Y',
                    'Ⴙ': 'C', 'Ⴚ': 'CH', 'Ⴛ': 'J', 'Ⴜ': 'X', 'Ⴝ': 'JH',
                    'Ⴞ': 'H', 'Ⴟ': 'E', 'Ⴠ': 'W', 'Ⴡ': 'H', 'Ⴢ': 'Y'
                }
            }
            self.save_character_mappings()
    
    def save_character_mappings(self):
        """Save character mappings"""
        mapping_file = self.data_root / "character_mappings.json"
        with open(mapping_file, 'w') as f:
            json.dump(self.character_mappings, f, indent=2)
    
    def register_system_font(self, alphabet_name, font_path, script_type):
        """Register a system font for an alphabet"""
        alphabet_id = alphabet_name.lower().replace(' ', '_')
        
        self.alphabet_registry[alphabet_id] = {
            'name': alphabet_name,
            'type': 'system_font',
            'script_type': script_type,
            'font_path': font_path,
            'character_mapping': script_type,
            'enabled': True
        }
        
        self.save_registry()
        print(f"Registered system font: {alphabet_name}")
    
    def register_custom_font(self, alphabet_name, font_file_path, script_type, character_mapping=None):
        """Register a custom TTF font file"""
        # Copy font to fonts directory
        font_file_path = Path(font_file_path)
        target_path = self.fonts_dir / font_file_path.name
        
        if font_file_path != target_path:
            import shutil
            shutil.copy2(font_file_path, target_path)
        
        alphabet_id = alphabet_name.lower().replace(' ', '_')
        
        self.alphabet_registry[alphabet_id] = {
            'name': alphabet_name,
            'type': 'custom_font',
            'script_type': script_type,
            'font_path': str(target_path),
            'character_mapping': character_mapping or script_type,
            'enabled': True
        }
        
        self.save_registry()
        print(f"Registered custom font: {alphabet_name}")
    
    def register_user_alphabet(self, alphabet_name, image_directory, character_mapping):
        """Register user-designed alphabet from image files"""
        alphabet_id = alphabet_name.lower().replace(' ', '_')
        
        self.alphabet_registry[alphabet_id] = {
            'name': alphabet_name,
            'type': 'user_images',
            'script_type': 'custom',
            'image_directory': str(image_directory),
            'character_mapping': character_mapping,
            'enabled': True
        }
        
        self.save_registry()
        print(f"Registered user alphabet: {alphabet_name}")
    
    def render_character_from_font(self, font_path, character, size=(64, 64)):
        """Render single character from font file"""
        try:
            font = ImageFont.truetype(font_path, 48)
            img = Image.new('L', size, 255)  # White background
            draw = ImageDraw.Draw(img)
            
            # Get character bounds
            bbox = font.getbbox(character)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            
            # Center character
            x = (size[0] - char_width) // 2 - bbox[0]
            y = (size[1] - char_height) // 2 - bbox[1]
            
            draw.text((x, y), character, fill=0, font=font)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=-1)
            
            return img_array
        except Exception as e:
            print(f"Error rendering {character} from {font_path}: {e}")
            return None
    
    def load_character_images(self, image_directory):
        """Load character images from directory"""
        images = {}
        image_dir = Path(image_directory)
        
        if image_dir.exists():
            for img_file in image_dir.glob('*.png'):
                char_name = img_file.stem
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    img = img.astype(np.float32) / 255.0
                    img = np.expand_dims(img, axis=-1)
                    images[char_name] = img
        
        return images
    
    def get_alphabet_data(self, alphabet_id):
        """Get character data for a registered alphabet"""
        if alphabet_id not in self.alphabet_registry:
            raise ValueError(f"Alphabet {alphabet_id} not registered")
        
        alphabet_info = self.alphabet_registry[alphabet_id]
        mapping = self.character_mappings[alphabet_info['character_mapping']]
        characters = []
        
        if alphabet_info['type'] in ['system_font', 'custom_font']:
            font_path = alphabet_info['font_path']
            
            for native_char, az_char in mapping.items():
                img = self.render_character_from_font(font_path, native_char)
                if img is not None:
                    characters.append({
                        'image': img,
                        'native_char': native_char,
                        'az_char': az_char,
                        'source': 'font'
                    })
        
        elif alphabet_info['type'] == 'user_images':
            images = self.load_character_images(alphabet_info['image_directory'])
            
            for native_char, az_char in mapping.items():
                if native_char in images:
                    characters.append({
                        'image': images[native_char],
                        'native_char': native_char,
                        'az_char': az_char,
                        'source': 'user_image'
                    })
        
        return characters
    
    def list_registered_alphabets(self):
        """List all registered alphabets"""
        print("Registered Alphabets:")
        for alphabet_id, info in self.alphabet_registry.items():
            status = "✓" if info['enabled'] else "✗"
            print(f"  {status} {alphabet_id}: {info['name']} ({info['type']})")
    
    def get_enabled_alphabets(self):
        """Get list of enabled alphabet IDs"""
        enabled = [aid for aid, info in self.alphabet_registry.items() if info.get('enabled', True)]
        print(f"Found {len(enabled)} enabled alphabets: {enabled}")
        return enabled
    
    def generate_training_batch(self, batch_size=32, alphabet_ids=None):
        """Generate training batch from enabled alphabets"""
        if alphabet_ids is None:
            alphabet_ids = self.get_enabled_alphabets()
        
        if not alphabet_ids:
            raise ValueError("No enabled alphabets available")
        
        batch_images = []
        batch_labels = []
        
        for _ in range(batch_size):
            # Random alphabet
            alphabet_id = np.random.choice(alphabet_ids)
            characters = self.get_alphabet_data(alphabet_id)
            
            if characters:
                char_data = np.random.choice(characters)
                batch_images.append(char_data['image'])
                batch_labels.append(f"{alphabet_id}_{char_data['az_char']}")
        
        return np.array(batch_images), np.array(batch_labels)

if __name__ == "__main__":
    # Example usage
    loader = ExtensibleAlphabetLoader()
    
    # Register some system fonts
    loader.register_system_font("Greek System", "/System/Library/Fonts/Arial.ttf", "greek")
    loader.register_system_font("Armenian System", "/System/Library/Fonts/Arial.ttf", "armenian")
    
    # List what we have
    loader.list_registered_alphabets()
    
    # Test batch generation
    try:
        images, labels = loader.generate_training_batch(4)
        print(f"Generated batch: {images.shape}, labels: {labels}")
    except Exception as e:
        print(f"Error: {e}")
