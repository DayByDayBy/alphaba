"""
Hybrid data loader for printed alphabets
Combines font-rendered characters with authentic examples
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import json

class HybridAlphabetLoader:
    def __init__(self, data_dir="alphabet_data"):
        self.data_dir = data_dir
        self.alphabet_data = {}
        self.font_mappings = {}
        self.character_mappings = {}
        
    def load_font_character(self, font_path, char, size=(64, 64)):
        """Render single character from font"""
        try:
            font = ImageFont.truetype(font_path, 48)
            img = Image.new('L', size, 255)  # White background
            draw = ImageDraw.Draw(img)
            
            # Center character
            bbox = font.getbbox(char)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            
            x = (size[0] - char_width) // 2
            y = (size[1] - char_height) // 2
            
            draw.text((x, y), char, fill=0, font=font)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=-1)
            
            return img_array
        except Exception as e:
            print(f"Error rendering {char} from {font_path}: {e}")
            return None
    
    def load_authentic_examples(self, alphabet_name):
        """Load authentic character images from dataset"""
        authentic_dir = os.path.join(self.data_dir, "authentic", alphabet_name)
        examples = {}
        
        if os.path.exists(authentic_dir):
            for char_file in os.listdir(authentic_dir):
                if char_file.endswith('.png'):
                    char_name = char_file.split('.')[0]
                    img_path = os.path.join(authentic_dir, char_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        # Resize and normalize
                        img = cv2.resize(img, (64, 64))
                        img = img.astype(np.float32) / 255.0
                        img = np.expand_dims(img, axis=-1)
                        examples[char_name] = img
        
        return examples
    
    def create_alphabet_dataset(self, alphabet_name, font_path, char_to_az_mapping):
        """Create dataset for one alphabet combining fonts and authentic examples"""
        print(f"Creating dataset for {alphabet_name}...")
        
        # Load authentic examples if available
        authentic_examples = self.load_authentic_examples(alphabet_name)
        
        alphabet_chars = []
        
        for native_char, az_char in char_to_az_mapping.items():
            # Render from font
            font_img = self.load_font_character(font_path, native_char)
            
            if font_img is not None:
                # Store both font and authentic versions if available
                char_data = {
                    'font_rendered': font_img,
                    'native_char': native_char,
                    'az_char': az_char,
                    'authentic_example': authentic_examples.get(native_char)
                }
                alphabet_chars.append(char_data)
        
        self.alphabet_data[alphabet_name] = alphabet_chars
        print(f"Created {len(alphabet_chars)} character entries for {alphabet_name}")
    
    def load_character_mappings(self, mapping_file="character_mappings.json"):
        """Load character-to-A-Z mapping definitions"""
        if os.path.exists(mapping_file):
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
                }
            }
    
    def generate_training_batch(self, batch_size=32):
        """Generate training batch combining font and authentic data"""
        if not self.alphabet_data:
            raise ValueError("No alphabet data loaded. Call create_alphabet_dataset() first.")
        
        batch_images = []
        batch_labels = []
        
        for _ in range(batch_size):
            # Random alphabet and character
            alphabet_name = np.random.choice(list(self.alphabet_data.keys()))
            alphabet_chars = self.alphabet_data[alphabet_name]
            char_data = np.random.choice(alphabet_chars)
            
            # Choose font or authentic version
            if char_data['authentic_example'] is not None and np.random.random() < 0.3:
                # Use authentic example 30% of time
                img = char_data['authentic_example']
            else:
                # Use font-rendered version
                img = char_data['font_rendered']
            
            batch_images.append(img)
            batch_labels.append(f"{alphabet_name}_{char_data['az_char']}")
        
        return np.array(batch_images), np.array(batch_labels)

if __name__ == "__main__":
    # Example usage
    loader = HybridAlphabetLoader()
    loader.load_character_mappings()
    
    # Create sample dataset for Greek
    loader.create_alphabet_dataset(
        'greek', 
        '/System/Library/Fonts/Helvetica.ttc',  # Example font
        loader.character_mappings['greek']
    )
