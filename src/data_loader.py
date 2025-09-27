import os
import cv2
import numpy as np
import random
from collections import defaultdict

class OmniglotTripletLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.alphabet_data = {}  # {alphabet_name: [(image, char_id), ...]}
        self.alphabet_names = []
        self.load_alphabet_data()
        
    def load_alphabet_data(self):
        """Load all images organized by alphabet."""
        background_path = os.path.join(self.data_path, "images_background")
        
        for alphabet_name in os.listdir(background_path):
            if alphabet_name.startswith('.'):
                continue
                
            alphabet_path = os.path.join(background_path, alphabet_name)
            if not os.path.isdir(alphabet_path):
                continue
                
            self.alphabet_data[alphabet_name] = []
            
            for char_folder in os.listdir(alphabet_path):
                if char_folder.startswith('.'):
                    continue
                    
                char_path = os.path.join(alphabet_path, char_folder)
                for img_file in os.listdir(char_path):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(char_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = img.astype(np.float32) / 255.0  # Normalize
                            img = np.expand_dims(img, axis=-1)  # Add channel dim
                            self.alphabet_data[alphabet_name].append((img, char_folder))
            
            if len(self.alphabet_data[alphabet_name]) > 0:
                self.alphabet_names.append(alphabet_name)
        
        print(f"Loaded {len(self.alphabet_names)} alphabets")
        for name in self.alphabet_names:
            print(f"  {name}: {len(self.alphabet_data[name])} images")
    
    def sample_triplet(self):
        """Sample one triplet: (anchor, positive, negative)."""
        # pick random alphabet for anchor
        anchor_alphabet = random.choice(self.alphabet_names)
        
        # pick random image from that alphabet as anchor
        anchor_img, anchor_char = random.choice(self.alphabet_data[anchor_alphabet])
        
        # find a 'positiv'e': ie different character from same alphabet
        same_alphabet_candidates = [
            (img, char) for img, char in self.alphabet_data[anchor_alphabet] 
            if char != anchor_char
        ]
        
        if len(same_alphabet_candidates) == 0:
            # fallback: same character, different instance
            same_alphabet_candidates = [
                (img, char) for img, char in self.alphabet_data[anchor_alphabet]
            ]
        
        positive_img, _ = random.choice(same_alphabet_candidates)
        
        # find a 'negative': ie any image from different alphabet
        negative_alphabet = random.choice([
            name for name in self.alphabet_names if name != anchor_alphabet
        ])
        negative_img, _ = random.choice(self.alphabet_data[negative_alphabet])
        
        return anchor_img, positive_img, negative_img
    
    def generate_batch(self, batch_size):
        """Generate a batch of triplets."""
        anchors, positives, negatives = [], [], []
        
        for _ in range(batch_size):
            anchor, positive, negative = self.sample_triplet()
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
            
        return (np.array(anchors), np.array(positives), np.array(negatives))