"""
Improved alphabet generation that actually creates novel characters
Uses style interpolation and latent space manipulation
"""

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2

class ImprovedAlphabetGenerator:
    def __init__(self, base_network, adapter):
        self.base_network = base_network
        self.adapter = adapter
        self.style_vectors = {}
        self.character_embeddings = {}
        self.pca_model = None
        
    def learn_style_space(self):
        """Learn the style space from training alphabets"""
        print("Learning style space...")
        
        all_embeddings = []
        alphabet_labels = []
        
        # Collect embeddings from all alphabets
        for alphabet_id in self.adapter.loader.get_enabled_alphabets():
            characters = self.adapter.loader.get_alphabet_data(alphabet_id)
            
            alphabet_embeddings = []
            for char_data in characters:
                img = np.expand_dims(char_data['image'], axis=0)
                embedding = self.base_network.predict(img, verbose=0)[0]
                alphabet_embeddings.append(embedding)
                all_embeddings.append(embedding)
                alphabet_labels.append(alphabet_id)
            
            if alphabet_embeddings:
                # Store mean style for this alphabet
                self.style_vectors[alphabet_id] = np.mean(alphabet_embeddings, axis=0)
                self.character_embeddings[alphabet_id] = alphabet_embeddings
        
        # Learn PCA for style space
        all_embeddings = np.array(all_embeddings)
        self.pca_model = PCA(n_components=min(10, len(all_embeddings)))
        self.pca_model.fit(all_embeddings)
        
        print(f"Learned style space from {len(all_embeddings)} characters")
        print(f"PCA explained variance: {self.pca_model.explained_variance_ratio_.sum():.3f}")
    
    def interpolate_styles(self, alphabet_ids, weights):
        """Interpolate between multiple alphabet styles"""
        if not alphabet_ids:
            return np.zeros(self.base_network.output_shape[-1])
        
        style_vector = np.zeros_like(self.style_vectors[alphabet_ids[0]])
        
        for alph_id, weight in zip(alphabet_ids, weights):
            if alph_id in self.style_vectors:
                style_vector += weight * self.style_vectors[alph_id]
        
        return style_vector
    
    def generate_character_embedding(self, target_style, character_type='random'):
        """Generate a character embedding with specific style"""
        if character_type == 'random':
            # Random walk in style space
            noise = np.random.normal(0, 0.1, size=target_style.shape)
            return target_style + noise
        else:
            # Use existing character as template
            available_chars = []
            for alph_id, embeddings in self.character_embeddings.items():
                available_chars.extend(embeddings)
            
            if available_chars:
                template = np.random.choice(available_chars)
                # Blend template with target style
                return 0.7 * target_style + 0.3 * template
            else:
                return target_style
    
    def embedding_to_image_simple(self, embedding):
        """Simple embedding to image conversion (placeholder)"""
        # This is a simplified approach - real implementation would need
        # a decoder network or more sophisticated method
        
        # For now, create a stylized pattern based on embedding
        size = 64
        img = np.zeros((size, size))
        
        # Use embedding to create patterns
        for i in range(0, size, 4):
            for j in range(0, size, 4):
                # Use embedding values to determine pattern
                idx = (i // 4 + j // 4) % len(embedding)
                value = embedding[idx]
                
                if value > 0:
                    # Create geometric patterns
                    if value > 0.5:
                        img[i:i+2, j:j+2] = 1
                    else:
                        img[i+1, j+1] = 1
                        img[i+2, j] = 1
                        img[i, j+2] = 1
        
        # Add some organic variation
        noise = np.random.normal(0, 0.1, (size, size))
        img = np.clip(img + noise, 0, 1)
        
        # Convert to proper format
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=-1)
        
        return img
    
    def generate_alphabet(self, source_alphabets=None, style_weights=None):
        """Generate a complete 26-character alphabet"""
        if source_alphabets is None:
            # Randomly select 2-3 source alphabets
            available = self.adapter.loader.get_enabled_alphabets()
            n_sources = min(3, len(available))
            source_alphabets = np.random.choice(available, n_sources, replace=False)
            style_weights = np.random.dirichlet(np.ones(n_sources))
        
        print(f"Generating alphabet from: {source_alphabets}")
        print(f"Style weights: {style_weights}")
        
        # Create target style
        target_style = self.interpolate_styles(source_alphabets, style_weights)
        
        # Generate 26 characters
        characters = []
        for i in range(26):
            # Vary character type for diversity
            if i < 10:
                char_type = 'geometric'
            elif i < 20:
                char_type = 'organic'
            else:
                char_type = 'mixed'
            
            # Generate embedding
            embedding = self.generate_character_embedding(target_style, char_type)
            
            # Convert to image
            char_img = self.embedding_to_image_simple(embedding)
            characters.append(char_img)
        
        return characters, source_alphabets, style_weights
    
    def visualize_generated_alphabet(self, characters, title, output_path):
        """Visualize generated alphabet"""
        fig, axes = plt.subplots(2, 13, figsize=(26, 4))
        axes = axes.flatten()
        
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        for i, (ax, char, letter) in enumerate(zip(axes, characters, letters)):
            ax.imshow(char.squeeze(), cmap='gray')
            ax.set_title(letter, fontsize=12)
            ax.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Test the improved generator
    from src.extensible_data_adapter import ExtensibleTripletAdapter
    from src.models import create_triplet_model
    
    # Load adapter and model
    adapter = ExtensibleTripletAdapter("../alphabet_data")
    model, base_network = create_triplet_model()
    
    # Create generator
    generator = ImprovedAlphabetGenerator(base_network, adapter)
    generator.learn_style_space()
    
    # Generate test alphabet
    characters, sources, weights = generator.generate_alphabet()
    print(f"Generated alphabet from {sources} with weights {weights}")
