"""
Alphabet generation module for Alphaba
Creates 26-character fictional alphabets mapped to Roman letters
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import os


class AlphabetGenerator:
    """Generate 26-character fictional alphabets from learned embeddings"""
    
    def __init__(self, base_network, embedding_dim=64):
        self.base_network = base_network
        self.embedding_dim = embedding_dim
        self.alphabet_embeddings = {}
        self.character_styles = {}
        
    def learn_alphabet_styles(self, data_loader, target_alphabets=None):
        """Learn style embeddings from source alphabets"""
        if target_alphabets is None:
            target_alphabets = data_loader.alphabet_names[:12]  # Use first 12
        
        print(f"Learning styles from {len(target_alphabets)} alphabets...")
        
        for alphabet in target_alphabets:
            alphabet_data = data_loader.alphabet_data[alphabet]
            
            # Sample characters from this alphabet
            n_samples = min(20, len(alphabet_data))
            indices = np.random.choice(len(alphabet_data), n_samples, replace=False)
            
            embeddings = []
            for idx in indices:
                img, _ = alphabet_data[idx]
                img_batch = np.expand_dims(img, axis=0)
                embedding = self.base_network.predict(img_batch, verbose=0)[0]
                embeddings.append(embedding)
            
            # Store mean style vector for this alphabet
            style_vector = np.mean(embeddings, axis=0)
            self.alphabet_embeddings[alphabet] = style_vector
            
        print(f"Learned styles for {len(self.alphabet_embeddings)} alphabets")
    
    def interpolate_styles(self, style_a, style_b, alpha=0.5):
        """Interpolate between two alphabet styles"""
        if style_a not in self.alphabet_embeddings or style_b not in self.alphabet_embeddings:
            raise ValueError("Style not found in learned embeddings")
        
        vec_a = self.alphabet_embeddings[style_a]
        vec_b = self.alphabet_embeddings[style_b]
        
        return alpha * vec_a + (1 - alpha) * vec_b
    
    def generate_random_style(self, variance=0.1):
        """Generate a random style vector based on learned statistics"""
        if not self.alphabet_embeddings:
            raise ValueError("No styles learned yet")
        
        # Compute statistics of learned styles
        all_styles = np.array(list(self.alphabet_embeddings.values()))
        mean_style = np.mean(all_styles, axis=0)
        std_style = np.std(all_styles, axis=0)
        
        # Generate random style
        random_style = mean_style + np.random.randn(self.embedding_dim) * std_style * variance
        return random_style
    
    def create_decoder(self):
        """Create decoder network for generating images from embeddings"""
        decoder = models.Sequential([
            layers.Input(shape=(self.embedding_dim,)),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(7 * 7 * 256, activation='relu'),
            layers.BatchNormalization(),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid'),
        ])
        return decoder
    
    def train_decoder(self, data_loader, epochs=50, batch_size=32):
        """Train decoder to reconstruct characters"""
        print("Training decoder...")
        
        decoder = self.create_decoder()
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = keras.losses.MeanSquaredError()
        
        # Prepare training data
        train_images = []
        train_embeddings = []
        
        for alphabet in data_loader.alphabet_names[:10]:  # Use subset for training
            alphabet_data = data_loader.alphabet_data[alphabet]
            n_samples = min(30, len(alphabet_data))
            indices = np.random.choice(len(alphabet_data), n_samples, replace=False)
            
            for idx in indices:
                img, _ = alphabet_data[idx]
                embedding = self.base_network.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                
                train_images.append(img)
                train_embeddings.append(embedding)
        
        train_images = np.array(train_images)
        train_embeddings = np.array(train_embeddings)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            n_batches = len(train_images) // batch_size
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_embeddings = train_embeddings[start_idx:end_idx]
                batch_images = train_images[start_idx:end_idx]
                
                with tf.GradientTape() as tape:
                    reconstructed = decoder(batch_embeddings, training=True)
                    loss = loss_fn(batch_images, reconstructed)
                
                gradients = tape.gradient(loss, decoder.trainable_variables)
                optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))
                total_loss += loss
            
            if epoch % 10 == 0:
                avg_loss = total_loss / n_batches
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        self.decoder = decoder
        return decoder
    
    def generate_alphabet(self, style_vector=None, source_alphabets=None, interpolation_weights=None):
        """Generate a 26-character alphabet"""
        if style_vector is None:
            if source_alphabets and interpolation_weights:
                # Create interpolated style
                if len(source_alphabets) != len(interpolation_weights):
                    raise ValueError("Source alphabets and weights must have same length")
                
                style_vector = np.zeros(self.embedding_dim)
                total_weight = sum(interpolation_weights)
                
                for alphabet, weight in zip(source_alphabets, interpolation_weights):
                    if alphabet not in self.alphabet_embeddings:
                        raise ValueError(f"Alphabet {alphabet} not found in learned styles")
                    style_vector += (weight / total_weight) * self.alphabet_embeddings[alphabet]
            else:
                # Generate random style
                style_vector = self.generate_random_style()
        
        if not hasattr(self, 'decoder'):
            raise ValueError("Decoder not trained. Call train_decoder() first.")
        
        # Generate 26 characters with controlled variation
        characters = []
        base_style = style_vector
        
        for i in range(26):
            # Add controlled variation for each character
            character_variation = np.random.randn(self.embedding_dim) * 0.05
            character_embedding = base_style + character_variation
            
            # Generate character image
            character_img = self.decoder.predict(np.expand_dims(character_embedding, axis=0), verbose=0)[0]
            characters.append(character_img)
        
        return characters
    
    def visualize_alphabet(self, characters, title="Generated Alphabet", save_path=None):
        """Visualize generated alphabet"""
        fig, axes = plt.subplots(4, 7, figsize=(14, 8))
        fig.suptitle(title, fontsize=16)
        
        roman_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        for i, char_img in enumerate(characters):
            row = i // 7
            col = i % 7
            
            if row < 4 and col < 7:
                axes[row, col].imshow(char_img.squeeze(), cmap='gray')
                axes[row, col].set_title(roman_alphabet[i])
                axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(26, 28):
            row = i // 7
            col = i % 7
            if row < 4 and col < 7:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_alphabet_images(self, characters, output_dir, alphabet_name="generated"):
        """Save individual character images"""
        os.makedirs(output_dir, exist_ok=True)
        
        roman_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        for i, char_img in enumerate(characters):
            # Convert to 0-255 range
            img_uint8 = (char_img.squeeze() * 255).astype(np.uint8)
            
            # Save image
            filename = f"{alphabet_name}_{roman_alphabet[i]}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, img_uint8)
        
        print(f"Saved {len(characters)} character images to {output_dir}")


def create_sample_alphabets(generator, output_dir="generated_alphabets"):
    """Generate sample alphabets with different styles"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Random style alphabet
    random_chars = generator.generate_alphabet()
    generator.visualize_alphabet(random_chars, "Random Style Alphabet", 
                                os.path.join(output_dir, "random_alphabet.png"))
    generator.save_alphabet_images(random_chars, output_dir, "random")
    
    # 2. Interpolated style (if we have learned styles)
    if len(generator.alphabet_embeddings) >= 2:
        alphabets = list(generator.alphabet_embeddings.keys())
        style_chars = generator.generate_alphabet(
            source_alphabets=[alphabets[0], alphabets[1]], 
            interpolation_weights=[0.7, 0.3]
        )
        generator.visualize_alphabet(style_chars, f"Interpolated ({alphabets[0]} + {alphabets[1]})", 
                                    os.path.join(output_dir, "interpolated_alphabet.png"))
        generator.save_alphabet_images(style_chars, output_dir, "interpolated")
    
    print(f"Sample alphabets saved to {output_dir}")
