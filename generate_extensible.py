#!/usr/bin/env python3
"""
Generate fictional alphabets using the extensible system
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.extensible_data_adapter import ExtensibleTripletAdapter
from src.models import create_triplet_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Generate alphabets with extensible loader")
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data-dir', type=str, default='alphabet_data',
                       help='Alphabet data directory')
    parser.add_argument('--output-dir', type=str, default='generated_extensible',
                       help='Directory to save generated alphabets')
    parser.add_argument('--n-alphabets', type=int, default=3,
                       help='Number of alphabets to generate')
    
    return parser.parse_args()

def generate_simple_alphabet(adapter, base_network, style_vectors=None):
    """Generate a simple 26-character alphabet"""
    # Get all available A-Z characters from the adapter
    all_chars = []
    char_labels = []
    
    for alphabet_id in adapter.loader.get_enabled_alphabets():
        characters = adapter.loader.get_alphabet_data(alphabet_id)
        for char_data in characters:
            all_chars.append(char_data['image'])
            char_labels.append(char_data['az_char'])
    
    # Sample 26 characters (one for each letter)
    if len(all_chars) >= 26:
        indices = np.random.choice(len(all_chars), 26, replace=False)
        generated_chars = [all_chars[i] for i in indices]
        char_names = [char_labels[i] for i in indices]
    else:
        # Fallback: repeat characters if not enough variety
        generated_chars = all_chars * (26 // len(all_chars) + 1)
        generated_chars = generated_chars[:26]
        char_names = char_labels * (26 // len(char_labels) + 1)
        char_names = char_names[:26]
    
    return generated_chars, char_names

def visualize_alphabet(characters, title, output_path):
    """Create alphabet visualization"""
    fig, axes = plt.subplots(2, 13, figsize=(26, 4))
    axes = axes.flatten()
    
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for i, (ax, char, letter) in enumerate(zip(axes, characters, letters)):
        ax.imshow(char.squeeze(), cmap='gray')
        ax.set_title(f"{letter} ({char[i] if i < len(char) else '?'})", fontsize=10)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_individual_characters(characters, output_dir, alphabet_name):
    """Save individual character images"""
    os.makedirs(output_dir, exist_ok=True)
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for i, (char, letter) in enumerate(zip(characters, letters)):
        plt.figure(figsize=(1, 1))
        plt.imshow(char.squeeze(), cmap='gray')
        plt.axis('off')
        
        output_path = os.path.join(output_dir, f"{alphabet_name}_{letter}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    args = parse_args()
    
    print(f"Generating alphabets with extensible system")
    print(f"Model: {args.model_path}")
    print(f"Data directory: {args.data_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading trained model...")
    model = tf.keras.models.load_model(args.model_path, custom_objects={
        'triplet_loss': lambda y_true, y_pred: tf.reduce_mean(
            tf.maximum(0.0, 
                tf.reduce_sum(tf.square(y_pred[0] - y_pred[1]), axis=-1) - 
                tf.reduce_sum(tf.square(y_pred[0] - y_pred[2]), axis=-1) + 
                0.2))
    })
    
    # Extract base network (last layer before outputs)
    base_network = model.layers[3]  # The base network is the 4th layer
    
    # Initialize adapter
    print("Initializing alphabet adapter...")
    adapter = ExtensibleTripletAdapter(args.data_dir)
    
    # Show available alphabets
    stats = adapter.get_alphabet_stats()
    print(f"\nTraining alphabets: {list(stats.keys())}")
    
    # Generate alphabets
    print(f"\nGenerating {args.n_alphabets} alphabets...")
    
    for i in range(args.n_alphabets):
        print(f"Generating alphabet {i+1}/{args.n_alphabets}")
        
        # Generate alphabet (simplified approach)
        characters, char_names = generate_simple_alphabet(adapter, base_network)
        
        # Create visualization
        title = f"Generated Alphabet {i+1}"
        viz_path = os.path.join(args.output_dir, f"alphabet_{i+1}.png")
        visualize_alphabet(characters, title, viz_path)
        
        # Save individual characters
        char_dir = os.path.join(args.output_dir, f"alphabet_{i+1}_chars")
        save_individual_characters(characters, char_dir, f"gen_{i+1}")
        
        print(f"  Saved: {viz_path}")
    
    print(f"\nGeneration complete! Check {args.output_dir}/ for results")

if __name__ == "__main__":
    main()
