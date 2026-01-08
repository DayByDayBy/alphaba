#!/usr/bin/env python3
"""
Training script using the extensible alphabet loader
Replaces Omniglot with the new font-based system
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.extensible_data_adapter import ExtensibleTripletAdapter
from src.training import train_triplet_model_custom
from src.models import create_triplet_model
import tensorflow as tf
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train with extensible alphabet loader")
    parser.add_argument('--data-dir', type=str, default='alphabet_data',
                       help='Alphabet data directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='outputs_extensible',
                       help='Directory to save results')
    parser.add_argument('--alphabets', type=str, nargs='*',
                       help='Specific alphabets to use (default: all enabled)')
    
    return parser.parse_args()

class ExtensibleDataGenerator:
    """Data generator compatible with existing training pipeline"""
    
    def __init__(self, adapter, batch_size=32, alphabet_ids=None):
        self.adapter = adapter
        self.batch_size = batch_size
        self.alphabet_ids = alphabet_ids or adapter.loader.get_enabled_alphabets()
        
    def __iter__(self):
        while True:
            yield self.adapter.generate_batch(self.batch_size, self.alphabet_ids)

def main():
    args = parse_args()
    
    print(f"Training with extensible alphabet loader")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize adapter
    print("Initializing extensible alphabet adapter...")
    adapter = ExtensibleTripletAdapter(args.data_dir)
    
    # Show alphabet statistics
    stats = adapter.get_alphabet_stats()
    print("\nAvailable alphabets:")
    for alph_id, stat in stats.items():
        print(f"  {alph_id}: {stat['character_count']} chars ({stat['type']})")
    
    # Filter alphabets if specified
    if args.alphabets:
        available_alphabets = set(stats.keys())
        selected_alphabets = []
        for alph in args.alphabets:
            if alph in available_alphabets:
                selected_alphabets.append(alph)
            else:
                print(f"Warning: Alphabet '{alph}' not found, skipping")
        
        if not selected_alphabets:
            print("Error: No valid alphabets selected")
            return
        
        print(f"\nUsing alphabets: {selected_alphabets}")
    else:
        selected_alphabets = list(stats.keys())
        print(f"\nUsing all {len(selected_alphabets)} alphabets")
    
    # Create model
    print(f"\nCreating triplet network (embedding_dim={args.embedding_dim})...")
    model, base_network = create_triplet_model(embedding_dim=args.embedding_dim)
    
    # Create a simple data loader wrapper for training
    class SimpleDataLoader:
        def __init__(self, adapter, batch_size, alphabet_ids):
            self.adapter = adapter
            self.batch_size = batch_size
            self.alphabet_ids = alphabet_ids
        
        def generate_batch(self):
            return self.adapter.generate_batch(self.batch_size, self.alphabet_ids)
    
    data_loader = SimpleDataLoader(adapter, args.batch_size, selected_alphabets)
    
    # Simple training loop
    print(f"Training for {args.epochs} epochs...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 10  # Small number for demo
        
        for batch in range(num_batches):
            anchors, positives, negatives = adapter.generate_batch(args.batch_size)
            
            with tf.GradientTape() as tape:
                anchor_emb, positive_emb, negative_emb = model([anchors, positives, negatives], training=True)
                
                pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=-1)
                neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=-1)
                loss = tf.reduce_mean(tf.maximum(0.0, pos_dist - neg_dist + 0.2))
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    model_path = os.path.join(args.output_dir, "triplet_model_extensible.h5")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save training info
    training_info = {
        'alphabets_used': selected_alphabets,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'embedding_dim': args.embedding_dim,
        'alphabet_stats': stats
    }
    
    import json
    info_path = os.path.join(args.output_dir, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"Training info saved to {info_path}")
    print("\nTraining complete! You can now generate alphabets with:")
    print(f"python main.py generate --model-path {model_path} --data-path {args.data_dir}")

if __name__ == "__main__":
    main()
