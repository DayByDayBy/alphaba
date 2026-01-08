#!/usr/bin/env python3
"""
Train alphabet model using Unicode-based loader
Clean, simple approach using Noto fonts and Unicode ranges
"""

import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.unicode_alphabet_loader import UnicodeAlphabetLoader
from src.models import create_triplet_model
import tensorflow as tf
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Train with Unicode alphabet loader")
    parser.add_argument('--font-dir', type=str, default='font_samples',
                       help='Font samples directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--steps-per-epoch', type=int, default=50,
                       help='Training steps per epoch')
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='outputs_unicode',
                       help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("ALPHABA - Unicode Alphabet Training")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load alphabets
    print(f"\nLoading alphabets from {args.font_dir}...")
    loader = UnicodeAlphabetLoader(font_samples_dir=args.font_dir)
    loader.load_all_alphabets()
    
    # Show stats
    stats = loader.get_alphabet_stats()
    total_chars = sum(s['character_count'] for s in stats.values())
    print(f"\nLoaded {len(stats)} alphabets with {total_chars} total characters")
    
    for font_name, stat in stats.items():
        print(f"  {font_name}: {stat['character_count']} chars")
    
    # Create model
    print(f"\nCreating triplet network (embedding_dim={args.embedding_dim})...")
    model, base_network = create_triplet_model(embedding_dim=args.embedding_dim)
    
    # Training
    print(f"\nTraining for {args.epochs} epochs ({args.steps_per_epoch} steps/epoch)...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    history = {'loss': [], 'epoch': []}
    
    for epoch in range(args.epochs):
        epoch_losses = []
        
        for step in range(args.steps_per_epoch):
            anchors, positives, negatives = loader.generate_batch(args.batch_size)
            
            with tf.GradientTape() as tape:
                anchor_emb, positive_emb, negative_emb = model(
                    [anchors, positives, negatives], training=True
                )
                
                pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=-1)
                neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=-1)
                loss = tf.reduce_mean(tf.maximum(0.0, pos_dist - neg_dist + 0.2))
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_losses.append(float(loss))
        
        avg_loss = np.mean(epoch_losses)
        history['loss'].append(avg_loss)
        history['epoch'].append(epoch + 1)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")
    
    # Save model
    model_path = os.path.join(args.output_dir, "triplet_model.keras")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save base network separately (for generation)
    base_path = os.path.join(args.output_dir, "base_network.keras")
    base_network.save(base_path)
    print(f"Base network saved to {base_path}")
    
    # Save training info
    training_info = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'steps_per_epoch': args.steps_per_epoch,
        'embedding_dim': args.embedding_dim,
        'learning_rate': args.learning_rate,
        'alphabets': list(stats.keys()),
        'total_characters': total_chars,
        'final_loss': history['loss'][-1],
        'alphabet_stats': {k: {'count': v['character_count'], 'scripts': v['scripts']} 
                          for k, v in stats.items()}
    }
    
    info_path = os.path.join(args.output_dir, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # Save loss history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Output: {args.output_dir}/")

if __name__ == "__main__":
    main()
