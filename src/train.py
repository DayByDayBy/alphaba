#!/usr/bin/env python3
"""
Main training script for Alphaba
Replaces notebook-based training with proper CLI interface
"""

import argparse
import os
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_loader import OmniglotTripletLoader
from models import create_triplet_model, normalize_embeddings
from training import train_triplet_model_custom, evaluate_embeddings
from data_augmentation import create_augmented_data_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Alphaba triplet network")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to Omniglot dataset (python folder)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--steps-per-epoch", type=int, default=100,
                        help="Training steps per epoch")
    parser.add_argument("--embedding-dim", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Triplet loss margin")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save models and plots")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation, skip training")
    parser.add_argument("--model-path", type=str,
                        help="Path to saved model for evaluation")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--aug-rotation", type=float, default=15.0,
                        help="Augmentation rotation range in degrees")
    parser.add_argument("--aug-zoom", type=float, default=0.1,
                        help="Augmentation zoom range")
    parser.add_argument("--aug-noise", type=float, default=0.01,
                        help="Augmentation noise standard deviation")
    return parser.parse_args()


def create_output_directory(output_dir):
    Path(output_dir).mkdir(exist_ok=True)
    return output_dir


def save_model_and_history(model, history, output_dir):
    """Save trained model and training history"""
    model_path = os.path.join(output_dir, "triplet_model.h5")
    model.save(model_path)
    
    history_path = os.path.join(output_dir, "training_history.npy")
    np.save(history_path, history)
    
    print(f"Model saved to {model_path}")
    print(f"History saved to {history_path}")


def plot_training_history(history, output_dir):
    """Plot and save training loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Loss')
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training plot saved to {plot_path}")


def run_evaluation(model, data_loader, output_dir, num_samples=500):
    """Run embedding evaluation and save results"""
    print("Running embedding evaluation...")
    embeddings, labels = evaluate_embeddings(model, data_loader, num_samples)
    
    # Save embeddings and labels
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    labels_path = os.path.join(output_dir, "labels.npy")
    
    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)
    
    print(f"Embeddings saved to {embeddings_path}")
    print(f"Labels saved to {labels_path}")
    
    return embeddings, labels


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data_loader = OmniglotTripletLoader(args.data_path)
    
    # Setup augmentation if enabled
    if not args.no_augmentation:
        print("Setting up data augmentation...")
        augmentation_config = {
            'rotation_range': args.aug_rotation,
            'zoom_range': args.aug_zoom,
            'noise_std': args.aug_noise,
            'elastic_alpha': 1.0,
            'elastic_sigma': 8.0
        }
        data_loader = create_augmented_data_loader(data_loader, augmentation_config)
        print(f"Augmentation enabled: rotation={args.aug_rotation}°, zoom={args.aug_zoom}, noise={args.aug_noise}")
    else:
        print("Data augmentation disabled")
    
    if args.eval_only:
        # Evaluation only mode
        if not args.model_path:
            print("Error: --model-path required for evaluation mode")
            sys.exit(1)
        
        print(f"Loading model from {args.model_path}")
        model = tf.keras.models.load_model(args.model_path, custom_objects={
            'triplet_loss': lambda: lambda y_true, y_pred: tf.reduce_mean(
                tf.maximum(0.0, 
                    tf.reduce_sum(tf.square(y_pred[0] - y_pred[1]), axis=-1) - 
                    tf.reduce_sum(tf.square(y_pred[0] - y_pred[2]), axis=-1) + 
                    args.margin))
        })
        
        run_evaluation(model, data_loader, output_dir)
        return
    
    # Training mode
    print("Creating model...")
    triplet_model, base_network = create_triplet_model(embedding_dim=args.embedding_dim)
    
    print(f"Training for {args.epochs} epochs...")
    history = train_triplet_model_custom(
        triplet_model,
        data_loader,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        margin=args.margin,
        use_augmentation=not args.no_augmentation
    )
    
    # Save results
    save_model_and_history(triplet_model, history, output_dir)
    plot_training_history(history, output_dir)
    
    # Run evaluation
    run_evaluation(triplet_model, data_loader, output_dir)
    
    print(f"Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
