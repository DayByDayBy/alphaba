#!/usr/bin/env python3
"""
Alphaba: Machine Learning Alphabet Project
Main entry point for training and generating fictional alphabets
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import main as train_main
from src.generator import AlphabetGenerator, create_sample_alphabets
from src.evaluation import comprehensive_evaluation
from src.data_loader import OmniglotTripletLoader
from src.config import AlphabaConfig
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Alphaba - Fictional Alphabet Generation")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the triplet network')
    train_parser.add_argument('--data-path', type=str, required=True,
                             help='Path to Omniglot dataset (python folder)')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size for training')
    train_parser.add_argument('--embedding-dim', type=int, default=64,
                             help='Embedding dimension')
    train_parser.add_argument('--output-dir', type=str, default='outputs',
                             help='Directory to save results')
    
    # Generation command
    gen_parser = subparsers.add_parser('generate', help='Generate fictional alphabets')
    gen_parser.add_argument('--model-path', type=str, required=True,
                           help='Path to trained model')
    gen_parser.add_argument('--data-path', type=str, required=True,
                           help='Path to Omniglot dataset')
    gen_parser.add_argument('--output-dir', type=str, default='generated_alphabets',
                           help='Directory to save generated alphabets')
    gen_parser.add_argument('--n-alphabets', type=int, default=5,
                           help='Number of alphabets to generate')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to trained model')
    eval_parser.add_argument('--data-path', type=str, required=True,
                            help='Path to Omniglot dataset')
    eval_parser.add_argument('--output-dir', type=str, default='evaluation',
                            help='Directory to save evaluation results')
    
    # Demo command - quick training + generation
    demo_parser = subparsers.add_parser('demo', help='Run quick demo with default settings')
    demo_parser.add_argument('--data-path', type=str, required=True,
                            help='Path to Omniglot dataset')
    demo_parser.add_argument('--epochs', type=int, default=10,
                            help='Number of training epochs for demo')
    
    return parser.parse_args()


def run_generation(args):
    """Run alphabet generation"""
    print(f"Loading model from {args.model_path}")
    
    # Load model
    model = tf.keras.models.load_model(args.model_path, custom_objects={
        'triplet_loss': lambda: lambda y_true, y_pred: tf.reduce_mean(
            tf.maximum(0.0, 
                tf.reduce_sum(tf.square(y_pred[0] - y_pred[1]), axis=-1) - 
                tf.reduce_sum(tf.square(y_pred[0] - y_pred[2]), axis=-1) + 
                0.2))
    })
    
    # Extract base network
    base_network = model.layers[3]
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data_loader = OmniglotTripletLoader(args.data_path)
    
    # Create generator
    generator = AlphabetGenerator(base_network)
    
    # Learn styles from data
    generator.learn_alphabet_styles(data_loader)
    
    # Train decoder
    generator.train_decoder(data_loader)
    
    # Generate alphabets
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i in range(args.n_alphabets):
        print(f"Generating alphabet {i+1}/{args.n_alphabets}")
        
        # Generate with different approaches
        if i == 0:
            characters = generator.generate_alphabet()  # Random style
            title = "Random Style Alphabet"
            name = "random"
        elif i == 1 and len(generator.alphabet_embeddings) >= 2:
            alphabets = list(generator.alphabet_embeddings.keys())
            characters = generator.generate_alphabet(
                source_alphabets=[alphabets[0], alphabets[1]], 
                interpolation_weights=[0.6, 0.4]
            )
            title = f"Interpolated ({alphabets[0]} + {alphabets[1]})"
            name = "interpolated"
        else:
            characters = generator.generate_alphabet()  # Random style
            title = f"Generated Alphabet {i+1}"
            name = f"generated_{i+1}"
        
        # Visualize and save
        generator.visualize_alphabet(characters, title, 
                                    os.path.join(args.output_dir, f'{name}.png'))
        generator.save_alphabet_images(characters, args.output_dir, name)
    
    print(f"Generated {args.n_alphabets} alphabets in {args.output_dir}")


def run_evaluation(args):
    """Run model evaluation"""
    print(f"Evaluating model {args.model_path}")
    
    # Load model
    model = tf.keras.models.load_model(args.model_path, custom_objects={
        'triplet_loss': lambda: lambda y_true, y_pred: tf.reduce_mean(
            tf.maximum(0.0, 
                tf.reduce_sum(tf.square(y_pred[0] - y_pred[1]), axis=-1) - 
                tf.reduce_sum(tf.square(y_pred[0] - y_pred[2]), axis=-1) + 
                0.2))
    })
    
    # Load data
    data_loader = OmniglotTripletLoader(args.data_path)
    
    # Run evaluation
    from src.training import evaluate_embeddings
    embeddings, labels = evaluate_embeddings(model, data_loader)
    
    # Comprehensive evaluation
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = comprehensive_evaluation(embeddings, labels, args.output_dir)
    
    print(f"Evaluation complete! Results saved to {args.output_dir}")


def run_demo(args):
    """Run quick demo"""
    print("Running Alphaba demo...")
    
    # Quick training
    output_dir = "demo_outputs"
    train_args = [
        '--data-path', args.data_path,
        '--epochs', str(args.epochs),
        '--output-dir', output_dir
    ]
    
    # Temporarily set sys.argv for training
    original_argv = sys.argv
    sys.argv = ['train'] + train_args
    
    try:
        train_main()
        model_path = os.path.join(output_dir, "triplet_model.h5")
        
        if os.path.exists(model_path):
            print("Training complete! Now generating alphabets...")
            
            # Generate alphabets
            gen_args = argparse.Namespace(
                model_path=model_path,
                data_path=args.data_path,
                output_dir=os.path.join(output_dir, "generated"),
                n_alphabets=3
            )
            run_generation(gen_args)
            
            print("Demo complete! Check demo_outputs/ for results.")
        else:
            print("Training failed - model not found")
    finally:
        sys.argv = original_argv


def main():
    args = parse_args()
    
    if not args.command:
        print("Please specify a command: train, generate, evaluate, or demo")
        sys.exit(1)
    
    if args.command == 'train':
        # Convert to train.py format
        train_args = [
            '--data-path', args.data_path,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--embedding-dim', str(args.embedding_dim),
            '--output-dir', args.output_dir
        ]
        
        original_argv = sys.argv
        sys.argv = ['train'] + train_args
        try:
            train_main()
        finally:
            sys.argv = original_argv
    
    elif args.command == 'generate':
        run_generation(args)
    
    elif args.command == 'evaluate':
        run_evaluation(args)
    
    elif args.command == 'demo':
        run_demo(args)


if __name__ == "__main__":
    main()
