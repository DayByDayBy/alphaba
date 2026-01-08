#!/usr/bin/env python3
"""
Generate a fictional alphabet by blending characters from trained sources
"""

import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.unicode_alphabet_loader import UnicodeAlphabetLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Generate fictional alphabet")
    parser.add_argument('--font-dir', type=str, default='font_samples',
                       help='Font samples directory')
    parser.add_argument('--output-dir', type=str, default='generated',
                       help='Output directory')
    parser.add_argument('--name', type=str, default=None,
                       help='Alphabet name (default: auto-generated)')
    parser.add_argument('--sources', type=str, nargs='*', default=None,
                       help='Source alphabets to blend (default: random selection)')
    parser.add_argument('--style', type=str, default='mixed',
                       choices=['mixed', 'consistent', 'chaotic'],
                       help='Generation style')
    return parser.parse_args()

def select_source_alphabets(loader, requested_sources=None, n_sources=4):
    """Select source alphabets for blending"""
    available = list(loader.registered_alphabets.keys())
    
    if requested_sources:
        # Filter to requested sources that exist
        sources = [s for s in requested_sources if s in available]
        if not sources:
            print(f"None of requested sources found, using random")
            sources = list(np.random.choice(available, min(n_sources, len(available)), replace=False))
    else:
        # Random selection, prefer diverse scripts
        sources = list(np.random.choice(available, min(n_sources, len(available)), replace=False))
    
    return sources

def generate_fictional_alphabet(loader, source_alphabets, style='mixed'):
    """Generate 26 characters by blending from source alphabets"""
    
    # Collect all characters from sources
    all_chars = []
    for alph in source_alphabets:
        chars = loader.registered_alphabets.get(alph, [])
        for c in chars:
            c['source_alphabet'] = alph
        all_chars.extend(chars)
    
    if len(all_chars) < 26:
        print(f"Warning: Only {len(all_chars)} source characters available")
    
    # Generate 26 characters based on style
    generated = []
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    if style == 'consistent':
        # Pick one dominant alphabet, supplement with others
        dominant = source_alphabets[0]
        dominant_chars = [c for c in all_chars if c['source_alphabet'] == dominant]
        other_chars = [c for c in all_chars if c['source_alphabet'] != dominant]
        
        for i, letter in enumerate(letters):
            if i < len(dominant_chars):
                char = dominant_chars[i % len(dominant_chars)]
            else:
                char = np.random.choice(other_chars) if other_chars else np.random.choice(all_chars)
            generated.append({
                'letter': letter,
                'image': char['image'],
                'source': char['source_alphabet'],
                'original_char': char['char']
            })
    
    elif style == 'chaotic':
        # Pure random sampling
        for letter in letters:
            char = np.random.choice(all_chars)
            generated.append({
                'letter': letter,
                'image': char['image'],
                'source': char['source_alphabet'],
                'original_char': char['char']
            })
    
    else:  # mixed - balanced sampling from all sources
        chars_per_source = max(1, 26 // len(source_alphabets))
        selected = []
        
        for alph in source_alphabets:
            alph_chars = [c for c in all_chars if c['source_alphabet'] == alph]
            if alph_chars:
                n_select = min(chars_per_source, len(alph_chars))
                indices = np.random.choice(len(alph_chars), n_select, replace=False)
                selected.extend([alph_chars[i] for i in indices])
        
        # Fill remaining slots
        while len(selected) < 26:
            selected.append(np.random.choice(all_chars))
        
        # Shuffle and assign to letters
        np.random.shuffle(selected)
        for i, letter in enumerate(letters):
            char = selected[i]
            generated.append({
                'letter': letter,
                'image': char['image'],
                'source': char['source_alphabet'],
                'original_char': char['char']
            })
    
    return generated

def visualize_alphabet(generated, title, output_path):
    """Create alphabet reference sheet"""
    fig, axes = plt.subplots(2, 13, figsize=(20, 4))
    axes = axes.flatten()
    
    for i, (ax, char_data) in enumerate(zip(axes, generated)):
        ax.imshow(char_data['image'].squeeze(), cmap='gray')
        ax.set_title(char_data['letter'], fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

def save_individual_chars(generated, output_dir, prefix):
    """Save individual character images"""
    char_dir = os.path.join(output_dir, f"{prefix}_chars")
    os.makedirs(char_dir, exist_ok=True)
    
    for char_data in generated:
        plt.figure(figsize=(2, 2))
        plt.imshow(char_data['image'].squeeze(), cmap='gray')
        plt.axis('off')
        
        path = os.path.join(char_dir, f"{char_data['letter']}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

def save_metadata(generated, source_alphabets, output_path):
    """Save alphabet metadata"""
    metadata = {
        'sources': source_alphabets,
        'characters': [
            {
                'letter': c['letter'],
                'source_alphabet': c['source'],
                'original_char': c['original_char']
            }
            for c in generated
        ]
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def main():
    args = parse_args()
    
    print("=" * 60)
    print("ALPHABA - Fictional Alphabet Generator")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load alphabets
    print(f"\nLoading alphabets from {args.font_dir}...")
    loader = UnicodeAlphabetLoader(font_samples_dir=args.font_dir)
    loader.load_all_alphabets()
    
    stats = loader.get_alphabet_stats()
    print(f"Loaded {len(stats)} alphabets")
    
    # Select sources
    sources = select_source_alphabets(loader, args.sources)
    print(f"\nSource alphabets: {sources}")
    
    # Generate alphabet
    print(f"\nGenerating alphabet (style: {args.style})...")
    generated = generate_fictional_alphabet(loader, sources, args.style)
    
    # Create name
    if args.name:
        name = args.name
    else:
        timestamp = datetime.now().strftime("%H%M")
        source_short = '_'.join([s.split('_')[-1][:3] for s in sources[:2]])
        name = f"Alphabet_{source_short}_{timestamp}"
    
    # Save outputs
    print(f"\nSaving {name}...")
    
    # Reference sheet
    viz_path = os.path.join(args.output_dir, f"{name}.png")
    visualize_alphabet(generated, name, viz_path)
    print(f"  Reference sheet: {viz_path}")
    
    # Individual characters
    save_individual_chars(generated, args.output_dir, name)
    print(f"  Individual chars: {args.output_dir}/{name}_chars/")
    
    # Metadata
    meta_path = os.path.join(args.output_dir, f"{name}_meta.json")
    save_metadata(generated, sources, meta_path)
    print(f"  Metadata: {meta_path}")
    
    print(f"\n✓ Generated: {name}")
    print(f"  Sources: {', '.join(sources)}")
    print(f"  Style: {args.style}")

if __name__ == "__main__":
    main()
