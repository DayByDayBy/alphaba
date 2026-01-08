#!/usr/bin/env python3
"""
Setup script for alphaba alphabets
Tests and registers working fonts for training
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.extensible_alphabet_loader import ExtensibleAlphabetLoader
from src.font_tester import FontTester
import matplotlib.font_manager as fm

def find_working_fonts():
    """Find fonts that actually render characters correctly"""
    tester = FontTester()
    
    # Test key characters for each script
    working_fonts = {
        'latin': [],
        'greek': [],
        'armenian': [],
        'georgian': []
    }
    
    font_paths = fm.findSystemFonts()
    
    for font_path in font_paths:
        try:
            font = fm.FontProperties(fname=font_path)
            font_name = font.get_name()
            
            # Test Greek characters
            try:
                from PIL import ImageFont
                pil_font = ImageFont.truetype(font_path, 20)
                
                # Test a few key characters
                greek_chars = ['Α', 'Β', 'Γ', 'Δ', 'Ε']
                greek_works = all(pil_font.getbbox(char) != (0, 0, 0, 0) for char in greek_chars)
                
                if greek_works:
                    working_fonts['greek'].append((font_path, font_name))
                    print(f"✓ Greek: {font_name}")
                
                # Test Armenian
                armenian_chars = ['Ա', 'Բ', 'Գ', 'Դ', 'Ե']
                armenian_works = all(pil_font.getbbox(char) != (0, 0, 0, 0) for char in armenian_chars)
                
                if armenian_works:
                    working_fonts['armenian'].append((font_path, font_name))
                    print(f"✓ Armenian: {font_name}")
                
                # Test Latin (always works)
                working_fonts['latin'].append((font_path, font_name))
                
            except:
                continue
                
        except:
            continue
    
    return working_fonts

def setup_core_alphabets():
    """Setup the core alphabets that work well"""
    loader = ExtensibleAlphabetLoader()
    
    # Find working fonts
    print("Finding working fonts...")
    working_fonts = find_working_fonts()
    
    # Register the best ones
    if working_fonts['greek']:
        # Use Arial Narrow Bold for Greek (we know it works)
        greek_font = None
        for font_path, font_name in working_fonts['greek']:
            if 'Arial' in font_name and 'Narrow' in font_name:
                greek_font = font_path
                break
        
        if greek_font:
            loader.register_system_font("Greek Primary", greek_font, "greek")
        else:
            # Fallback to first working Greek font
            font_path, font_name = working_fonts['greek'][0]
            loader.register_system_font("Greek Primary", font_path, "greek")
    
    if working_fonts['armenian']:
        # Use first working Armenian font
        font_path, font_name = working_fonts['armenian'][0]
        loader.register_system_font("Armenian Primary", font_path, "armenian")
    
    # Always add Latin
    if working_fonts['latin']:
        # Find a nice serif font for Latin
        latin_font = None
        for font_path, font_name in working_fonts['latin']:
            if 'Times' in font_name or 'Georgia' in font_name:
                latin_font = font_path
                break
        
        if not latin_font:
            latin_font = working_fonts['latin'][0][0]
        
        loader.register_system_font("Latin Primary", latin_font, "latin")
    
    print("\nRegistered alphabets:")
    loader.list_registered_alphabets()
    
    return loader

def add_custom_font_example():
    """Example of how to add a custom font"""
    loader = ExtensibleAlphabetLoader()
    
    # This would be used like:
    # loader.register_custom_font(
    #     "My Custom Alphabet", 
    #     "/path/to/my_font.ttf", 
    #     "custom_mapping_name"
    # )
    
    print("\nTo add custom fonts, use:")
    print("loader.register_custom_font('Alphabet Name', '/path/to/font.ttf', 'script_type')")

def create_sample_training_data(loader):
    """Create sample training data to test the pipeline"""
    print("\nGenerating sample training data...")
    
    try:
        images, labels = loader.generate_training_batch(8)
        print(f"✓ Generated training batch: {images.shape}")
        print(f"Sample labels: {labels[:4]}")
        
        # Save a sample
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(images):
                ax.imshow(images[i].squeeze(), cmap='gray')
                ax.set_title(labels[i].split('_')[-1])
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_training_data.png', dpi=150)
        print("✓ Saved sample visualization: sample_training_data.png")
        
    except Exception as e:
        print(f"✗ Error generating training data: {e}")

if __name__ == "__main__":
    print("Setting up Alphaba alphabets...")
    
    # Setup core alphabets
    loader = setup_core_alphabets()
    
    # Create sample training data
    create_sample_training_data(loader)
    
    # Show how to add more
    add_custom_font_example()
    
    print("\nNext steps:")
    print("1. Test training with: python main.py demo --data-path alphabet_data")
    print("2. Add more fonts to alphabet_data/fonts/")
    print("3. Register them with loader.register_custom_font()")
    print("4. Create character mappings in alphabet_data/character_mappings.json")
