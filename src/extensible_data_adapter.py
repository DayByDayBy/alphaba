"""
Adapter to connect extensible alphabet loader with existing training pipeline
Makes the new alphabet system work with current triplet network
"""

import numpy as np
import random
from extensible_alphabet_loader import ExtensibleAlphabetLoader
from collections import defaultdict

class ExtensibleTripletAdapter:
    def __init__(self, alphabet_data_dir="alphabet_data"):
        self.loader = ExtensibleAlphabetLoader(alphabet_data_dir)
        self.alphabet_groups = self._organize_by_alphabet()
        
    def _organize_by_alphabet(self):
        """Organize characters by alphabet for triplet sampling"""
        groups = defaultdict(list)
        
        enabled_alphabets = self.loader.get_enabled_alphabets()
        print(f"Enabled alphabets: {enabled_alphabets}")
        
        for alphabet_id in enabled_alphabets:
            try:
                characters = self.loader.get_alphabet_data(alphabet_id)
                print(f"Alphabet {alphabet_id}: {len(characters)} characters")
                for char_data in characters:
                    groups[alphabet_id].append(char_data)
            except Exception as e:
                print(f"Error loading {alphabet_id}: {e}")
        
        result = dict(groups)
        print(f"Organized {len(result)} alphabet groups")
        return result
    
    def sample_triplet(self):
        """Sample one triplet: (anchor, positive, negative)"""
        # Pick random alphabet for anchor
        anchor_alphabet = random.choice(list(self.alphabet_groups.keys()))
        anchor_chars = self.alphabet_groups[anchor_alphabet]
        
        # Pick random character as anchor
        anchor_data = random.choice(anchor_chars)
        anchor_img = anchor_data['image']
        anchor_az = anchor_data['az_char']
        
        # Find positive: different character from same alphabet
        same_alphabet_candidates = [
            char_data for char_data in anchor_chars 
            if char_data['az_char'] != anchor_az
        ]
        
        if len(same_alphabet_candidates) == 0:
            # Fallback: same character, different instance if available
            same_alphabet_candidates = anchor_chars
        
        positive_data = random.choice(same_alphabet_candidates)
        positive_img = positive_data['image']
        
        # Find negative: character from different alphabet
        negative_alphabet = random.choice([
            alph for alph in self.alphabet_groups.keys() 
            if alph != anchor_alphabet
        ])
        negative_chars = self.alphabet_groups[negative_alphabet]
        negative_data = random.choice(negative_chars)
        negative_img = negative_data['image']
        
        return anchor_img, positive_img, negative_img
    
    def generate_batch(self, batch_size):
        """Generate a batch of triplets"""
        anchors, positives, negatives = [], [], []
        
        for _ in range(batch_size):
            anchor, positive, negative = self.sample_triplet()
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
        
        return (np.array(anchors), np.array(positives), np.array(negatives))
    
    def get_alphabet_names(self):
        """Get list of alphabet names for compatibility"""
        return list(self.alphabet_groups.keys())
    
    def get_alphabet_stats(self):
        """Get statistics about loaded alphabets"""
        stats = {}
        for alphabet_id, characters in self.alphabet_groups.items():
            stats[alphabet_id] = {
                'name': self.loader.alphabet_registry[alphabet_id]['name'],
                'type': self.loader.alphabet_registry[alphabet_id]['type'],
                'character_count': len(characters),
                'unique_az_chars': len(set(char['az_char'] for char in characters))
            }
        return stats

# Test the adapter
if __name__ == "__main__":
    adapter = ExtensibleTripletAdapter("../alphabet_data")
    
    print("Alphabet Statistics:")
    for alph_id, stats in adapter.get_alphabet_stats().items():
        print(f"  {alph_id}: {stats['character_count']} chars, {stats['unique_az_chars']} unique A-Z")
    
    # Test batch generation
    anchors, positives, negatives = adapter.generate_batch(4)
    print(f"\nGenerated batch: {anchors.shape}")
    print(f"Batch range: [{anchors.min():.3f}, {anchors.max():.3f}]")
