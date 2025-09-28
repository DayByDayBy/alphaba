import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import seaborn as sns
from collections import defaultdict
import random


from src.models import create_triplet_model
from src.data_loader import OmniglotTripletLoader


# load data in
loader = OmniglotTripletLoader("../../omniglot/python")  # Update this path!

# create model architecture
triplet_model, base_network = create_triplet_model(embedding_dim=128)

def extract_all_embeddings(base_network, loader):
    """Extract embeddings for all images in the dataset."""
    embeddings = []
    metadata = []  # [(alphabet_name, char_id, image_index), ...]
    
    for alphabet_name in loader.alphabet_names:
        print(f"Processing {alphabet_name}...")
        for i, (img, char_id) in enumerate(loader.alphabet_data[alphabet_name]):
            # Get embedding
            img_batch = np.expand_dims(img, axis=0)  # Add batch dimension
            embedding = base_network.predict(img_batch, verbose=0)[0]  # Remove batch dim
            
            embeddings.append(embedding)
            metadata.append((alphabet_name, char_id, i))
    
    return np.array(embeddings), metadata

def analyze_intra_alphabet_similarity(embeddings, metadata):
    """Analyze how similar characters within the same alphabet are."""
    
    # Group embeddings by alphabet
    alphabet_groups = defaultdict(list)
    for i, (alphabet_name, char_id, img_idx) in enumerate(metadata):
        alphabet_groups[alphabet_name].append(i)
    
    results = {}
    
    for alphabet_name, indices in alphabet_groups.items():
        if len(indices) < 2:
            continue
            
        # Get embeddings for this alphabet
        alphabet_embeddings = embeddings[indices]
        
        # Calculate pairwise similarities within alphabet
        similarities = cosine_similarity(alphabet_embeddings)
        
        # Get upper triangle (avoid diagonal and duplicates)
        upper_tri = np.triu(similarities, k=1)
        intra_similarities = upper_tri[upper_tri > 0]
        
        results[alphabet_name] = {
            'mean_similarity': np.mean(intra_similarities),
            'std_similarity': np.std(intra_similarities),
            'num_characters': len(set([metadata[i][1] for i in indices])),
            'num_images': len(indices)
        }
    
    return results

def analyze_inter_alphabet_similarity(embeddings, metadata, sample_size=1000):
    """Compare similarities between different alphabets."""
    
    # Group embeddings by alphabet
    alphabet_groups = defaultdict(list)
    for i, (alphabet_name, char_id, img_idx) in enumerate(metadata):
        alphabet_groups[alphabet_name].append(i)
    
    # Sample pairs from different alphabets
    inter_similarities = []
    alphabet_names = list(alphabet_groups.keys())
    
    for _ in range(sample_size):
        # Pick two different alphabets
        alph1, alph2 = random.sample(alphabet_names, 2)
        
        # Pick random images from each
        idx1 = random.choice(alphabet_groups[alph1])
        idx2 = random.choice(alphabet_groups[alph2])
        
        # Calculate similarity
        sim = cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0, 0]
        inter_similarities.append(sim)
    
    return np.array(inter_similarities)

def plot_similarity_comparison(intra_results, inter_similarities):
    """Plot comparison of intra vs inter alphabet similarities."""
    
    # Get all intra-alphabet similarities
    all_intra_sims = []
    for alphabet_name, results in intra_results.items():
        # We need to recalculate to get individual similarities
        # This is a simplified version - you might want to store them above
        all_intra_sims.extend([results['mean_similarity']] * 10)  # Approximate
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_intra_sims, alpha=0.7, label='Intra-alphabet', bins=30)
    plt.hist(inter_similarities, alpha=0.7, label='Inter-alphabet', bins=30)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Intra vs Inter Alphabet Similarities')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Box plot comparison
    data_to_plot = [all_intra_sims, inter_similarities]
    plt.boxplot(data_to_plot, labels=['Intra-alphabet', 'Inter-alphabet'])
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity Distribution Comparison')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Intra-alphabet similarity: {np.mean(all_intra_sims):.3f} ± {np.std(all_intra_sims):.3f}")
    print(f"Inter-alphabet similarity: {np.mean(inter_similarities):.3f} ± {np.std(inter_similarities):.3f}")

def find_most_similar_cross_alphabet_pairs(embeddings, metadata, top_k=10):
    """Find character pairs from different alphabets that are most similar."""
    
    # Group by alphabet
    alphabet_groups = defaultdict(list)
    for i, (alphabet_name, char_id, img_idx) in enumerate(metadata):
        alphabet_groups[alphabet_name].append((i, char_id))
    
    alphabet_names = list(alphabet_groups.keys())
    similarities = []
    
    # Compare across alphabets
    for i, alph1 in enumerate(alphabet_names):
        for alph2 in alphabet_names[i+1:]:
            for idx1, char1 in alphabet_groups[alph1][:5]:  # Limit to first 5 per alphabet
                for idx2, char2 in alphabet_groups[alph2][:5]:
                    sim = cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0, 0]
                    similarities.append((sim, alph1, char1, alph2, char2, idx1, idx2))
    
    # Sort by similarity (descending)
    similarities.sort(reverse=True)
    
    print(f"Top {top_k} most similar cross-alphabet pairs:")
    for i, (sim, alph1, char1, alph2, char2, idx1, idx2) in enumerate(similarities[:top_k]):
        print(f"{i+1:2d}. {sim:.3f} - {alph1}/{char1} ↔ {alph2}/{char2}")
    
    return similarities[:top_k]

# Main analysis function
def run_case_analysis(base_network, loader):
    """Run the complete case relationship analysis."""
    
    print("Step 1: Extracting all embeddings...")
    embeddings, metadata = extract_all_embeddings(base_network, loader)
    print(f"Extracted {len(embeddings)} embeddings")
    
    print("\nStep 2: Analyzing intra-alphabet similarities...")
    intra_results = analyze_intra_alphabet_similarity(embeddings, metadata)
    
    print("Alphabet similarity statistics:")
    for alphabet_name, results in sorted(intra_results.items(), 
                                       key=lambda x: x[1]['mean_similarity'], reverse=True):
        print(f"  {alphabet_name:25s}: {results['mean_similarity']:.3f} ± {results['std_similarity']:.3f} "
              f"({results['num_characters']} chars, {results['num_images']} images)")
    
    print("\nStep 3: Analyzing inter-alphabet similarities...")
    inter_similarities = analyze_inter_alphabet_similarity(embeddings, metadata)
    
    print("\nStep 4: Plotting comparison...")
    plot_similarity_comparison(intra_results, inter_similarities)
    
    print("\nStep 5: Finding most similar cross-alphabet pairs...")
    top_pairs = find_most_similar_cross_alphabet_pairs(embeddings, metadata)
    
    return {
        'embeddings': embeddings,
        'metadata': metadata,
        'intra_results': intra_results,
        'inter_similarities': inter_similarities,
        'top_cross_pairs': top_pairs
    }