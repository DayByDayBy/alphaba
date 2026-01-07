"""
Evaluation metrics for Alphaba
Focus on intra-alphabet and inter-alphabet variation metrics
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns


def compute_pairwise_distances(embeddings, labels):
    """Compute pairwise distances grouped by alphabet"""
    distances = {}
    
    # Group embeddings by alphabet
    alphabet_groups = {}
    for i, label in enumerate(labels):
        if label not in alphabet_groups:
            alphabet_groups[label] = []
        alphabet_groups[label].append(embeddings[i])
    
    # Compute intra-alphabet distances
    intra_distances = []
    for alphabet, group in alphabet_groups.items():
        if len(group) > 1:
            group_array = np.array(group)
            pair_dists = pdist(group_array, metric='euclidean')
            intra_distances.extend(pair_dists)
    
    # Compute inter-alphabet distances
    inter_distances = []
    alphabets = list(alphabet_groups.keys())
    for i in range(len(alphabets)):
        for j in range(i + 1, len(alphabets)):
            group1 = np.array(alphabet_groups[alphabets[i]])
            group2 = np.array(alphabet_groups[alphabets[j]])
            
            # Sample to avoid memory issues
            n_samples = min(50, len(group1), len(group2))
            idx1 = np.random.choice(len(group1), n_samples, replace=False)
            idx2 = np.random.choice(len(group2), n_samples, replace=False)
            
            cross_dists = np.linalg.norm(
                group1[idx1][:, np.newaxis] - group2[idx2], axis=2
            )
            inter_distances.extend(cross_dists.flatten())
    
    return {
        'intra_distances': np.array(intra_distances),
        'inter_distances': np.array(inter_distances),
        'intra_mean': np.mean(intra_distances),
        'inter_mean': np.mean(inter_distances),
        'intra_std': np.std(intra_distances),
        'inter_std': np.std(inter_distances)
    }


def compute_clustering_metrics(embeddings, labels):
    """Compute clustering quality metrics"""
    # Silhouette score (higher is better)
    silhouette_avg = silhouette_score(embeddings, labels)
    
    # K-means clustering accuracy
    n_clusters = len(set(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return {
        'silhouette_score': silhouette_avg,
        'n_clusters_found': n_clusters,
        'kmeans_inertia': kmeans.inertia_
    }


def compute_separation_ratio(intra_mean, inter_mean):
    """Compute ratio of inter to intra alphabet distances"""
    if intra_mean == 0:
        return float('inf')
    return inter_mean / intra_mean


def plot_distance_distributions(intra_distances, inter_distances, output_path=None):
    """Plot distribution of intra vs inter alphabet distances"""
    plt.figure(figsize=(10, 6))
    
    sns.histplot(intra_distances, label='Intra-alphabet', alpha=0.7, bins=50)
    sns.histplot(inter_distances, label='Inter-alphabet', alpha=0.7, bins=50)
    
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Character Distances')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def comprehensive_evaluation(embeddings, labels, output_dir=None):
    """Run comprehensive evaluation with all metrics"""
    print("Running comprehensive evaluation...")
    
    # Compute distance metrics
    distance_metrics = compute_pairwise_distances(embeddings, labels)
    
    # Compute clustering metrics
    clustering_metrics = compute_clustering_metrics(embeddings, labels)
    
    # Compute separation ratio
    separation_ratio = compute_separation_ratio(
        distance_metrics['intra_mean'], 
        distance_metrics['inter_mean']
    )
    
    # Combine all metrics
    all_metrics = {
        **distance_metrics,
        **clustering_metrics,
        'separation_ratio': separation_ratio
    }
    
    # Print summary
    print("\n=== Evaluation Results ===")
    print(f"Intra-alphabet distance: {distance_metrics['intra_mean']:.3f} ± {distance_metrics['intra_std']:.3f}")
    print(f"Inter-alphabet distance: {distance_metrics['inter_mean']:.3f} ± {distance_metrics['inter_std']:.3f}")
    print(f"Separation ratio: {separation_ratio:.3f}")
    print(f"Silhouette score: {clustering_metrics['silhouette_score']:.3f}")
    print(f"K-means inertia: {clustering_metrics['kmeans_inertia']:.3f}")
    
    # Plot distributions
    if output_dir:
        plot_distance_distributions(
            distance_metrics['intra_distances'],
            distance_metrics['inter_distances'],
            os.path.join(output_dir, 'distance_distributions.png')
        )
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.npy')
        np.save(metrics_path, all_metrics)
        print(f"Evaluation metrics saved to {metrics_path}")
    
    return all_metrics
