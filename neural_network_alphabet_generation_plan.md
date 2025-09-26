# Neural Network Alphabet Generation Plan

This document outlines the approach for creating a neural network that can generate new fictional alphabets for worldbuilding projects. The goal is to create visually cohesive 26-character alphabets that map 1:1 to the English alphabet (A-Z, a-z, 0-9, and punctuation), suitable for encoding English text in fictional writing systems.

## Project Goals

**Primary Objectives:**
- Learn neural network fundamentals through hands-on implementation
- Generate first usable fictional alphabet quickly
- Create a foundation for future alphabet generation experiments
- Build a worldbuilding tool for creative projects

**Target Output:** A neural network that can generate complete alphabet sets where:
- All 26 characters feel visually consistent (like they belong to the same writing system)
- Characters are distinctive enough to be readable/usable
- The style draws inspiration from real writing systems without copying them
- Think Tengwar, Klingon, or fantasy game alphabets

## High-Level Approach

This is a **representation learning** problem. We'll train a model to understand the visual patterns and systematic relationships within writing systems, then use that knowledge to generate new, cohesive alphabet sets.

### Phase 1: Learn Alphabet Representations

**Data Source:** Complete Omniglot dataset (50 alphabets, 1,623 characters, 20 examples each)

**Strategy:** Start with the full dataset to maximize learning, then iteratively refine by removing alphabets that don't contribute to desired aesthetics.

**Architecture:** Siamese Network with Triplet Loss
- Learn embeddings where characters from the same alphabet cluster together
- Characters from different alphabets are pushed apart in embedding space
- Creates structured representation space perfect for generation

### Phase 2: Generate New Alphabets

Use the learned representations to generate new alphabet sets that maintain visual consistency while being functionally complete for English encoding.

## Technical Implementation

### 1. Data Preparation

**Dataset:** Omniglot (Brenden Lake's version)
- Already preprocessed: 105x105 pixel grayscale images
- Organized by alphabet and character
- Perfect train/validation splits included

**Data Augmentation:**
- Rotation (±15 degrees)
- Small translations
- Scale variations
- Light noise addition
- Maintains alphabet coherence while teaching variation tolerance

### 2. Model Architecture

**Base Network:** Convolutional Neural Network for feature extraction

```python
def create_base_network(input_shape=(105, 105, 1)):
    model = models.Sequential([
        layers.Conv2D(64, (10, 10), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (7, 7), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (4, 4), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, (4, 4), activation='relu'),
        layers.Flatten(),
        layers.Dense(4096, activation='sigmoid')
    ])
    return model
```

**Training Framework:** Triplet Loss Network
- **Anchor:** Character from an alphabet
- **Positive:** Different character from same alphabet
- **Negative:** Character from different alphabet
- Minimizes anchor-positive distance, maximizes anchor-negative distance

### 3. Training Strategy

**Loss Function:** Triplet Loss with margin
```python
def triplet_loss(y_true, y_pred, margin=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    loss = tf.maximum(0.0, pos_dist - neg_dist + margin)
    return tf.reduce_mean(loss)
```

**Training Process:**
1. Sample triplets (anchor, positive, negative)
2. Generate embeddings through base network
3. Calculate triplet loss
4. Backpropagate and update weights

### 4. Evaluation and Visualization

**Embedding Quality:**
- t-SNE/UMAP visualization of learned embeddings
- Should show clear clusters by alphabet
- Quantitative clustering metrics (silhouette score)

**Alphabet Coherence:**
- Within-alphabet similarity scores
- Between-alphabet distinction measures
- Visual inspection of closest neighbors in embedding space

## Implementation Guide for Getting Started

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv alphabet_gen_env
source alphabet_gen_env/bin/activate  # On Windows: alphabet_gen_env\Scripts\activate

# Install dependencies
pip install tensorflow opencv-python numpy matplotlib scikit-learn seaborn
```

### Step 2: Dataset Preparation

```bash
# Clone Omniglot dataset into separate repo
git clone https://github.com/brendenlake/omniglot.git omniglot-dataset
```

Create data loading utilities:
```python
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_omniglot_data(data_path):
    """Load all Omniglot data with alphabet labels."""
    alphabets = []
    characters = []
    labels = []
    
    # Process background and evaluation sets
    for split in ['images_background', 'images_evaluation']:
        split_path = os.path.join(data_path, split)
        for alphabet in os.listdir(split_path):
            alphabet_path = os.path.join(split_path, alphabet)
            if os.path.isdir(alphabet_path):
                for character in os.listdir(alphabet_path):
                    character_path = os.path.join(alphabet_path, character)
                    if os.path.isdir(character_path):
                        for image_file in os.listdir(character_path):
                            if image_file.endswith('.png'):
                                img_path = os.path.join(character_path, image_file)
                                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                img = img.astype(np.float32) / 255.0
                                
                                characters.append(img)
                                labels.append(alphabet)
    
    return np.array(characters), np.array(labels)
```

### Step 3: Build and Train the Model

Start with a simple triplet network implementation:

```python
def create_triplet_model(base_network):
    """Create triplet network with shared weights."""
    anchor_input = layers.Input(shape=(105, 105, 1))
    positive_input = layers.Input(shape=(105, 105, 1))
    negative_input = layers.Input(shape=(105, 105, 1))
    
    # Shared embeddings
    anchor_embedding = base_network(anchor_input)
    positive_embedding = base_network(positive_input)
    negative_embedding = base_network(negative_input)
    
    # Concatenate for loss calculation
    output = layers.concatenate([anchor_embedding, positive_embedding, negative_embedding])
    
    model = models.Model(inputs=[anchor_input, positive_input, negative_input], 
                        outputs=output)
    return model
```

### Step 4: Visualization and Analysis

```python
def visualize_embeddings(model, test_data, test_labels):
    """Visualize learned embeddings with t-SNE."""
    # Generate embeddings
    base_network = model.layers[3]  # Extract base network
    embeddings = base_network.predict(test_data)
    
    # Reduce dimensionality
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot by alphabet
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(test_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, alphabet in enumerate(unique_labels):
        mask = test_labels == alphabet
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], label=alphabet, alpha=0.6)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Learned Alphabet Embeddings')
    plt.tight_layout()
    plt.show()
```

## Success Milestones

### Phase 1 Complete When:
- [ ] t-SNE visualization shows clear alphabet clusters
- [ ] Quantitative metrics confirm good separation
- [ ] Model can reliably identify alphabet family from character embedding
- [ ] Training loss converges and validation metrics are stable

### Phase 2 Goals:
- [ ] Generate first complete 26-character alphabet
- [ ] Characters appear visually consistent
- [ ] Alphabet is readable/usable for encoding English text
- [ ] Style reflects learned patterns from training data

## Next Steps and Extensions

**Immediate Next Phase:**
- Implement VAE or GAN using learned embeddings as conditioning
- Generate new characters that fit specific regions of embedding space
- Ensure generated alphabet maintains internal consistency

**Future Enhancements:**
- Modular training on alphabet subsets for style control
- Integration of custom alphabets (Tengwar, etc.)
- Multi-style generation (geometric vs. flowing)
- Interactive alphabet editing tools

## Tech Stack

- **Language:** Python 3.8+
- **Deep Learning:** TensorFlow/Keras
- **Image Processing:** OpenCV, Pillow
- **Data Science:** NumPy, pandas, scikit-learn
- **Visualization:** Matplotlib, seaborn
- **Development:** Jupyter notebooks for experimentation

---

*This project prioritizes learning neural network fundamentals while building toward a practical worldbuilding tool. Start simple, iterate quickly, and expand complexity as understanding grows.*