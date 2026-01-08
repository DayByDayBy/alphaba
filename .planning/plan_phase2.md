# Phase 2 Plan: Generative Architecture

**Goal**: Build a generative model that produces novel glyphs from a continuous latent space, conditioned on alphabet-level style variables.

**Prerequisites**: Phase 1 complete — alphabet tensors, data loader, validated pipeline.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALPHABET ENCODER                              │
│  Input: Full alphabet (52 glyphs)                               │
│  Output: Style vector z_style ∈ ℝ^d                             │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │ Glyph    │───▶│ Set      │───▶│ Style    │                   │
│  │ Encoder  │    │ Pooling  │    │ Vector   │                   │
│  │ (shared) │    │ (DeepSet)│    │ z_style  │                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GLYPH DECODER                                 │
│  Input: z_style + glyph_id (one-hot or embedding)               │
│  Output: Generated glyph (points, strokes, or raster)           │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │ Style +  │───▶│ Decoder  │───▶│ Output   │                   │
│  │ Glyph ID │    │ Network  │    │ Glyph    │                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key principle**: The decoder NEVER sees training glyphs directly. It only receives `z_style` and `glyph_id`. This prevents glyph copying.

---

## Step 1: Define Output Representation

**Duration**: 2-3 days  
**Decision required**

### Option A: Point Cloud (Recommended for Phase 2)

Output: `(256, 2)` — fixed number of 2D points (arc-length samples)

**Pros**: 
- Matches pipeline output
- Fixed dimension (easy to decode)
- Differentiable

**Cons**:
- No stroke ordering
- Reconstruction loss requires correspondence

```python
class PointCloudDecoder(tf.keras.Model):
    def __init__(self, n_points=256, style_dim=128, glyph_vocab=52):
        super().__init__()
        self.glyph_embedding = layers.Embedding(glyph_vocab, 32)
        self.decoder = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(n_points * 2),  # Output: flattened points
            layers.Reshape((n_points, 2))
        ])
    
    def call(self, style_vector, glyph_id):
        glyph_emb = self.glyph_embedding(glyph_id)
        combined = tf.concat([style_vector, glyph_emb], axis=-1)
        return self.decoder(combined)
```

### Option B: Stroke Sequence (Future)

Output: Variable-length `(T, 5)` — (x, y, pen_down, pen_up, end)

**Pros**:
- Natural drawing representation
- Captures stroke order

**Cons**:
- Variable length (needs RNN/Transformer)
- More complex training

### Option C: Raster (Simpler but limited)

Output: `(64, 64, 1)` — binary image

**Pros**:
- Well-understood (CNN decoders)
- Easy reconstruction loss (BCE)

**Cons**:
- Loses vector precision
- Harder to extract usable glyphs

**Recommendation**: Start with **Point Cloud** for Phase 2. Migrate to **Stroke Sequence** in Phase 3.

---

## Step 2: Implement Alphabet Encoder

**Duration**: 3-4 days  
**New file**: `src/alphabet_encoder.py`

### 2.1 Glyph Encoder (Shared)

Encodes individual glyphs to per-glyph features:

```python
class GlyphEncoder(tf.keras.Model):
    """Encode single glyph point cloud to feature vector."""
    
    def __init__(self, feature_dim=128):
        super().__init__()
        # PointNet-style encoder
        self.point_encoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
        ])
        self.global_pool = layers.GlobalMaxPooling1D()
        self.fc = layers.Dense(feature_dim)
    
    def call(self, points):
        # points: (batch, n_points, 2)
        x = self.point_encoder(points)  # (batch, n_points, 256)
        x = self.global_pool(x)         # (batch, 256)
        return self.fc(x)               # (batch, feature_dim)
```

### 2.2 Set Encoder (DeepSets)

Aggregates per-glyph features into alphabet-level style:

```python
class AlphabetEncoder(tf.keras.Model):
    """Encode full alphabet to style vector using DeepSets."""
    
    def __init__(self, style_dim=128, n_glyphs=52):
        super().__init__()
        self.glyph_encoder = GlyphEncoder(feature_dim=128)
        
        # Per-glyph transform (φ in DeepSets)
        self.phi = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
        ])
        
        # Post-aggregation transform (ρ in DeepSets)
        self.rho = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(style_dim),
        ])
    
    def call(self, alphabet):
        # alphabet: (batch, n_glyphs, n_points, 2)
        batch_size = tf.shape(alphabet)[0]
        n_glyphs = tf.shape(alphabet)[1]
        
        # Encode each glyph
        flat = tf.reshape(alphabet, (-1, alphabet.shape[2], 2))
        glyph_features = self.glyph_encoder(flat)  # (batch*n_glyphs, 128)
        glyph_features = tf.reshape(glyph_features, (batch_size, n_glyphs, -1))
        
        # Apply φ to each glyph
        transformed = self.phi(glyph_features)  # (batch, n_glyphs, 256)
        
        # Sum-pool over glyphs (permutation invariant)
        aggregated = tf.reduce_mean(transformed, axis=1)  # (batch, 256)
        
        # Apply ρ to get style vector
        return self.rho(aggregated)  # (batch, style_dim)
```

---

## Step 3: Implement Glyph Decoder

**Duration**: 3-4 days  
**New file**: `src/glyph_decoder.py`

### 3.1 Basic Decoder

```python
class GlyphDecoder(tf.keras.Model):
    """Decode style vector + glyph ID to point cloud."""
    
    def __init__(self, n_points=256, style_dim=128, n_glyphs=52):
        super().__init__()
        self.n_points = n_points
        
        # Learnable glyph embeddings
        self.glyph_embedding = layers.Embedding(n_glyphs, 64)
        
        # Decoder MLP
        self.decoder = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(n_points * 2, activation='sigmoid'),  # Normalized coords
            layers.Reshape((n_points, 2))
        ])
    
    def call(self, style_vector, glyph_id):
        # style_vector: (batch, style_dim)
        # glyph_id: (batch,) integers
        
        glyph_emb = self.glyph_embedding(glyph_id)  # (batch, 64)
        combined = tf.concat([style_vector, glyph_emb], axis=-1)
        return self.decoder(combined)
```

### 3.2 Reconstruction Loss

Point cloud matching via Chamfer distance:

```python
def chamfer_distance(pred, target):
    """Compute Chamfer distance between point clouds."""
    # pred, target: (batch, n_points, 2)
    
    # Expand for pairwise distances
    pred_exp = tf.expand_dims(pred, 2)      # (batch, n_pred, 1, 2)
    target_exp = tf.expand_dims(target, 1)  # (batch, 1, n_target, 2)
    
    # Pairwise squared distances
    dists = tf.reduce_sum(tf.square(pred_exp - target_exp), axis=-1)
    
    # Min distance from each pred point to target
    min_pred_to_target = tf.reduce_min(dists, axis=2)
    
    # Min distance from each target point to pred
    min_target_to_pred = tf.reduce_min(dists, axis=1)
    
    # Chamfer = mean of both directions
    return tf.reduce_mean(min_pred_to_target) + tf.reduce_mean(min_target_to_pred)
```

---

## Step 4: Training Loop

**Duration**: 3-4 days  
**New file**: `src/generative_training.py`

### 4.1 Combined Model

```python
class AlphabetVAE(tf.keras.Model):
    """Full alphabet-to-glyph generative model."""
    
    def __init__(self, style_dim=128, n_points=256, n_glyphs=52):
        super().__init__()
        self.encoder = AlphabetEncoder(style_dim=style_dim)
        self.decoder = GlyphDecoder(n_points=n_points, style_dim=style_dim)
        
    def call(self, alphabet, glyph_ids):
        # Encode full alphabet to style
        style = self.encoder(alphabet)
        
        # Decode requested glyphs
        return self.decoder(style, glyph_ids)
    
    def generate(self, style_vector, glyph_ids):
        """Generate glyphs from explicit style vector."""
        return self.decoder(style_vector, glyph_ids)
```

### 4.2 Training Step

```python
@tf.function
def train_step(model, optimizer, alphabet_batch, target_glyphs, glyph_ids):
    """
    alphabet_batch: (batch, 52, 256, 2) - full alphabets
    target_glyphs: (batch, 256, 2) - target glyph point clouds
    glyph_ids: (batch,) - which glyph to reconstruct
    """
    with tf.GradientTape() as tape:
        # Encode alphabet, decode specific glyph
        pred_glyphs = model(alphabet_batch, glyph_ids)
        
        # Reconstruction loss
        recon_loss = chamfer_distance(pred_glyphs, target_glyphs)
        
        # Style consistency loss (optional)
        # Ensure same alphabet → same style even with different glyph subsets
        
        total_loss = recon_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return {'recon_loss': recon_loss}
```

### 4.3 Training Protocol

```python
def train_generative_model(model, data_loader, epochs=100):
    optimizer = tf.keras.optimizers.Adam(1e-4)
    
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            # Sample batch of alphabets
            alphabets = data_loader.sample_alphabets(batch_size=16)
            
            # For each alphabet, pick random glyph to reconstruct
            glyph_ids = np.random.randint(0, 52, size=16)
            target_glyphs = alphabets[np.arange(16), glyph_ids]
            
            losses = train_step(model, optimizer, alphabets, target_glyphs, glyph_ids)
        
        # Validation: generate all glyphs for held-out alphabets
        if epoch % 10 == 0:
            validate_generation(model, val_alphabets)
```

---

## Step 5: Style Space Constraints

**Duration**: 2-3 days

### 5.1 Triplet Loss on Style Vectors

Integrate with existing triplet infrastructure:

```python
def style_triplet_loss(anchor_style, positive_style, negative_style, margin=0.5):
    """Ensure similar alphabets have similar styles."""
    pos_dist = tf.reduce_sum(tf.square(anchor_style - positive_style), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor_style - negative_style), axis=-1)
    return tf.reduce_mean(tf.maximum(0.0, pos_dist - neg_dist + margin))
```

### 5.2 VAE Regularization (Optional)

Add KL divergence to encourage smooth latent space:

```python
class AlphabetVAE(tf.keras.Model):
    def encode(self, alphabet):
        features = self.encoder_base(alphabet)
        mu = self.fc_mu(features)
        log_var = self.fc_logvar(features)
        
        # Reparameterization
        eps = tf.random.normal(tf.shape(mu))
        z = mu + tf.exp(0.5 * log_var) * eps
        
        return z, mu, log_var
    
    def kl_loss(self, mu, log_var):
        return -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
```

---

## Step 6: Novel Alphabet Generation

**Duration**: 2 days

### 6.1 Style Interpolation

```python
def interpolate_styles(model, alphabet1, alphabet2, n_steps=10):
    """Generate alphabets interpolating between two styles."""
    style1 = model.encoder(alphabet1)
    style2 = model.encoder(alphabet2)
    
    results = []
    for alpha in np.linspace(0, 1, n_steps):
        interp_style = (1 - alpha) * style1 + alpha * style2
        generated = model.generate(interp_style, tf.range(52))
        results.append(generated)
    
    return results
```

### 6.2 Random Sampling

```python
def sample_novel_alphabet(model, n_samples=1):
    """Generate completely novel alphabets from latent space."""
    # Sample from prior (unit Gaussian if VAE)
    z = tf.random.normal((n_samples, model.style_dim))
    
    # Generate all 52 glyphs
    all_glyphs = []
    for glyph_id in range(52):
        glyph = model.generate(z, tf.fill([n_samples], glyph_id))
        all_glyphs.append(glyph)
    
    return tf.stack(all_glyphs, axis=1)  # (n_samples, 52, 256, 2)
```

---

## Validation Criteria

### Must Pass

1. **Reconstruction**: Given full alphabet, can reconstruct any glyph with Chamfer < 0.01
2. **Style transfer**: Encoding alphabet A, decoding glyph 'B' produces A-styled B
3. **No copying**: Generated glyphs measurably different from all training glyphs
4. **Coherence**: Glyphs from same style vector visually consistent

### Metrics

```python
def evaluate_model(model, test_alphabets):
    metrics = {}
    
    # 1. Reconstruction error
    recon_errors = []
    for alphabet in test_alphabets:
        for glyph_id in range(52):
            pred = model(alphabet[None], tf.constant([glyph_id]))
            target = alphabet[glyph_id]
            recon_errors.append(chamfer_distance(pred, target[None]))
    metrics['mean_recon_error'] = np.mean(recon_errors)
    
    # 2. Novelty (distance to nearest training glyph)
    # ...
    
    # 3. Intra-alphabet consistency
    # ...
    
    return metrics
```

---

## Deliverables

| Item | File | Description |
|------|------|-------------|
| Alphabet Encoder | `src/alphabet_encoder.py` | DeepSets-style set encoder |
| Glyph Decoder | `src/glyph_decoder.py` | Conditional point cloud decoder |
| Full VAE | `src/alphabet_vae.py` | Combined model |
| Training | `src/generative_training.py` | Training loops |
| Evaluation | `src/generative_eval.py` | Metrics and visualization |
| Notebook | `notebooks/04_train_generative.ipynb` | Interactive training |
| Notebook | `notebooks/05_generate_alphabets.ipynb` | Generation demos |

---

## Timeline Estimate

| Step | Duration | Cumulative |
|------|----------|------------|
| Output representation | 2-3 days | 3 days |
| Alphabet encoder | 3-4 days | 7 days |
| Glyph decoder | 3-4 days | 11 days |
| Training loop | 3-4 days | 15 days |
| Style constraints | 2-3 days | 18 days |
| Novel generation | 2 days | 20 days |

**Total**: ~3-4 weeks for Phase 2
