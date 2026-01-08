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
        
        # Glyph-type embedding: distinguish uppercase (0-25) vs lowercase (26-51)
        # Helps model learn baseline/height differences
        self.glyph_type_embedding = layers.Embedding(2, 16)  # 0=upper, 1=lower
        
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
        
        # Add glyph-type embedding (upper=0 for indices 0-25, lower=1 for 26-51)
        glyph_types = tf.concat([
            tf.zeros(26, dtype=tf.int32),   # A-Z
            tf.ones(26, dtype=tf.int32)     # a-z
        ], axis=0)
        glyph_types = tf.tile(glyph_types[None, :], [batch_size, 1])  # (batch, 52)
        type_emb = self.glyph_type_embedding(glyph_types)  # (batch, 52, 16)
        
        # Concatenate glyph features with type embedding
        glyph_features = tf.concat([glyph_features, type_emb], axis=-1)  # (batch, 52, 144)
        
        # Apply φ to each glyph
        transformed = self.phi(glyph_features)  # (batch, n_glyphs, 256)
        
        # Sum-pool over glyphs (permutation invariant)
        aggregated = tf.reduce_mean(transformed, axis=1)  # (batch, 256)
        
        # Apply ρ to get style vector
        return self.rho(aggregated)  # (batch, style_dim)
```

**Note**: The glyph-type embedding helps the model distinguish uppercase/lowercase baseline and height differences, which is crucial for coherent alphabet generation.

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

### 3.3 Alternative: Earth Mover's Distance (EMD)

EMD can give more structured reconstructions but is computationally heavier:

```python
def earth_mover_distance(pred, target):
    """Compute approximate EMD via Sinkhorn iterations.
    
    More expensive than Chamfer but often gives better structure.
    Use for validation or final training stages.
    """
    # Compute pairwise distance matrix
    pred_exp = tf.expand_dims(pred, 2)
    target_exp = tf.expand_dims(target, 1)
    cost_matrix = tf.reduce_sum(tf.square(pred_exp - target_exp), axis=-1)
    
    # Sinkhorn iterations for optimal transport
    n = tf.shape(pred)[1]
    epsilon = 0.01  # Regularization
    
    K = tf.exp(-cost_matrix / epsilon)
    u = tf.ones((tf.shape(pred)[0], n, 1)) / tf.cast(n, tf.float32)
    
    for _ in range(50):  # Sinkhorn iterations
        v = 1.0 / (tf.matmul(K, u, transpose_a=True) + 1e-8)
        u = 1.0 / (tf.matmul(K, v) + 1e-8)
    
    transport = u * K * tf.transpose(v, [0, 2, 1])
    return tf.reduce_sum(transport * cost_matrix, axis=[1, 2])
```

**Recommendation**: Start with Chamfer (faster iteration), switch to EMD for fine-tuning if needed.

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

Add KL divergence to encourage smooth latent space.

**Important**: `mu` and `log_var` must be **separate heads** from the main encoder output to allow proper KL computation:

```python
class AlphabetVAE(tf.keras.Model):
    def __init__(self, style_dim=128, n_points=256, n_glyphs=52):
        super().__init__()
        
        # Base encoder (outputs features, not style directly)
        self.encoder_base = AlphabetEncoderBase(feature_dim=256)
        
        # Separate heads for mu and log_var
        self.fc_mu = layers.Dense(style_dim)
        self.fc_logvar = layers.Dense(style_dim)
        
        self.decoder = GlyphDecoder(n_points=n_points, style_dim=style_dim)
        self.style_dim = style_dim
    
    def encode(self, alphabet):
        """Encode alphabet to latent distribution parameters."""
        features = self.encoder_base(alphabet)  # (batch, 256)
        
        # Separate heads for mean and variance
        mu = self.fc_mu(features)               # (batch, style_dim)
        log_var = self.fc_logvar(features)      # (batch, style_dim)
        
        # Reparameterization trick
        eps = tf.random.normal(tf.shape(mu))
        z = mu + tf.exp(0.5 * log_var) * eps
        
        return z, mu, log_var
    
    def kl_loss(self, mu, log_var):
        """KL divergence from N(mu, sigma) to N(0, 1)."""
        return -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
        )
    
    def call(self, alphabet, glyph_ids, training=True):
        z, mu, log_var = self.encode(alphabet)
        pred_glyphs = self.decoder(z, glyph_ids)
        
        if training:
            return pred_glyphs, mu, log_var
        return pred_glyphs
```

**Training with KL loss**:

```python
@tf.function
def train_step_vae(model, optimizer, alphabet_batch, target_glyphs, glyph_ids, beta=0.1):
    with tf.GradientTape() as tape:
        pred_glyphs, mu, log_var = model(alphabet_batch, glyph_ids, training=True)
        
        recon_loss = chamfer_distance(pred_glyphs, target_glyphs)
        kl_loss = model.kl_loss(mu, log_var)
        
        # Beta-VAE style weighting (start low, increase gradually)
        total_loss = recon_loss + beta * kl_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return {'recon_loss': recon_loss, 'kl_loss': kl_loss, 'total_loss': total_loss}
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
def evaluate_model(model, test_alphabets, train_alphabets, topology_data):
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
    generated = sample_novel_alphabet(model, n_samples=10)
    min_dists = []
    for gen_alph in generated:
        for glyph in gen_alph:
            dists = [chamfer_distance(glyph[None], train_g[None]) 
                     for train_alph in train_alphabets for train_g in train_alph]
            min_dists.append(min(dists))
    metrics['mean_novelty'] = np.mean(min_dists)
    
    # 3. Intra-alphabet consistency (using style embeddings)
    style_vars = []
    for alphabet in test_alphabets:
        # Encode subsets of glyphs, check style variance
        styles = []
        for _ in range(10):
            subset_idx = np.random.choice(52, 26, replace=False)
            subset = alphabet[subset_idx]
            style = model.encode(subset[None])[0]  # Just mu
            styles.append(style)
        style_vars.append(np.var(styles, axis=0).mean())
    metrics['style_consistency'] = 1.0 / (1.0 + np.mean(style_vars))
    
    return metrics
```

### Stroke Consistency Metrics (from Phase 0)

Leverage topology data extracted in Phase 0:

```python
def evaluate_stroke_consistency(generated_glyphs, topology_data):
    """Compare generated glyph topology to training distribution.
    
    Uses junction counts, endpoint counts, and aspect ratios from Phase 0.
    """
    metrics = {}
    
    # Rasterize generated glyphs and extract skeletons
    gen_topology = []
    for glyph_points in generated_glyphs:
        # Rasterize points to image
        img = points_to_raster(glyph_points, size=512)
        skeleton = skeletonize(img > 0)
        topo = analyze_skeleton_topology(skeleton)
        gen_topology.append(topo)
    
    # Compare to training topology distributions
    train_junctions = [t['junctions'] for t in topology_data]
    gen_junctions = [t['junctions'] for t in gen_topology]
    
    train_endpoints = [t['endpoints'] for t in topology_data]
    gen_endpoints = [t['endpoints'] for t in gen_topology]
    
    # Wasserstein distance between distributions
    from scipy.stats import wasserstein_distance
    metrics['junction_dist'] = wasserstein_distance(train_junctions, gen_junctions)
    metrics['endpoint_dist'] = wasserstein_distance(train_endpoints, gen_endpoints)
    
    # Aspect ratio consistency within alphabet
    gen_aspects = [compute_aspect_ratio(g) for g in generated_glyphs]
    metrics['aspect_std'] = np.std(gen_aspects)  # Lower = more consistent
    
    return metrics
```

**Why this matters**: Generated alphabets should have similar topological properties to real alphabets — similar numbers of strokes, junctions, and consistent proportions.

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
