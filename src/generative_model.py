"""
Generative model for alphabet synthesis.

Architecture:
- AlphabetEncoder: Encodes full alphabet (52 glyphs) to style vector
- GlyphDecoder: Decodes style vector + glyph ID to point cloud
- AlphabetVAE: Combined encoder-decoder with optional VAE regularization
"""

import logging
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


# =============================================================================
# LOSSES
# =============================================================================

def chamfer_distance(pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    """
    Compute Chamfer distance between point clouds.
    
    Args:
        pred: (batch, n_points, 2) predicted points
        target: (batch, n_points, 2) target points
    
    Returns:
        Scalar loss value
    """
    pred_exp = tf.expand_dims(pred, 2)      # (batch, n_pred, 1, 2)
    target_exp = tf.expand_dims(target, 1)  # (batch, 1, n_target, 2)
    
    dists = tf.reduce_sum(tf.square(pred_exp - target_exp), axis=-1)
    
    min_pred_to_target = tf.reduce_min(dists, axis=2)
    min_target_to_pred = tf.reduce_min(dists, axis=1)
    
    return tf.reduce_mean(min_pred_to_target) + tf.reduce_mean(min_target_to_pred)


# =============================================================================
# GLYPH ENCODER (PointNet-style)
# =============================================================================

class GlyphEncoder(tf.keras.Model):
    """Encode single glyph point cloud to feature vector."""
    
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.point_encoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
        ])
        self.global_pool = layers.GlobalMaxPooling1D()
        self.fc = layers.Dense(feature_dim)
    
    def call(self, points: tf.Tensor) -> tf.Tensor:
        """
        Args:
            points: (batch, n_points, 2) point cloud
        
        Returns:
            (batch, feature_dim) feature vector
        """
        x = self.point_encoder(points)
        x = self.global_pool(x)
        return self.fc(x)


# =============================================================================
# ALPHABET ENCODER (DeepSets)
# =============================================================================

class AlphabetEncoder(tf.keras.Model):
    """Encode full alphabet to style vector using DeepSets architecture."""
    
    def __init__(self, style_dim: int = 128, n_glyphs: int = 52):
        super().__init__()
        self.style_dim = style_dim
        self.n_glyphs = n_glyphs
        
        self.glyph_encoder = GlyphEncoder(feature_dim=128)
        
        self.glyph_type_embedding = layers.Embedding(2, 16)
        
        self.phi = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
        ])
        
        self.rho = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(style_dim),
        ])
    
    def call(self, alphabet: tf.Tensor) -> tf.Tensor:
        """
        Args:
            alphabet: (batch, n_glyphs, n_points, 2) full alphabet
        
        Returns:
            (batch, style_dim) style vector
        """
        batch_size = tf.shape(alphabet)[0]
        n_glyphs = tf.shape(alphabet)[1]
        n_points = alphabet.shape[2]
        
        flat = tf.reshape(alphabet, (-1, n_points, 2))
        glyph_features = self.glyph_encoder(flat)
        glyph_features = tf.reshape(glyph_features, (batch_size, n_glyphs, -1))
        
        glyph_types = tf.concat([
            tf.zeros(26, dtype=tf.int32),
            tf.ones(26, dtype=tf.int32)
        ], axis=0)
        glyph_types = glyph_types[:n_glyphs]
        glyph_types = tf.tile(glyph_types[None, :], [batch_size, 1])
        type_emb = self.glyph_type_embedding(glyph_types)
        
        glyph_features = tf.concat([glyph_features, type_emb], axis=-1)
        
        transformed = self.phi(glyph_features)
        aggregated = tf.reduce_mean(transformed, axis=1)
        
        return self.rho(aggregated)


# =============================================================================
# GLYPH DECODER
# =============================================================================

class GlyphDecoder(tf.keras.Model):
    """Decode style vector + glyph ID to point cloud."""
    
    def __init__(self, n_points: int = 256, style_dim: int = 128, n_glyphs: int = 52):
        super().__init__()
        self.n_points = n_points
        self.n_glyphs = n_glyphs
        
        self.glyph_embedding = layers.Embedding(n_glyphs, 64)
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(n_points * 2, activation='sigmoid'),
            layers.Reshape((n_points, 2))
        ])
    
    def call(self, style_vector: tf.Tensor, glyph_id: tf.Tensor) -> tf.Tensor:
        """
        Args:
            style_vector: (batch, style_dim) style encoding
            glyph_id: (batch,) integer glyph indices
        
        Returns:
            (batch, n_points, 2) generated point cloud
        """
        glyph_emb = self.glyph_embedding(glyph_id)
        combined = tf.concat([style_vector, glyph_emb], axis=-1)
        return self.decoder(combined)


# =============================================================================
# COMBINED MODEL (VAE)
# =============================================================================

class AlphabetVAE(tf.keras.Model):
    """Full alphabet-to-glyph generative model with optional VAE regularization."""
    
    def __init__(
        self,
        style_dim: int = 128,
        n_points: int = 256,
        n_glyphs: int = 52,
        use_vae: bool = True
    ):
        super().__init__()
        self.style_dim = style_dim
        self.n_points = n_points
        self.n_glyphs = n_glyphs
        self.use_vae = use_vae
        
        self.encoder = AlphabetEncoder(style_dim=256, n_glyphs=n_glyphs)
        
        if use_vae:
            self.fc_mu = layers.Dense(style_dim)
            self.fc_logvar = layers.Dense(style_dim)
        else:
            self.fc_style = layers.Dense(style_dim)
        
        self.decoder = GlyphDecoder(
            n_points=n_points,
            style_dim=style_dim,
            n_glyphs=n_glyphs
        )
    
    def encode(self, alphabet: tf.Tensor) -> Tuple[tf.Tensor, ...]:
        """Encode alphabet to latent distribution."""
        features = self.encoder(alphabet)
        
        if self.use_vae:
            mu = self.fc_mu(features)
            log_var = self.fc_logvar(features)
            
            eps = tf.random.normal(tf.shape(mu))
            z = mu + tf.exp(0.5 * log_var) * eps
            
            return z, mu, log_var
        else:
            return (self.fc_style(features),)
    
    def decode(self, style_vector: tf.Tensor, glyph_id: tf.Tensor) -> tf.Tensor:
        """Decode style vector + glyph ID to point cloud."""
        return self.decoder(style_vector, glyph_id)
    
    def call(
        self,
        alphabet: tf.Tensor,
        glyph_id: tf.Tensor,
        training: bool = True
    ) -> Tuple[tf.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            alphabet: (batch, n_glyphs, n_points, 2)
            glyph_id: (batch,) integer glyph indices
            training: whether in training mode
        
        Returns:
            If VAE: (pred_glyphs, mu, log_var)
            Else: (pred_glyphs,)
        """
        encoded = self.encode(alphabet)
        z = encoded[0]
        pred_glyphs = self.decode(z, glyph_id)
        
        if self.use_vae and training:
            return (pred_glyphs,) + encoded[1:]
        return (pred_glyphs,)
    
    def generate(self, style_vector: tf.Tensor, glyph_id: tf.Tensor) -> tf.Tensor:
        """Generate glyphs from explicit style vector."""
        return self.decode(style_vector, glyph_id)
    
    def sample_prior(self, n_samples: int = 1) -> tf.Tensor:
        """Sample from prior distribution."""
        return tf.random.normal((n_samples, self.style_dim))
    
    @staticmethod
    def kl_loss(mu: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        """KL divergence from N(mu, sigma) to N(0, 1)."""
        return -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
        )


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class AlphabetTrainer:
    """Training loop for AlphabetVAE."""
    
    def __init__(
        self,
        model: AlphabetVAE,
        learning_rate: float = 1e-4,
        beta: float = 0.1
    ):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.beta = beta
        
        self.recon_loss_metric = tf.keras.metrics.Mean(name='recon_loss')
        self.kl_loss_metric = tf.keras.metrics.Mean(name='kl_loss')
        self.total_loss_metric = tf.keras.metrics.Mean(name='total_loss')
    
    @tf.function
    def train_step(
        self,
        alphabet: tf.Tensor,
        glyph_id: tf.Tensor,
        target_glyph: tf.Tensor
    ) -> dict:
        """Single training step."""
        with tf.GradientTape() as tape:
            outputs = self.model(alphabet, glyph_id, training=True)
            pred_glyphs = outputs[0]
            
            recon_loss = chamfer_distance(pred_glyphs, target_glyph)
            
            if self.model.use_vae and len(outputs) > 1:
                mu, log_var = outputs[1], outputs[2]
                kl_loss = AlphabetVAE.kl_loss(mu, log_var)
                total_loss = recon_loss + self.beta * kl_loss
            else:
                kl_loss = tf.constant(0.0)
                total_loss = recon_loss
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.recon_loss_metric.update_state(recon_loss)
        self.kl_loss_metric.update_state(kl_loss)
        self.total_loss_metric.update_state(total_loss)
        
        return {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss
        }
    
    def train_epoch(self, dataset: tf.data.Dataset) -> dict:
        """Train for one epoch."""
        self.recon_loss_metric.reset_state()
        self.kl_loss_metric.reset_state()
        self.total_loss_metric.reset_state()
        
        for alphabet, glyph_id, target_glyph in dataset:
            self.train_step(alphabet, glyph_id, target_glyph)
        
        return {
            'recon_loss': float(self.recon_loss_metric.result()),
            'kl_loss': float(self.kl_loss_metric.result()),
            'total_loss': float(self.total_loss_metric.result())
        }
    
    def fit(
        self,
        dataset: tf.data.Dataset,
        epochs: int = 100,
        steps_per_epoch: Optional[int] = None,
        verbose: bool = True
    ) -> list:
        """Full training loop."""
        history = []
        
        for epoch in range(epochs):
            if steps_per_epoch:
                epoch_dataset = dataset.take(steps_per_epoch)
            else:
                epoch_dataset = dataset
            
            metrics = self.train_epoch(epoch_dataset)
            history.append(metrics)
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"recon: {metrics['recon_loss']:.4f}, "
                    f"kl: {metrics['kl_loss']:.4f}, "
                    f"total: {metrics['total_loss']:.4f}"
                )
        
        return history


# =============================================================================
# GENERATION UTILITIES
# =============================================================================

def interpolate_styles(
    model: AlphabetVAE,
    alphabet1: tf.Tensor,
    alphabet2: tf.Tensor,
    n_steps: int = 10
) -> tf.Tensor:
    """Generate alphabets interpolating between two styles."""
    z1, *_ = model.encode(alphabet1)
    z2, *_ = model.encode(alphabet2)
    
    results = []
    for alpha in np.linspace(0, 1, n_steps):
        interp_style = (1 - alpha) * z1 + alpha * z2
        
        all_glyphs = []
        for glyph_id in range(model.n_glyphs):
            glyph = model.generate(interp_style, tf.fill([tf.shape(interp_style)[0]], glyph_id))
            all_glyphs.append(glyph)
        
        results.append(tf.stack(all_glyphs, axis=1))
    
    return tf.stack(results, axis=0)


def sample_novel_alphabet(model: AlphabetVAE, n_samples: int = 1) -> tf.Tensor:
    """Generate completely novel alphabets from latent space."""
    z = model.sample_prior(n_samples)
    
    all_glyphs = []
    for glyph_id in range(model.n_glyphs):
        glyph = model.generate(z, tf.fill([n_samples], glyph_id))
        all_glyphs.append(glyph)
    
    return tf.stack(all_glyphs, axis=1)
