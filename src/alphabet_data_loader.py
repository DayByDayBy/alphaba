"""
Data loader for alphabet tensors.

Loads processed font data from Phase 1 pipeline outputs for generative model training.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray
import tensorflow as tf

logger = logging.getLogger(__name__)


class FontEntry(TypedDict):
    name: str
    path: Path
    samples: NDArray[np.floating]
    glyph_order: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]


class AlphabetDataLoader:
    """Load and batch alphabet tensors for training."""
    
    def __init__(
        self,
        data_dir: str,
        n_points: int = 256,
        n_glyphs: int = 52
    ):
        """
        Args:
            data_dir: Directory containing processed font outputs
            n_points: Number of points per glyph (default 256)
            n_glyphs: Number of glyphs per alphabet (default 52 for A-Za-z)
        """
        self.data_dir = Path(data_dir)
        self.n_points = n_points
        self.n_glyphs = n_glyphs
        
        self.fonts: List[FontEntry] = []
        self._load_fonts()
    
    def _load_fonts(self) -> None:
        """Discover and load all processed fonts."""
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return
        
        for font_dir in self.data_dir.iterdir():
            if not font_dir.is_dir():
                continue
            
            samples_file = font_dir / 'alphabet_samples.npy'
            glyph_order_file = font_dir / 'glyph_order.json'
            metadata_file = font_dir / 'metadata.json'
            
            if not samples_file.exists():
                logger.debug(f"Skipping {font_dir.name}: no alphabet_samples.npy")
                continue
            
            try:
                samples = np.load(samples_file)
                
                # Validate tensor shape
                if samples.ndim != 3:
                    logger.warning(
                        f"Skipping {font_dir.name}: expected 3D tensor, got {samples.ndim}D"
                    )
                    continue
                
                n_glyphs_actual, n_points_actual, n_coords = samples.shape
                
                if n_coords != 2:
                    logger.warning(
                        f"Skipping {font_dir.name}: expected 2 coordinates, got {n_coords}"
                    )
                    continue
                
                if n_points_actual != self.n_points:
                    logger.warning(
                        f"Skipping {font_dir.name}: n_points mismatch (expected {self.n_points}, got {n_points_actual})"
                    )
                    continue
                
                if n_glyphs_actual != self.n_glyphs:
                    logger.warning(
                        f"Skipping {font_dir.name}: n_glyphs mismatch (expected {self.n_glyphs}, got {n_glyphs_actual})"
                    )
                    continue
                
                glyph_order = None
                if glyph_order_file.exists():
                    with open(glyph_order_file) as f:
                        glyph_order = json.load(f)
                
                metadata = None
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                
                self.fonts.append({
                    'name': font_dir.name,
                    'path': font_dir,
                    'samples': samples,
                    'glyph_order': glyph_order,
                    'metadata': metadata
                })
                
                logger.info(f"Loaded font: {font_dir.name} with shape {samples.shape}")
                
            except Exception as e:
                logger.error(f"Failed to load {font_dir.name}: {e}")
        
        logger.info(f"Loaded {len(self.fonts)} fonts")
    
    def __len__(self) -> int:
        return len(self.fonts)
    
    def get_alphabet(self, idx: int) -> NDArray[np.floating]:
        """Get single alphabet tensor by index."""
        return self.fonts[idx]['samples']
    
    def get_all_alphabets(self) -> NDArray[np.floating]:
        """
        Stack all alphabets into single tensor.
        
        Returns:
            (n_fonts, n_glyphs, n_points, 2) tensor
        
        Raises:
            ValueError: If no fonts are loaded
        """
        if not self.fonts:
            raise ValueError("No fonts loaded")
        return np.stack([f['samples'] for f in self.fonts], axis=0)
    
    def create_dataset(
        self,
        batch_size: int = 16,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset for training.
        
        Returns dataset yielding:
            - alphabet: (batch, n_glyphs, n_points, 2)
            - glyph_id: (batch,) random glyph indices
            - target_glyph: (batch, n_points, 2) corresponding glyph
        """
        all_alphabets = self.get_all_alphabets()  # Raises ValueError if empty
        
        n_alphabets = len(all_alphabets)
        
        def generator():
            rng = np.random.default_rng(seed)
            indices = np.arange(n_alphabets)
            
            while True:
                if shuffle:
                    rng.shuffle(indices)
                
                for idx in indices:
                    alphabet = all_alphabets[idx]
                    glyph_id = rng.integers(0, alphabet.shape[0])
                    target_glyph = alphabet[glyph_id]
                    
                    yield (
                        alphabet.astype(np.float32),
                        np.int32(glyph_id),
                        target_glyph.astype(np.float32)
                    )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.n_glyphs, self.n_points, 2), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(self.n_points, 2), dtype=tf.float32)
            )
        )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train_val_split(
        self,
        val_ratio: float = 0.2,
        seed: int = 42
    ) -> Tuple['AlphabetDataLoader', 'AlphabetDataLoader']:
        """Split into train and validation loaders."""
        rng = np.random.default_rng(seed)
        n_fonts = len(self.fonts)
        n_val = max(1, int(n_fonts * val_ratio))
        
        indices = rng.permutation(n_fonts)
        val_indices = set(indices[:n_val])
        
        train_loader = AlphabetDataLoader.__new__(AlphabetDataLoader)
        train_loader.data_dir = self.data_dir
        train_loader.n_points = self.n_points
        train_loader.n_glyphs = self.n_glyphs
        train_loader.fonts = [f for i, f in enumerate(self.fonts) if i not in val_indices]
        
        val_loader = AlphabetDataLoader.__new__(AlphabetDataLoader)
        val_loader.data_dir = self.data_dir
        val_loader.n_points = self.n_points
        val_loader.n_glyphs = self.n_glyphs
        val_loader.fonts = [f for i, f in enumerate(self.fonts) if i in val_indices]
        
        return train_loader, val_loader


def load_single_alphabet(font_path: str) -> Optional[NDArray[np.floating]]:
    """Load alphabet tensor from a single font output directory."""
    font_dir = Path(font_path)
    samples_file = font_dir / 'alphabet_samples.npy'
    
    if not samples_file.exists():
        logger.error(f"No alphabet_samples.npy in {font_path}")
        return None

    samples = np.load(samples_file)
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError(f"Invalid alphabet tensor shape {samples.shape} in {font_path}")

    return samples
