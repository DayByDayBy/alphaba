"""
Configuration management for Alphaba
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 50
    batch_size: int = 32
    steps_per_epoch: int = 100
    learning_rate: float = 0.001
    margin: float = 0.2
    embedding_dim: int = 64
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: float = 15.0
    zoom_range: float = 0.1
    noise_std: float = 0.01
    
    # Early stopping
    early_stopping_patience: int = 10
    min_delta: float = 0.001


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_shape: tuple = (105, 105, 1)
    embedding_dim: int = 64
    
    # CNN layers
    conv_filters: List[int] = None
    kernel_sizes: List[tuple] = None
    dense_units: int = 1024
    
    def __post_init__(self):
        if self.conv_filters is None:
            self.conv_filters = [64, 128, 128, 256]
        if self.kernel_sizes is None:
            self.kernel_sizes = [(10, 10), (7, 7), (4, 4), (4, 4)]


@dataclass
class DataConfig:
    """Data configuration"""
    omniglot_path: str = "omniglot/python"
    target_alphabets: Optional[List[str]] = None
    max_samples_per_alphabet: int = 50
    train_test_split: float = 0.8
    
    # Target alphabets for generation (from your shortlist)
    generation_alphabets: List[str] = None
    
    def __post_init__(self):
        if self.generation_alphabets is None:
            self.generation_alphabets = [
                "Asomtavruli_(Georgian)",
                "Mkhedruli_(Georgian)", 
                "Armenian",
                "Greek",
                "Latin",
                "Tifinagh"
            ]


@dataclass
class GenerationConfig:
    """Alphabet generation configuration"""
    n_characters: int = 26  # Map to Roman alphabet
    character_variation_scale: float = 0.05
    style_variance: float = 0.1
    
    # Decoder training
    decoder_epochs: int = 50
    decoder_batch_size: int = 32
    
    # Output settings
    output_format: str = "png"  # png, svg
    image_size: tuple = (105, 105)


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    n_evaluation_samples: int = 500
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "intra_distance", "inter_distance", "separation_ratio",
                "silhouette_score", "clustering_quality"
            ]


@dataclass
class AlphabaConfig:
    """Main configuration class"""
    training: TrainingConfig = None
    model: ModelConfig = None
    data: DataConfig = None
    generation: GenerationConfig = None
    evaluation: EvaluationConfig = None
    
    # Paths
    output_dir: str = "outputs"
    model_save_path: str = "outputs/triplet_model.h5"
    
    def __post_init__(self):
        if self.training is None:
            self.training = TrainingConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        config = cls()
        
        for key, value in config_dict.items():
            if hasattr(config, key):
                if isinstance(value, dict):
                    # Nested config
                    nested_config = getattr(config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(config, key, value)
        
        return config
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'training': self.training.__dict__,
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'generation': self.generation.__dict__,
            'evaluation': self.evaluation.__dict__,
            'output_dir': self.output_dir,
            'model_save_path': self.model_save_path
        }
    
    def save(self, filepath: str):
        """Save config to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load config from file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_default_config():
    """Get default configuration"""
    return AlphabaConfig()


def get_config_from_env():
    """Get configuration from environment variables"""
    config = get_default_config()
    
    # Override with environment variables
    if os.getenv('ALPHABA_OUTPUT_DIR'):
        config.output_dir = os.getenv('ALPHABA_OUTPUT_DIR')
    if os.getenv('ALPHABA_OMNIGLOT_PATH'):
        config.data.omniglot_path = os.getenv('ALPHABA_OMNIGLOT_PATH')
    if os.getenv('ALPHABA_EPOCHS'):
        config.training.epochs = int(os.getenv('ALPHABA_EPOCHS'))
    if os.getenv('ALPHABA_BATCH_SIZE'):
        config.training.batch_size = int(os.getenv('ALPHABA_BATCH_SIZE'))
    
    return config
