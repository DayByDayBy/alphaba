# Alphaba: Machine Learning Alphabet Project

## Overview

Alphaba is an alphabet research project with two main tracks:

- A **triplet-network** workflow to learn character embeddings.
- A **font geometry pipeline** that converts `.ttf` fonts into normalized vector/point/skeleton representations suitable for alphabet-level learning.

The longer-term goal is to understand structure/style in writing systems and enable generation of new, fictional alphabets.

## Features

- Triplet network implementation for learning character embeddings (Omniglot-style)
- CLI entrypoint for triplet training / generation / evaluation (`main.py`)
- Font geometry pipeline (`src/alphabet_pipeline.py`) producing:
  - normalized SVG path strings (`vectors/*.svgpath.txt`)
  - arc-length point samples (`samples/*_samples.npy`)
  - rasters and skeletons (`rasters/*.png`, `skeletons/*.png`)
  - per-font tensors (`alphabet_samples.npy`, `alphabet_skeletons.npy`) and metadata
- Generative architecture scaffolding (`src/generative_model.py`) consuming alphabet tensors

## Prerequisites

- Python 3.13 or higher
- uv package manager
- Git

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd alphaba
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up the development environment:
```bash
uv shell
```

## Dependencies

The project uses the following key dependencies:

- **tensorflow >= 2.20.0**: Deep learning framework for neural network implementation
- **numpy >= 2.3.3**: Numerical computing library for array operations
- **opencv-python >= 4.11.0.86**: Computer vision library for image processing
- **matplotlib >= 3.10.6**: Plotting library for data visualization
- **scikit-learn >= 1.7.2**: Machine learning utilities including t-SNE
- **jupyter >= 1.1.1**: Interactive computing environment for notebooks
- **seaborn >= 0.13.2**: Statistical data visualization

## Usage

### Triplet Training (Omniglot-Style)

The triplet pipeline expects an Omniglot-style dataset checkout.

#### CLI

See `README_CLI.md` for details. Example:

```bash
uv run python main.py train --data-path ../omniglot/python --epochs 20 --output-dir outputs
```

#### Python API

```python
from src.data_loader import OmniglotTripletLoader
from src.models import create_triplet_model
from src.training import train_triplet_model_custom

# Load data
data_loader = OmniglotTripletLoader("path/to/omniglot/python")

# Create model
triplet_model, base_network = create_triplet_model(embedding_dim=64)

# Train model
history = train_triplet_model_custom(
    triplet_model,
    data_loader,
    epochs=10,
    batch_size=32,
    steps_per_epoch=100,
    learning_rate=0.01
)
```

### Evaluation and Visualization

```python
from src.training import evaluate_embeddings

# Generate embeddings and create t-SNE visualization
embeddings, labels = evaluate_embeddings(triplet_model, data_loader)

# Plot training loss
from src.training import visualize_training
visualize_training(history)
```

### Font Geometry Pipeline (TTF → Alphabet Tensors)

Process a single `.ttf` font into per-glyph artifacts plus per-font tensors:

```bash
uv run python -c "from src.alphabet_pipeline import process_font; process_font('alphabet_data/fonts/GoogleSans-VariableFont_GRAD,opsz,wght.ttf','output/pipeline_runs')"
```

The output directory will contain a subfolder named after the font file stem, including:

```text
output/pipeline_runs/<FontName>/
  vectors/
  samples/
  rasters/
  skeletons/
  glyph_order.json
  alphabet_samples.npy
  alphabet_skeletons.npy
  metadata.json
```

Note: if some glyphs have no vector path data in a given font, the pipeline will skip them, so the leading dimension of `alphabet_samples.npy` / `alphabet_skeletons.npy` may be **less than 52**. Use `glyph_order.json` (or the tensor shape) as the source of truth.

### Generative Model (Alphabet Tensors)

The generative training code consumes `alphabet_samples.npy` tensors produced by the pipeline.

```bash
uv run python -c "
import numpy as np
from pathlib import Path
from src.alphabet_data_loader import AlphabetDataLoader
from src.generative_model import AlphabetVAE, AlphabetTrainer

base = Path('output/pipeline_runs') / 'GoogleSans-VariableFont_GRAD,opsz,wght'
n_glyphs = int(np.load(base / 'alphabet_samples.npy').shape[0])

loader = AlphabetDataLoader('output/pipeline_runs', n_points=256, n_glyphs=n_glyphs)
ds = loader.create_dataset(batch_size=1)

model = AlphabetVAE(style_dim=64, n_points=256, n_glyphs=n_glyphs, use_vae=True)
trainer = AlphabetTrainer(model, learning_rate=1e-3, beta=0.01)
trainer.fit(ds, epochs=1, steps_per_epoch=10, verbose=False)
print('ok')
"
```

### Running Notebooks

The project includes Jupyter notebooks for interactive development:

```bash
# Start Jupyter server
jupyter notebook

# Open notebooks in the following order:
# 1. notebooks/01_explore_omniglot.ipynb - Dataset exploration
# 2. notebooks/02_train_triplet_network.ipynb - Model training
# 3. notebooks/03_visualise_embeddings.ipynb - Results analysis
```

## Project Structure

```
alphaba/
├── src/
│   ├── alphabet_pipeline.py      # Font geometry pipeline (TTF → artifacts/tensors)
│   ├── alphabet_data_loader.py   # Loads alphabet tensors for training
│   ├── generative_model.py       # Generative architecture scaffolding (AlphabetVAE)
│   ├── data_loader.py            # Omniglot triplet dataset loader
│   ├── models.py                 # Triplet network model
│   ├── training.py               # Triplet training utilities
│   └── train.py                  # Triplet training CLI implementation
├── notebooks/
│   ├── 01_explore_omniglot.ipynb
│   ├── 02_train_triplet_network.ipynb
│   └── 03_visualise_embeddings.ipynb
├── alphabet_data/          # Fonts + registry for pipeline
├── omniglot/               # Dataset directory (clone separately, optional)
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock               # Dependency lock file
└── README.md             # This file
```

## Dataset Setup

The triplet workflow uses the Omniglot dataset, which must be obtained separately:

```bash
# Clone the Omniglot dataset
git clone https://github.com/brendenlake/omniglot.git
```

Update the data path in the notebooks to point to the correct location of the omniglot/python directory.

## Development

### Adding Dependencies

To add new dependencies to the project:

```bash
uv add package-name
```

### Running Tests

Currently, the project focuses on notebook-based development and experimentation. Unit tests can be added to a tests/ directory as the project matures.

### Code Style

The project follows standard Python conventions. Key modules are organized for clarity:

- `data_loader.py`: Handles dataset loading and triplet generation
- `models.py`: Defines the neural network architecture
- `training.py`: Contains training loops and evaluation functions

## Architecture

The project implements a triplet network with the following components:

- **Base Network**: Convolutional neural network for feature extraction
- **Triplet Loss**: Custom loss function that minimizes anchor-positive distance while maximizing anchor-negative distance
- **Embedding Space**: Learned representations where similar characters cluster together

The model learns to create meaningful embeddings that capture the visual similarities and differences between characters from various writing systems.

## Contributing

This project is developed for learning purposes. Contributions should focus on:

- Improving model architectures
- Enhancing training stability
- Adding evaluation metrics
- Expanding visualization capabilities

## License

This project is for educational purposes. Refer to the Omniglot dataset license for data usage terms.