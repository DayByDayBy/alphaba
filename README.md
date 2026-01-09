# Alphaba: Machine Learning Alphabet Project

## Overview

Alphaba is a machine learning project that uses triplet networks to learn character embeddings. The goal is to understand the visual patterns and systematic relationships within writing systems, with the eventual aim of generating new fictional alphabets for worldbuilding projects.

This project implements representation learning to create structured embedding spaces where characters from the same alphabet cluster together, while characters from different alphabets are pushed apart. The learned representations provide a foundation for understanding and generating cohesive writing systems.

## Features

- Triplet network implementation for learning character embeddings
- Custom training loops with triplet loss optimization
- t-SNE visualization of learned embedding spaces
- Jupyter notebook workflows for experimentation and analysis
- Comprehensive evaluation tools for embedding quality assessment

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

### Basic Training Example

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
│   ├── __init__.py
│   ├── data_loader.py      # Omniglot dataset loading and triplet sampling
│   ├── models.py          # Triplet network architecture
│   ├── training.py        # Training loops and evaluation functions
│   └── utils.py           # Utility functions
├── notebooks/
│   ├── 01_explore_omniglot.ipynb
│   ├── 02_train_triplet_network.ipynb
│   └── 03_visualise_embeddings.ipynb
├── omniglot/              # Dataset directory (clone separately)
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock               # Dependency lock file
└── README.md             # This file
```

## Dataset Setup

The project uses the Omniglot dataset, which must be obtained separately:

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