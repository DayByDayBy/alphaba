# LLM-Friendly Project Report: alphaba

## Project Goal

To create a "machine that can make alphabets" by training a neural network on the Omniglot dataset (and eventually custom datasets) to generate novel, coherent fictional alphabets.

## Current State

The project is currently set up for **representation learning**, not **generation**. It uses a Triplet Network to learn a vector-space embedding of the Omniglot characters. This means it can tell if characters are similar, but cannot create new ones.

- **`data_loader.py`**: Loads images from the Omniglot dataset and creates triplets (anchor, positive, negative) for training.
- **`models.py`**: Defines a CNN-based Triplet Network model.
- **`training.py`**: Contains a custom TensorFlow training loop for the Triplet Network.
- **`notebooks/`**: Jupyter notebooks for exploring the data, training the model, and visualizing the learned embeddings with t-SNE.

## Analysis of a Mismatch

The current implementation, while functional for learning embeddings, is **not aligned with the primary goal of generating alphabets**. A triplet network learns to *organize* existing data, but it does not learn how to *create* new data. To generate alphabets, a generative model is required.

## What Needs To Be Done Next: Action Plan

To pivot the project towards its generative goal, the following steps are recommended:

### Phase 1: Foundational Improvements

1.  **Isolate Configuration**: Create a `config.py` file to store all hardcoded values (e.g., data path, learning rate, batch size, epochs). This is crucial for flexibility and for training on custom datasets.
2.  **Create a `main.py` Entrypoint**: Convert the logic from the notebooks into a runnable `main.py` script that takes command-line arguments (e.g., `--train`, `--generate`).
3.  **Add Unit Tests**: Create a `tests/` directory and add basic tests for the data loader to ensure it handles different datasets correctly.

### Phase 2: Transition to a Generative Model

1.  **Choose a Generative Model**: A **Variational Autoencoder (VAE)** is a good choice for this task. It learns a compressed, continuous latent space of the data, which is ideal for generating new, similar data points.
2.  **Implement the VAE**:
    - **Update `data_loader.py`**: Modify the data loader to yield individual images instead of triplets.
    - **Create a VAE Model in `models.py`**: Define the VAE architecture, consisting of an encoder and a decoder.
    - **Implement the VAE Training Loop in `training.py`**: Write a new training loop for the VAE, which will involve a reconstruction loss and a KL divergence loss.

### Phase 3: Alphabet Generation

1.  **Implement a Generation Script**: Create a script (`generate.py`) that loads the trained VAE and samples from its latent space to generate new character images.
2.  **Develop Alphabet Generation Logic**: Create a function that generates a full alphabet by sampling a region of the latent space. This will allow for the generation of coherent sets of characters.
3.  **(Optional) Build a Simple UI**: Use a library like `Gradio` or `Streamlit` to create a simple web interface for generating alphabets without writing code. This would be a great way to share your project with comic book artists and other creatives.
