# Code Review: alphaba

This document provides a review of the `alphaba` codebase, including suggestions for improvement and a list of development tickets.

## Overall Assessment

The project is a good starting point for a Siamese/Triplet network implementation for one-shot learning with the Omniglot dataset. The code is reasonably well-structured into data loading, model definition, and training components. The use of Jupyter notebooks for exploration and visualization is a good practice.

However, the project lacks some key elements for maintainability, robustness, and ease of use, such as documentation, testing, and configuration management.

## Suggestions for Improvement

### 1. Documentation

- **README.md:** The `README.md` is empty. It should be populated with a project description, setup instructions, and usage examples.
- **Docstrings and Comments:** The code lacks docstrings and comments, making it difficult to understand the purpose of functions and classes without reading the implementation.

### 2. Configuration

- **Hardcoded Paths:** The `data_loader.py` contains a hardcoded path to the `omniglot` dataset. This should be externalized to a configuration file or passed as a command-line argument.
- **Hyperparameters:** Hyperparameters like learning rate, batch size, and epochs are hardcoded in the training script and notebooks. These should also be moved to a configuration file.

### 3. Code Quality and Structure

- **Redundant Training Functions:** `training.py` has two training functions: `train_triplet_model_custom` and `train_triplet_model`. The custom training loop is more flexible and should be the default. The other function should be removed to avoid confusion.
- **`main.py`:** The `main.py` file is a placeholder and doesn't do anything. It should be updated to be the main entry point for training and evaluation.
- **Testing:** There are no unit tests. Adding tests would improve the robustness of the code and make it easier to refactor.
- **Code Duplication:** The notebooks and the `src` files have some duplicated code. The notebooks should import the code from the `src` directory to avoid this.
- **Trailing Whitespace:** The `training.py` file has a lot of trailing whitespace that should be removed.

## Development Tickets

Here is a prioritized list of development tickets based on the review:

- [ ] **Ticket 1: Populate README.md**
  - **Description:** Add a project description, setup instructions, and usage examples to the `README.md` file.
  - **Priority:** High

- [ ] **Ticket 2: Externalize data path**
  - **Description:** Remove the hardcoded path to the Omniglot dataset in `data_loader.py` and replace it with a configurable option (e.g., environment variable, config file, or command-line argument).
  - **Priority:** High

- [ ] **Ticket 3: Add unit tests**
  - **Description:** Create a `tests` directory and add unit tests for the data loader, model, and training functions.
  - **Priority:** High

- [ ] **Ticket 4: Refactor `main.py`**
  - **Description:** Update `main.py` to be a proper entry point for the application, allowing users to run training and evaluation from the command line.
  - **Priority:** Medium

- [ ] **Ticket 5: Add a configuration file**
  - **Description:** Create a configuration file (e.g., `config.py` or `config.yaml`) to store hyperparameters and other settings.
  - **Priority:** Medium

- [ ] **Ticket 6: Refactor training script**
  - **Description:** Remove the redundant training function from `training.py` and clean up the file.
  - **Priority:** Medium

- [ ] **Ticket 7: Add docstrings and comments**
  - **Description:** Add docstrings and comments to the code to improve readability.
  - **Priority:** Low
