# Alphaba CLI Usage Guide

## Quick Start

### 1. Train a Model
```bash
python main.py train --data-path ../omniglot/python --epochs 50 --output-dir my_model
```

### 2. Generate Fictional Alphabets
```bash
python main.py generate --model-path my_model/triplet_model.h5 --data-path ../omniglot/python --n-alphabets 5
```

### 3. Quick Demo (Train + Generate)
```bash
python main.py demo --data-path ../omniglot/python --epochs 20
```

## Detailed Usage

### Training Commands

**Basic Training:**
```bash
python main.py train --data-path PATH/TO/OMNIGLOT/python
```

**Advanced Training:**
```bash
python main.py train \
  --data-path ../omniglot/python \
  --epochs 100 \
  --batch-size 64 \
  --embedding-dim 64 \
  --learning-rate 0.001 \
  --margin 0.2 \
  --output-dir experiments/run1
```

**Training with Augmentation:**
```bash
python main.py train \
  --data-path ../omniglot/python \
  --epochs 50 \
  --aug-rotation 20.0 \
  --aug-zoom 0.15 \
  --aug-noise 0.02
```

**Disable Augmentation:**
```bash
python main.py train --data-path ../omniglot/python --no-augmentation
```

### Generation Commands

**Generate Alphabets:**
```bash
python main.py generate \
  --model-path outputs/triplet_model.h5 \
  --data-path ../omniglot/python \
  --n-alphabets 10 \
  --output-dir my_alphabets
```

### Evaluation Commands

**Evaluate Model:**
```bash
python main.py evaluate \
  --model-path outputs/triplet_model.h5 \
  --data-path ../omniglot/python \
  --output-dir evaluation_results
```

## Key Features

### 26-Character Alphabets
- All generated alphabets map to A-Z Roman characters
- Easy for writers to use for fictional text
- Individual character images saved for easy use

### Style Interpolation
- Blend characteristics of source alphabets
- Create unique "alphatish" writing systems
- Consistent internal variation

### Evaluation Metrics
- **Intra-alphabet distance**: How similar characters are within the same alphabet
- **Inter-alphabet distance**: How different alphabets are from each other  
- **Separation ratio**: Key metric for alphabet distinctiveness
- **Silhouette score**: Clustering quality in embedding space

## Output Structure

```
outputs/
├── triplet_model.h5          # Trained model
├── training_history.npy     # Training loss history
├── training_loss.png         # Training plot
├── embeddings.npy           # Character embeddings
├── labels.npy               # Alphabet labels
└── evaluation_metrics.npy   # Performance metrics

generated_alphabets/
├── random_alphabet.png      # Visualization
├── random_A.png             # Individual characters
├── random_B.png
├── ...
├── interpolated_alphabet.png
└── interpolated_A.png
    ...
```

## Tips for Fiction Writers

1. **Quick Demo**: Start with `python main.py demo` to see results quickly
2. **Custom Styles**: Train longer (50+ epochs) for more coherent alphabets
3. **Consistency**: Save your model - you can generate more alphabets in the same style later
4. **Visual Reference**: Each alphabet comes with a full A-Z reference sheet
5. **Easy Integration**: Individual character images can be imported into any graphics software

## Advanced Configuration

### Environment Variables
```bash
export ALPHABA_OUTPUT_DIR="my_experiments"
export ALPHABA_OMNIGLOT_PATH="/path/to/omniglot/python"
export ALPHABA_EPOCHS="100"
```

### Configuration Files
Create `config.json`:
```json
{
  "training": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "data": {
    "omniglot_path": "../omniglot/python"
  }
}
```

## Troubleshooting

**Common Issues:**
1. **High training loss**: Try reducing learning rate or increasing epochs
2. **Poor generation quality**: Train longer with augmentation enabled
3. **Memory errors**: Reduce batch size or embedding dimension
4. **Dataset not found**: Ensure omniglot/python path is correct

**Performance Tips:**
- Use GPU for faster training
- Enable augmentation for better generalization
- Start with demo mode to test setup
- Monitor training loss - should decrease steadily
