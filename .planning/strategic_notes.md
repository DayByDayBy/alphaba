# Strategic Notes

Collected recommendations and decisions for Phase 1→2→3 progression.

---

## Phase 1 → Phase 2 Bridge

### Export Skeleton Graph Data

Even though Phase 2 uses point clouds, export skeleton graphs now to enable quick pivot to stroke-based decoding in Phase 3:

```python
@dataclass
class SkeletonGraph:
    nodes: np.ndarray           # (n_nodes, 2) positions
    edges: List[Tuple[int, int]]
    node_types: List[str]       # 'endpoint', 'junction', 'curve'

def skeleton_to_graph(skeleton: np.ndarray) -> SkeletonGraph:
    """Extract graph structure from skeleton image."""
    # Find special points
    neighbor_count = compute_neighbor_count(skeleton)
    
    endpoints = np.argwhere(neighbor_count == 1)
    junctions = np.argwhere(neighbor_count >= 3)
    
    # Trace paths between special points
    edges = trace_skeleton_paths(skeleton, endpoints, junctions)
    
    nodes = np.vstack([endpoints, junctions])
    node_types = ['endpoint'] * len(endpoints) + ['junction'] * len(junctions)
    
    return SkeletonGraph(nodes=nodes, edges=edges, node_types=node_types)
```

**Save alongside tensors**:
```python
# In process_font()
graph = skeleton_to_graph(skeleton)
np.savez(base_path / 'graphs' / f'{glyph_name}_graph.npz',
         nodes=graph.nodes,
         edges=np.array(graph.edges),
         node_types=graph.node_types)
```

---

## Augmentation Strategy

### Key Principle: Consistency Over Variety

All augmentations must be applied **consistently across the entire alphabet** and **after normalization**.

**Allowed** (Phase 1-2):
- Small rotations (±5°)
- Uniform scaling (0.95-1.05×)
- Translation (±5% of unit box)

**Forbidden** (until Phase 3):
- Per-glyph independent transforms
- Asymmetric warping
- Elastic deformations
- Stroke-level perturbations

**Rationale**: Inconsistent augmentation breaks the fundamental assumption that alphabet style is shared across all glyphs.

---

## Loss Function Progression

| Phase | Primary Loss | Secondary Loss | Notes |
|-------|-------------|----------------|-------|
| 2 (early) | Chamfer | Style triplet | Fast iteration |
| 2 (mid) | Chamfer | Triplet + KL | Add VAE regularization |
| 2 (late) | EMD | Triplet + KL | Refine structure |
| 3 | Stroke-based | Topology consistency | Full stroke model |

**Beta schedule for KL loss**:
```python
def get_beta(epoch, warmup=20, max_beta=0.5):
    """Gradually increase KL weight to avoid posterior collapse."""
    if epoch < warmup:
        return 0.0
    return min(max_beta, (epoch - warmup) / 50 * max_beta)
```

---

## Data Requirements

### Minimum Viable Dataset

| Stage | Alphabets | Notes |
|-------|-----------|-------|
| Phase 1 validation | 10-20 | Verify pipeline outputs |
| Phase 2 training | 100+ | Diverse styles needed |
| Phase 2 robust | 500+ | Prevent overfitting |

### Data Sources (Priority Order)

1. **Google Fonts** — 1400+ families, open license, high quality
2. **Noto fonts** — Excellent coverage, already partially integrated
3. **Adobe Fonts** (if licensed) — Professional quality
4. **Omniglot** — Hand-drawn, different domain but useful for diversity

### Font Selection Criteria

- Complete A-Za-z coverage (reject if >10% missing)
- Distinct style (avoid near-duplicates)
- Clean outlines (reject corrupted/malformed fonts)

---

## Validation Checkpoints

### Phase 1 Exit Criteria

Before starting Phase 2, verify:

- [ ] Pipeline runs on 50+ fonts without errors
- [ ] Visual inspection: 'M' > 'i' in normalized output
- [ ] Visual inspection: Same letter looks consistent across fonts
- [ ] Skeleton topology numbers match visual inspection
- [ ] Alphabet tensor loads in <1s

### Phase 2 Exit Criteria

Before declaring Phase 2 complete:

- [ ] Reconstruction Chamfer < 0.01 on test set
- [ ] Style interpolation produces smooth transitions
- [ ] Random samples look "letter-like" (subjective but necessary)
- [ ] Novelty metric > threshold (not copying training data)
- [ ] Generated alphabets pass topology consistency check

---

## Risk Mitigations

### Mode Collapse

**Symptom**: All generated glyphs look similar regardless of glyph_id.

**Mitigations**:
1. Strong glyph identity embedding (64+ dims)
2. Per-glyph reconstruction loss (not just mean)
3. Diversity regularization in latent space

### Style Leakage

**Symptom**: Generated glyphs recognizably from specific training fonts.

**Mitigations**:
1. VAE regularization (smooth latent space)
2. Style triplet loss (meaningful style distances)
3. Novelty metric in validation

### Incoherence

**Symptom**: Glyphs from same style vector don't look like same alphabet.

**Mitigations**:
1. Glyph-type embedding (upper/lower distinction)
2. Style consistency loss
3. Alphabet-level (not glyph-level) training batches

---

## Phase 3 Preview: Stroke-Based Decoding

When Phase 2 is stable, migrate to stroke sequences:

```python
class StrokeDecoder(tf.keras.Model):
    """Decode style + glyph_id to stroke sequence."""
    
    def __init__(self, style_dim=128, max_strokes=20, points_per_stroke=32):
        super().__init__()
        self.max_strokes = max_strokes
        self.points_per_stroke = points_per_stroke
        
        # Predict number of strokes
        self.stroke_count_head = layers.Dense(max_strokes, activation='softmax')
        
        # Predict stroke parameters
        self.stroke_decoder = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(max_strokes * points_per_stroke * 2),
            layers.Reshape((max_strokes, points_per_stroke, 2))
        ])
```

**Data requirement**: Skeleton graphs from Phase 0 provide ground truth for stroke structure.

---

## Quick Reference: Tensor Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| Alphabet (points) | `(52, 256, 2)` | Full alphabet, arc-length samples |
| Alphabet (combined) | `(52, 256, 4)` | Points + skeleton samples |
| Glyph embedding | `(52, 64)` | Learnable glyph identity |
| Type embedding | `(2, 16)` | Upper/lower distinction |
| Style vector | `(batch, 128)` | Alphabet-level style |
| Decoded glyph | `(batch, 256, 2)` | Generated point cloud |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Phase 0 | Use 256 arc-length samples | Balance detail vs. computation |
| Phase 0 | 512×512 rasters for skeletonization | Sufficient resolution for topology |
| Phase 1 | x_height/cap_height normalization | Preserve upper/lower relationships |
| Phase 2 | Start with point clouds, not strokes | Simpler, faster iteration |
| Phase 2 | Chamfer before EMD | Faster training, add EMD later |
| Phase 2 | Separate mu/log_var heads | Proper VAE architecture |
