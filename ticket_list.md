# Project Tickets - Alphaba ML Alphabet Project

## Priority Order (Max 5 tickets) - Updated

### 📊 **High - Add Evaluation Metrics**
- **Issue**: No quantitative metrics to measure embedding quality beyond visualization
- **Impact**: Cannot objectively assess model performance or track improvements
- **Action**: Implement silhouette score, intra/inter-alphabet distance calculations, and embedding quality metrics

### 🏗️ **High - Implement Batch Normalization**
- **Issue**: Model lacks batch normalization for training stability
- **Impact**: Slower convergence and less stable training
- **Action**: Add BatchNormalization layers after each Conv2D layer in base network

### 🔍 **Medium - Fix Loss Function Consistency**
- **Issue**: Two different training functions with inconsistent loss implementations
- **Impact**: Confusing training behavior and maintenance overhead
- **Action**: Unify training approach and ensure consistent loss calculation

### 📝 **Medium - Add Model Compilation Function**
- **Issue**: `compile_triplet_model()` function exists but may not be properly integrated
- **Impact**: Model training might not use optimal optimizer settings
- **Action**: Ensure proper model compilation and verify optimizer configuration

### 🧪 **Medium - Add Embedding Quality Tests**
- **Issue**: No automated tests to verify embedding quality improvements
- **Impact**: Cannot systematically validate that changes improve performance
- **Action**: Create test functions to validate embedding clustering and quality metrics

---
*Updated tickets based on current codebase state - core bugs have been resolved*