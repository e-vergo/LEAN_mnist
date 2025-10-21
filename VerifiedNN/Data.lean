/-
# Data Loading and Preprocessing

This module provides data loading and preprocessing utilities for neural network training.

**Components:**
- `MNIST`: MNIST dataset loading from IDX format
- `Preprocessing`: Normalization and data transformation utilities
- `Iterator`: Memory-efficient data iteration for training

**Verification status:** Implementation only, no formal verification.
-/

import VerifiedNN.Data.MNIST
import VerifiedNN.Data.Preprocessing
import VerifiedNN.Data.Iterator
