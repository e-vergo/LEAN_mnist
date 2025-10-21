/-
# Optimizer Module

Gradient descent optimizers for neural network training.

This module provides:
- Stochastic Gradient Descent (SGD)
- SGD with Momentum (classical and Nesterov variants)
- Learning rate scheduling (constant, step, exponential, cosine)
- Gradient clipping for stability
- Gradient accumulation for large effective batch sizes
- Unified optimizer interface for flexibility

**Usage:**
```lean
import VerifiedNN.Optimizer

-- Use individual components
open VerifiedNN.Optimizer
let state := initSGD params learningRate
let updated := sgdStep state gradient

-- Or use unified interface
let optimizer := OptimizerState.sgd (initSGD params lr)
let updated := Update.optimizerStep optimizer gradient
```

**Module Structure:**
- `SGD`: Basic stochastic gradient descent
- `Momentum`: Momentum-based optimization
- `Update`: Learning rate scheduling and utilities

**See Also:**
- HEALTH_REPORT.md - Comprehensive module health assessment
- MATHLIB_INTEGRATION.md - Future verification roadmap
- FIXES_APPLIED.md - Recent improvements and fixes
-/

import VerifiedNN.Optimizer.SGD
import VerifiedNN.Optimizer.Momentum
import VerifiedNN.Optimizer.Update
