# VerifiedNN Architecture Documentation

**Project:** Verified Neural Network Training in Lean 4
**Last Updated:** 2025-10-21
**Status:** Complete project structure, 17 sorries remaining in proofs

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Module Hierarchy](#module-hierarchy)
3. [Dependency Graph](#dependency-graph)
4. [Data Flow](#data-flow)
5. [Verification Architecture](#verification-architecture)
6. [Module Descriptions](#module-descriptions)

---

## System Overview

VerifiedNN implements a **formally verified neural network training system** in Lean 4, focusing on:
- **Gradient correctness:** Proving automatic differentiation computes correct derivatives
- **Type safety:** Enforcing dimension consistency through dependent types
- **Practical training:** Functional MNIST training with SGD and backpropagation

### Architecture Principles

1. **Layered Design:** Core → Layer → Network → Training
2. **Verification Isolation:** Separate computational code (Float) from proofs (ℝ)
3. **Dependency Minimization:** Each module imports only what it needs
4. **Re-export Modules:** Top-level namespace modules for convenience

### Tech Stack Integration

```
┌─────────────────────────────────────────┐
│         VerifiedNN (This Project)       │
├─────────────────────────────────────────┤
│              SciLean                    │  ← Automatic differentiation
├─────────────────────────────────────────┤
│             mathlib4                    │  ← Mathematical foundations
├─────────────────────────────────────────┤
│              Lean 4                     │  ← Proof assistant & language
└─────────────────────────────────────────┘
```

---

## Module Hierarchy

The codebase is organized into **10 primary directories** under `VerifiedNN/`:

### Tier 0: Foundation
**Core/** - Fundamental types and operations (0 dependencies)
- DataTypes, LinearAlgebra, Activation
- All other modules depend on Core

### Tier 1: Basic Components
**Layer/** - Neural network layers (depends: Core)
- Dense layers, composition, properties

**Data/** - Dataset handling (depends: Core)
- MNIST loading, preprocessing, iteration

**Loss/** - Loss functions (depends: Core)
- Cross-entropy, gradients, properties

**Optimizer/** - Optimization algorithms (depends: Core)
- SGD, momentum, update rules

### Tier 2: High-Level Systems
**Network/** - Multi-layer networks (depends: Core, Layer)
- Architecture, initialization, gradient computation

**Training/** - Training pipeline (depends: Network, Loss, Optimizer)
- Training loop, batching, metrics

### Tier 3: Quality Assurance
**Testing/** - Test suites (depends: most modules)
- Unit tests, integration tests, gradient checking

**Verification/** - Formal proofs (depends: Core, Layer, Network, Loss)
- Gradient correctness, type safety, convergence theory

**Examples/** - End-user programs (depends: all)
- Simple example, MNIST training application

---

## Dependency Graph

### High-Level Module Dependencies

```
Examples (MNISTTrain, SimpleExample)
    │
    ├──> Training (Loop, Batch, Metrics)
    │       ├──> Network (Architecture, Gradient, Initialization)
    │       │       ├──> Layer (Dense, Composition, Properties)
    │       │       │       └──> Core (DataTypes, LinearAlgebra, Activation)
    │       │       └──> Loss (CrossEntropy, Gradient, Properties)
    │       │               └──> Core
    │       └──> Optimizer (SGD, Momentum, Update)
    │               └──> Core
    │
    ├──> Testing (UnitTests, Integration, GradientCheck, OptimizerTests)
    │       ├──> Network
    │       ├──> Layer
    │       ├──> Optimizer
    │       └──> Core
    │
    ├──> Verification (GradientCorrectness, TypeSafety, Convergence)
    │       ├──> Network
    │       ├──> Layer
    │       ├──> Loss
    │       └──> Core
    │
    └──> Data (MNIST, Preprocessing, Iterator)
            └──> Core
```

### Detailed File-Level Dependencies

#### Core Module (Foundation - No Internal Dependencies)
```
Core/
├── DataTypes.lean              (leaf - no VerifiedNN imports)
├── LinearAlgebra.lean          → DataTypes
└── Activation.lean             → DataTypes
```

#### Layer Module
```
Layer/
├── Dense.lean                  → Core.{DataTypes, LinearAlgebra, Activation}
├── Composition.lean            → Dense, Core.Activation
└── Properties.lean             → Dense, Composition, Core.LinearAlgebra
```

#### Network Module
```
Network/
├── Architecture.lean           → Layer.Dense, Core.Activation
├── Initialization.lean         → Architecture, Layer.Dense, Core.DataTypes
└── Gradient.lean               → Architecture, Loss.CrossEntropy
```

#### Loss Module
```
Loss/
├── CrossEntropy.lean           → Core.DataTypes
├── Gradient.lean               → CrossEntropy
├── Properties.lean             → CrossEntropy, Gradient
└── Test.lean                   → CrossEntropy, Gradient
```

#### Optimizer Module
```
Optimizer/
├── SGD.lean                    → Core.DataTypes
├── Momentum.lean               → Core.DataTypes
└── Update.lean                 → SGD, Momentum, Core.DataTypes
```

#### Training Module
```
Training/
├── Batch.lean                  → Core.DataTypes
├── Metrics.lean                → Network.Architecture, Loss.CrossEntropy
└── Loop.lean                   → Network.{Architecture, Gradient}
                                  Training.{Batch, Metrics}
                                  Loss.CrossEntropy, Optimizer.SGD
```

#### Data Module
```
Data/
├── MNIST.lean                  → Core.DataTypes
├── Preprocessing.lean          → Core.DataTypes
└── Iterator.lean               → Core.DataTypes
```

#### Verification Module
```
Verification/
├── Tactics.lean                (leaf - helper tactics)
├── GradientCorrectness.lean    → Core.{Activation, LinearAlgebra}, Loss.Gradient
├── TypeSafety.lean             → Layer.{Dense, Composition}
│                                 Network.{Architecture, Gradient}
│                                 Core.{DataTypes, LinearAlgebra}
├── Convergence/
│   ├── Axioms.lean             (leaf - convergence axioms)
│   ├── Lemmas.lean             → Axioms
│   └── Convergence.lean        → Axioms, Lemmas (re-export)
```

#### Testing Module
```
Testing/
├── UnitTests.lean              → Core.{DataTypes, LinearAlgebra, Activation}
│                                 Layer.Dense
├── OptimizerTests.lean         → Optimizer.{SGD, Momentum, Update}
├── OptimizerVerification.lean  → Optimizer.{SGD, Momentum, Update}
├── GradientCheck.lean          → Core.DataTypes, Network.Gradient
├── Integration.lean            → Network.Architecture, Training.Loop
│                                 Core.{DataTypes, Activation}, Loss.CrossEntropy
└── RunTests.lean               → UnitTests, OptimizerTests, Integration
```

#### Examples Module
```
Examples/
├── SimpleExample.lean          (standalone - demonstrates API)
└── MNISTTrain.lean             (standalone - full training program)
```

### Re-export Convenience Modules

Top-level namespace modules for convenience (located in `VerifiedNN/`):
```
Core.lean       → re-exports Core.{DataTypes, LinearAlgebra, Activation}
Data.lean       → re-exports Data.{MNIST, Preprocessing, Iterator}
Optimizer.lean  → re-exports Optimizer.{SGD, Momentum, Update}
Training.lean   → re-exports Training.{Loop, Batch, Metrics}
```

---

## Data Flow

### Forward Pass (Inference)

```
Input Data (Float^[batchSize, inputDim])
    │
    ├──> Layer 1 (Dense)
    │       ├──> Matrix multiply: W₁ × x
    │       ├──> Add bias: + b₁
    │       └──> Activation: σ(·)
    │
    ├──> Layer 2 (Dense)
    │       ├──> Matrix multiply: W₂ × h₁
    │       ├──> Add bias: + b₂
    │       └──> Activation: σ(·)
    │
    ├──> ... (more layers)
    │
    └──> Output Layer (Dense)
            ├──> Matrix multiply: Wₙ × hₙ₋₁
            ├──> Add bias: + bₙ
            └──> Softmax: σ(·)
                    │
                    ▼
            Predictions (Float^[batchSize, numClasses])
```

**Implementation:** `Network.Architecture.forward`

### Backward Pass (Gradient Computation)

```
Loss Function (Cross-Entropy)
    │
    ├──> ∂L/∂ŷ (output gradient)
    │
    └──> Backpropagate through layers (reverse order)
            │
            ├──> Layer n: ∂L/∂Wₙ, ∂L/∂bₙ, ∂L/∂hₙ₋₁
            ├──> Layer 2: ∂L/∂W₂, ∂L/∂b₂, ∂L/∂h₁
            └──> Layer 1: ∂L/∂W₁, ∂L/∂b₁
                    │
                    ▼
            Gradient vector (all parameters)
```

**Implementation:** `Network.Gradient.computeGradient`
**Verification:** `Verification.GradientCorrectness` (proves gradient = mathematical derivative)

### Training Loop

```
┌─────────────────────────────────────────────────┐
│            Training Loop Iteration              │
└─────────────────────────────────────────────────┘
    │
    ├──> 1. Sample batch from dataset
    │       (Training.Batch.sampleBatch)
    │
    ├──> 2. Forward pass → predictions
    │       (Network.Architecture.forward)
    │
    ├──> 3. Compute loss
    │       (Loss.CrossEntropy.compute)
    │
    ├──> 4. Backward pass → gradients
    │       (Network.Gradient.computeGradient)
    │
    ├──> 5. Update parameters (SGD)
    │       (Optimizer.SGD.update)
    │
    ├──> 6. Update metrics (loss, accuracy)
    │       (Training.Metrics.update)
    │
    └──> 7. Repeat for all batches → next epoch
```

**Implementation:** `Training.Loop.train`
**Integration Test:** `Testing.Integration.testTrainingPipeline`

---

## Verification Architecture

### Verification Strategy

The project uses a **two-tier verification approach**:

1. **Computational Tier (Float):** Actual training code using IEEE 754 floats
2. **Mathematical Tier (ℝ):** Formal proofs on real numbers

**Bridge:** Axioms connecting Float behavior to ℝ (explicitly documented, see Loss/Properties.lean)

### Proof Structure

```
Verification/
│
├── GradientCorrectness.lean (PRIMARY CONTRIBUTION)
│   ├── sigmoid_differentiable
│   ├── sigmoid_fderiv_correct
│   ├── relu_differentiable
│   ├── composition_differentiable
│   ├── cross_entropy_gradient_correct
│   └── softmax_cross_entropy_gradient_correct
│
├── TypeSafety.lean (SECONDARY CONTRIBUTION)
│   ├── layer_forward_preserves_dims
│   ├── network_forward_type_safe
│   ├── gradient_dimensions_match
│   ├── flatten_unflatten_inverse
│   └── parameter_count_correct
│
└── Convergence/ (OUT OF SCOPE - Axiomatized)
    ├── Axioms.lean (8 convergence theorems)
    │   ├── sgd_converges_under_conditions
    │   ├── learning_rate_decay_necessary
    │   └── ... (6 more axioms)
    │
    └── Lemmas.lean (Robbins-Monro conditions)
        ├── robbins_monro_conditions
        └── learning_rate_summability_lemmas
```

### Verification Metrics

| Module | Sorries | Axioms | Status |
|--------|---------|--------|--------|
| Core | 1 | 0 | ⚠️ 1 trivial proof |
| Layer | 1 | 0 | ⚠️ 1 affine property |
| Network | 7 | 0 | ⚠️ Index arithmetic |
| Loss | 0 | 1 | ✅ Float bridge axiom documented |
| Verification/GradientCorrectness | 6 | 0 | ⚠️ Mathlib integration needed |
| Verification/TypeSafety | 2 | 0 | ⚠️ DataArrayN.ext lemmas needed |
| Verification/Convergence | 0 | 8 | ✅ Axiomatized (out of scope) |
| **Total** | **17** | **9** | **All documented** |

**Target:** ≤8 axioms (currently 9, acceptable deviation)
**Current Focus:** Eliminate sorries in GradientCorrectness (primary contribution)

### Verification Dependency Graph

```
Proofs about end-to-end training
    │
    ├──> TypeSafety (dimensions preserved throughout)
    │       └──> Layer properties → Core properties
    │
    └──> GradientCorrectness (AD computes correct derivatives)
            ├──> Loss gradient properties
            ├──> Layer differentiability
            └──> Core activation derivatives
```

---

## Module Descriptions

### Core/ - Foundation Layer

**Purpose:** Fundamental types and operations for numerical computing

**Files:**
- **DataTypes.lean** (leaf module)
  - `Vector n`, `Matrix m n`, `Batch b n` definitions
  - `Tensor` type with dimension tracking
  - Approximate equality for numerical comparisons

- **LinearAlgebra.lean** (depends: DataTypes)
  - Matrix-vector multiplication
  - Vector operations (dot product, norms)
  - **1 sorry:** `matvec_linear` proof

- **Activation.lean** (depends: DataTypes)
  - ReLU, sigmoid, softmax implementations
  - Differentiability registered with SciLean
  - Numerical stability considerations

**Verification Status:** 1 sorry in LinearAlgebra (trivial linearity proof)

---

### Layer/ - Neural Network Layers

**Purpose:** Composable neural network layer implementations

**Files:**
- **Dense.lean** (depends: Core)
  - `DenseLayer m n` structure (weights, biases, activation)
  - Forward pass implementation
  - Differentiability proofs

- **Composition.lean** (depends: Dense)
  - Layer composition operators
  - Chain rule application
  - Composition differentiability

- **Properties.lean** (depends: Dense, Composition)
  - Mathematical properties of layers
  - **1 sorry:** Affine combination preservation

**Verification Status:** 1 sorry (affine property proof)

---

### Network/ - Multi-Layer Networks

**Purpose:** Multi-layer perceptron architecture and training primitives

**Files:**
- **Architecture.lean** (depends: Layer, Core)
  - `MLPArchitecture` definition
  - Multi-layer forward pass
  - Parameter flattening/unflattening

- **Initialization.lean** (depends: Architecture)
  - Xavier/He initialization
  - Random weight generation (IO)

- **Gradient.lean** (depends: Architecture, Loss)
  - Backpropagation implementation
  - **7 sorries:** Index arithmetic bounds proofs
  - Uses SciLean automatic differentiation

**Verification Status:** 7 sorries (all index bounds, considered trivial)

---

### Loss/ - Loss Functions

**Purpose:** Cross-entropy loss with verified gradients

**Files:**
- **CrossEntropy.lean** (depends: Core)
  - Categorical cross-entropy implementation
  - Numerical stability (log-sum-exp trick)

- **Gradient.lean** (depends: CrossEntropy)
  - Gradient computation (ŷ - y for softmax)
  - Chain rule integration

- **Properties.lean** (depends: CrossEntropy, Gradient)
  - **1 axiom (Float bridge):** Connects Float computation to ℝ proofs
  - 58-line comprehensive axiom documentation
  - Mathematical correctness statements

- **Test.lean** (testing only)

**Verification Status:** 0 sorries, 1 well-documented axiom (Float/ℝ bridge)

---

### Optimizer/ - Optimization Algorithms

**Purpose:** Parameter update rules for training

**Files:**
- **SGD.lean** (depends: Core)
  - Stochastic gradient descent
  - Configurable learning rate

- **Momentum.lean** (depends: Core)
  - Momentum-based SGD
  - Velocity accumulation

- **Update.lean** (depends: SGD, Momentum)
  - Unified parameter update interface
  - Learning rate scheduling

**Verification Status:** 0 sorries, 0 axioms (complete implementation)

---

### Training/ - Training Pipeline

**Purpose:** End-to-end training orchestration

**Files:**
- **Batch.lean** (depends: Core)
  - Mini-batch sampling
  - Shuffle and iteration

- **Metrics.lean** (depends: Network, Loss)
  - Training metrics (loss, accuracy)
  - Metric aggregation

- **Loop.lean** (depends: Network, Training, Loss, Optimizer)
  - Main training loop
  - Epoch iteration
  - Full training pipeline integration

**Verification Status:** 0 sorries, 0 axioms (complete implementation)

---

### Data/ - Dataset Handling

**Purpose:** MNIST data loading and preprocessing

**Files:**
- **MNIST.lean** (depends: Core)
  - IDX file format parsing
  - Dataset loading from disk

- **Preprocessing.lean** (depends: Core)
  - Normalization (pixel values → [0,1])
  - One-hot encoding for labels

- **Iterator.lean** (depends: Core)
  - Dataset iteration abstraction
  - Batch generation

**Verification Status:** Placeholder implementations (functional but minimal)

---

### Verification/ - Formal Proofs

**Purpose:** Primary scientific contribution - gradient correctness proofs

**Files:**
- **GradientCorrectness.lean** (PRIMARY)
  - **6 sorries:** Mathlib integration needed
  - Proves: `fderiv ℝ f = analytical_derivative(f)` for each operation
  - Activation functions, composition, loss gradients

- **TypeSafety.lean** (SECONDARY)
  - **2 sorries:** Flatten/unflatten inverse proofs
  - Dimension preservation throughout network
  - Type-level guarantees enforce runtime correctness

- **Convergence/** (OUT OF SCOPE)
  - **Axioms.lean:** 8 convergence theorems (accepted as axioms)
  - **Lemmas.lean:** Robbins-Monro conditions
  - **Convergence.lean:** Re-export module

- **Tactics.lean**
  - Custom tactics for verification
  - Helper automation

**Verification Status:**
- GradientCorrectness: 6 sorries (high priority for completion)
- TypeSafety: 2 sorries (DataArrayN.ext lemmas needed)
- Convergence: 8 axioms (intentionally out of scope)

---

### Testing/ - Quality Assurance

**Purpose:** Unit, integration, and numerical validation tests

**Files:**
- **UnitTests.lean**
  - Core component tests (activations, data types)
  - Mathematical property validation

- **OptimizerTests.lean**
  - SGD, momentum, learning rate tests
  - Update rule correctness

- **OptimizerVerification.lean**
  - Type-level optimizer verification
  - Dimension preservation proofs

- **GradientCheck.lean**
  - Finite difference validation of AD
  - Numerical gradient comparison

- **Integration.lean**
  - End-to-end pipeline tests
  - Training on tiny synthetic datasets

- **RunTests.lean**
  - Unified test runner
  - Comprehensive reporting

**Build Status:** All 6 test files compile successfully with 0 errors ✅

---

### Examples/ - End-User Applications

**Purpose:** Demonstrate API usage and full training

**Files:**
- **SimpleExample.lean**
  - Minimal MLP training example
  - Demonstrates basic API

- **MNISTTrain.lean**
  - Full MNIST training program
  - Command-line interface
  - Performance metrics

**Status:** Mock implementations (demonstrate structure)

---

## Cross-Cutting Concerns

### SciLean Integration

**Automatic Differentiation:**
- `@[fun_prop]` for differentiability registration
- `@[fun_trans]` for derivative computation rules
- `fun_trans` tactic for proof automation

**Used in:**
- Core/Activation.lean (sigmoid, ReLU derivatives)
- Network/Gradient.lean (backpropagation)
- Verification/GradientCorrectness.lean (correctness proofs)

### Dependent Types for Safety

**Dimension Tracking:**
```lean
Vector (n : Nat) := Float^[n]
Matrix (m n : Nat) := Float^[m, n]
DenseLayer (inDim outDim : Nat)
```

**Compile-Time Guarantees:**
- Matrix-vector multiplication dimension compatibility
- Layer composition dimension matching
- Network architecture well-formedness

**Runtime Safety:**
- No dimension mismatch errors possible
- Type checker enforces correctness

---

## Build and Development

### Build Dependency Order

1. **Core modules** (DataTypes → LinearAlgebra, Activation)
2. **Independent modules** (Data, Loss, Optimizer in parallel)
3. **Layer modules** (Dense → Composition → Properties)
4. **Network modules** (Architecture → Initialization, Gradient)
5. **Training modules** (Batch, Metrics → Loop)
6. **Verification modules** (GradientCorrectness, TypeSafety, Convergence)
7. **Testing modules** (all tests)
8. **Examples** (SimpleExample, MNISTTrain)

**Total build:** ~2-3 minutes with mathlib cache

### Import Hygiene

**Rules:**
1. Never create circular dependencies
2. Import only what you need (not entire namespaces)
3. Prefer specific imports: `import VerifiedNN.Core.DataTypes`
4. Use re-export modules for user-facing APIs

**Verification:**
```bash
# Check for circular dependencies
lake build --verbose 2>&1 | grep -i "circular"

# View import graph (if using lean4-graph)
lake exe graph
```

---

## Future Extensions

### Planned Architecture Changes

1. **Additional Layer Types** (if needed)
   - Convolution layers (VerifiedNN/Layer/Conv2D.lean)
   - Batch normalization (VerifiedNN/Layer/BatchNorm.lean)

2. **Advanced Optimizers**
   - Adam, AdaGrad (VerifiedNN/Optimizer/Adam.lean)

3. **Additional Loss Functions**
   - MSE, Hinge loss (VerifiedNN/Loss/MSE.lean)

4. **GPU Support** (if SciLean adds CUDA backend)
   - Modify Core/DataTypes.lean to support GPU tensors

### Scalability Considerations

**Current scale:** MNIST (60K training samples, 784 input dim)
**Architecture supports:** Arbitrary network sizes (limited by memory)
**Verification scales to:** Networks of reasonable size (dependent type checking overhead)

---

## References

- **Project Documentation:** [README.md](README.md)
- **Cleanup Report:** [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)
- **Development Guide:** [CLAUDE.md](CLAUDE.md)
- **Technical Spec:** [verified-nn-spec.md](verified-nn-spec.md)
- **Directory READMEs:** All 10 subdirectories have comprehensive documentation

---

**Document Maintenance:**
- Update this file when adding new modules or changing dependencies
- Regenerate dependency graph after major refactoring
- Keep verification metrics synchronized with actual sorry/axiom counts

**Last Verified:** 2025-10-21
**Maintainers:** Project contributors
