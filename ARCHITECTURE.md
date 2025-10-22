# VerifiedNN Architecture

A comprehensive guide to the system design, module dependencies, and architectural decisions of the VerifiedNN project.

## Table of Contents

1. [System Overview](#system-overview)
2. [Module Dependency Graph](#module-dependency-graph)
3. [Core Design Principles](#core-design-principles)
4. [Module Descriptions](#module-descriptions)
5. [Data Flow](#data-flow)
6. [Verification Architecture](#verification-architecture)
7. [Design Decisions](#design-decisions)
8. [Extension Points](#extension-points)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VerifiedNN System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐      │
│  │   Examples  │  │  Verification│  │    Testing    │      │
│  │  (Entry)    │  │   (Proofs)   │  │  (Validation) │      │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘      │
│         │                 │                   │               │
│         └─────────────────┴───────────────────┘               │
│                           │                                   │
│         ┌─────────────────┴─────────────────┐                │
│         │                                     │                │
│  ┌──────▼──────┐  ┌──────────┐  ┌──────────▼──────┐        │
│  │  Training   │  │ Optimizer│  │    Network      │        │
│  │   (Loop)    │  │   (SGD)  │  │ (Architecture)  │        │
│  └──────┬──────┘  └────┬─────┘  └────────┬────────┘        │
│         │              │                   │                  │
│         └──────────────┴───────────────────┘                  │
│                        │                                      │
│         ┌──────────────┴──────────────┐                      │
│         │                               │                      │
│  ┌──────▼─────┐  ┌──────────┐  ┌─────▼─────┐               │
│  │    Loss    │  │  Layer   │  │   Data    │               │
│  │(CrossEntropy)│ │ (Dense)  │  │  (MNIST)  │               │
│  └──────┬─────┘  └────┬─────┘  └───────────┘               │
│         │             │                                       │
│         └─────────────┘                                       │
│                 │                                             │
│         ┌───────▼────────┐                                   │
│         │      Core      │                                   │
│         │  (Foundation)  │                                   │
│         └────────────────┘                                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Component Layers

**Layer 0: Foundation (Core)**
- Data types (Vector, Matrix, Batch)
- Linear algebra operations
- Activation functions
- No external dependencies except SciLean and mathlib

**Layer 1: Building Blocks (Layer, Loss, Data)**
- Dense layer implementation
- Cross-entropy loss
- MNIST data loading
- Depends on Core

**Layer 2: Orchestration (Network, Optimizer, Training)**
- MLP architecture
- SGD and variants
- Training loop
- Depends on Layer 1 and Core

**Layer 3: Application (Examples, Testing, Verification)**
- Runnable examples
- Test suites
- Formal proofs
- Depends on all lower layers

---

## Module Dependency Graph

### Text Representation

```
Examples ────────────────┐
  ├─ SimpleExample       │
  └─ MNISTTrain          │
                          ├──> Training ──┐
Testing ─────────────────┤                │
  ├─ UnitTests           │                ├──> Network ──┐
  ├─ Integration         │                │               │
  └─ GradientCheck       ├──> Optimizer ─┤               ├──> Layer ──┐
                          │                               │             │
Verification ────────────┤                               │             ├──> Core
  ├─ GradientCorrectness │                Loss ──────────┤             │
  ├─ TypeSafety          │                                │             │
  └─ Convergence         │                                              │
                                                                        │
                         Data ────────────────────────────────────────┘
```

### Dependency Matrix

| Module        | Core | Data | Layer | Loss | Network | Optimizer | Training | Examples | Testing | Verification |
|---------------|------|------|-------|------|---------|-----------|----------|----------|---------|--------------|
| Core          |  -   |  ✗   |   ✗   |  ✗   |    ✗    |     ✗     |    ✗     |    ✗     |    ✗    |      ✗       |
| Data          |  ✓   |  -   |   ✗   |  ✗   |    ✗    |     ✗     |    ✗     |    ✗     |    ✗    |      ✗       |
| Layer         |  ✓   |  ✗   |   -   |  ✗   |    ✗    |     ✗     |    ✗     |    ✗     |    ✗    |      ✗       |
| Loss          |  ✓   |  ✗   |   ✗   |  -   |    ✗    |     ✗     |    ✗     |    ✗     |    ✗    |      ✗       |
| Network       |  ✓   |  ✗   |   ✓   |  ✓   |    -    |     ✗     |    ✗     |    ✗     |    ✗    |      ✗       |
| Optimizer     |  ✓   |  ✗   |   ✗   |  ✗   |    ✗    |     -     |    ✗     |    ✗     |    ✗    |      ✗       |
| Training      |  ✓   |  ✓   |   ✓   |  ✓   |    ✓    |     ✓     |    -     |    ✗     |    ✗    |      ✗       |
| Examples      |  ✓   |  ✓   |   ✓   |  ✓   |    ✓    |     ✓     |    ✓     |    -     |    ✗    |      ✗       |
| Testing       |  ✓   |  ✓   |   ✓   |  ✓   |    ✓    |     ✓     |    ✓     |    ✗     |    -    |      ✗       |
| Verification  |  ✓   |  ✗   |   ✓   |  ✓   |    ✓    |     ✗     |    ✗     |    ✗     |    ✗    |      -       |

✓ = Imports directly, ✗ = No dependency

### Critical Import Paths

**Forward Pass Path:**
```
Examples/SimpleExample.lean
  → Training/Loop.lean
    → Network/Architecture.lean
      → Layer/Dense.lean
        → Core/LinearAlgebra.lean
          → Core/DataTypes.lean
```

**Gradient Computation Path:**
```
Examples/SimpleExample.lean
  → Training/Loop.lean
    → Network/Gradient.lean
      → Loss/CrossEntropy.lean
        → Core/Activation.lean (softmax)
```

**Verification Path:**
```
Verification/GradientCorrectness.lean
  → Network/Architecture.lean
  → Layer/Properties.lean
  → Loss/Properties.lean
  → Core/Activation.lean (with proofs)
```

---

## Core Design Principles

### 1. Layered Architecture

**Principle:** Lower layers never depend on higher layers.

**Rationale:** Enables independent testing, modularity, and prevents circular dependencies.

**Example:**
- `Core` modules can be tested without knowing about `Network`
- `Layer` can be used independently of `Training`
- `Verification` sits at top; proves properties of lower layers

### 2. Dependent Types for Safety

**Principle:** Use compile-time dimension checking wherever practical.

**Implementation:**
```lean
structure DenseLayer (inDim outDim : Nat) where
  weights : Matrix outDim inDim
  bias : Vector outDim
```

**Benefits:**
- Dimension mismatches caught at compile time
- Self-documenting code (dimensions in type signature)
- Impossible to construct invalid layer compositions

### 3. Separation of Concerns

**Principle:** Each module has a single, well-defined responsibility.

**Module Responsibilities:**

| Module | Responsibility | Does NOT Handle |
|--------|----------------|-----------------|
| Core | Foundation types and operations | Network architecture |
| Layer | Single layer transformations | Multi-layer composition |
| Network | Architecture and forward pass | Training logic |
| Loss | Loss computation | Optimization |
| Optimizer | Parameter updates | Training orchestration |
| Training | Training loop coordination | Proof obligations |
| Verification | Formal proofs | Executable code |

### 4. Verification-First Design

**Principle:** Prove properties on ℝ, implement in Float, bridge the gap explicitly.

**Workflow:**
1. Define operation on ℝ with formal properties
2. Implement in Float for computation
3. Document Float ≈ ℝ correspondence (axiomatized)
4. Numerically validate with gradient checking

**Example:**
```lean
-- Mathematical property (proven on ℝ)
theorem loss_nonneg_real {n : Nat} (predictions : Fin n → ℝ) (target : Fin n) :
  crossEntropyLossReal predictions target ≥ 0 := by
  -- 26-line proof
  ...

-- Computational implementation (Float)
def crossEntropyLoss {n : Nat} (predictions : Vector n) (target : Nat) : Float :=
  -- Numerically stable implementation
  ...

-- Bridge (axiomatized)
axiom float_crossEntropy_preserves_nonneg : ...
```

### 5. Documentation as Code

**Principle:** Every public definition has comprehensive docstrings.

**Standard:**
- Module docstrings: `/-!` format
- Definition docstrings: `/--` format
- Minimum content: Purpose, parameters, returns, properties, references

**Enforcement:**
- Pre-commit checklist in CLAUDE.md
- All 10 subdirectories have READMEs
- Quality gate: Zero undocumented public definitions

---

## Module Descriptions

### Core/ (Foundation Layer)

**Purpose:** Provide fundamental types and operations for neural network computations.

**Files:**
- `DataTypes.lean` (182 lines) - Vector, Matrix, Batch type aliases
- `LinearAlgebra.lean` (503 lines) - Matrix operations with 5 proven properties
- `Activation.lean` (390 lines) - ReLU, softmax, sigmoid with derivatives

**Key Abstractions:**
- `Vector n := Float^[n]` - Fixed-size vector
- `Matrix m n := Float^[m, n]` - Fixed-size matrix
- `matvec`, `matmul`, `vadd` - Linear algebra operations

**Verification Status:**
- 5 proven theorems (commutativity, associativity, distributivity, linearity)
- Zero sorries
- Zero axioms

**Dependencies:**
- SciLean (for `DataArrayN`)
- mathlib4 (for analysis and algebra)

### Layer/ (Neural Network Layers)

**Purpose:** Implement dense layers with type-safe transformations.

**Files:**
- `Dense.lean` (431 lines) - Dense layer implementation
- `Composition.lean` (213 lines) - Layer composition utilities
- `Properties.lean` (268 lines) - 13 proven layer properties

**Key Abstractions:**
- `DenseLayer inDim outDim` - Parameterized dense layer
- `forwardLinear` - Pre-activation output (Wx + b)
- `forward` - Full forward pass with activation

**Verification Status:**
- 13 proven theorems (dimension preservation, linearity, differentiability)
- Zero sorries
- Zero axioms

**Dependencies:**
- Core (DataTypes, LinearAlgebra, Activation)

### Network/ (Architecture)

**Purpose:** Define MLP architecture and coordinate forward/backward passes.

**Files:**
- `Architecture.lean` (213 lines) - MLP structure and forward pass
- `Initialization.lean` (265 lines) - He/Xavier initialization
- `Gradient.lean` (491 lines) - Gradient computation and parameter marshalling

**Key Abstractions:**
- `MLPArchitecture` - 2-layer network (784 → 128 → 10)
- `forwardPass` - End-to-end forward computation
- `flattenParams` / `unflattenParams` - Parameter marshalling

**Verification Status:**
- 2 axioms (flatten/unflatten inverses, inherited from SciLean DataArray.ext)
- Comprehensive 80+ line proof strategies documented

**Dependencies:**
- Core, Layer, Loss

### Loss/ (Loss Functions)

**Purpose:** Implement cross-entropy loss with mathematical properties.

**Files:**
- `CrossEntropy.lean` (202 lines) - Numerically stable implementation
- `Properties.lean` (272 lines) - 2 proven properties + Float bridge
- `Gradient.lean` (178 lines) - Loss gradient computation

**Key Abstractions:**
- `crossEntropyLoss` - CE(ŷ, y) = -log(ŷ_y)
- `batchCrossEntropyLoss` - Average over mini-batch
- `crossEntropyGrad` - Gradient: ŷ - one_hot(y)

**Verification Status:**
- 2 proven theorems (non-negativity on ℝ, log-sum-exp inequality)
- 1 axiom (Float ≈ ℝ bridge for non-negativity)
- 58-line justification for axiom

**Dependencies:**
- Core (Activation for softmax)

### Optimizer/ (Optimization Algorithms)

**Purpose:** Implement SGD and variants for parameter updates.

**Files:**
- `SGD.lean` (336 lines) - Basic SGD implementation
- `Momentum.lean` (168 lines) - SGD with momentum
- `LRSchedule.lean` (216 lines) - 5 learning rate schedules

**Key Abstractions:**
- `SGDOptimizer` - Basic SGD state
- `MomentumOptimizer` - Momentum-augmented SGD
- `LRSchedule` - Learning rate scheduling strategies

**Verification Status:**
- Zero sorries
- Zero axioms
- Type-level dimension preservation proven by construction

**Dependencies:**
- Core (Vector operations)

### Training/ (Training Orchestration)

**Purpose:** Coordinate the training loop and batch processing.

**Files:**
- `Loop.lean` (527 lines) - Training loop and epoch management
- `Batch.lean` (289 lines) - Mini-batch creation and iteration
- `Metrics.lean` (332 lines) - Accuracy and loss tracking

**Key Abstractions:**
- `trainEpoch` - Single epoch of training
- `createBatches` - Split dataset into mini-batches
- `evaluateFull` - Compute metrics on full dataset

**Verification Status:**
- Zero sorries
- Zero axioms
- Integration with Network and Optimizer modules

**Dependencies:**
- Core, Layer, Network, Loss, Optimizer, Data

### Data/ (Dataset Management)

**Purpose:** Load and preprocess MNIST dataset.

**Files:**
- `MNIST.lean` (408 lines) - IDX binary format parser
- `Preprocessing.lean` (229 lines) - Normalization and augmentation
- `Iterator.lean` (220 lines) - Efficient data iteration

**Key Abstractions:**
- `loadMNISTImages` - Parse IDX image files
- `loadMNISTLabels` - Parse IDX label files
- `normalize` - Scale pixels to [0, 1]

**Verification Status:**
- Zero sorries
- Zero axioms
- Fully computable (70,000 images load successfully)

**Dependencies:**
- Core (Vector, Batch)

### Verification/ (Formal Proofs)

**Purpose:** Prove gradient correctness and type safety properties.

**Files:**
- `GradientCorrectness.lean` (492 lines) - 11 gradient theorems
- `TypeSafety.lean` (340 lines) - 14 dimension consistency theorems
- `Convergence/Axioms.lean` (390 lines) - 8 convergence axioms (out of scope)
- `Tactics.lean` (156 lines) - Proof automation

**Key Theorems:**
- `network_gradient_correct` - Main theorem: end-to-end differentiability
- `chain_rule_preserves_correctness` - Composition preserves gradients
- `layer_preserves_affine_combination` - Dense layers are affine

**Verification Status:**
- 26 proven theorems (11 gradient + 14 type safety + 1 convergence lemma)
- 8 convergence axioms (explicitly out of scope per spec)
- Main contribution: Gradient correctness proof

**Dependencies:**
- Core, Layer, Network, Loss
- mathlib4 (calculus and analysis)

### Testing/ (Validation)

**Purpose:** Validate implementation correctness through tests.

**Files:**
- `UnitTests.lean` (11,879 bytes) - Component unit tests
- `OptimizerTests.lean` (8,137 bytes) - Optimizer validation
- `GradientCheck.lean` (11,013 bytes) - Finite difference validation
- `Integration.lean` (14,448 bytes) - End-to-end tests
- `SmokeTest.lean` (2,558 bytes) - Fast CI/CD test

**Key Test Suites:**
- Activation function properties
- Optimizer update correctness
- Gradient numerical validation
- MNIST data loading
- Full training pipeline

**Test Coverage:**
- Unit tests: 6/8 suites working (75%)
- Optimizer tests: 6/6 suites working (100%)
- Integration: Partial (dataset generation working)

**Dependencies:**
- All modules (tests entire stack)

### Examples/ (Demonstrations)

**Purpose:** Provide pedagogical and production examples.

**Files:**
- `SimpleExample.lean` (699 lines) - Minimal training demo
- `MNISTTrain.lean` - Production CLI (mock backend)

**Key Examples:**
- Synthetic data training (16 samples, 20 epochs)
- Command-line argument parsing
- Training progress display
- Network initialization and evaluation

**Status:**
- SimpleExample: ✅ Fully functional
- MNISTTrain: ⚠️ CLI complete, backend mock

**Dependencies:**
- Core, Data, Layer, Network, Loss, Optimizer, Training

### Util/ (Utilities)

**Purpose:** Supporting utilities (ASCII renderer, etc.)

**Files:**
- `ImageRenderer.lean` (380 lines) - ASCII art renderer
- `README.md` - Renderer technical documentation

**Key Feature:**
- First fully computable executable
- Manual unrolling workaround (28 match cases, 784 indices)
- Processes real MNIST images

**Status:**
- ✅ Fully functional and computable

**Dependencies:**
- Data (MNIST loading)

---

## Data Flow

### Training Data Flow

```
┌─────────────────────┐
│  MNIST IDX Files    │
│  (train-images.idx) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Data.MNIST         │
│  loadMNISTImages    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Data.Preprocessing │
│  normalize          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Training.Batch     │
│  createBatches      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Training.Loop      │
│  trainEpoch         │
└──────────┬──────────┘
           │
           ├──────────────────────────────┐
           │                              │
           ▼                              ▼
┌─────────────────────┐      ┌──────────────────────┐
│  Network.forward    │      │  Network.gradient     │
│  (loss computation) │      │  (backpropagation)    │
└──────────┬──────────┘      └───────────┬──────────┘
           │                              │
           │                              ▼
           │                  ┌──────────────────────┐
           │                  │  Optimizer.update    │
           │                  │  (SGD step)          │
           │                  └───────────┬──────────┘
           │                              │
           └──────────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Updated Network     │
                   └──────────────────────┘
```

### Forward Pass Data Flow

```
Input: Vector 784
     │
     ▼
┌────────────────────────┐
│  Layer 1: Dense        │
│  784 → 128             │
│  W₁x + b₁              │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Activation: ReLU      │
│  max(0, x)             │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Layer 2: Dense        │
│  128 → 10              │
│  W₂h + b₂              │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Activation: Softmax   │
│  exp(xᵢ) / Σexp(xⱼ)    │
└────────┬───────────────┘
         │
         ▼
Output: Vector 10 (probabilities)
```

### Gradient Computation Data Flow (Backpropagation)

```
Loss: ℓ = CrossEntropy(ŷ, y)
     │
     ▼
┌────────────────────────┐
│  ∂ℓ/∂ŷ = ŷ - y         │
│  (softmax + CE grad)   │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  ∂ℓ/∂h₂ (Layer 2 grad) │
│  via chain rule        │
└────────┬───────────────┘
         │
         ├──────────────────────┐
         │                      │
         ▼                      ▼
┌────────────────┐  ┌──────────────────┐
│  ∂ℓ/∂W₂        │  │  ∂ℓ/∂b₂          │
│  (weight grad) │  │  (bias grad)     │
└────────────────┘  └──────────────────┘
         │
         ▼
┌────────────────────────┐
│  ∂ℓ/∂h₁ (ReLU grad)    │
│  δ * (h₁ > 0)          │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  ∂ℓ/∂h₀ (Layer 1 grad) │
│  via chain rule        │
└────────┬───────────────┘
         │
         ├──────────────────────┐
         │                      │
         ▼                      ▼
┌────────────────┐  ┌──────────────────┐
│  ∂ℓ/∂W₁        │  │  ∂ℓ/∂b₁          │
│  (weight grad) │  │  (bias grad)     │
└────────────────┘  └──────────────────┘
         │
         ▼
All gradients collected
```

---

## Verification Architecture

### Proof Strategy

```
┌─────────────────────────────────────────────────────────┐
│                    Verification Goals                    │
├─────────────────────────────────────────────────────────┤
│  1. Gradient Correctness (Primary)                      │
│  2. Type Safety (Secondary)                             │
│  3. Convergence (Out of Scope - Axiomatized)            │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌───────────────────┐           ┌───────────────────┐
│ Component Proofs  │           │  Composition      │
│ (Per-operation)   │           │  Proofs           │
└────────┬──────────┘           └────────┬──────────┘
         │                                │
         ├──> relu_gradient_correct       ├──> chain_rule_preserves
         ├──> sigmoid_gradient_correct    ├──> layer_composition_correct
         ├──> matvec_gradient_correct     └──> network_gradient_correct
         ├──> softmax_gradient_correct
         └──> crossentropy_gradient_correct
                          │
                          ▼
              ┌──────────────────────┐
              │  Main Theorem        │
              │  network_gradient_   │
              │  correct             │
              └──────────────────────┘
```

### Verification Layers

**Layer 1: Primitive Operations**
- Prove `fderiv ℝ f = analytical_derivative(f)` for each operation
- ReLU, sigmoid, matrix multiply, vector add, etc.
- Uses SciLean's `fun_trans` tactic

**Layer 2: Layer Composition**
- Prove dense layer is differentiable
- Prove composition preserves differentiability
- Uses mathlib's `Differentiable.comp`

**Layer 3: Network Integration**
- Prove full network is differentiable
- Establish end-to-end gradient correctness
- Main theorem: `network_gradient_correct`

**Layer 4: Type Safety**
- Prove dimension specifications match runtime
- Prove operations preserve dimensions
- Leverages dependent type system

### Proof Techniques

**1. Type-Level Proofs (Compile-Time)**
```lean
-- Proof by construction: if it type-checks, dimensions are correct
theorem layer_output_dim {m n : Nat} (layer : DenseLayer n m) (x : Vector n) :
  (layer.forward x).size = m := by
  rfl  -- Trivial: type system guarantees this
```

**2. Symbolic Differentiation (SciLean)**
```lean
-- Automatic differentiation with fun_trans
theorem relu_gradient_correct :
  fderiv ℝ relu = ... := by
  unfold relu
  fun_trans
  simp [...]
```

**3. Mathematical Analysis (mathlib)**
```lean
-- Use calculus library from mathlib
theorem composition_differentiable
  (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) :
  Differentiable ℝ (g ∘ f) := by
  apply Differentiable.comp hg hf
```

**4. Numerical Validation (Testing)**
```lean
-- Gradient checking: compare AD vs finite differences
def checkGradient (f : Vector n → Float) (x : Vector n) : Bool :=
  let adGrad := (∇ x', f x') x |>.rewrite_by fun_trans
  let fdGrad := finiteDifferenceGradient f x
  vectorApproxEq adGrad fdGrad 1e-5
```

---

## Design Decisions

### Decision 1: SciLean vs Custom AD

**Options Considered:**
1. Build custom automatic differentiation
2. Use SciLean's AD infrastructure

**Decision:** Use SciLean

**Rationale:**
- SciLean provides proven `fun_trans` tactics
- Integration with mathlib's calculus library
- Focus project on neural network verification, not AD infrastructure

**Trade-offs:**
- ✅ Leverages existing proofs
- ❌ Noncomputable (gradients don't compile to binaries)
- ❌ Early-stage library with evolving API

### Decision 2: Dependent Types for Dimensions

**Options Considered:**
1. Runtime dimension checking
2. Compile-time dimension checking via dependent types

**Decision:** Dependent types

**Rationale:**
- Catches errors at compile time
- Self-documenting code
- Proof obligations become type-level guarantees

**Trade-offs:**
- ✅ Impossible to construct invalid layer compositions
- ✅ Dimension proofs often trivial (`rfl`)
- ❌ More verbose type signatures
- ❌ Less flexible for dynamic architectures

### Decision 3: Float vs ℝ Separation

**Options Considered:**
1. Verify Float arithmetic directly
2. Prove on ℝ, implement in Float, bridge explicitly

**Decision:** Separate Float and ℝ

**Rationale:**
- Lean 4 lacks comprehensive Float theory (no Flocq equivalent)
- Mathematical properties easier to prove on ℝ
- Standard practice in verified numerical computing

**Trade-offs:**
- ✅ Clean mathematical proofs
- ✅ Follows precedent (Certigrad)
- ❌ Float ≈ ℝ bridge axiomatized
- ❌ Numerical stability not formally verified

### Decision 4: Axiomatize Convergence Theory

**Options Considered:**
1. Prove convergence theorems formally
2. State convergence theorems, mark as out of scope

**Decision:** Axiomatize convergence

**Rationale:**
- Convergence proofs are a separate research project (multi-year effort)
- Not necessary for gradient correctness (primary goal)
- Well-established results in literature

**Trade-offs:**
- ✅ Focus on gradient correctness
- ✅ Clearly documented with references
- ❌ Convergence not formally verified
- ❌ 8 axioms in axiom catalog

### Decision 5: Module Granularity

**Options Considered:**
1. Few large files (monolithic)
2. Many small files (fine-grained)

**Decision:** Medium granularity (10 directories, ~46 files)

**Rationale:**
- Balance between navigability and cohesion
- Each module has clear responsibility
- File sizes: 150-500 lines (guideline)

**Trade-offs:**
- ✅ Easy to navigate
- ✅ Clear separation of concerns
- ❌ More imports to manage
- ❌ Potential for circular dependencies (avoided through layering)

---

## Extension Points

### Adding New Activation Functions

1. **Define activation in `Core/Activation.lean`:**
```lean
def myActivation (x : Float) : Float := ...
def myActivationVec {n : Nat} (v : Vector n) : Vector n :=
  ⊞ i => myActivation v[i]
```

2. **Register with SciLean:**
```lean
@[fun_prop]
theorem myActivation_differentiable : Differentiable ℝ myActivation := by
  -- Proof

@[fun_trans]
theorem myActivation_fderiv : fderiv ℝ myActivation = ... := by
  -- Proof
```

3. **Add tests in `Testing/UnitTests.lean`:**
```lean
def testMyActivation := test "myActivation properties" $ ...
```

4. **Update documentation:**
   - Add docstring to activation function
   - Update `Core/README.md`

### Adding New Layer Types

1. **Create file `Layer/MyLayer.lean`:**
```lean
structure MyLayer (inDim outDim : Nat) where
  -- Parameters

def MyLayer.forward {m n : Nat} (layer : MyLayer n m) (x : Vector n) : Vector m :=
  -- Forward pass
```

2. **Prove properties in `Layer/Properties.lean`:**
```lean
theorem myLayer_preserves_dimensions : ...
theorem myLayer_differentiable : ...
```

3. **Integrate with `Network/Architecture.lean`:**
```lean
structure ExtendedMLP where
  layer1 : DenseLayer 784 128
  myLayer : MyLayer 128 64
  layer2 : DenseLayer 64 10
```

4. **Add tests:**
   - Unit tests in `Testing/UnitTests.lean`
   - Integration test in `Testing/Integration.lean`

### Adding New Optimizers

1. **Define optimizer in `Optimizer/MyOptimizer.lean`:**
```lean
structure MyOptimizerState (nParams : Nat) where
  params : Vector nParams
  -- Optimizer-specific state

def myOptimizerStep (state : MyOptimizerState n) (grad : Vector n) : MyOptimizerState n :=
  -- Update rule
```

2. **Add tests in `Testing/OptimizerTests.lean`:**
```lean
def testMyOptimizer := test "myOptimizer update" $ ...
```

3. **Update `Training/Loop.lean` to support new optimizer:**
```lean
-- Add optimizer as parameter to training loop
```

### Adding New Loss Functions

1. **Implement in `Loss/MyLoss.lean`:**
```lean
def myLoss {n : Nat} (predictions : Vector n) (target : Nat) : Float :=
  -- Loss computation
```

2. **Prove properties in `Loss/Properties.lean`:**
```lean
theorem myLoss_nonneg : ∀ pred target, myLoss pred target ≥ 0 := by
  -- Proof
```

3. **Implement gradient in `Loss/Gradient.lean`:**
```lean
def myLossGrad {n : Nat} (predictions : Vector n) (target : Nat) : Vector n :=
  -- Gradient computation
```

4. **Add verification:**
```lean
theorem myLoss_gradient_correct :
  fderiv ℝ myLoss = ... := by
  -- Proof
```

### Adding New Datasets

1. **Create loader in `Data/MyDataset.lean`:**
```lean
def loadMyDataset (path : System.FilePath) : IO (Array (Vector n × Label)) :=
  -- Data loading logic
```

2. **Add preprocessing if needed:**
```lean
def preprocessMyDataset (data : Array (Vector n × Label)) : Array (Vector n × Label) :=
  -- Preprocessing
```

3. **Add test in `Testing/Integration.lean`:**
```lean
def testMyDatasetLoad := test "myDataset loads correctly" $ ...
```

---

## Performance Considerations

### Compilation Performance

**Challenge:** Large mathlib dependency causes slow builds.

**Mitigations:**
- Use `lake exe cache get` to download precompiled mathlib
- Incremental compilation (Lake only rebuilds changed modules)
- Module precompilation enabled in lakefile.lean

### Runtime Performance

**Challenge:** Lean 4 numerical code slower than optimized frameworks.

**Mitigations:**
- OpenBLAS integration for matrix operations
- `@[inline]` and `@[specialize]` annotations on hot paths
- `DataArrayN` instead of `Array Float` for numerical arrays

**Current status:** Acceptable for proof-of-concept, not production-ready.

### Memory Management

**Challenge:** Lean LSP can consume significant memory.

**Mitigations:**
- Monitor with `pgrep -af lean`
- Restart LSP when memory grows: `pkill -f "lean --server"`
- Work on one module at a time during development

---

## Future Architecture Improvements

### 1. Modular Activation System

**Current:** Hardcoded activation in forward pass
**Future:** Parameterize layers with activation functions

```lean
structure DenseLayer (inDim outDim : Nat) (activation : Vector outDim → Vector outDim) where
  weights : Matrix outDim inDim
  bias : Vector outDim
```

### 2. Generic Network Architecture

**Current:** Fixed 2-layer MLP
**Future:** Flexible layer stacking with heterogeneous lists

```lean
inductive NetworkArch : Nat → Nat → Type
  | single : DenseLayer inDim outDim → NetworkArch inDim outDim
  | cons : DenseLayer inDim midDim → NetworkArch midDim outDim → NetworkArch inDim outDim
```

### 3. Computable Gradients

**Current:** Noncomputable AD (SciLean limitation)
**Future:** Explore computable AD alternatives or contribute to SciLean

### 4. GPU Acceleration

**Current:** CPU-only via OpenBLAS
**Future:** FFI to CUDA for GPU execution

### 5. Convolutional Layers

**Current:** Dense layers only
**Future:** Add Conv2D layers for image classification

---

## Conclusion

The VerifiedNN architecture balances:
- **Modularity:** Clear separation of concerns across 10 directories
- **Verification:** Formal proofs at top layer, type safety throughout
- **Executability:** Data loading and forward pass computable
- **Extensibility:** Clear extension points for new components

**Design Philosophy Summary:**
1. Prove properties on ℝ, implement in Float
2. Use dependent types for compile-time safety
3. Layer architecture from foundation to application
4. Document everything at mathlib quality

**Next Steps for Architecture Evolution:**
- See `Extension Points` section for how to add components
- Consult `Future Improvements` for long-term roadmap
- Review `Design Decisions` for rationale behind current choices

---

**Last Updated:** 2025-10-22
**Maintained by:** Project contributors
**For Questions:** See CLAUDE.md or open an issue
