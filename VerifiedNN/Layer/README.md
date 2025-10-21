# VerifiedNN Layer Module

Neural network layer abstractions with compile-time dimension safety.

## Overview

The Layer module provides type-safe implementations of dense (fully-connected) layers and utilities for composing them into multi-layer networks. All dimension compatibility is enforced at compile time through Lean's dependent type system.

## Module Structure

### Dense.lean
Core implementation of dense layers.

**Key Definitions:**
- `DenseLayer n m` - Dense layer structure with weights and biases
- `forwardLinear` - Linear transformation: `Wx + b`
- `forward` - Forward pass with optional activation: `activation(Wx + b)`
- `forwardBatch` - Batched forward pass for efficient training
- `forwardReLU` / `forwardBatchReLU` - Convenience functions with ReLU activation

**Properties:**
- Type-safe: dimension compatibility guaranteed by type system
- Efficient: uses SciLean's `DataArrayN` for vectorized operations
- Batched operations for training throughput

### Composition.lean
Layer composition utilities for building multi-layer networks.

**Key Definitions:**
- `stack` - Compose two layers sequentially with optional activations
- `stackLinear` - Pure affine composition without activations
- `stackReLU` - Composition with ReLU activations
- `stackBatch` / `stackBatchReLU` - Batched composition
- `stack3` - Three-layer composition

**Properties:**
- Type-safe composition prevents dimension mismatches
- Batched variants for efficient training
- Differentiability preserved through composition (proof planned)

### Properties.lean
Mathematical properties and formal verification theorems.

**Theorem Categories:**

1. **Dimension Consistency** (lines 39-81)
   - Type-level guarantees for forward pass dimensions
   - Batch processing dimension preservation
   - Composition dimension correctness

2. **Linearity Properties** (lines 83-170)
   - `forwardLinear_is_affine` - Dense layer computes affine transformation
   - `matvec_is_linear` - Matrix multiplication is linear
   - `layer_preserves_affine_combination` - Affine maps preserve weighted sums (1 sorry)
   - `stackLinear_preserves_affine_combination` - Composition preserves affine structure

3. **Differentiability Properties** (lines 172-182)
   - Planned: gradient correctness via SciLean's automatic differentiation
   - Planned: chain rule for composition

4. **Type Safety Examples** (lines 184-220)
   - Demonstrations of compile-time dimension tracking

5. **Examples** (lines 222-245)
   - MNIST-typical architectures (784 → 128 → 10)
   - Batched processing examples

## Key Theorems

### Proven

- **Dimension Safety** (by type system):
  - `forward_dimension_typesafe` - Forward pass outputs correct dimension
  - `forwardBatch_dimension_typesafe` - Batched operations preserve dimensions
  - `composition_dimension_typesafe` - Layer composition maintains compatibility

- **Linearity**:
  - `forwardLinear_is_affine` - Dense layer is affine transformation
  - `matvec_is_linear` - Matrix-vector multiplication is linear
  - `forwardLinear_spec` - Forward pass computes `Wx + b`

- **Composition**:
  - `stackLinear_preserves_affine_combination` - Composition of affine maps is affine

### Incomplete (1 sorry)

- `layer_preserves_affine_combination` (Properties.lean:140)
  - **Claim**: Affine transformations preserve weighted sums when coefficients sum to 1
  - **Blocked by**: Requires distributivity lemmas from `Core.LinearAlgebra`
  - **Strategy**: Apply `matvec_linear` + scalar distributivity over addition
  - **Impact**: Used by `stackLinear_preserves_affine_combination`

## Verification Status

| Property | Status | Module |
|----------|--------|--------|
| Type-level dimension safety | ✓ Proven | All |
| Forward pass correctness | ✓ Implemented | Dense.lean |
| Batched operations | ✓ Implemented | Dense.lean |
| Affine transformation properties | Partial (1 sorry) | Properties.lean |
| Differentiability | Planned | See Verification/ |
| Gradient correctness | Planned | See Verification/ |

## Dependencies

```
Layer/
├── Dense.lean
│   ├── Core.DataTypes (Vector, Matrix, Batch)
│   ├── Core.LinearAlgebra (matvec, vadd, smul)
│   └── Core.Activation (reluVec, reluBatch)
├── Composition.lean
│   ├── Layer.Dense
│   └── Core.Activation
└── Properties.lean
    ├── Layer.Dense
    ├── Layer.Composition
    └── Core.LinearAlgebra
```

## Usage Examples

### Creating a Dense Layer
```lean
import VerifiedNN.Layer.Dense

-- MNIST hidden layer (784 → 128)
def hiddenLayer : DenseLayer 784 128 := {
  weights := ⊞ i j => 0.01  -- Initialize from Network.Initialization
  bias := ⊞ i => 0.0
}

-- Forward pass with ReLU
def output (x : Vector 784) : Vector 128 :=
  hiddenLayer.forwardReLU x
```

### Composing Layers
```lean
import VerifiedNN.Layer.Composition

-- Two-layer network (784 → 128 → 10)
def predict (x : Vector 784) : Vector 10 :=
  let layer1 : DenseLayer 784 128 := ...
  let layer2 : DenseLayer 128 10 := ...
  stackReLU layer1 layer2 x
```

### Batched Training
```lean
-- Process batch of 32 samples
def forwardBatch (X : Batch 32 784) : Batch 32 10 :=
  stackBatchReLU layer1 layer2 X
```

## Design Philosophy

### Type Safety First
Dimension errors are prevented at compile time. If code type-checks, dimensions are guaranteed to be compatible. This eliminates an entire class of runtime errors common in numerical computing.

### Verification Boundaries
- **Type-level properties**: Enforced by Lean's dependent types ✓
- **Mathematical properties**: Proven on abstract types (ℝ) with some dependencies incomplete
- **Numerical properties**: Implementation uses `Float`, verification uses `ℝ` (symbolic correctness)

### Performance Considerations
- Use `@[inline]` on hot-path functions for optimization
- Prefer batched operations for training (better cache utilization)
- Leverage SciLean's `DataArrayN` for efficient array operations

## Future Work

1. **Complete Core.LinearAlgebra proofs** - Enable `layer_preserves_affine_combination`
2. **Differentiability theorems** - Integrate SciLean's automatic differentiation
3. **Gradient correctness proofs** - Verify backpropagation computes correct derivatives
4. **Chain rule formalization** - Prove composition preserves differentiability

See `verified-nn-spec.md` Section 5.1 for detailed verification roadmap.

## Related Modules

- **Core/** - Fundamental types and operations
- **Network/** - Full MLP architecture and initialization
- **Verification/** - Formal proofs of gradient correctness
- **Training/** - Training loop and optimization

---

**Last Updated**: 2025-10-21
**Maintainers**: Project contributors
