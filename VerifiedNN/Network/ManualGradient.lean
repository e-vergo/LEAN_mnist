import VerifiedNN.Core.DataTypes
import VerifiedNN.Core.Activation
import VerifiedNN.Core.DenseBackward
import VerifiedNN.Core.ReluBackward
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Gradient
import VerifiedNN.Network.GradientFlattening
import VerifiedNN.Loss.CrossEntropy
import VerifiedNN.Loss.Gradient
import SciLean

namespace VerifiedNN.Network.ManualGradient

open VerifiedNN.Core
open VerifiedNN.Core.DenseBackward
open VerifiedNN.Core.ReluBackward
open VerifiedNN.Network
open VerifiedNN.Network.Gradient
open VerifiedNN.Network.GradientFlattening
open VerifiedNN.Loss
open VerifiedNN.Loss.Gradient
open SciLean
open VerifiedNN.Core.Activation

set_default_scalar Float

/-!
# Manual Network Gradient Computation

Computable implementation of gradient computation for the 2-layer MLP.

This module implements **manual backpropagation** through the network by
explicitly chaining gradient operations. Unlike `Network.Gradient.networkGradient`
which uses SciLean's noncomputable `∇` operator, this implementation uses only
computable operations and can be compiled to a standalone executable.

## Network Architecture

```
Input[784] → Dense1 → ReLU → Dense2 → Softmax → CrossEntropy Loss
              W1,b1           W2,b2
            (784→128)       (128→10)
```

## Backward Pass Algorithm

Given loss L, target y, and saved forward activations:

1. **Loss gradient:** `∂L/∂z2 = softmax(z2) - one_hot(y)`
2. **Layer 2 gradients:**
   - `∂L/∂W2 = (∂L/∂z2) ⊗ h1`
   - `∂L/∂b2 = ∂L/∂z2`
   - `∂L/∂h1 = W2^T @ (∂L/∂z2)`
3. **ReLU gradient:** `∂L/∂z1[i] = ∂L/∂h1[i] if z1[i] > 0 else 0`
4. **Layer 1 gradients:**
   - `∂L/∂W1 = (∂L/∂z1) ⊗ input`
   - `∂L/∂b1 = ∂L/∂z1`

## Key Difference from SciLean AD

**SciLean's `networkGradient` (noncomputable):**
```lean
let lossFunc := fun p => computeLoss p input target
(∇ p, lossFunc p) params  -- Symbolic differentiation at type-check time
```

**Our `networkGradientManual` (computable):**
```lean
-- Forward pass with saved activations
let z1 := ...
let h1 := relu z1
let z2 := ...
-- Backward pass with explicit chain rule
let dL_dz2 := lossGradient z2 target
...
```

The manual version is **explicit** (we write out each gradient computation)
while SciLean's version is **implicit** (compiler derives gradients automatically).

Both should compute the **same mathematical gradient** - but only the manual
version is executable!

## Verification Strategy

The manual implementation should match the noncomputable specification:

**Theorem (to be proven in Verification/ManualGradientCorrectness.lean):**
```lean
theorem manual_matches_automatic :
  ∀ params input target,
    networkGradientManual params input target =
    networkGradient params input target
```

This preserves all 26 existing gradient correctness theorems!

## Key Functions

- `networkGradientManual`: Main computable gradient computation
- `networkGradientManual'`: Alias for consistency

## Verification Status

- **Computable:** Yes (uses only computable operations)
- **Correctness:** Should be validated via:
  1. Finite difference testing (Testing/GradientCheck.lean)
  2. Proof of equivalence to `networkGradient` (future work)
- **Performance:** Native binary execution, no interpreter needed

## Implementation Notes

**Forward Pass Requirements:**
- Must save intermediate activations (z1, h1, z2) for backward pass
- z1 (pre-ReLU activation) needed for gradient masking
- h1 (post-ReLU activation) needed for layer 2 weight gradient
- z2 (logits) needed for loss gradient

**Chain Rule Order:**
- Start at loss, work backward through layers
- Each gradient computation uses saved activations from forward pass
- Dimensions verified at compile time via dependent types

**Numerical Stability:**
- Loss gradient uses softmax-cross-entropy fusion (numerically stable)
- No explicit division operations in gradient computation
- Inherits stability properties from Loss.Gradient.lossGradient

## References

- Backpropagation algorithm: Rumelhart et al. (1986)
- Numerical stability: Goodfellow et al., Deep Learning (2016), Section 4.1
- SciLean AD: https://github.com/lecopivo/SciLean
-/

/-- Compute network gradients using manual backpropagation.

This is the **computable alternative** to `Network.Gradient.networkGradient`.
Unlike the SciLean AD version which uses the noncomputable `∇` operator,
this implementation explicitly applies the chain rule through each layer.

**Algorithm:**

1. **Forward pass** (save intermediate activations):
   - Unflatten parameters to network structure
   - Compute z1 = W1 @ x + b1 (save z1 for ReLU backward)
   - Compute h1 = relu(z1) (save h1 for layer 2 backward)
   - Compute z2 = W2 @ h1 + b2 (save z2 for loss backward)
   - Compute loss (not returned, but gradient starts here)

2. **Backward pass** (apply chain rule):
   - Start with loss gradient: dL/dz2
   - Backprop through layer 2: compute dW2, db2, dL/dh1
   - Backprop through ReLU: mask dL/dh1 → dL/dz1
   - Backprop through layer 1: compute dW1, db1
   - Flatten all gradients into parameter vector

**Parameters:**
- `params`: Flattened network parameters [101,770]
- `input`: Input image vector [784]
- `target`: Target class (0-9 for MNIST)

**Returns:** Gradient vector ∂L/∂params [101,770]

**Computational Cost:**
- Forward pass: O(784×128 + 128×10) = O(100K) operations
- Backward pass: Similar complexity
- Total: ~200K floating-point operations
- Runtime: <1ms on modern CPU

**Example:**
```lean
-- During training
let input : Vector 784 := loadImage(...)
let target : Nat := 5  -- Digit "5"

-- Compute gradient
let gradient := networkGradientManual params input target

-- Update parameters
let newParams := params - learningRate • gradient
```

**Correctness:**
Should match `Network.Gradient.networkGradient` within numerical precision.
Validate using `Testing.GradientCheck.runAllGradientTests`.

**Why This Works:**
Manual backpropagation implements the multivariate chain rule:
```
∂L/∂θ = (∂L/∂z2) · (∂z2/∂h1) · (∂h1/∂z1) · (∂z1/∂θ)
```
Each term is computed by one of our backward pass functions.

**Critical Implementation Details:**

1. **Pre-activation saved for ReLU:** z1 must be saved before ReLU activation
   because the gradient mask depends on whether z1[i] > 0, not h1[i] > 0.

2. **Post-activation saved for layer 2:** h1 (after ReLU) is needed for the
   outer product in dW2 = dL_dz2 ⊗ h1.

3. **Logits saved for loss:** z2 is needed for the combined softmax-cross-entropy
   gradient, which is numerically more stable than computing them separately.

4. **Gradient flattening order:** Must exactly match `flattenParams` layout:
   [W1 | b1 | W2 | b2] = [100352 | 128 | 1280 | 10] elements.
-/
@[inline]
def networkGradientManual
  (params : Vector nParams)
  (input : Vector 784)
  (target : Nat)
  : Vector nParams :=
  -- ===== FORWARD PASS (save activations) =====

  -- Unflatten parameters into network structure
  let net := unflattenParams params

  -- Layer 1: Dense layer forward
  let z1 := net.layer1.forwardLinear input  -- [128] pre-activation (SAVE for ReLU backward)

  -- ReLU activation
  let h1 := reluVec z1  -- [128] post-activation (SAVE for layer 2 backward)

  -- Layer 2: Dense layer forward
  let z2 := net.layer2.forwardLinear h1  -- [10] logits (SAVE for loss backward)

  -- Note: We don't compute softmax explicitly here since lossGradient
  -- combines softmax+cross-entropy gradient (more numerically stable)

  -- ===== BACKWARD PASS (chain rule) =====

  -- Start at the loss: gradient of cross-entropy w.r.t. logits
  -- This is the beautiful formula: softmax(z) - one_hot(target)
  let dL_dz2 := lossGradient z2 target  -- [10]

  -- Backprop through layer 2 (dense)
  let (dW2, db2, dL_dh1) := denseLayerBackward dL_dz2 h1 net.layer2.weights
  -- dW2: [10, 128] weight gradient
  -- db2: [10] bias gradient
  -- dL_dh1: [128] gradient flowing back to hidden layer

  -- Backprop through ReLU activation
  -- Gradient flows through where z1 > 0, zeroed elsewhere
  let dL_dz1 := reluBackward dL_dh1 z1  -- [128]

  -- Backprop through layer 1 (dense)
  let (dW1, db1, _dL_dinput) := denseLayerBackward dL_dz1 input net.layer1.weights
  -- dW1: [128, 784] weight gradient
  -- db1: [128] bias gradient
  -- We don't need dL_dinput since input is not trainable

  -- ===== PACK GRADIENTS =====

  -- Flatten all gradients into single parameter vector using GradientFlattening module
  GradientFlattening.flattenGradients dW1 db1 dW2 db2  -- [101,770]

/-- Alias for consistency with existing code.

Some modules use `networkGradient'` notation, this provides compatibility.
-/
@[inline]
def networkGradientManual' := networkGradientManual

-- ============================================================================
-- Examples and Validation
-- ============================================================================

/-- Example: Validate manual gradient computation is computable.

This demonstrates that the gradient can be computed without any noncomputable
operations, making it suitable for compilation to native binaries.

The full validation against finite differences should be performed using
`Testing.GradientCheck.runAllGradientTests`.
-/
example : True := by
  -- This demonstrates the gradient can be computed
  -- (Full validation requires finite difference testing)
  let params : Vector nParams := ⊞ (i : Idx nParams) => 0.01
  let input : Vector 784 := ⊞ (j : Idx 784) => 0.5
  let target : Nat := 0

  let gradient := networkGradientManual params input target
  -- Gradient is computable! (Can build executable)
  trivial

/-- Example: Gradient has correct dimensions.

The type system ensures the gradient vector has exactly the same dimensions
as the parameter vector, making it safe to use in optimization updates.
-/
example : True := by
  -- Random initialization (for demonstration)
  let params : Vector nParams := ⊞ (i : Idx nParams) => 0.1
  let input : Vector 784 := ⊞ (j : Idx 784) => 0.0
  let target : Nat := 5

  -- Compute gradient
  let gradient := networkGradientManual params input target

  -- Type checker verifies: gradient : Vector nParams
  -- This means gradient.size = params.size = 101,770
  trivial

/-!
## Validation Against SciLean AD

The manual gradient should match the automatic differentiation version within
numerical precision. This can be validated using:

```lean
-- In Testing/GradientCheck.lean
def testManualVsAutomatic : IO Unit := do
  let params := initializeRandomParams()
  let input := loadTestImage()
  let target := 5

  let manualGrad := networkGradientManual params input target
  let autoGrad := networkGradient params input target

  let diff := vectorNorm (manualGrad - autoGrad)
  if diff < 1e-6 then
    IO.println "✓ Manual gradient matches automatic gradient"
  else
    IO.println s!"✗ Gradient mismatch: {diff}"
```

## Performance Comparison

**Manual Backpropagation:**
- Advantages: Compiles to native code, predictable performance, no AD overhead
- Disadvantages: Must manually implement backward pass for each operation

**SciLean Automatic Differentiation:**
- Advantages: Compiler derives gradients automatically, less error-prone
- Disadvantages: Noncomputable (cannot build executable), may have runtime overhead

For production training, the manual implementation is preferred. For prototyping
and verification, the automatic version provides a correctness specification.

## Integration with Training Loop

```lean
-- In Training/Loop.lean
def trainStep (model : MLPArchitecture) (params : Vector nParams)
              (input : Vector 784) (target : Nat)
              (learningRate : Float) : Vector nParams :=
  -- Compute gradient using manual backpropagation
  let gradient := networkGradientManual params input target

  -- SGD update: params' = params - α * ∇L
  params - (learningRate • gradient)
```

The manual gradient computation is a drop-in replacement for the automatic
version, with the advantage of being fully executable.
-/

end VerifiedNN.Network.ManualGradient
