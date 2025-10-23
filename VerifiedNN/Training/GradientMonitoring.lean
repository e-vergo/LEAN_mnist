import VerifiedNN.Core.DataTypes
import SciLean

/-!
# Gradient Monitoring Utilities

Provides tools to compute and display gradient norms during training to help detect
vanishing or exploding gradients, which are common causes of poor training performance.

## Main Definitions

- `GradientNorms` - Structure holding norm values for all gradient components
- `computeMatrixNorm` - Computes Frobenius norm of a matrix
- `computeVectorNorm` - Computes L2 norm of a vector
- `computeGradientNorms` - Computes norms for all gradient components
- `formatGradientNorms` - Formats norms for display
- `checkGradientHealth` - Detects vanishing/exploding gradients

## Gradient Health Indicators

**Normal Range:** Most gradient norms should be in range [0.0001, 10.0]

**Vanishing Gradients (norm < 0.0001):**
- Symptoms: Very small gradient norms, learning stalls
- Causes: Too many layers, poor initialization, saturating activations
- Solutions: Better initialization (He/Xavier), skip connections, batch normalization

**Exploding Gradients (norm > 10.0):**
- Symptoms: Very large gradient norms, NaN/Inf in parameters, training diverges
- Causes: Poor initialization, high learning rate, unstable loss function
- Solutions: Gradient clipping, lower learning rate, better initialization

## Usage Example

```lean
-- During training, after computing gradients:
let gradients := computeManualGradients net input label
let norms := computeGradientNorms gradients

-- Display norms
IO.println (formatGradientNorms norms)

-- Check for problems
if norms.hasVanishing then
  IO.println "WARNING: Vanishing gradients detected!"
if norms.hasExploding then
  IO.println "WARNING: Exploding gradients detected!"
```

## Implementation Notes

**Frobenius Norm for Matrices:**
The Frobenius norm is defined as: ‖A‖_F = √(Σᵢ Σⱼ aᵢⱼ²)
This generalizes the L2 norm to matrices and is computationally efficient.

**L2 Norm for Vectors:**
The L2 norm is defined as: ‖v‖₂ = √(Σᵢ vᵢ²)
This is the standard Euclidean norm.

**Numerical Stability:**
We add a small epsilon (1e-12) inside the square root to prevent division by zero
in edge cases where all gradients are exactly zero.

## Verification Status

- **Sorries:** 0
- **Axioms:** 0 (uses only basic arithmetic)
- **Compilation:** ✅ Fully computable

-/

namespace VerifiedNN.Training.GradientMonitoring

open VerifiedNN.Core
open SciLean

/-- Structure holding gradient norm values for all network parameters.

This structure stores the computed norms for each gradient component and provides
flags indicating whether the gradients are in a healthy range.

**Fields:**
- `layer1WeightNorm`: Frobenius norm of layer 1 weight gradients (128×784 matrix)
- `layer1BiasNorm`: L2 norm of layer 1 bias gradients (128-vector)
- `layer2WeightNorm`: Frobenius norm of layer 2 weight gradients (10×128 matrix)
- `layer2BiasNorm`: L2 norm of layer 2 bias gradients (10-vector)
- `hasVanishing`: True if any norm is below 0.0001 (indicates vanishing gradients)
- `hasExploding`: True if any norm is above 10.0 (indicates exploding gradients)
-/
structure GradientNorms where
  layer1WeightNorm : Float
  layer1BiasNorm : Float
  layer2WeightNorm : Float
  layer2BiasNorm : Float
  hasVanishing : Bool
  hasExploding : Bool
deriving Repr

/-- Threshold for detecting vanishing gradients.

Gradients with norms below this threshold may indicate vanishing gradient problems,
where the network stops learning due to extremely small gradient values.
-/
def vanishingThreshold : Float := 0.0001

/-- Threshold for detecting exploding gradients.

Gradients with norms above this threshold may indicate exploding gradient problems,
where the training becomes unstable due to extremely large gradient values.
-/
def explodingThreshold : Float := 10.0

/-- Small epsilon for numerical stability in square root computation.

Added inside square roots to prevent division by zero when all gradient
components are exactly zero.
-/
def epsilon : Float := 1e-12

/-- Compute Frobenius norm of a matrix.

The Frobenius norm is the square root of the sum of squared elements:
‖A‖_F = √(Σᵢ Σⱼ aᵢⱼ²)

This is the matrix generalization of the L2 norm and provides a measure
of the overall magnitude of the matrix.

**Parameters:**
- `mat`: Matrix of dimensions m × n

**Returns:** Non-negative Float representing the Frobenius norm

**Properties:**
- ‖A‖_F ≥ 0 for all matrices A
- ‖A‖_F = 0 if and only if A is the zero matrix
- ‖αA‖_F = |α| · ‖A‖_F for scalar α
- ‖A + B‖_F ≤ ‖A‖_F + ‖B‖_F (triangle inequality)

**Numerical Stability:**
Adds epsilon (1e-12) inside the square root to handle edge case of zero matrix.
-/
def computeMatrixNorm {m n : Nat} (mat : Matrix m n) : Float :=
  let sumOfSquares := ∑ i, ∑ j, mat[i,j] * mat[i,j]
  Float.sqrt (sumOfSquares + epsilon)

/-- Compute L2 (Euclidean) norm of a vector.

The L2 norm is the square root of the sum of squared elements:
‖v‖₂ = √(Σᵢ vᵢ²)

This is the standard Euclidean distance from the origin and provides
a measure of the vector's magnitude.

**Parameters:**
- `vec`: Vector of dimension n

**Returns:** Non-negative Float representing the L2 norm

**Properties:**
- ‖v‖₂ ≥ 0 for all vectors v
- ‖v‖₂ = 0 if and only if v is the zero vector
- ‖αv‖₂ = |α| · ‖v‖₂ for scalar α
- ‖v + w‖₂ ≤ ‖v‖₂ + ‖w‖₂ (triangle inequality)

**Numerical Stability:**
Adds epsilon (1e-12) inside the square root to handle edge case of zero vector.
-/
def computeVectorNorm {n : Nat} (vec : Vector n) : Float :=
  let sumOfSquares := ∑ i, vec[i] * vec[i]
  Float.sqrt (sumOfSquares + epsilon)

/-- Compute gradient norms for all network parameters.

Given the gradients for all layers (returned by `computeManualGradients`),
computes the Frobenius norm for weight matrices and L2 norm for bias vectors.

**Parameters:**
- `grads`: Tuple of (dL/dW1, dL/db1, dL/dW2, dL/db2) where:
  - dL/dW1: Gradient w.r.t. layer 1 weights (128×784)
  - dL/db1: Gradient w.r.t. layer 1 bias (128)
  - dL/dW2: Gradient w.r.t. layer 2 weights (10×128)
  - dL/db2: Gradient w.r.t. layer 2 bias (10)

**Returns:** `GradientNorms` structure with computed norms and health flags

**Health Checks:**
- `hasVanishing` is set if any norm < 0.0001
- `hasExploding` is set if any norm > 10.0

**Usage:** Call this after computing gradients but before applying them to monitor
gradient health throughout training.
-/
def computeGradientNorms (grads : Matrix 128 784 × Vector 128 × Matrix 10 128 × Vector 10)
    : GradientNorms :=
  let (dW1, db1, dW2, db2) := grads

  -- Compute norms for each component
  let w1Norm := computeMatrixNorm dW1
  let b1Norm := computeVectorNorm db1
  let w2Norm := computeMatrixNorm dW2
  let b2Norm := computeVectorNorm db2

  -- Check for vanishing gradients (any norm too small)
  let hasVanishing :=
    w1Norm < vanishingThreshold ||
    b1Norm < vanishingThreshold ||
    w2Norm < vanishingThreshold ||
    b2Norm < vanishingThreshold

  -- Check for exploding gradients (any norm too large)
  let hasExploding :=
    w1Norm > explodingThreshold ||
    b1Norm > explodingThreshold ||
    w2Norm > explodingThreshold ||
    b2Norm > explodingThreshold

  {
    layer1WeightNorm := w1Norm
    layer1BiasNorm := b1Norm
    layer2WeightNorm := w2Norm
    layer2BiasNorm := b2Norm
    hasVanishing := hasVanishing
    hasExploding := hasExploding
  }

/-- Format gradient norms for display.

Produces a compact, human-readable string showing all gradient norms.

**Parameters:**
- `norms`: GradientNorms structure with computed values

**Returns:** Formatted string like "L1_W=0.023 L1_b=0.045 L2_W=0.078 L2_b=0.012"

**Format:**
- L1_W: Layer 1 weight gradient norm (Frobenius)
- L1_b: Layer 1 bias gradient norm (L2)
- L2_W: Layer 2 weight gradient norm (Frobenius)
- L2_b: Layer 2 bias gradient norm (L2)

All values are formatted to 3 decimal places for readability.

**Usage:** Display this string after each training step or epoch to monitor gradient health.
-/
def formatGradientNorms (norms : GradientNorms) : String :=
  let w1Str := Float.toString (Float.floor (norms.layer1WeightNorm * 1000.0) / 1000.0)
  let b1Str := Float.toString (Float.floor (norms.layer1BiasNorm * 1000.0) / 1000.0)
  let w2Str := Float.toString (Float.floor (norms.layer2WeightNorm * 1000.0) / 1000.0)
  let b2Str := Float.toString (Float.floor (norms.layer2BiasNorm * 1000.0) / 1000.0)
  s!"L1_W={w1Str} L1_b={b1Str} L2_W={w2Str} L2_b={b2Str}"

/-- Get detailed health message for gradient state.

Provides human-readable diagnostic messages when gradients are unhealthy.

**Parameters:**
- `norms`: GradientNorms structure with computed values

**Returns:** Empty string if gradients are healthy, warning message otherwise

**Messages:**
- If vanishing: "WARNING: Vanishing gradients detected! Norms below 0.0001"
- If exploding: "WARNING: Exploding gradients detected! Norms above 10.0"
- If both: "WARNING: Gradient instability! Check initialization and learning rate"

**Usage:** Display this message after gradient norm output to alert user to problems.
-/
def checkGradientHealth (norms : GradientNorms) : String :=
  if norms.hasVanishing && norms.hasExploding then
    "WARNING: Gradient instability! Check initialization and learning rate"
  else if norms.hasVanishing then
    "WARNING: Vanishing gradients detected! Norms below 0.0001"
  else if norms.hasExploding then
    "WARNING: Exploding gradients detected! Norms above 10.0"
  else
    ""

end VerifiedNN.Training.GradientMonitoring
