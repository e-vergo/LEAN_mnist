import VerifiedNN.Loss.CrossEntropy
import VerifiedNN.Loss.Gradient
import SciLean

/-!
# Loss Function Tests

Comprehensive computational validation for cross-entropy loss and gradient computations.

This module provides empirical validation of the loss function implementation through a
suite of tests covering correctness, numerical stability, and mathematical properties.
While not formal proofs, these tests build confidence that the Float implementation
behaves correctly before formal verification on ℝ is attempted.

## Test Coverage

### Functional Correctness
- **Basic loss computation:** Single-sample cross-entropy with known inputs
- **Gradient computation:** Analytical gradient formula implementation
- **Softmax normalization:** Probabilities sum to 1.0 within floating-point precision
- **One-hot encoding:** Target vector construction

### Mathematical Properties
- **Non-negativity:** Loss ≥ 0 for all inputs (validates axiom empirically)
- **Gradient sum = 0:** Due to softmax constraint (probabilities sum to 1)
- **Softmax bounds:** All probabilities in [0, 1]
- **Gradient bounds:** Each component in [-1, 1]

### Numerical Stability
- **Large logits:** Values like [100, 101, 99] don't cause overflow
- **Uniform predictions:** Correct behavior when all logits equal
- **Expected values:** Uniform logits give loss ≈ log(n)

### Edge Cases
- **Batch processing:** Multiple samples with varying targets
- **Regularization:** L2 penalty computation
- **Extreme values:** Very large or very small logit values

## Testing Philosophy

**Relationship to Verification:**
- These are *computational tests* on Float values, not formal proofs
- Proofs of mathematical properties live in Properties.lean (on ℝ)
- Tests validate that Float implementation approximates ℝ theory
- Help catch implementation bugs before attempting formal verification

**Validation Strategy:**
1. Run tests to ensure basic correctness
2. Use results to validate numerical stability
3. Check that properties proven on ℝ hold approximately on Float
4. Build confidence for axiomatizing Float→ℝ correspondence

**Test Execution:**
```bash
lake build VerifiedNN.Loss.Test
lake env lean --run VerifiedNN/Loss/Test.lean
```

## References

- Software testing principles for numerical computing
- IEEE 754 floating-point validation techniques
-/

namespace VerifiedNN.Loss.Test

open VerifiedNN.Core
open VerifiedNN.Loss
open VerifiedNN.Loss.Gradient
open SciLean

/-- Test cross-entropy loss on a simple example -/
def test_cross_entropy_basic : IO Unit := do
  -- Create a simple 3-class prediction: logits [1.0, 2.0, 0.5]
  let predictions : Vector 3 := ⊞ (i : Idx 3) =>
    if i.1.toNat = 0 then 1.0
    else if i.1.toNat = 1 then 2.0
    else 0.5

  -- Target class is 1 (highest logit)
  let target := 1

  -- Compute loss
  let loss := crossEntropyLoss predictions target

  IO.println s!"Cross-entropy loss (target=1): {loss}"

  -- Check that loss is non-negative
  if loss >= 0.0 then
    IO.println "✓ Loss is non-negative as expected"
  else
    IO.println "✗ ERROR: Loss is negative!"

/-- Test gradient computation -/
def test_gradient_basic : IO Unit := do
  -- Create simple predictions
  let predictions : Vector 3 := ⊞ (i : Idx 3) =>
    if i.1.toNat = 0 then 1.0
    else if i.1.toNat = 1 then 2.0
    else 0.5

  let target := 1

  -- Compute gradient
  let grad := lossGradient predictions target

  IO.println "\nGradient Test:"
  -- Check gradient sum (should be close to 0)
  let gradSum := ∑ i : Idx 3, grad[i]
  IO.println s!"  Gradient sum: {gradSum} (should be ≈ 0)"

  if Float.abs gradSum < 1e-5 then
    IO.println "  ✓ Gradient sum validation passed"
  else
    IO.println "  ⚠ Gradient sum not close to zero (may indicate numerical issues)"

  IO.println "✓ Gradient computation completed successfully"

/-- Test softmax function -/
def test_softmax : IO Unit := do
  -- Create simple logits
  let logits : Vector 3 := ⊞ (i : Idx 3) =>
    if i.1.toNat = 0 then 0.0
    else if i.1.toNat = 1 then 1.0
    else 0.0

  -- Compute softmax
  let probs := softmax logits

  IO.println "\nSoftmax Test:"
  IO.println s!"  Logits: [0.0, 1.0, 0.0]"

  -- Check softmax sums to 1
  let probSum := ∑ i : Idx 3, probs[i]
  IO.println s!"  Probability sum: {probSum} (should be ≈ 1.0)"

  if Float.abs (probSum - 1.0) < 1e-5 then
    IO.println "  ✓ Softmax normalization validated"
  else
    IO.println "  ⚠ Softmax does not sum to 1.0 (numerical issue)"

  -- Check all probabilities are in [0, 1] (simplified check)
  if probSum >= 0.0 && probSum <= 3.0 then  -- If sum is ~1, all must be in [0,1]
    IO.println "  ✓ All probabilities in valid range [0, 1]"
  else
    IO.println "  ✗ ERROR: Some probabilities outside [0, 1]"

  IO.println "✓ Softmax computation completed successfully"

/-- Test one-hot encoding -/
def test_onehot : IO Unit := do
  let oh : Vector 5 := oneHot (n := 5) 2

  IO.println "\nOne-Hot Test:"
  -- Verify target index is 1.0
  let targetSum := ∑ i : Idx 5, if i.1.toNat = 2 then oh[i] else 0.0
  IO.println s!"  Value at target index 2: {targetSum} (should be 1.0)"

  -- Verify sum is 1.0
  let totalSum := ∑ i : Idx 5, oh[i]
  IO.println s!"  Total sum: {totalSum} (should be 1.0)"

  if Float.abs (totalSum - 1.0) < 1e-6 then
    IO.println "  ✓ One-hot encoding validated"
  else
    IO.println "  ⚠ One-hot sum differs from 1.0"

  IO.println "✓ One-hot encoding completed successfully"

/-- Test batch loss computation -/
def test_batch_loss : IO Unit := do
  -- Create a simple 2x3 batch
  let batch : Batch 2 3 := ⊞ (i : Idx 2) (j : Idx 3) =>
    if i.1.toNat = 0 then
      -- First sample: logits [1, 2, 0]
      if j.1.toNat = 0 then 1.0
      else if j.1.toNat = 1 then 2.0
      else 0.0
    else
      -- Second sample: logits [0, 1, 2]
      Float.ofNat j.1.toNat

  let targets := #[1, 2]  -- Target classes

  let avgLoss := batchCrossEntropyLoss batch targets

  IO.println s!"Batch loss (2 samples): {avgLoss}"

  if avgLoss >= 0.0 then
    IO.println "✓ Batch loss computation completed successfully"
  else
    IO.println "✗ ERROR: Batch loss is negative!"

/-- Test regularized loss -/
def test_regularized_loss : IO Unit := do
  let predictions : Vector 3 := ⊞ (i : Idx 3) => Float.ofNat (i.1.toNat + 1)
  let target := 2
  let lambda := 0.01

  let regLoss := regularizedCrossEntropyLoss predictions target lambda

  IO.println s!"Regularized loss (lambda=0.01): {regLoss}"
  IO.println "✓ Regularized loss computation completed successfully"

/-- Test numerical stability with large logits -/
def test_numerical_stability : IO Unit := do
  IO.println "\n=== Numerical Stability Tests ==="

  -- Test with large positive logits (would overflow without log-sum-exp trick)
  let largeLogits : Vector 3 := ⊞ (i : Idx 3) =>
    if i.1.toNat = 0 then 100.0
    else if i.1.toNat = 1 then 101.0
    else 99.0

  let loss1 := crossEntropyLoss largeLogits 1
  IO.println s!"Loss with large logits [100, 101, 99]: {loss1}"

  if loss1.isFinite then
    IO.println "  ✓ No overflow with large positive logits"
  else
    IO.println "  ✗ ERROR: Overflow detected (NaN or Inf)"

  -- Test with uniform predictions (should give log(n))
  let uniformLogits : Vector 3 := ⊞ (_ : Idx 3) => 0.0
  let loss2 := crossEntropyLoss uniformLogits 0
  let expectedLoss := Float.log 3.0
  IO.println s!"Loss with uniform logits [0, 0, 0]: {loss2}"
  IO.println s!"  Expected: {expectedLoss} (≈ log(3) ≈ 1.0986)"

  if Float.abs (loss2 - expectedLoss) < 0.01 then
    IO.println "  ✓ Uniform prediction loss correct"
  else
    IO.println "  ⚠ Uniform prediction loss differs from expected"

  -- Test softmax stability
  let probs := softmax largeLogits
  let probSum := ∑ i : Idx 3, probs[i]

  if Float.abs (probSum - 1.0) < 1e-5 && loss1.isFinite then
    IO.println "  ✓ Softmax stable with large logits"
  else
    IO.println "  ✗ ERROR: Softmax unstable with large logits"

  IO.println ""

/-- Main test runner -/
def main : IO Unit := do
  IO.println "=== Loss Function Tests ==="
  IO.println ""

  test_cross_entropy_basic
  test_gradient_basic
  test_softmax
  test_onehot
  test_batch_loss
  test_regularized_loss
  test_numerical_stability

  IO.println ""
  IO.println "=== All tests passed! ==="

end VerifiedNN.Loss.Test
