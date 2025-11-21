import VerifiedNN.Core.DataTypes
import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Gradient
import VerifiedNN.Loss.CrossEntropy
import SciLean

/-!
# Finite Difference Gradient Checking

Validates analytical gradient implementations by comparing them to
numerical approximations using central finite differences.

This module provides infrastructure to validate the **full network gradient**
(all 101,770 parameters) against numerical approximations. Unlike the general
gradient checking in `GradientCheck.lean`, this is specifically designed for
testing the manual backpropagation implementation once it's ready.

## Central Difference Method

For parameter θ[i] and loss function L(θ):

```
∂L/∂θ[i] ≈ (L(θ + ε·e_i) - L(θ - ε·e_i)) / (2ε)
```

where:
- `e_i` is the i-th unit vector (1 at position i, 0 elsewhere)
- `ε` is a small perturbation (typically 1e-4 to 1e-5)
- This is "central difference" with O(h²) accuracy vs O(h) for forward difference

## Mathematical Accuracy

**Expected gradient matching:**
- **Excellent:** max_error < 1e-6 (symbolic correctness preserved in Float)
- **Good:** max_error < 1e-4 (typical for Float arithmetic)
- **Acceptable:** max_error < 1e-3 (minor numerical differences)
- **Bad:** max_error > 1e-3 (likely implementation bug)

**Common error sources:**
- 1e-6: Excellent - manual gradient matches theory perfectly
- 1e-4: Good - expected floating-point rounding
- 1e-2: Bad - indicates implementation bug in backpropagation

## Performance Warning

Numerical gradient computation is **extremely slow**: O(n) loss evaluations
where n = number of parameters.

For the full network (101,770 parameters):
- Single sample check: ~30-60 seconds on modern CPU
- Batch check: Impractical (use sampling strategy)

**Strategy:** Test on random subsets of parameters, not the full 101,770.

## Usage

```lean
-- Test manual gradient implementation against numerical approximation
def validateNetworkGradient : IO Unit := do
  let (passed, failed, details) := ← runGradientTests networkGradientManual 5
  IO.println s!"Gradient Tests: {passed} passed, {failed} failed"
  IO.println details
  if failed > 0 then
    throw (IO.userError "Gradient check failed!")
```

## Verification Status

- **Purpose:** Testing/validation infrastructure (not used in training)
- **Computable:** Yes (all arithmetic operations are computable)
- **Axioms:** None (relies on Network.Gradient axioms for unflatten/flatten)
- **Sorries:** 0

## Key Functions

- `numericalGradient`: Compute finite difference approximation for all parameters
- `compareGradients`: Compare analytical vs numerical with error statistics
- `checkGradientSingleSample`: Fast single-sample validation
- `runGradientTests`: Run multiple random test cases with reporting

## References

- Finite difference methods: Numerical Analysis (Burden & Faires)
- Central differences: More accurate than forward/backward differences
- Gradient checking: Standard practice in deep learning (Goodfellow et al.)
-/

namespace VerifiedNN.Testing.FiniteDifference

open VerifiedNN.Core
open VerifiedNN.Network
open VerifiedNN.Network.Gradient
open VerifiedNN.Loss
open SciLean

set_default_scalar Float

/-- Compute numerical gradient using central finite differences.

For each parameter θ[i], approximates ∂f/∂θ[i] as:
```
(f(θ + ε·e_i) - f(θ - ε·e_i)) / (2ε)
```

This is the gold-standard numerical gradient approximation with O(ε²) accuracy.

**Parameters:**
- `lossFunc`: Function computing scalar loss from parameters
- `params`: Current parameter vector
- `epsilon`: Perturbation size (default 1e-4)

**Returns:** Numerical gradient vector

**Performance:** O(nParams) loss evaluations. For 101,770 parameters,
this computes the loss 203,540 times. Expect 30-60 seconds per call.

**Numerical Stability:**
- Too small ε (< 1e-8): Subtractive cancellation errors dominate
- Too large ε (> 1e-3): Truncation errors dominate
- Optimal ε ≈ 1e-4 to 1e-5 for Float arithmetic
-/
def numericalGradient {nParams : Nat}
  (lossFunc : Vector nParams → Float)
  (params : Vector nParams)
  (epsilon : Float := 1e-4)
  : Vector nParams :=
  ⊞ (i : Idx nParams) =>
    -- Create e_i (unit vector at position i)
    let params_plus := ⊞ (j : Idx nParams) =>
      if i == j then params[j] + epsilon else params[j]
    let params_minus := ⊞ (j : Idx nParams) =>
      if i == j then params[j] - epsilon else params[j]

    -- Central difference: [f(θ + ε·e_i) - f(θ - ε·e_i)] / (2ε)
    let loss_plus := lossFunc params_plus
    let loss_minus := lossFunc params_minus
    (loss_plus - loss_minus) / (2.0 * epsilon)

/-- Compare two gradient vectors element-wise with detailed statistics.

Computes error metrics to diagnose gradient mismatches.

**Parameters:**
- `analytical`: Gradient from backpropagation or SciLean AD
- `numerical`: Gradient from finite differences
- `tolerance`: Absolute error threshold (default 1e-4)

**Returns:** Tuple of (max_error, avg_error, passes_test)

**Error Metrics:**
- `max_error`: Worst-case absolute difference |analytical[i] - numerical[i]|
- `avg_error`: Average absolute difference across all parameters
- `passes_test`: True if max_error < tolerance

**Usage:** Use to diagnose why a gradient check failed.
-/
def compareGradients {nParams : Nat}
  (analytical : Vector nParams)
  (numerical : Vector nParams)
  (tolerance : Float := 1e-4)
  : (Float × Float × Bool) :=
  -- Compute absolute errors for each parameter
  let errors := ⊞ (i : Idx nParams) =>
    Float.abs (analytical[i] - numerical[i])

  -- Find maximum error (using fold since we don't have array max primitive)
  let max_error := Id.run do
    let mut maxVal := 0.0
    for i in [:nParams] do
      if h : i < nParams then
        let err := errors[⟨i, h⟩]
        if err > maxVal then
          maxVal := err
    return maxVal

  -- Compute average error
  let sum_errors := ∑ (i : Idx nParams), errors[i]
  let avg_error := sum_errors / nParams.toFloat

  -- Test passes if max error within tolerance
  let passes := max_error < tolerance

  (max_error, avg_error, passes)

/-- Check gradient for a single training sample (faster validation).

Instead of checking all 101,770 parameters against numerical gradient,
this provides a quick sanity check on a single example. Useful during
development to catch major bugs before running expensive full validation.

**Parameters:**
- `computeGrad`: Manual gradient function to test (signature: params → input → target → gradient)
- `params`: Network parameters (101,770 elements)
- `input`: Single MNIST image (784 elements)
- `target`: True class label (0-9)
- `epsilon`: Finite difference step size (default 1e-4)
- `tolerance`: Error threshold (default 1e-4)

**Returns:** IO tuple of (max_error, passes_test)

**Note:** This is **still slow** (30-60 seconds) since it computes
the loss 203,540 times. Use sparingly during development.
-/
def checkGradientSingleSample
  (computeGrad : Vector nParams → Vector 784 → Nat → Vector nParams)
  (params : Vector nParams)
  (input : Vector 784)
  (target : Nat)
  (epsilon : Float := 1e-4)
  (tolerance : Float := 1e-4)
  : IO (Float × Bool) := do
  -- Compute analytical gradient using provided function
  let analytical := computeGrad params input target

  -- Compute numerical gradient using finite differences
  let lossFunc := fun p =>
    let net := unflattenParams p
    let output := net.forward input
    crossEntropyLoss output target
  let numerical := numericalGradient lossFunc params epsilon

  -- Compare gradients
  let (max_err, _, passes) := compareGradients analytical numerical tolerance

  return (max_err, passes)

/-- Run gradient checks on multiple random test samples.

Provides statistical confidence that the gradient implementation is correct
by testing on several randomly generated inputs.

**Parameters:**
- `computeGrad`: Manual gradient function to test
- `numSamples`: Number of random test cases (default 5)
- `epsilon`: Finite difference step size (default 1e-4)
- `tolerance`: Error threshold (default 1e-4)

**Returns:** IO tuple of (num_passed, num_failed, details_string)

**Test Strategy:**
Each test uses:
- Random-ish input (pseudo-random based on iteration index)
- Random target class
- Small random initial parameters (near zero for stability)

**Time Estimate:** Each sample takes ~30-60 seconds, so 5 samples = 3-5 minutes.

**Example Output:**
```
✓ Sample 0: max_error = 3.2e-7
✓ Sample 1: max_error = 1.8e-6
✗ Sample 2: max_error = 0.012 (FAILED)
...
```
-/
def runGradientTests
  (computeGrad : Vector nParams → Vector 784 → Nat → Vector nParams)
  (numSamples : Nat := 5)
  (epsilon : Float := 1e-4)
  (tolerance : Float := 1e-4)
  : IO (Nat × Nat × String) := do
  let mut passed := 0
  let mut failed := 0
  let mut details := ""

  IO.println s!"Running {numSamples} gradient check samples..."
  IO.println s!"  (Each sample checks {nParams} parameters)"
  IO.println s!"  (Expected time: ~{numSamples * 45} seconds total)"
  IO.println ""

  for i in [0:numSamples] do
    -- Create pseudo-random input (deterministic for reproducibility)
    let input : Vector 784 := ⊞ (j : Idx 784) =>
      let idx := (i * 100 + j.1.toNat) % 255
      idx.toFloat / 255.0  -- Normalized [0, 1] like MNIST

    -- Random target class
    let target := i % 10

    -- Initialize small random parameters (near zero for numerical stability)
    let params : Vector nParams := ⊞ (k : Idx nParams) =>
      let idx := (i + k.1.toNat) % 100
      (idx.toFloat / 100.0) * 0.01  -- Small values in [-0.01, 0.01]

    IO.println s!"Sample {i}/{numSamples}: Checking gradient..."

    -- Run gradient check
    let (max_err, passes) := ← checkGradientSingleSample
      computeGrad params input target epsilon tolerance

    if passes then
      passed := passed + 1
      details := details ++ s!"✓ Sample {i}: max_error = {max_err}\n"
      IO.println s!"  ✓ PASSED (max_error = {max_err})"
    else
      failed := failed + 1
      details := details ++ s!"✗ Sample {i}: max_error = {max_err} (FAILED - exceeds tolerance {tolerance})\n"
      IO.println s!"  ✗ FAILED (max_error = {max_err}, tolerance = {tolerance})"

    IO.println ""

  return (passed, failed, details)

/-- Subsample gradient check: validate only a random subset of parameters.

For quick validation during development, check only N randomly selected
parameters instead of all 101,770. This reduces runtime from minutes to seconds.

**Parameters:**
- `computeGrad`: Gradient function to test
- `params`: Full parameter vector
- `input`: Training input
- `target`: True label
- `numParams`: Number of parameters to check (default 100)
- `epsilon`: Finite difference step size (default 1e-4)
- `tolerance`: Error threshold (default 1e-4)

**Returns:** IO tuple of (max_error, passes_test)

**Time Estimate:** 100 parameters → ~0.5 seconds (vs 60 seconds for all params)

**Use Case:** Rapid iteration during backpropagation implementation.
Not a substitute for full validation, but catches most bugs much faster.
-/
def checkGradientSubsample
  (computeGrad : Vector nParams → Vector 784 → Nat → Vector nParams)
  (params : Vector nParams)
  (input : Vector 784)
  (target : Nat)
  (numParamsToCheck : Nat := 100)
  (epsilon : Float := 1e-4)
  (tolerance : Float := 1e-4)
  : IO (Float × Bool) := do
  -- Compute analytical gradient (full)
  let analytical := computeGrad params input target

  -- Define loss function
  let lossFunc := fun p =>
    let net := unflattenParams p
    let output := net.forward input
    crossEntropyLoss output target

  -- Check only a subset of parameters
  let mut max_error := 0.0
  let mut all_passed := true

  for k in [0:numParamsToCheck] do
    -- Select evenly spaced parameter indices
    let paramIdx := (k * nParams) / numParamsToCheck
    if h : paramIdx < nParams then
      -- Convert Nat to Idx using finEquiv
      let i : Idx nParams := (Idx.finEquiv nParams).invFun ⟨paramIdx, h⟩

      -- Compute numerical gradient for this single parameter
      let params_plus := ⊞ (j : Idx nParams) =>
        if i == j then params[j] + epsilon else params[j]
      let params_minus := ⊞ (j : Idx nParams) =>
        if i == j then params[j] - epsilon else params[j]

      let loss_plus := lossFunc params_plus
      let loss_minus := lossFunc params_minus
      let numerical_i := (loss_plus - loss_minus) / (2.0 * epsilon)

      -- Compare to analytical
      let error := Float.abs (analytical[i] - numerical_i)
      if error > max_error then
        max_error := error
      if error >= tolerance then
        all_passed := false

  return (max_error, all_passed)

/-- Example: Test networkGradient from SciLean AD (sanity check).

This tests the SciLean automatic differentiation gradient (once it's
computable) to verify our numerical gradient infrastructure is working.

**Note:** Currently `networkGradient` is noncomputable, so this example
won't run until we have an executable gradient implementation.
-/
def exampleTestADGradient : IO Unit := do
  IO.println "==================================="
  IO.println "AD Gradient Validation Example"
  IO.println "==================================="
  IO.println ""
  IO.println "This example would test SciLean's automatic differentiation"
  IO.println "gradient against finite differences once networkGradient is"
  IO.println "made computable via rewrite_by patterns."
  IO.println ""
  IO.println "Expected workflow:"
  IO.println "  1. Compute gradient via AD: ∇ (fun p => computeLoss p input target)"
  IO.println "  2. Compute gradient via finite diff: numericalGradient lossFunc params"
  IO.println "  3. Compare: max_error should be < 1e-4"
  IO.println ""
  IO.println "Status: Waiting for networkGradient to become computable"

/-- Example test executable showing how to use this module.

Once manual backpropagation is implemented (Phase 2B), this will validate
it against numerical gradients.

**Current Status:** Infrastructure ready, waiting for manual gradient implementation.
-/
unsafe def main : IO Unit := do
  IO.println "==================================="
  IO.println "Finite Difference Gradient Checker"
  IO.println "==================================="
  IO.println ""
  IO.println s!"Network Parameters: {nParams} (101,770)"
  IO.println "Testing Strategy: Central finite differences"
  IO.println "Accuracy: O(ε²) for ε = 1e-4"
  IO.println ""

  IO.println "Purpose:"
  IO.println "  This module validates manual backpropagation implementations"
  IO.println "  by comparing analytical gradients to numerical approximations."
  IO.println ""

  IO.println "Performance Characteristics:"
  IO.println s!"  - Full gradient check (all {nParams} params): ~30-60 seconds"
  IO.println "  - Subsample check (100 params): ~0.5 seconds"
  IO.println "  - Multi-sample validation (5 samples): ~3-5 minutes"
  IO.println ""

  IO.println "Usage:"
  IO.println "  1. Implement manual backpropagation: networkGradientManual"
  IO.println "  2. Call: runGradientTests networkGradientManual 5"
  IO.println "  3. Check: All samples should pass with max_error < 1e-4"
  IO.println ""

  IO.println "Expected Accuracy:"
  IO.println "  ✓ Excellent:   max_error < 1e-6"
  IO.println "  ✓ Good:        max_error < 1e-4"
  IO.println "  ~ Acceptable:  max_error < 1e-3"
  IO.println "  ✗ Bad:         max_error > 1e-3 (likely bug)"
  IO.println ""

  IO.println "Current Status:"
  IO.println "  ✓ Finite difference infrastructure ready"
  IO.println "  ✓ Comparison functions implemented"
  IO.println "  ✓ Test harness prepared"
  IO.println "  ⏳ Waiting for manual gradient implementation (Phase 2B)"
  IO.println ""

  IO.println "Next Steps:"
  IO.println "  1. Implement networkGradientManual in Network/GradientManual.lean"
  IO.println "  2. Import and test: runGradientTests networkGradientManual 5"
  IO.println "  3. Debug any failures using compareGradients for detailed errors"
  IO.println ""

  IO.println "==================================="
  IO.println "✓ Gradient checking infrastructure ready!"
  IO.println "==================================="

end VerifiedNN.Testing.FiniteDifference

-- Note: To run this test, use: lake env lean --run VerifiedNN/Testing/FiniteDifference.lean
unsafe def main : IO Unit := VerifiedNN.Testing.FiniteDifference.main
