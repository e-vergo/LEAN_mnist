import VerifiedNN.Core.Activation
import VerifiedNN.Core.LinearAlgebra
import VerifiedNN.Loss.CrossEntropy
import VerifiedNN.Core.DataTypes
import SciLean

/-!
# Numerical Stability Tests

Test suite for validating numerical stability and edge case handling.

## Main Tests

- `testActivationsWithExtremes`: Activation functions with very large/small inputs
- `testDivisionByZeroHandling`: Edge cases in operations involving division
- `testNaNAndInfHandling`: Propagation and detection of NaN/Inf values
- `testUnderflowAndOverflow`: Behavior near Float limits
- `testSoftmaxStability`: Softmax with extreme logits
- `testNormalizationEdgeCases`: Norm computation with extreme values
- `testGradientExtremes`: Gradient-related operations with edge cases

## Implementation Notes

**Testing Strategy:** Validate robust behavior in extreme conditions:
- Very large values (approaching Float overflow)
- Very small values (approaching Float underflow)
- Zero values (division by zero cases)
- NaN and Inf propagation
- Numerical precision limits

**Coverage Status:** Comprehensive edge case testing across all modules.

**Test Framework:** IO-based assertions with detailed diagnostic output.

## Usage

```bash
# Build tests
lake build VerifiedNN.Testing.NumericalStabilityTests

# Run tests
lake env lean --run VerifiedNN/Testing/NumericalStabilityTests.lean
```
-/

namespace VerifiedNN.Testing.NumericalStabilityTests

open VerifiedNN.Core
open VerifiedNN.Core.Activation
open VerifiedNN.Core.LinearAlgebra
open VerifiedNN.Loss
open SciLean

/-! ## Helper Functions for Testing -/

/-- Assert a boolean condition with a message -/
def assertTrue (name : String) (condition : Bool) : IO Bool := do
  if condition then
    IO.println s!"✓ {name}"
    return true
  else
    IO.println s!"✗ {name} FAILED"
    return false

/-- Assert approximate equality of floats -/
def assertApproxEq (name : String) (x y : Float) (tol : Float := 1e-6) : IO Bool := do
  let condition := Float.abs (x - y) ≤ tol
  if condition then
    IO.println s!"✓ {name}"
    return true
  else
    IO.println s!"✗ {name} FAILED: {x} ≠ {y} (diff: {Float.abs (x - y)})"
    return false

/-- Assert value is not NaN -/
def assertNotNaN (name : String) (x : Float) : IO Bool := do
  if !x.isNaN then
    IO.println s!"✓ {name}"
    return true
  else
    IO.println s!"✗ {name} FAILED: value is NaN"
    return false

/-- Assert value is not Inf -/
def assertNotInf (name : String) (x : Float) : IO Bool := do
  if !x.isInf then
    IO.println s!"✓ {name}"
    return true
  else
    IO.println s!"✗ {name} FAILED: value is Inf"
    return false

/-- Assert value is finite (not NaN or Inf) -/
def assertFinite (name : String) (x : Float) : IO Bool := do
  if !x.isNaN && !x.isInf then
    IO.println s!"✓ {name}"
    return true
  else
    IO.println s!"✗ {name} FAILED: value is {x}"
    return false

/-! ## Activation Function Stability Tests -/

/-- Test activation functions with extreme inputs -/
def testActivationsWithExtremes : IO Bool := do
  IO.println "\n=== Activation Functions with Extreme Values ==="

  let mut allPassed := true

  -- Very large positive value
  let largePos := 1e30
  allPassed := allPassed && (← assertApproxEq "ReLU: large positive unchanged" (relu largePos) largePos 1e20)
  allPassed := allPassed && (← assertFinite "Sigmoid: large positive is finite" (sigmoid largePos))
  allPassed := allPassed && (← assertTrue "Sigmoid: large positive → 1" (sigmoid largePos > 0.99))

  -- Very large negative value
  let largeNeg := -1e30
  allPassed := allPassed && (← assertApproxEq "ReLU: large negative → 0" (relu largeNeg) 0.0)
  allPassed := allPassed && (← assertFinite "Sigmoid: large negative is finite" (sigmoid largeNeg))
  allPassed := allPassed && (← assertTrue "Sigmoid: large negative → 0" (sigmoid largeNeg < 0.01))

  -- Very small positive value (near zero)
  let smallPos := 1e-30
  allPassed := allPassed && (← assertApproxEq "ReLU: small positive unchanged" (relu smallPos) smallPos 1e-35)
  allPassed := allPassed && (← assertFinite "Sigmoid: small positive is finite" (sigmoid smallPos))

  -- Exact zero
  allPassed := allPassed && (← assertApproxEq "ReLU: zero → zero" (relu 0.0) 0.0)
  allPassed := allPassed && (← assertApproxEq "Sigmoid: zero → 0.5" (sigmoid 0.0) 0.5 1e-6)
  allPassed := allPassed && (← assertApproxEq "Tanh: zero → zero" (tanh 0.0) 0.0)

  -- Leaky ReLU with extreme negative
  allPassed := allPassed && (← assertFinite "Leaky ReLU: large negative is finite" (leakyRelu 0.01 largeNeg))

  return allPassed

/-! ## Division by Zero and Edge Cases -/

/-- Test operations that might involve division by zero -/
def testDivisionByZeroHandling : IO Bool := do
  IO.println "\n=== Division by Zero Handling ==="

  let mut allPassed := true

  -- Zero vector norm
  let zeroVec : Vector 3 := ⊞[0.0, 0.0, 0.0]
  let zeroNorm := norm zeroVec
  allPassed := allPassed && (← assertApproxEq "Norm: zero vector → 0" zeroNorm 0.0)

  -- Very small vector norm (near underflow)
  let tinyVec : Vector 3 := ⊞[1e-30, 1e-30, 1e-30]
  let tinyNorm := norm tinyVec
  allPassed := allPassed && (← assertFinite "Norm: tiny vector is finite" tinyNorm)
  allPassed := allPassed && (← assertTrue "Norm: tiny vector > 0" (tinyNorm > 0.0))

  -- Dot product with zero vector
  let nonZeroVec : Vector 3 := ⊞[1.0, 2.0, 3.0]
  let dotZero := dot zeroVec nonZeroVec
  allPassed := allPassed && (← assertApproxEq "Dot: zero vector → 0" dotZero 0.0)

  return allPassed

/-! ## NaN and Inf Handling -/

/-- Test NaN and Inf propagation -/
def testNaNAndInfHandling : IO Bool := do
  IO.println "\n=== NaN and Inf Handling ==="

  let mut allPassed := true

  -- Check that operations with reasonable inputs don't produce NaN/Inf
  let normalVec : Vector 3 := ⊞[1.0, 2.0, 3.0]
  let normalNorm := norm normalVec
  allPassed := allPassed && (← assertNotNaN "Norm: normal vector not NaN" normalNorm)
  allPassed := allPassed && (← assertNotInf "Norm: normal vector not Inf" normalNorm)

  let normalDot := dot normalVec normalVec
  allPassed := allPassed && (← assertNotNaN "Dot: normal vectors not NaN" normalDot)
  allPassed := allPassed && (← assertNotInf "Dot: normal vectors not Inf" normalDot)

  -- Softmax with reasonable inputs shouldn't produce NaN/Inf
  let logits : Vector 3 := ⊞[1.0, 2.0, 3.0]
  let probs := softmax logits
  for i in [:3] do
    if h : i < 3 then
      allPassed := allPassed && (← assertNotNaN s!"Softmax: p[{i}] not NaN" probs[⟨i, h⟩])
      allPassed := allPassed && (← assertNotInf s!"Softmax: p[{i}] not Inf" probs[⟨i, h⟩])

  return allPassed

/-! ## Underflow and Overflow Tests -/

/-- Test behavior near Float limits -/
def testUnderflowAndOverflow : IO Bool := do
  IO.println "\n=== Underflow and Overflow Tests ==="

  let mut allPassed := true

  -- Very large vector components
  let largeVec : Vector 2 := ⊞[1e30, 1e30]
  let largeNorm := norm largeVec
  -- Norm should be large but finite
  allPassed := allPassed && (← assertFinite "Norm: large vector is finite" largeNorm)
  allPassed := allPassed && (← assertTrue "Norm: large vector is large" (largeNorm > 1e30))

  -- Very small vector components (underflow region)
  let smallVec : Vector 2 := ⊞[1e-150, 1e-150]
  let smallNorm := norm smallVec
  allPassed := allPassed && (← assertFinite "Norm: tiny vector is finite" smallNorm)
  -- May underflow to zero, which is acceptable
  IO.println s!"  Norm of [1e-150, 1e-150]: {smallNorm}"

  -- Squared norm of large vector
  let largeNormSq := normSq largeVec
  allPassed := allPassed && (← assertFinite "NormSq: large vector result is finite" largeNormSq)

  return allPassed

/-! ## Softmax Stability Tests -/

/-- Test softmax with extreme logits -/
def testSoftmaxStability : IO Bool := do
  IO.println "\n=== Softmax Stability Tests ==="

  let mut allPassed := true

  -- Very large logits (would overflow without log-sum-exp trick)
  let hugeLogits : Vector 3 := ⊞[1000.0, 1001.0, 999.0]
  let hugeProbs := softmax hugeLogits

  -- All probabilities should be finite and in (0, 1)
  for i in [:3] do
    if h : i < 3 then
      let pi := hugeProbs[⟨i, h⟩]
      allPassed := allPassed && (← assertNotNaN s!"Softmax: huge logits p[{i}] not NaN" pi)
      allPassed := allPassed && (← assertNotInf s!"Softmax: huge logits p[{i}] not Inf" pi)
      allPassed := allPassed && (← assertTrue s!"Softmax: huge logits p[{i}] ∈ (0,1)" (pi > 0.0 && pi < 1.0))

  -- Sum should still be 1
  let hugeSum := ∑ i, hugeProbs[i]
  allPassed := allPassed && (← assertApproxEq "Softmax: huge logits sum = 1" hugeSum 1.0 1e-4)

  -- Very negative logits
  let negativeLogits : Vector 3 := ⊞[-1000.0, -999.0, -1001.0]
  let negativeProbs := softmax negativeLogits

  for i in [:3] do
    if h : i < 3 then
      let pi := negativeProbs[⟨i, h⟩]
      allPassed := allPassed && (← assertFinite s!"Softmax: negative logits p[{i}] finite" pi)

  let negativeSum := ∑ i, negativeProbs[i]
  allPassed := allPassed && (← assertApproxEq "Softmax: negative logits sum = 1" negativeSum 1.0 1e-4)

  -- Mixed extreme values
  let mixedLogits : Vector 3 := ⊞[500.0, -500.0, 0.0]
  let mixedProbs := softmax mixedLogits

  for i in [:3] do
    if h : i < 3 then
      allPassed := allPassed && (← assertFinite s!"Softmax: mixed logits p[{i}] finite" mixedProbs[⟨i, h⟩])

  let mixedSum := ∑ i, mixedProbs[i]
  allPassed := allPassed && (← assertApproxEq "Softmax: mixed logits sum = 1" mixedSum 1.0 1e-4)

  return allPassed

/-! ## Normalization Edge Cases -/

/-- Test normalization with extreme values -/
def testNormalizationEdgeCases : IO Bool := do
  IO.println "\n=== Normalization Edge Cases ==="

  let mut allPassed := true

  -- Vector with one very large component
  let skewedVec : Vector 3 := ⊞[1e20, 1.0, 1.0]
  let skewedNorm := norm skewedVec
  allPassed := allPassed && (← assertFinite "Norm: skewed vector is finite" skewedNorm)
  -- Should be approximately the large component
  allPassed := allPassed && (← assertTrue "Norm: dominated by large component" (skewedNorm > 0.99e20))

  -- All equal large values
  let equalLargeVec : Vector 3 := ⊞[1e10, 1e10, 1e10]
  let equalLargeNorm := norm equalLargeVec
  allPassed := allPassed && (← assertFinite "Norm: equal large values is finite" equalLargeNorm)

  -- Mix of positive and negative large values
  let mixedSignVec : Vector 3 := ⊞[1e10, -1e10, 1e10]
  let mixedSignNorm := norm mixedSignVec
  allPassed := allPassed && (← assertFinite "Norm: mixed signs is finite" mixedSignNorm)

  return allPassed

/-! ## Gradient-Related Stability -/

/-- Test gradient-related operations with extreme values -/
def testGradientExtremes : IO Bool := do
  IO.println "\n=== Gradient Extremes Tests ==="

  let mut allPassed := true

  -- Sigmoid derivative at extreme values
  -- At large positive: σ'(x) → 0
  let sigDeriv_large := sigmoidDerivative 100.0
  allPassed := allPassed && (← assertFinite "Sigmoid derivative: large input is finite" sigDeriv_large)
  allPassed := allPassed && (← assertTrue "Sigmoid derivative: large input → 0" (sigDeriv_large < 0.01))

  -- At large negative: σ'(x) → 0
  let sigDeriv_neg := sigmoidDerivative (-100.0)
  allPassed := allPassed && (← assertFinite "Sigmoid derivative: large negative is finite" sigDeriv_neg)
  allPassed := allPassed && (← assertTrue "Sigmoid derivative: large negative → 0" (sigDeriv_neg < 0.01))

  -- Tanh derivative at extreme values
  -- At large magnitude: tanh'(x) → 0
  let tanhDeriv_large := tanhDerivative 50.0
  allPassed := allPassed && (← assertFinite "Tanh derivative: large input is finite" tanhDeriv_large)
  allPassed := allPassed && (← assertTrue "Tanh derivative: large input → 0" (tanhDeriv_large < 0.01))

  -- ReLU derivative is always 0 or 1 (stable)
  let reluDeriv_pos := reluDerivative 1e10
  allPassed := allPassed && (← assertApproxEq "ReLU derivative: large positive = 1" reluDeriv_pos 1.0)

  let reluDeriv_neg := reluDerivative (-1e10)
  allPassed := allPassed && (← assertApproxEq "ReLU derivative: large negative = 0" reluDeriv_neg 0.0)

  return allPassed

/-! ## Test Runner -/

/-- Run all numerical stability tests and report results -/
def runAllTests : IO Unit := do
  IO.println "=========================================="
  IO.println "Running Numerical Stability Tests"
  IO.println "=========================================="

  let mut totalPassed := 0
  let mut totalTests := 0

  let testSuites : List (String × IO Bool) := [
    ("Activations with Extreme Values", testActivationsWithExtremes),
    ("Division by Zero Handling", testDivisionByZeroHandling),
    ("NaN and Inf Handling", testNaNAndInfHandling),
    ("Underflow and Overflow", testUnderflowAndOverflow),
    ("Softmax Stability", testSoftmaxStability),
    ("Normalization Edge Cases", testNormalizationEdgeCases),
    ("Gradient Extremes", testGradientExtremes)
  ]

  for (name, test) in testSuites do
    totalTests := totalTests + 1
    let passed ← test
    if passed then
      totalPassed := totalPassed + 1

  IO.println "\n=========================================="
  IO.println s!"Test Summary: {totalPassed}/{totalTests} suites passed"
  IO.println "=========================================="

  if totalPassed == totalTests then
    IO.println "✓ All numerical stability tests passed!"
  else
    IO.println s!"✗ {totalTests - totalPassed} test suite(s) failed"

end VerifiedNN.Testing.NumericalStabilityTests

/-- Top-level main for execution -/
def main : IO Unit := VerifiedNN.Testing.NumericalStabilityTests.runAllTests
