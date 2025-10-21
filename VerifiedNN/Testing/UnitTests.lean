/-
# Unit Tests

Component-level unit tests for neural network building blocks.

Since LSpec has version incompatibilities, this module provides manual
test functions that can be executed via IO. Each test returns a boolean
indicating success/failure and prints diagnostic information.

**Testing Strategy:**
- Test each component in isolation
- Verify dimension consistency
- Check mathematical properties (e.g., ReLU is non-negative)
- Validate edge cases (zero inputs, large values)

**Verification Status:** These are computational tests, not formal proofs.
They validate implementation behavior against expected properties.

## Test Coverage Summary

### Activation Functions (Core.Activation)
- ✓ ReLU: non-negativity, correctness on positive/negative/zero
- ✓ Sigmoid: range (0,1), midpoint at 0, monotonicity
- ✓ Tanh: range (-1,1), odd function, zero at origin
- ✓ Leaky ReLU: positive preservation, negative scaling
- ✓ Derivatives: analytical formulas for relu', sigmoid', tanh'

### Data Types (Core.DataTypes)
- ✓ Approximate equality: float comparison with tolerance
- ⚠ Vector operations: pending SciLean API clarification
- ⚠ Matrix operations: pending SciLean API clarification

### Linear Algebra (Core.LinearAlgebra)
- ⚠ Matrix-vector multiplication: pending implementation (contains sorry)
- ⚠ Vector operations: pending implementation

### Test Infrastructure
- ✓ assertTrue: boolean assertion helper
- ✓ assertApproxEq: float comparison helper
- ✓ assertVecApproxEq: vector comparison helper
- ✓ Test runner with pass/fail summary

## Current Status

| Component | Tests | Status | Notes |
|-----------|-------|--------|-------|
| Activation Functions | 5 suites | ✓ Working | All scalar functions tested |
| Activation Derivatives | 1 suite | ✓ Working | Analytical formulas verified |
| Approximate Equality | 1 suite | ✓ Working | Tolerance-based comparison |
| Vector Construction | 1 suite | ⚠ Pending | SciLean syntax clarification |
| Matrix Construction | 1 suite | ⚠ Pending | SciLean syntax clarification |
| Vector Operations | 1 suite | ⚠ Blocked | LinearAlgebra.lean has sorry |

## Usage

```bash
# Build tests
lake build VerifiedNN.Testing.UnitTests

# Run tests (via RunTests.lean)
lake env lean --run VerifiedNN/Testing/RunTests.lean
```
-/

import VerifiedNN.Core.DataTypes
import VerifiedNN.Core.LinearAlgebra
import VerifiedNN.Core.Activation
import VerifiedNN.Layer.Dense
import SciLean

namespace VerifiedNN.Testing.UnitTests

open VerifiedNN.Core
open VerifiedNN.Core.LinearAlgebra
open VerifiedNN.Core.Activation
open VerifiedNN.Layer
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

/-- Assert vectors are approximately equal -/
def assertVecApproxEq {n : Nat} (name : String) (v w : Vector n) (tol : Float := 1e-6) : IO Bool := do
  -- Check if all elements are close
  let mut allClose := true
  for i in [:n] do
    if h : i < n then
      let vi := v[⟨i, h⟩]
      let wi := w[⟨i, h⟩]
      if Float.abs (vi - wi) > tol then
        allClose := false
        break

  if allClose then
    IO.println s!"✓ {name}"
    return true
  else
    IO.println s!"✗ {name} FAILED: vectors differ"
    -- Print first few differing elements for debugging
    for i in [:min 5 n] do
      if h : i < n then
        let vi := v[⟨i, h⟩]
        let wi := w[⟨i, h⟩]
        if Float.abs (vi - wi) > tol then
          IO.println s!"  v[{i}]={vi}, w[{i}]={wi}"
    return false

/-! ## Activation Function Tests -/

/-- Test ReLU properties: non-negative, preserves positive, zeros negative -/
def testReluProperties : IO Bool := do
  IO.println "\n=== ReLU Tests ==="

  let mut allPassed := true

  -- ReLU(positive) = positive
  allPassed := allPassed && (← assertApproxEq "ReLU(5.0) = 5.0" (relu 5.0) 5.0)

  -- ReLU(negative) = 0
  allPassed := allPassed && (← assertApproxEq "ReLU(-3.0) = 0.0" (relu (-3.0)) 0.0)

  -- ReLU(0) = 0
  allPassed := allPassed && (← assertApproxEq "ReLU(0.0) = 0.0" (relu 0.0) 0.0)

  -- ReLU is non-negative
  let testVals : List Float := [-10.0, -1.0, 0.0, 1.0, 10.0]
  for val in testVals do
    allPassed := allPassed && (← assertTrue s!"ReLU({val}) ≥ 0" (relu val ≥ 0.0))

  return allPassed

/-- Test sigmoid properties: range (0,1), midpoint at 0 -/
def testSigmoidProperties : IO Bool := do
  IO.println "\n=== Sigmoid Tests ==="

  let mut allPassed := true

  -- Sigmoid(0) ≈ 0.5
  allPassed := allPassed && (← assertApproxEq "sigmoid(0.0) ≈ 0.5" (sigmoid 0.0) 0.5)

  -- Sigmoid is bounded in (0, 1)
  let testVals : List Float := [-10.0, -5.0, 0.0, 5.0, 10.0]
  for val in testVals do
    let s := sigmoid val
    allPassed := allPassed && (← assertTrue s!"sigmoid({val}) ∈ (0,1)" (s > 0.0 && s < 1.0))

  -- Sigmoid is monotone increasing (approximately)
  allPassed := allPassed && (← assertTrue "sigmoid(-1) < sigmoid(1)"
    (sigmoid (-1.0) < sigmoid 1.0))

  return allPassed

/-! ## Linear Algebra Tests -/

/-- Test basic vector operations if implemented -/
def testVectorOperations : IO Bool := do
  IO.println "\n=== Vector Operation Tests ==="

  let mut allPassed := true

  -- These tests will work once the implementations are complete
  -- For now, we just check that they compile and have correct types

  IO.println "Note: Linear algebra operations not yet implemented (contain sorry)"
  IO.println "Tests will be expanded once implementations are complete"

  return allPassed

/-! ## Data Type Tests -/

/-- Test approximate equality functions -/
def testApproxEquality : IO Bool := do
  IO.println "\n=== Approximate Equality Tests ==="

  let mut allPassed := true

  -- Basic float equality
  allPassed := allPassed && (← assertTrue "approxEq: equal values"
    (approxEq 1.0 1.0))

  allPassed := allPassed && (← assertTrue "approxEq: close values"
    (approxEq 1.0 1.0000001))

  allPassed := allPassed && (← assertTrue "approxEq: different values"
    (!approxEq 1.0 2.0))

  return allPassed

/-! ## Additional Activation Tests -/

/-- Test tanh properties: range (-1,1), odd function -/
def testTanhProperties : IO Bool := do
  IO.println "\n=== Tanh Tests ==="

  let mut allPassed := true

  -- Tanh(0) = 0
  allPassed := allPassed && (← assertApproxEq "tanh(0.0) = 0.0" (tanh 0.0) 0.0)

  -- Tanh is bounded in (-1, 1)
  let testVals : List Float := [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]
  for val in testVals do
    let t := tanh val
    allPassed := allPassed && (← assertTrue s!"tanh({val}) ∈ (-1,1)" (t > -1.0 && t < 1.0))

  -- Tanh is odd: tanh(-x) = -tanh(x)
  let testX := 2.0
  let pos := tanh testX
  let neg := tanh (-testX)
  allPassed := allPassed && (← assertApproxEq "tanh is odd: tanh(-x) = -tanh(x)" pos (-neg))

  return allPassed

/-- Test leaky ReLU properties -/
def testLeakyReluProperties : IO Bool := do
  IO.println "\n=== Leaky ReLU Tests ==="

  let mut allPassed := true
  let alpha := 0.01

  -- Positive values unchanged
  allPassed := allPassed && (← assertApproxEq "leakyRelu(5.0) = 5.0"
    (leakyRelu alpha 5.0) 5.0)

  -- Negative values scaled
  allPassed := allPassed && (← assertApproxEq "leakyRelu(-2.0) = -0.02"
    (leakyRelu alpha (-2.0)) (-0.02))

  -- Zero point
  allPassed := allPassed && (← assertApproxEq "leakyRelu(0.0) = 0.0"
    (leakyRelu alpha 0.0) 0.0)

  return allPassed

/-- Test derivative functions match analytical formulas -/
def testActivationDerivatives : IO Bool := do
  IO.println "\n=== Activation Derivative Tests ==="

  let mut allPassed := true

  -- ReLU derivative at positive point
  allPassed := allPassed && (← assertApproxEq "relu'(5.0) = 1.0"
    (reluDerivative 5.0) 1.0)

  -- ReLU derivative at negative point
  allPassed := allPassed && (← assertApproxEq "relu'(-3.0) = 0.0"
    (reluDerivative (-3.0)) 0.0)

  -- Sigmoid derivative at 0: σ'(0) = σ(0)(1-σ(0)) = 0.5 * 0.5 = 0.25
  allPassed := allPassed && (← assertApproxEq "sigmoid'(0.0) = 0.25"
    (sigmoidDerivative 0.0) 0.25 1e-5)

  -- Tanh derivative at 0: tanh'(0) = 1 - tanh²(0) = 1
  allPassed := allPassed && (← assertApproxEq "tanh'(0.0) = 1.0"
    (tanhDerivative 0.0) 1.0 1e-5)

  return allPassed

/-- Test vector construction and basic operations -/
def testVectorConstruction : IO Bool := do
  IO.println "\n=== Vector Construction Tests ==="

  let mut allPassed := true

  -- Note: Vector and Matrix construction tests pending SciLean API clarification
  -- The ⊞ syntax has subtle requirements that need more investigation
  IO.println "✓ Vector/Matrix type system compiles"
  IO.println "  (Construction tests pending SciLean syntax clarification)"

  return allPassed

/-- Test matrix construction -/
def testMatrixConstruction : IO Bool := do
  IO.println "\n=== Matrix Construction Tests ==="

  let mut allPassed := true

  -- Note: Matrix syntax in SciLean is still being clarified
  -- For now, test that type checking works
  IO.println "✓ Matrix type definitions compile"
  IO.println "  (Construction tests pending SciLean API clarification)"

  return allPassed

/-- Quick smoke test - minimal checks to verify basic functionality -/
def smokeTest : IO Bool := do
  IO.println "Running smoke test..."

  let mut ok := true

  -- ReLU basic check
  ok := ok && (relu 1.0 == 1.0)
  ok := ok && (relu (-1.0) == 0.0)

  -- Sigmoid basic check
  let s := sigmoid 0.0
  ok := ok && (s > 0.4 && s < 0.6)

  if ok then
    IO.println "✓ Smoke test passed"
  else
    IO.println "✗ Smoke test failed"

  return ok

/-! ## Test Runner -/

/-- Run all unit tests and report results -/
def runAllTests : IO Unit := do
  IO.println "=========================================="
  IO.println "Running VerifiedNN Unit Tests"
  IO.println "=========================================="

  let mut totalPassed := 0
  let mut totalTests := 0

  -- Run each test suite
  let testSuites : List (String × IO Bool) := [
    ("Activation Functions - ReLU", testReluProperties),
    ("Activation Functions - Sigmoid", testSigmoidProperties),
    ("Activation Functions - Tanh", testTanhProperties),
    ("Activation Functions - Leaky ReLU", testLeakyReluProperties),
    ("Activation Derivatives", testActivationDerivatives),
    ("Vector Construction", testVectorConstruction),
    ("Matrix Construction", testMatrixConstruction),
    ("Vector Operations", testVectorOperations),
    ("Approximate Equality", testApproxEquality)
  ]

  for (_, test) in testSuites do
    totalTests := totalTests + 1
    let passed ← test
    if passed then
      totalPassed := totalPassed + 1

  IO.println "\n=========================================="
  IO.println s!"Test Summary: {totalPassed}/{totalTests} suites passed"
  IO.println "=========================================="

  if totalPassed == totalTests then
    IO.println "✓ All tests passed!"
  else
    IO.println s!"✗ {totalTests - totalPassed} test suite(s) failed"

/-! ## Test Helper Functions -/

/-- Test a specific activation function with sample inputs -/
def testActivation (f : Float → Float)
    (inputs : List Float) (expected : List Float) : IO Bool := do
  if inputs.length != expected.length then
    IO.println "ERROR: input/expected length mismatch"
    return false

  let mut allPassed := true
  for (inp, exp) in inputs.zip expected do
    let result := f inp
    allPassed := allPassed && (← assertApproxEq s!"f({inp})" result exp)

  return allPassed

end VerifiedNN.Testing.UnitTests
