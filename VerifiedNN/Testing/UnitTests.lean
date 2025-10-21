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
  let allClose := (Fin n).all fun i => Float.abs (v[i] - w[i]) ≤ tol
  if allClose then
    IO.println s!"✓ {name}"
    return true
  else
    IO.println s!"✗ {name} FAILED: vectors differ"
    -- Print first few differing elements for debugging
    for i in [0:min 5 n] do
      if i < n then
        let idx : Fin n := ⟨i, by omega⟩
        if Float.abs (v[idx] - w[idx]) > tol then
          IO.println s!"  v[{i}]={v[idx]}, w[{i}]={w[idx]}"
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

/-! ## Integration: Run All Tests -/

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
    ("Vector Operations", testVectorOperations),
    ("Approximate Equality", testApproxEquality)
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
    IO.println "✓ All tests passed!"
  else
    IO.println s!"✗ {totalTests - totalPassed} test suite(s) failed"

/-! ## Individual Component Test Functions

These can be called independently for targeted testing during development.
-/

/-- Test a specific activation function with sample inputs -/
def testActivation (name : String) (f : Float → Float)
    (inputs : List Float) (expected : List Float) : IO Bool := do
  IO.println s!"\nTesting {name}:"

  if inputs.length != expected.length then
    IO.println "ERROR: input/expected length mismatch"
    return false

  let mut allPassed := true
  for (inp, exp) in inputs.zip expected do
    let result := f inp
    allPassed := allPassed && (← assertApproxEq s!"{name}({inp})" result exp)

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

end VerifiedNN.Testing.UnitTests
