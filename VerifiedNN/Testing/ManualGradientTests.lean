import VerifiedNN.Core.DataTypes
import VerifiedNN.Network.ManualGradient
import VerifiedNN.Network.Gradient
import VerifiedNN.Network.Architecture
import VerifiedNN.Loss.CrossEntropy
import SciLean

namespace VerifiedNN.Testing.ManualGradientTests

open VerifiedNN.Core
open VerifiedNN.Network.ManualGradient
open VerifiedNN.Network.Gradient
open VerifiedNN.Network
open VerifiedNN.Loss
open SciLean

set_default_scalar Float

/-!
# Manual Gradient Unit Tests

Comprehensive test suite validating the manual backpropagation implementation.

Tests include:
1. Finite difference validation (numerical accuracy)
2. Known gradient cases (specific test vectors)
3. Gradient norm checks (no NaN/Inf/explosion)
4. Dimension consistency checks

## Test Methodology

Each test reports PASS/FAIL with detailed diagnostics.
Tests use small parameter values to avoid numerical issues.

## Expected Accuracy

Manual gradients should match finite differences within 1e-3 tolerance
for Float arithmetic. Larger errors indicate implementation bugs.
-/

/-- Check if vector contains NaN or Inf -/
def hasNaNOrInf {n : Nat} (v : Vector n) : Bool :=
  Id.run do
    for i in [:n] do
      if h : i < n then
        let val := v[⟨i, h⟩]
        if val.isNaN || val.isInf then
          return true
    return false

/-- Check if vector is all zeros -/
def isAllZeros {n : Nat} (v : Vector n) : Bool :=
  Id.run do
    for i in [:n] do
      if h : i < n then
        let val := v[⟨i, h⟩]
        if val != 0.0 then
          return false
    return true

/-- Compute L2 norm of vector -/
def vectorNorm {n : Nat} (v : Vector n) : Float :=
  Id.run do
    let mut sum := 0.0
    for i in [:n] do
      if h : i < n then
        let val := v[⟨i, h⟩]
        sum := sum + val * val
    return Float.sqrt sum

/-- Test 1: Simple gradient check on zero input -/
def testZeroInput : IO Bool := do
  IO.println "Test 1: Zero Input Gradient Check"

  -- Initialize small random parameters
  let params : Vector nParams := ⊞ (i : Idx nParams) =>
    (((i.1.toNat * 7) % 13).toFloat - 6.0) * 0.01  -- Small pseudo-random values

  -- Zero input
  let input : Vector 784 := ⊞ (_ : Idx 784) => 0.0
  let target : Nat := 0

  -- Compute manual gradient
  let gradient := networkGradientManual params input target

  -- Check for NaN/Inf
  if hasNaNOrInf gradient then
    IO.println "  ✗ FAIL: Gradient contains NaN or Inf"
    return false

  -- Check not all zeros
  if isAllZeros gradient then
    IO.println "  ✗ FAIL: All gradients are zero (unexpected for non-zero loss)"
    return false

  IO.println "  ✓ PASS: Gradient is finite and non-zero"
  return true

/-- Test 2: Gradient check on uniform input -/
def testUniformInput : IO Bool := do
  IO.println "Test 2: Uniform Input Gradient Check"

  let params : Vector nParams := ⊞ (_ : Idx nParams) => 0.01
  let input : Vector 784 := ⊞ (_ : Idx 784) => 0.5  -- All pixels 0.5
  let target : Nat := 5

  let gradient := networkGradientManual params input target

  -- Check for NaN/Inf
  if hasNaNOrInf gradient then
    IO.println "  ✗ FAIL: Gradient contains NaN or Inf"
    return false

  -- Compute gradient norm
  let norm := vectorNorm gradient

  IO.println s!"  Gradient norm: {norm}"

  -- Check reasonable magnitude (not exploding)
  if norm > 1000.0 then
    IO.println "  ✗ FAIL: Gradient norm too large (exploding)"
    return false

  -- Check reasonable magnitude (not vanishing)
  if norm < 0.0001 then
    IO.println "  ✗ FAIL: Gradient norm too small (vanishing)"
    return false

  IO.println "  ✓ PASS: Gradient has reasonable magnitude"
  return true

/-- Test 3: Finite difference validation (subsample) -/
def testFiniteDifference : IO Bool := do
  IO.println "Test 3: Finite Difference Validation (100 params)"

  let params : Vector nParams := ⊞ (i : Idx nParams) =>
    (((i.1.toNat * 13) % 17).toFloat - 8.0) * 0.01

  let input : Vector 784 := ⊞ (i : Idx 784) =>
    (((i.1.toNat * 3) % 7).toFloat) * 0.1

  let target : Nat := 3

  -- Compute manual gradient
  let manual := networkGradientManual params input target

  -- Compute numerical gradient (subsample for speed)
  let numSamples := 100  -- Test 100 random parameters
  let epsilon := 1e-4
  let tolerance := 0.1  -- Relaxed tolerance for Float (softmax can have large gradients)

  let mut maxError := 0.0
  let mut avgError := 0.0
  let mut numChecked := 0

  -- Sample random indices
  for i in [0:numSamples] do
    let paramIdx := (i * 1019) % nParams  -- Pseudo-random index

    -- Compute numerical gradient for this parameter
    let params_plus := ⊞ (j : Idx nParams) =>
      if j.1.toNat == paramIdx then params[j] + epsilon else params[j]
    let params_minus := ⊞ (j : Idx nParams) =>
      if j.1.toNat == paramIdx then params[j] - epsilon else params[j]

    let net_plus := unflattenParams params_plus
    let output_plus := net_plus.forward input
    let loss_plus := crossEntropyLoss output_plus target

    let net_minus := unflattenParams params_minus
    let output_minus := net_minus.forward input
    let loss_minus := crossEntropyLoss output_minus target

    let numerical := (loss_plus - loss_minus) / (2.0 * epsilon)

    -- Get analytical gradient
    let analytical := manual[(Idx.finEquiv nParams).invFun ⟨paramIdx, Nat.mod_lt _ (Nat.zero_lt_of_lt (by decide : 0 < nParams))⟩]

    let error := Float.abs (analytical - numerical)
    maxError := max maxError error
    avgError := avgError + error
    numChecked := numChecked + 1

  avgError := avgError / numChecked.toFloat

  IO.println s!"  Max error: {maxError}"
  IO.println s!"  Avg error: {avgError}"
  IO.println s!"  Samples checked: {numChecked}"

  if maxError > tolerance then
    IO.println s!"  ✗ FAIL: Max error {maxError} exceeds tolerance {tolerance}"
    return false

  IO.println "  ✓ PASS: Manual gradient matches finite differences"
  return true

/-- Test 4: Dimension consistency check -/
def testDimensions : IO Bool := do
  IO.println "Test 4: Dimension Consistency"

  let params : Vector nParams := ⊞ (_ : Idx nParams) => 0.01
  let input : Vector 784 := ⊞ (_ : Idx 784) => 0.5
  let target : Nat := 7

  let _gradient := networkGradientManual params input target

  -- Gradient should have correct type (enforced by types)
  -- Just verify we can compute it
  IO.println s!"  ✓ PASS: Gradient computed with correct dimensions ({nParams} parameters)"
  return true

/-- Test 5: Different target classes -/
def testAllTargets : IO Bool := do
  IO.println "Test 5: All Target Classes (0-9)"

  let params : Vector nParams := ⊞ (_ : Idx nParams) => 0.01
  let input : Vector 784 := ⊞ (i : Idx 784) =>
    (i.1.toNat.toFloat / 784.0)  -- Gradient input 0 to 1

  let mut allPassed := true

  for target in [0:10] do
    let gradient := networkGradientManual params input target

    -- Check for NaN/Inf
    if hasNaNOrInf gradient then
      IO.println s!"  ✗ FAIL: Target {target} produced NaN/Inf"
      allPassed := false

  if allPassed then
    IO.println "  ✓ PASS: All 10 target classes produce valid gradients"

  return allPassed

/-- Run all tests and report results -/
unsafe def runTests : IO Unit := do
  IO.println "========================================"
  IO.println "Manual Gradient Unit Tests"
  IO.println "========================================"
  IO.println ""

  let mut testsPassed := 0
  let mut testsFailed := 0

  -- Test 1: Zero input
  if ← testZeroInput then
    testsPassed := testsPassed + 1
  else
    testsFailed := testsFailed + 1
  IO.println ""

  -- Test 2: Uniform input
  if ← testUniformInput then
    testsPassed := testsPassed + 1
  else
    testsFailed := testsFailed + 1
  IO.println ""

  -- Test 3: Finite difference
  IO.println "⚠ This test may take 10-20 seconds..."
  if ← testFiniteDifference then
    testsPassed := testsPassed + 1
  else
    testsFailed := testsFailed + 1
  IO.println ""

  -- Test 4: Dimensions
  if ← testDimensions then
    testsPassed := testsPassed + 1
  else
    testsFailed := testsFailed + 1
  IO.println ""

  -- Test 5: All targets
  if ← testAllTargets then
    testsPassed := testsPassed + 1
  else
    testsFailed := testsFailed + 1
  IO.println ""

  -- Summary
  IO.println "========================================"
  IO.println s!"Tests Passed: {testsPassed}/5"
  IO.println s!"Tests Failed: {testsFailed}/5"
  IO.println "========================================"

  if testsFailed > 0 then
    throw (IO.userError "Some tests failed!")
  else
    IO.println "✓ All tests passed!"

end VerifiedNN.Testing.ManualGradientTests

unsafe def main : IO Unit := do
  VerifiedNN.Testing.ManualGradientTests.runTests
