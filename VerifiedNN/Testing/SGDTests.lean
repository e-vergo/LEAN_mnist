import VerifiedNN.Optimizer.SGD
import VerifiedNN.Core.DataTypes

/-!
# SGD Optimizer Unit Tests

Tests verify that the SGD update rule `params_new = params_old - lr * gradient`
is implemented correctly with hand-calculable examples.

## Test Cases

1. Simple 3-parameter update with known result
2. Verify negative sign (gradient descent, not ascent)
3. Verify scalar multiplication by learning rate
4. Zero gradient produces no change
5. Large gradient still works (no overflow)

## Implementation Notes

These tests use simple, hand-calculable examples to validate the basic
arithmetic of SGD updates. Each test includes:
- Clear setup documentation
- Expected result
- Pass/fail validation with explicit error messages
- Diagnostic output showing actual vs expected values

## Usage

```bash
lake build VerifiedNN.Testing.SGDTests
lake exe sgdTests
```

If all tests pass, SGD update rule is correctly implemented.
If any test fails, the diagnostic output will identify the specific bug.
-/

namespace VerifiedNN.Testing.SGDTests

open VerifiedNN.Optimizer
open VerifiedNN.Core
open SciLean

set_default_scalar Float

-- Helper to check if two floats are approximately equal
def approxEqual (x y : Float) (tolerance : Float := 1e-6) : Bool :=
  Float.abs (x - y) < tolerance

-- Helper to check if two vectors are approximately equal
def vectorApproxEqual {n : Nat} (v1 v2 : Vector n) (tolerance : Float := 1e-6) : IO Bool := do
  let mut allClose := true
  for i in [:n] do
    if h : i < n then
      let vi := v1[⟨i, h⟩]
      let wi := v2[⟨i, h⟩]
      if Float.abs (vi - wi) > tolerance then
        allClose := false
        break
  return allClose

/-- Test 1: Simple 3-parameter update with known values

Setup: params = [1.0, 2.0, 3.0], gradient = [0.1, -0.2, 0.3], lr = 0.1
Expected: params - lr * gradient = [0.99, 2.02, 2.97]
-/
def test1_simpleUpdate : IO Bool := do
  IO.println "\n[Test 1] Simple 3-parameter update"
  IO.println "  Setup: params = [1.0, 2.0, 3.0]"
  IO.println "         gradient = [0.1, -0.2, 0.3]"
  IO.println "         learning rate = 0.1"
  IO.println "  Expected: [0.99, 2.02, 2.97]"

  let params1 : Vector 3 := ⊞[1.0, 2.0, 3.0]
  let grad1 : Vector 3 := ⊞[0.1, -0.2, 0.3]
  let state1 := initSGD params1 0.1
  let newState1 := sgdStep state1 grad1
  let expected1 : Vector 3 := ⊞[0.99, 2.02, 2.97]

  IO.println s!"  Actual: [{newState1.params[0]}, {newState1.params[1]}, {newState1.params[2]}]"

  let isMatch ← vectorApproxEqual newState1.params expected1
  if isMatch then
    IO.println "  ✓ PASS"
    return true
  else
    IO.println "  ✗ FAIL"
    IO.println s!"  Expected: [{expected1[0]}, {expected1[1]}, {expected1[2]}]"
    return false

/-- Test 2: Verify gradient descent direction (negative sign)

Setup: params = [5.0], gradient = [2.0], lr = 0.1
Expected: [4.8] (descent, not ascent to 5.2)

This test catches sign errors: params + lr*grad would give 5.2 (wrong).
Correct formula: params - lr*grad gives 4.8.
-/
def test2_descentDirection : IO Bool := do
  IO.println "\n[Test 2] Gradient descent direction"
  IO.println "  Setup: params = [5.0], gradient = [2.0], lr = 0.1"
  IO.println "  Expected: [4.8] (decreased, not increased to 5.2)"

  let params2 : Vector 1 := ⊞[5.0]
  let grad2 : Vector 1 := ⊞[2.0]
  let state2 := initSGD params2 0.1
  let newState2 := sgdStep state2 grad2

  IO.println s!"  Actual: [{newState2.params[0]}]"

  if approxEqual newState2.params[0] 4.8 then
    IO.println "  ✓ PASS (correct descent direction)"
    return true
  else
    IO.println "  ✗ FAIL"
    if approxEqual newState2.params[0] 5.2 then
      IO.println "  ERROR: Gradient ASCENT detected! Should be params - lr*grad, not +"
    else
      IO.println s!"  Expected: 4.8, got: {newState2.params[0]}"
    return false

/-- Test 3: Learning rate scaling

Setup: params = [1.0], gradient = [1.0]
With lr=0.01: expect [0.99]
With lr=0.1:  expect [0.9]
With lr=1.0:  expect [0.0]

Validates that learning rate correctly scales the gradient step.
-/
def test3_learningRateScaling : IO Bool := do
  IO.println "\n[Test 3] Learning rate scaling"
  IO.println "  Setup: params = [1.0], gradient = [1.0]"
  IO.println "  With lr=0.01: expect [0.99]"
  IO.println "  With lr=0.1:  expect [0.9]"
  IO.println "  With lr=1.0:  expect [0.0]"

  let params3 : Vector 1 := ⊞[1.0]
  let grad3 : Vector 1 := ⊞[1.0]

  let state3a := initSGD params3 0.01
  let newState3a := sgdStep state3a grad3
  IO.println s!"  lr=0.01: [{newState3a.params[0]}]"
  let pass3a := approxEqual newState3a.params[0] 0.99

  let state3b := initSGD params3 0.1
  let newState3b := sgdStep state3b grad3
  IO.println s!"  lr=0.1:  [{newState3b.params[0]}]"
  let pass3b := approxEqual newState3b.params[0] 0.9

  let state3c := initSGD params3 1.0
  let newState3c := sgdStep state3c grad3
  IO.println s!"  lr=1.0:  [{newState3c.params[0]}]"
  let pass3c := approxEqual newState3c.params[0] 0.0

  if pass3a && pass3b && pass3c then
    IO.println "  ✓ PASS (all learning rates correct)"
    return true
  else
    IO.println "  ✗ FAIL"
    if !pass3a then IO.println "    lr=0.01 failed"
    if !pass3b then IO.println "    lr=0.1 failed"
    if !pass3c then IO.println "    lr=1.0 failed"
    return false

/-- Test 4: Zero gradient produces no change

Setup: params = [1.5, 2.5], gradient = [0.0, 0.0], lr = 0.1
Expected: [1.5, 2.5] (unchanged)

Validates that zero gradients don't modify parameters.
-/
def test4_zeroGradient : IO Bool := do
  IO.println "\n[Test 4] Zero gradient produces no change"
  IO.println "  Setup: params = [1.5, 2.5], gradient = [0.0, 0.0], lr = 0.1"
  IO.println "  Expected: [1.5, 2.5] (unchanged)"

  let params4 : Vector 2 := ⊞[1.5, 2.5]
  let grad4 : Vector 2 := ⊞[0.0, 0.0]
  let state4 := initSGD params4 0.1
  let newState4 := sgdStep state4 grad4

  IO.println s!"  Actual: [{newState4.params[0]}, {newState4.params[1]}]"

  let isMatch ← vectorApproxEqual newState4.params params4
  if isMatch then
    IO.println "  ✓ PASS"
    return true
  else
    IO.println "  ✗ FAIL"
    IO.println s!"  Expected: [{params4[0]}, {params4[1]}]"
    return false

/-- Test 5: Large gradient handling

Setup: params = [10.0], gradient = [100.0], lr = 0.01
Expected: [9.0] (large step but no overflow)

Validates numerical stability with large gradients.
-/
def test5_largeGradient : IO Bool := do
  IO.println "\n[Test 5] Large gradient handling"
  IO.println "  Setup: params = [10.0], gradient = [100.0], lr = 0.01"
  IO.println "  Expected: [9.0] (large step but no overflow)"

  let params5 : Vector 1 := ⊞[10.0]
  let grad5 : Vector 1 := ⊞[100.0]
  let state5 := initSGD params5 0.01
  let newState5 := sgdStep state5 grad5

  IO.println s!"  Actual: [{newState5.params[0]}]"

  if approxEqual newState5.params[0] 9.0 && !newState5.params[0].isNaN && !newState5.params[0].isInf then
    IO.println "  ✓ PASS (no overflow, correct result)"
    return true
  else
    IO.println "  ✗ FAIL"
    if newState5.params[0].isNaN then
      IO.println "  ERROR: NaN detected"
    else if newState5.params[0].isInf then
      IO.println "  ERROR: Infinity detected"
    else
      IO.println s!"  Expected: 9.0, got: {newState5.params[0]}"
    return false

/-- Test 6: Epoch counter increments

Validates that SGD step increments the epoch counter.
-/
def test6_epochCounter : IO Bool := do
  IO.println "\n[Test 6] Epoch counter increments"
  IO.println "  Setup: Initial epoch = 0"
  IO.println "  Expected: After 1 step, epoch = 1"

  let params : Vector 2 := ⊞[1.0, 2.0]
  let grad : Vector 2 := ⊞[0.1, 0.2]
  let state := initSGD params 0.1

  IO.println s!"  Initial epoch: {state.epoch}"

  let state1 := sgdStep state grad
  IO.println s!"  After step 1: {state1.epoch}"

  let state2 := sgdStep state1 grad
  IO.println s!"  After step 2: {state2.epoch}"

  if state.epoch == 0 && state1.epoch == 1 && state2.epoch == 2 then
    IO.println "  ✓ PASS (epoch counter works)"
    return true
  else
    IO.println "  ✗ FAIL"
    return false

/-- Main test runner -/
unsafe def main : IO Unit := do
  IO.println "=== SGD Optimizer Unit Tests ==="

  let result1 ← test1_simpleUpdate
  let result2 ← test2_descentDirection
  let result3 ← test3_learningRateScaling
  let result4 ← test4_zeroGradient
  let result5 ← test5_largeGradient
  let result6 ← test6_epochCounter

  -- Summary
  IO.println "\n=== Test Summary ==="
  let passCount := [result1, result2, result3, result4, result5, result6].filter id |>.length
  let totalCount := 6
  IO.println s!"Passed: {passCount}/{totalCount}"

  if passCount == totalCount then
    IO.println "\n✓ All tests passed! SGD update rule is mathematically correct."
  else
    IO.println "\n✗ Some tests failed. SGD implementation has bugs."
    IO.println "\nDiagnostic hints:"
    IO.println "  - If Test 2 fails: Check sign (should be params - lr*grad, not +)"
    IO.println "  - If Test 3 fails: Check learning rate multiplication"
    IO.println "  - If Test 1 fails: Check vector arithmetic operations"
    IO.println "  - If Test 4 fails: Check zero gradient handling"
    IO.println "  - If Test 5 fails: Check numerical stability"

  IO.println ""
  IO.println "=== SGD Tests Complete ==="

end VerifiedNN.Testing.SGDTests

/-- Top-level main function for running the SGD tests. -/
unsafe def main : IO Unit := VerifiedNN.Testing.SGDTests.main
