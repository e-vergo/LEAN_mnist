import VerifiedNN.Core.DataTypes
import VerifiedNN.Core.LinearAlgebra
import VerifiedNN.Core.DenseBackward
import SciLean

/-!
# Dense Layer Backward Pass Tests

Test suite for validating the dense layer backward pass implementation.

## Main Tests

- `testSimpleBackward`: Verify backward pass on a small example with known values
- `testWeightGradient`: Verify weight gradient computation
- `testBiasGradient`: Verify bias gradient (identity)
- `testInputGradient`: Verify input gradient via transpose multiply

## Implementation Notes

**Testing Strategy:**
- Validate on small examples with manually computed expected gradients
- Check dimension consistency (enforced by type system)
- Verify mathematical properties (outer product, transpose multiplication)

**Coverage Status:** Comprehensive tests for backward pass correctness.

## Usage

```bash
# Build tests
lake build VerifiedNN.Testing.DenseBackwardTests

# Run tests
lake env lean --run VerifiedNN/Testing/DenseBackwardTests.lean
```
-/

namespace VerifiedNN.Testing.DenseBackwardTests

open VerifiedNN.Core
open VerifiedNN.Core.LinearAlgebra
open VerifiedNN.Core.DenseBackward
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

/-- Assert vectors are approximately equal -/
def assertVecApproxEq {n : Nat} (name : String) (v w : Vector n) (tol : Float := 1e-6) : IO Bool := do
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
    for i in [:min 5 n] do
      if h : i < n then
        let vi := v[⟨i, h⟩]
        let wi := w[⟨i, h⟩]
        if Float.abs (vi - wi) > tol then
          IO.println s!"  v[{i}]={vi}, w[{i}]={wi}"
    return false

/-- Assert matrices are approximately equal -/
def assertMatApproxEq {m n : Nat} (name : String) (A B : Matrix m n) (tol : Float := 1e-6) : IO Bool := do
  let mut allClose := true
  for i in [:m] do
    if hi : i < m then
      for j in [:n] do
        if hj : j < n then
          let Aij := A[⟨i, hi⟩, ⟨j, hj⟩]
          let Bij := B[⟨i, hi⟩, ⟨j, hj⟩]
          if Float.abs (Aij - Bij) > tol then
            allClose := false
            break

  if allClose then
    IO.println s!"✓ {name}"
    return true
  else
    IO.println s!"✗ {name} FAILED: matrices differ"
    for i in [:min 3 m] do
      if hi : i < m then
        for j in [:min 3 n] do
          if hj : j < n then
            let Aij := A[⟨i, hi⟩, ⟨j, hj⟩]
            let Bij := B[⟨i, hi⟩, ⟨j, hj⟩]
            if Float.abs (Aij - Bij) > tol then
              IO.println s!"  A[{i},{j}]={Aij}, B[{i},{j}]={Bij}"
    return false

/-! ## Dense Backward Pass Tests -/

/-- Test simple backward pass with known values.

Example setup:
- Weight matrix W: 2x3
- Input x: 3-dimensional
- Gradient from above: 2-dimensional

Expected:
- dW = gradOutput ⊗ input (outer product)
- db = gradOutput (identity)
- dInput = W^T @ gradOutput
-/
def testSimpleBackward : IO Bool := do
  IO.println "\n=== Simple Dense Backward Test ==="

  -- Setup: 2 outputs, 3 inputs
  let weights : Matrix 2 3 := ⊞[1.0, 2.0, 3.0; 4.0, 5.0, 6.0]
  let input : Vector 3 := ⊞[1.0, 0.5, 0.25]
  let gradOutput : Vector 2 := ⊞[1.0, 2.0]

  -- Compute backward pass
  let (dW, db, dInput) := denseLayerBackward gradOutput input weights

  -- Expected weight gradient: outer product
  -- dW[i,j] = gradOutput[i] * input[j]
  -- dW = [1.0 * 1.0, 1.0 * 0.5, 1.0 * 0.25]
  --      [2.0 * 1.0, 2.0 * 0.5, 2.0 * 0.25]
  let expectedDW : Matrix 2 3 := ⊞[1.0, 0.5, 0.25; 2.0, 1.0, 0.5]

  -- Expected bias gradient: identity
  let expectedDB : Vector 2 := ⊞[1.0, 2.0]

  -- Expected input gradient: W^T @ gradOutput
  -- dInput[j] = Σ_i W[i,j] * gradOutput[i]
  -- dInput[0] = 1.0*1.0 + 4.0*2.0 = 9.0
  -- dInput[1] = 2.0*1.0 + 5.0*2.0 = 12.0
  -- dInput[2] = 3.0*1.0 + 6.0*2.0 = 15.0
  let expectedDInput : Vector 3 := ⊞[9.0, 12.0, 15.0]

  -- Verify all gradients
  let mut allPassed := true
  if !(← assertMatApproxEq "Weight gradient (dW)" dW expectedDW) then
    allPassed := false
  if !(← assertVecApproxEq "Bias gradient (db)" db expectedDB) then
    allPassed := false
  if !(← assertVecApproxEq "Input gradient (dInput)" dInput expectedDInput) then
    allPassed := false

  return allPassed

/-- Test weight gradient is correct outer product -/
def testWeightGradient : IO Bool := do
  IO.println "\n=== Weight Gradient Test ==="

  let weights : Matrix 3 2 := ⊞[1.0, 2.0; 3.0, 4.0; 5.0, 6.0]
  let input : Vector 2 := ⊞[2.0, 3.0]
  let gradOutput : Vector 3 := ⊞[1.0, 1.0, 1.0]

  let (dW, _, _) := denseLayerBackward gradOutput input weights

  -- Expected: outer product
  -- dW[i,j] = gradOutput[i] * input[j]
  let expectedDW : Matrix 3 2 := ⊞[2.0, 3.0; 2.0, 3.0; 2.0, 3.0]

  assertMatApproxEq "Weight gradient is outer product" dW expectedDW

/-- Test bias gradient is identity -/
def testBiasGradient : IO Bool := do
  IO.println "\n=== Bias Gradient Test ==="

  let weights : Matrix 4 3 := ⊞[1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0; 10.0, 11.0, 12.0]
  let input : Vector 3 := ⊞[1.0, 1.0, 1.0]
  let gradOutput : Vector 4 := ⊞[0.5, 1.5, 2.5, 3.5]

  let (_, db, _) := denseLayerBackward gradOutput input weights

  -- Expected: db = gradOutput (identity)
  let expectedDB : Vector 4 := ⊞[0.5, 1.5, 2.5, 3.5]

  assertVecApproxEq "Bias gradient is identity" db expectedDB

/-- Test input gradient via transpose multiply -/
def testInputGradient : IO Bool := do
  IO.println "\n=== Input Gradient Test ==="

  -- Setup: 2x2 for simple manual verification
  let weights : Matrix 2 2 := ⊞[1.0, 2.0; 3.0, 4.0]
  let input : Vector 2 := ⊞[1.0, 1.0]
  let gradOutput : Vector 2 := ⊞[1.0, 1.0]

  let (_, _, dInput) := denseLayerBackward gradOutput input weights

  -- Expected: W^T @ gradOutput
  -- W^T = [1.0, 3.0]
  --       [2.0, 4.0]
  -- dInput[0] = 1.0*1.0 + 3.0*1.0 = 4.0
  -- dInput[1] = 2.0*1.0 + 4.0*1.0 = 6.0
  let expectedDInput : Vector 2 := ⊞[4.0, 6.0]

  assertVecApproxEq "Input gradient via transpose multiply" dInput expectedDInput

/-- Test zero gradient propagation -/
def testZeroGradient : IO Bool := do
  IO.println "\n=== Zero Gradient Test ==="

  let weights : Matrix 2 3 := ⊞[1.0, 2.0, 3.0; 4.0, 5.0, 6.0]
  let input : Vector 3 := ⊞[1.0, 2.0, 3.0]
  let gradOutput : Vector 2 := ⊞[0.0, 0.0]  -- Zero gradient

  let (dW, db, dInput) := denseLayerBackward gradOutput input weights

  -- Expected: all gradients should be zero
  let expectedDW : Matrix 2 3 := ⊞[0.0, 0.0, 0.0; 0.0, 0.0, 0.0]
  let expectedDB : Vector 2 := ⊞[0.0, 0.0]
  let expectedDInput : Vector 3 := ⊞[0.0, 0.0, 0.0]

  let mut allPassed := true
  if !(← assertMatApproxEq "Zero weight gradient" dW expectedDW) then
    allPassed := false
  if !(← assertVecApproxEq "Zero bias gradient" db expectedDB) then
    allPassed := false
  if !(← assertVecApproxEq "Zero input gradient" dInput expectedDInput) then
    allPassed := false

  return allPassed

/-! ## Main Test Runner -/

/-- Run all dense backward pass tests and report results -/
def runAllTests : IO Unit := do
  IO.println "=========================================="
  IO.println "Dense Layer Backward Pass Tests"
  IO.println "=========================================="

  let mut totalTests := 0
  let mut totalPassed := 0

  let testSuites : List (String × IO Bool) := [
    ("Simple Backward Pass", testSimpleBackward),
    ("Weight Gradient", testWeightGradient),
    ("Bias Gradient", testBiasGradient),
    ("Input Gradient", testInputGradient),
    ("Zero Gradient Propagation", testZeroGradient)
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
    IO.println "✓ All dense backward tests passed!"
  else
    IO.println s!"✗ {totalTests - totalPassed} test suite(s) failed"

end VerifiedNN.Testing.DenseBackwardTests

-- Main entry point for standalone execution
def main : IO Unit := do
  VerifiedNN.Testing.DenseBackwardTests.runAllTests
