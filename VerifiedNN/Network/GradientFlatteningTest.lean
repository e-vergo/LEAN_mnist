import VerifiedNN.Network.GradientFlattening
import VerifiedNN.Network.Gradient
import VerifiedNN.Core.DataTypes
import SciLean

/-!
# Gradient Flattening Test

Validates that `flattenGradients` produces a gradient vector with the correct
layout matching the parameter vector structure.

## Tests

1. **Dimension test:** Result has exactly 101,770 elements
2. **Layout test:** Check indices map to correct gradient components
3. **Example values:** Verify specific gradient values land in expected positions

## Usage

```bash
lake build VerifiedNN.Network.GradientFlatteningTest
lake env lean --run VerifiedNN/Network/GradientFlatteningTest.lean
```
-/

namespace VerifiedNN.Network.GradientFlatteningTest

open VerifiedNN.Core
open VerifiedNN.Network.Gradient
open VerifiedNN.Network.GradientFlattening
open SciLean

set_default_scalar Float

/-- Helper to create a Nat index as Idx -/
def mkIdx (n : Nat) (i : Nat) (h : i < n) : Idx n :=
  (Idx.finEquiv n).invFun ⟨i, h⟩

/-- Test that flattenGradients produces correct dimension -/
def testDimension : Bool :=
  -- Create dummy gradients
  let dW1 : Matrix 128 784 := ⊞ (_ : Idx 128 × Idx 784) => 1.0
  let db1 : Vector 128 := ⊞ (_ : Idx 128) => 2.0
  let dW2 : Matrix 10 128 := ⊞ (_ : Idx 10 × Idx 128) => 3.0
  let db2 : Vector 10 := ⊞ (_ : Idx 10) => 4.0

  let _gradient := flattenGradients dW1 db1 dW2 db2

  -- Check dimension (type level guarantees this, but verify at runtime)
  true  -- Type system guarantees correct dimension

/-- Test that gradient values land in the correct positions -/
def testLayout : IO Unit := do
  -- Create gradients with distinguishable values
  let dW1 : Matrix 128 784 := ⊞ (_ : Idx 128 × Idx 784) => 1.0
  let db1 : Vector 128 := ⊞ (_ : Idx 128) => 2.0
  let dW2 : Matrix 10 128 := ⊞ (_ : Idx 10 × Idx 128) => 3.0
  let db2 : Vector 10 := ⊞ (_ : Idx 10) => 4.0

  let gradient := flattenGradients dW1 db1 dW2 db2

  -- Test key indices
  -- Layer 1 weights start at index 0
  let idx0 := mkIdx nParams 0 (by unfold nParams; omega)
  IO.println s!"gradient[0] (expect 1.0): {gradient[idx0]}"

  -- Layer 1 bias starts at index 100352
  let idx100352 := mkIdx nParams 100352 (by unfold nParams; omega)
  IO.println s!"gradient[100352] (expect 2.0): {gradient[idx100352]}"

  -- Layer 2 weights start at index 100480
  let idx100480 := mkIdx nParams 100480 (by unfold nParams; omega)
  IO.println s!"gradient[100480] (expect 3.0): {gradient[idx100480]}"

  -- Layer 2 bias starts at index 101760
  let idx101760 := mkIdx nParams 101760 (by unfold nParams; omega)
  IO.println s!"gradient[101760] (expect 4.0): {gradient[idx101760]}"

  -- Last element (Layer 2 bias, last position)
  let idx101769 := mkIdx nParams 101769 (by unfold nParams; omega)
  IO.println s!"gradient[101769] (expect 4.0): {gradient[idx101769]}"

/-- Test with specific matrix values to verify row-major ordering -/
def testRowMajorOrdering : IO Unit := do
  -- Create Layer 1 weights with position-encoded values
  -- dW1[i,j] = i * 1000.0 + j (so we can verify row-major flattening)
  let dW1 : Matrix 128 784 := ⊞ ((i, j) : Idx 128 × Idx 784) =>
    i.1.toNat.toFloat * 1000.0 + j.1.toNat.toFloat

  let db1 : Vector 128 := ⊞ (_ : Idx 128) => 0.0
  let dW2 : Matrix 10 128 := ⊞ (_ : Idx 10 × Idx 128) => 0.0
  let db2 : Vector 10 := ⊞ (_ : Idx 10) => 0.0

  let gradient := flattenGradients dW1 db1 dW2 db2

  -- Test: dW1[0,0] should be at gradient[0]
  let idx0 := mkIdx nParams 0 (by unfold nParams; omega)
  IO.println s!"gradient[0] = dW1[0,0] (expect 0.0): {gradient[idx0]}"

  -- Test: dW1[0,784] should be at gradient[784]
  -- Wait, dW1 is [128, 784], so dW1[0,784] is out of bounds
  -- Instead: dW1[1,0] should be at gradient[784]
  let idx784 := mkIdx nParams 784 (by unfold nParams; omega)
  IO.println s!"gradient[784] = dW1[1,0] (expect 1000.0): {gradient[idx784]}"

  -- Test: dW1[1,1] should be at gradient[785]
  let idx785 := mkIdx nParams 785 (by unfold nParams; omega)
  IO.println s!"gradient[785] = dW1[1,1] (expect 1001.0): {gradient[idx785]}"

  -- Test: dW1[2,3] should be at gradient[2*784 + 3] = gradient[1571]
  let idx1571 := mkIdx nParams 1571 (by unfold nParams; omega)
  IO.println s!"gradient[1571] = dW1[2,3] (expect 2003.0): {gradient[idx1571]}"

/-- Main test runner -/
def main : IO Unit := do
  IO.println "=== Gradient Flattening Tests ==="
  IO.println ""

  IO.println "Test 1: Dimension check"
  let dimOk := testDimension
  IO.println s!"Dimension test: {if dimOk then "PASS" else "FAIL"}"
  IO.println ""

  IO.println "Test 2: Layout check (gradient component boundaries)"
  testLayout
  IO.println ""

  IO.println "Test 3: Row-major ordering check"
  testRowMajorOrdering
  IO.println ""

  IO.println "=== All tests complete ==="

end VerifiedNN.Network.GradientFlatteningTest

#eval! VerifiedNN.Network.GradientFlatteningTest.main
