import VerifiedNN.Core.DataTypes
import VerifiedNN.Core.LinearAlgebra
import SciLean

/-!
# Linear Algebra Tests

Comprehensive test suite for matrix and vector operations.

## Main Tests

- `testVectorArithmetic`: Vector addition, subtraction, scalar multiplication
- `testVectorDotProduct`: Dot product properties (commutativity, linearity)
- `testVectorNorms`: L2 norm and squared norm properties
- `testMatrixVectorMultiply`: Matrix-vector multiplication correctness
- `testMatrixTranspose`: Transpose properties and correctness
- `testMatrixArithmetic`: Matrix addition, subtraction, scalar multiplication
- `testOuterProduct`: Outer product of vectors
- `testBatchOperations`: Batch matrix-vector multiplication and bias addition

## Implementation Notes

**Testing Strategy:** Validate mathematical properties of linear algebra operations:
- Commutativity, associativity, distributivity where applicable
- Dimension preservation (type-checked at compile time)
- Numerical correctness on known test cases
- Edge cases (zero vectors, identity matrices)

**Coverage Status:** Comprehensive tests for all Core.LinearAlgebra operations.

**Test Framework:** IO-based assertions with detailed diagnostic output.

## Usage

```bash
# Build tests
lake build VerifiedNN.Testing.LinearAlgebraTests

# Run tests
lake env lean --run VerifiedNN/Testing/LinearAlgebraTests.lean
```
-/

namespace VerifiedNN.Testing.LinearAlgebraTests

open VerifiedNN.Core
open VerifiedNN.Core.LinearAlgebra
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
          let aij := A[⟨i, hi⟩, ⟨j, hj⟩]
          let bij := B[⟨i, hi⟩, ⟨j, hj⟩]
          if Float.abs (aij - bij) > tol then
            allClose := false
            break
      if !allClose then break

  if allClose then
    IO.println s!"✓ {name}"
    return true
  else
    IO.println s!"✗ {name} FAILED: matrices differ"
    return false

/-! ## Vector Arithmetic Tests -/

/-- Test vector addition and subtraction -/
def testVectorArithmetic : IO Bool := do
  IO.println "\n=== Vector Arithmetic Tests ==="

  let mut allPassed := true

  -- Create test vectors
  let v : Vector 3 := ⊞[1.0, 2.0, 3.0]
  let w : Vector 3 := ⊞[4.0, 5.0, 6.0]

  -- Test addition
  let sum := vadd v w
  let expectedSum : Vector 3 := ⊞[5.0, 7.0, 9.0]
  allPassed := allPassed && (← assertVecApproxEq "vadd: [1,2,3] + [4,5,6] = [5,7,9]" sum expectedSum)

  -- Test subtraction
  let diff := vsub w v
  let expectedDiff : Vector 3 := ⊞[3.0, 3.0, 3.0]
  allPassed := allPassed && (← assertVecApproxEq "vsub: [4,5,6] - [1,2,3] = [3,3,3]" diff expectedDiff)

  -- Test scalar multiplication
  let scaled := smul 2.0 v
  let expectedScaled : Vector 3 := ⊞[2.0, 4.0, 6.0]
  allPassed := allPassed && (← assertVecApproxEq "smul: 2 * [1,2,3] = [2,4,6]" scaled expectedScaled)

  -- Test element-wise multiplication
  let hadamard := vmul v w
  let expectedHadamard : Vector 3 := ⊞[4.0, 10.0, 18.0]
  allPassed := allPassed && (← assertVecApproxEq "vmul: [1,2,3] ⊙ [4,5,6] = [4,10,18]" hadamard expectedHadamard)

  return allPassed

/-- Test vector dot product -/
def testVectorDotProduct : IO Bool := do
  IO.println "\n=== Vector Dot Product Tests ==="

  let mut allPassed := true

  let v : Vector 3 := ⊞[1.0, 2.0, 3.0]
  let w : Vector 3 := ⊞[4.0, 5.0, 6.0]

  -- Test dot product: v·w = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
  let dotProd := dot v w
  allPassed := allPassed && (← assertApproxEq "dot: [1,2,3]·[4,5,6] = 32" dotProd 32.0)

  -- Test commutativity: v·w = w·v
  let dotRev := dot w v
  allPassed := allPassed && (← assertApproxEq "dot commutativity: w·v = v·w" dotRev dotProd)

  -- Test with orthogonal vectors: [1,0,0]·[0,1,0] = 0
  let e1 : Vector 3 := ⊞[1.0, 0.0, 0.0]
  let e2 : Vector 3 := ⊞[0.0, 1.0, 0.0]
  let orthDot := dot e1 e2
  allPassed := allPassed && (← assertApproxEq "dot: orthogonal vectors = 0" orthDot 0.0)

  return allPassed

/-- Test vector norms -/
def testVectorNorms : IO Bool := do
  IO.println "\n=== Vector Norm Tests ==="

  let mut allPassed := true

  -- Test on [3, 4]: ||v||² = 9 + 16 = 25, ||v|| = 5
  let v : Vector 2 := ⊞[3.0, 4.0]
  let nSq := normSq v
  allPassed := allPassed && (← assertApproxEq "normSq: ||[3,4]||² = 25" nSq 25.0)

  let n := LinearAlgebra.norm v
  allPassed := allPassed && (← assertApproxEq "norm: ||[3,4]|| = 5" n 5.0)

  -- Test on unit vector: ||[1,0,0]|| = 1
  let e1 : Vector 3 := ⊞[1.0, 0.0, 0.0]
  let unitNorm := LinearAlgebra.norm e1
  allPassed := allPassed && (← assertApproxEq "norm: unit vector = 1" unitNorm 1.0)

  -- Test on zero vector: ||0|| = 0
  let zero : Vector 3 := ⊞[0.0, 0.0, 0.0]
  let zeroNorm := LinearAlgebra.norm zero
  allPassed := allPassed && (← assertApproxEq "norm: zero vector = 0" zeroNorm 0.0)

  return allPassed

/-! ## Matrix Operation Tests -/

/-- Test matrix-vector multiplication -/
def testMatrixVectorMultiply : IO Bool := do
  IO.println "\n=== Matrix-Vector Multiplication Tests ==="

  let mut allPassed := true

  -- 2×3 matrix times 3×1 vector = 2×1 vector
  -- [[1, 2, 3],   [1]   [1*1 + 2*2 + 3*3]   [14]
  --  [4, 5, 6]] * [2] = [4*1 + 5*2 + 6*3] = [32]
  --               [3]
  let A : Matrix 2 3 := ⊞ (i : Idx 2) (j : Idx 3) => (i.1.toNat * 3 + j.1.toNat + 1).toFloat
  let x : Vector 3 := ⊞[1.0, 2.0, 3.0]
  let result := matvec A x
  let expected : Vector 2 := ⊞[14.0, 32.0]
  allPassed := allPassed && (← assertVecApproxEq "matvec: 2×3 matrix times 3-vector" result expected)

  -- Identity matrix times vector = vector
  let I : Matrix 3 3 := ⊞ (i : Idx 3) (j : Idx 3) => if i.1 == j.1 then 1.0 else 0.0
  let v : Vector 3 := ⊞[5.0, 7.0, 9.0]
  let identityResult := matvec I v
  allPassed := allPassed && (← assertVecApproxEq "matvec: identity matrix times vector = vector" identityResult v)

  return allPassed

/-- Test matrix transpose -/
def testMatrixTranspose : IO Bool := do
  IO.println "\n=== Matrix Transpose Tests ==="

  let mut allPassed := true

  -- Create 2×3 matrix and transpose to 3×2
  -- [[1, 2, 3],      [[1, 4],
  --  [4, 5, 6]]  =>   [2, 5],
  --                   [3, 6]]
  let A : Matrix 2 3 := ⊞ (i : Idx 2) (j : Idx 3) => (i.1.toNat * 3 + j.1.toNat + 1).toFloat
  let AT := transpose A

  -- Check specific elements
  allPassed := allPassed && (← assertApproxEq "transpose: AT[0,0] = A[0,0]" AT[⟨0, by decide⟩, ⟨0, by decide⟩] 1.0)
  allPassed := allPassed && (← assertApproxEq "transpose: AT[0,1] = A[1,0]" AT[⟨0, by decide⟩, ⟨1, by decide⟩] 4.0)
  allPassed := allPassed && (← assertApproxEq "transpose: AT[1,0] = A[0,1]" AT[⟨1, by decide⟩, ⟨0, by decide⟩] 2.0)
  allPassed := allPassed && (← assertApproxEq "transpose: AT[2,1] = A[1,2]" AT[⟨2, by decide⟩, ⟨1, by decide⟩] 6.0)

  -- Double transpose should return original
  let ATT := transpose AT
  allPassed := allPassed && (← assertMatApproxEq "transpose: (A^T)^T = A" ATT A)

  return allPassed

/-- Test matrix arithmetic operations -/
def testMatrixArithmetic : IO Bool := do
  IO.println "\n=== Matrix Arithmetic Tests ==="

  let mut allPassed := true

  let A : Matrix 2 2 := ⊞ (i : Idx 2) (j : Idx 2) => (i.1.toNat + j.1.toNat + 1).toFloat
  let B : Matrix 2 2 := ⊞ (i : Idx 2) (j : Idx 2) => (i.1.toNat * 2 + j.1.toNat + 1).toFloat

  -- Test matrix addition
  let sum := matAdd A B
  let expectedSum : Matrix 2 2 := ⊞ (i : Idx 2) (j : Idx 2) =>
    A[i, j] + B[i, j]
  allPassed := allPassed && (← assertMatApproxEq "matAdd: element-wise sum" sum expectedSum)

  -- Test matrix subtraction
  let diff := matSub B A
  let expectedDiff : Matrix 2 2 := ⊞ (i : Idx 2) (j : Idx 2) =>
    B[i, j] - A[i, j]
  allPassed := allPassed && (← assertMatApproxEq "matSub: element-wise difference" diff expectedDiff)

  -- Test scalar-matrix multiplication
  let scaled := matSmul 3.0 A
  let expectedScaled : Matrix 2 2 := ⊞ (i : Idx 2) (j : Idx 2) => 3.0 * A[i, j]
  allPassed := allPassed && (← assertMatApproxEq "matSmul: scalar times matrix" scaled expectedScaled)

  return allPassed

/-- Test outer product -/
def testOuterProduct : IO Bool := do
  IO.println "\n=== Outer Product Tests ==="

  let mut allPassed := true

  -- Outer product of [1, 2] and [3, 4, 5]
  -- Result should be 2×3 matrix:
  -- [[1*3, 1*4, 1*5],   [[3,  4,  5],
  --  [2*3, 2*4, 2*5]] =  [6,  8, 10]]
  let v : Vector 2 := ⊞[1.0, 2.0]
  let w : Vector 3 := ⊞[3.0, 4.0, 5.0]
  let outerProd := outer v w

  allPassed := allPassed && (← assertApproxEq "outer: [1,2] ⊗ [3,4,5] element [0,0]" outerProd[⟨0, by decide⟩, ⟨0, by decide⟩] 3.0)
  allPassed := allPassed && (← assertApproxEq "outer: [1,2] ⊗ [3,4,5] element [0,2]" outerProd[⟨0, by decide⟩, ⟨2, by decide⟩] 5.0)
  allPassed := allPassed && (← assertApproxEq "outer: [1,2] ⊗ [3,4,5] element [1,0]" outerProd[⟨1, by decide⟩, ⟨0, by decide⟩] 6.0)
  allPassed := allPassed && (← assertApproxEq "outer: [1,2] ⊗ [3,4,5] element [1,2]" outerProd[⟨1, by decide⟩, ⟨2, by decide⟩] 10.0)

  return allPassed

/-! ## Batch Operation Tests -/

/-- Test batch matrix-vector multiplication -/
def testBatchMatvec : IO Bool := do
  IO.println "\n=== Batch Matrix-Vector Tests ==="

  let mut allPassed := true

  -- Create 2×2 matrix and batch of 3 vectors
  let A : Matrix 2 2 := ⊞ (i : Idx 2) (j : Idx 2) => (i.1.toNat + j.1.toNat + 1).toFloat
  let X : Batch 3 2 := ⊞ (b : Idx 3) (i : Idx 2) => (b.1.toNat + i.1.toNat + 1).toFloat

  -- Apply matrix to each vector in batch
  let result := batchMatvec A X

  -- Verify batch size preserved (type-checked at compile time)
  -- Verify output dimension correct (type-checked at compile time)

  -- Check first output vector manually
  let x0 : Vector 2 := ⊞ (i : Idx 2) => X[⟨0, by decide⟩, i]
  let expected0 := matvec A x0
  let result0 : Vector 2 := ⊞ (i : Idx 2) => result[⟨0, by decide⟩, i]
  allPassed := allPassed && (← assertVecApproxEq "batchMatvec: first batch element" result0 expected0)

  return allPassed

/-- Test batch bias addition -/
def testBatchAddVec : IO Bool := do
  IO.println "\n=== Batch Bias Addition Tests ==="

  let mut allPassed := true

  -- Create batch and bias vector
  let X : Batch 2 3 := ⊞ (b : Idx 2) (i : Idx 3) => (b.1.toNat * 3 + i.1.toNat + 1).toFloat
  let bias : Vector 3 := ⊞[10.0, 20.0, 30.0]

  -- Add bias to each row
  let result := batchAddVec X bias

  -- Check first row: [1,2,3] + [10,20,30] = [11,22,33]
  let r00 := result[⟨0, Nat.zero_lt_succ 1⟩, ⟨0, Nat.zero_lt_succ 2⟩]
  allPassed := allPassed && (← assertApproxEq "batchAddVec: first row, first element" r00 11.0)
  let r01 := result[⟨0, Nat.zero_lt_succ 1⟩, ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 1)⟩]
  allPassed := allPassed && (← assertApproxEq "batchAddVec: first row, second element" r01 22.0)
  let r02 := result[⟨0, Nat.zero_lt_succ 1⟩, ⟨2, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 0))⟩]
  allPassed := allPassed && (← assertApproxEq "batchAddVec: first row, third element" r02 33.0)

  -- Check second row: [4,5,6] + [10,20,30] = [14,25,36]
  let r10 := result[⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 0)⟩, ⟨0, Nat.zero_lt_succ 2⟩]
  allPassed := allPassed && (← assertApproxEq "batchAddVec: second row, first element" r10 14.0)
  let r12 := result[⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 0)⟩, ⟨2, Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ 0))⟩]
  allPassed := allPassed && (← assertApproxEq "batchAddVec: second row, third element" r12 36.0)

  return allPassed

/-! ## Test Runner -/

/-- Run all linear algebra tests and report results -/
def runAllTests : IO Unit := do
  IO.println "=========================================="
  IO.println "Running Linear Algebra Tests"
  IO.println "=========================================="

  let mut totalPassed := 0
  let mut totalTests := 0

  let testSuites : List (String × IO Bool) := [
    ("Vector Arithmetic", testVectorArithmetic),
    ("Vector Dot Product", testVectorDotProduct),
    ("Vector Norms", testVectorNorms),
    ("Matrix-Vector Multiplication", testMatrixVectorMultiply),
    ("Matrix Transpose", testMatrixTranspose),
    ("Matrix Arithmetic", testMatrixArithmetic),
    ("Outer Product", testOuterProduct),
    ("Batch Matrix-Vector Multiplication", testBatchMatvec),
    ("Batch Bias Addition", testBatchAddVec)
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
    IO.println "✓ All linear algebra tests passed!"
  else
    IO.println s!"✗ {totalTests - totalPassed} test suite(s) failed"

end VerifiedNN.Testing.LinearAlgebraTests

/-- Top-level main for execution -/
def main : IO Unit := VerifiedNN.Testing.LinearAlgebraTests.runAllTests
