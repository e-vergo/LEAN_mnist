/-
# Gradient Checking

Numerical validation of automatic differentiation using finite differences.

This module provides utilities for validating that automatic differentiation
computes correct gradients by comparing them against finite difference
approximations.

**Verification Status:** Implementation-level testing only. The finite
difference method itself is not formally verified, but serves as a numerical
sanity check for the symbolic gradient computation.

## Test Coverage

This module provides:
- ✓ Finite difference gradient approximation (central differences)
- ✓ Vector approximate equality with relative/absolute tolerance
- ✓ Gradient correctness checking against numerical approximation
- ✓ Gradient relative error computation for debugging
- ✓ Test suite for basic functions: quadratic, linear, polynomial, product
- ✓ All tests use IO-based test framework (no LSpec dependency)

## Functions Tested

| Test | Function | Gradient | Status |
|------|----------|----------|--------|
| testQuadraticGradient | f(x) = ‖x‖² | ∇f(x) = 2x | ✓ Implemented |
| testLinearGradient | f(x) = a·x | ∇f(x) = a | ✓ Implemented |
| testPolynomialGradient | f(x) = Σ(xᵢ² + 3xᵢ + 2) | ∇f(x) = 2x + 3 | ✓ Implemented |
| testProductGradient | f(x₀,x₁) = x₀·x₁ | ∇f = (x₁, x₀) | ✓ Implemented |

## Usage

```bash
# Build
lake build VerifiedNN.Testing.GradientCheck

# Run tests (once Network.Gradient is implemented)
lake env lean --run VerifiedNN/Testing/GradientCheck.lean
```
-/

import VerifiedNN.Core.DataTypes
import VerifiedNN.Network.Gradient
import SciLean

namespace VerifiedNN.Testing.GradientCheck

open VerifiedNN.Core
open SciLean

/-- Compute finite difference approximation of gradient using central differences.

Central difference formula: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)

This provides O(h²) accuracy compared to O(h) for forward differences.

**Parameters:**
- `f`: Scalar-valued function on n-dimensional vectors
- `x`: Point at which to approximate gradient
- `h`: Step size for finite differences (default 1e-5)

**Returns:** Approximate gradient vector of dimension n

**Note:** This is a numerical approximation, not a verified computation.
-/
def finiteDifferenceGradient {n : Nat}
    (f : Vector n → Float)
    (x : Vector n)
    (h : Float := 1e-5) : Vector n :=
  -- For each dimension i, compute (f(x + h*e_i) - f(x - h*e_i)) / (2h)
  -- where e_i is the i-th unit vector
  ⊞ i =>
    let x_plus := ⊞ j => if i == j then x[j] + h else x[j]
    let x_minus := ⊞ j => if i == j then x[j] - h else x[j]
    (f x_plus - f x_minus) / (2 * h)

/-- Check if two vectors are approximately equal within tolerance.

Uses relative error when values are large, absolute error when small.

**Parameters:**
- `v`: First vector
- `w`: Second vector
- `tolerance`: Maximum allowed difference per component
- `relTol`: Relative tolerance for large values (default 1e-5)

**Returns:** True if all components match within tolerance
-/
def vectorsApproxEq {n : Nat}
    (v w : Vector n)
    (tolerance : Float := 1e-5)
    (relTol : Float := 1e-5) : Bool :=
  let maxDiff := IndexType.foldl (fun maxVal i =>
    let diff := Float.abs (v[i] - w[i])
    let scale := Float.max (Float.abs v[i]) (Float.abs w[i])
    let relativeDiff := if scale > 1.0 then diff / scale else diff
    Float.max maxVal relativeDiff) 0.0 (Fin n)
  maxDiff ≤ tolerance || maxDiff ≤ relTol

/-- Check if automatic gradient matches finite difference approximation.

Computes both the analytical gradient (via automatic differentiation) and
numerical gradient (via finite differences), then compares them.

**Parameters:**
- `f`: Scalar-valued function to differentiate
- `grad_f`: Analytical gradient function (should be `∇ f`)
- `x`: Point at which to check gradient
- `tolerance`: Maximum allowed difference (default 1e-5)
- `h`: Step size for finite differences (default 1e-5)

**Returns:** True if gradients match within tolerance

**Usage Example:**
```lean
def myFunc (x : Vector 3) : Float := x[0] * x[0] + x[1] * x[2]
def myGrad := ∇ myFunc
let testPoint : Vector 3 := ⊞[1.0, 2.0, 3.0]
#eval checkGradient myFunc myGrad testPoint
```
-/
def checkGradient {n : Nat}
    (f : Vector n → Float)
    (grad_f : Vector n → Vector n)
    (x : Vector n)
    (tolerance : Float := 1e-5)
    (h : Float := 1e-5) : Bool :=
  let analytical := grad_f x
  let numerical := finiteDifferenceGradient f x h
  vectorsApproxEq analytical numerical tolerance

/-- Compute relative error between analytical and numerical gradients.

**Parameters:**
- `f`: Scalar-valued function
- `grad_f`: Analytical gradient function
- `x`: Point at which to compute error
- `h`: Step size for finite differences

**Returns:** Maximum relative error across all dimensions

**Note:** Useful for debugging when gradients don't match - tells you
how far off they are.
-/
def gradientRelativeError {n : Nat}
    (f : Vector n → Float)
    (grad_f : Vector n → Vector n)
    (x : Vector n)
    (h : Float := 1e-5) : Float :=
  let analytical := grad_f x
  let numerical := finiteDifferenceGradient f x h
  IndexType.foldl (fun maxErr i =>
    let diff := Float.abs (analytical[i] - numerical[i])
    let scale := Float.max (Float.abs analytical[i]) (Float.abs numerical[i])
    let relErr := if scale > 1e-10 then diff / scale else diff
    Float.max maxErr relErr) 0.0 (Fin n)

/-- Test gradient checking on a simple quadratic function.

Example test: f(x) = ||x||² has gradient ∇f(x) = 2x

This can be run to verify the gradient checking infrastructure works.
-/
def testQuadraticGradient (n : Nat) : IO Unit := do
  let f : Vector n → Float := fun x =>
    IndexType.foldl (fun sum i => sum + x[i] * x[i]) 0.0 (Fin n)

  let grad_f : Vector n → Vector n := fun x =>
    ⊞ i => 2.0 * x[i]

  let testPoint : Vector n := ⊞ i => (i.val.toFloat + 1.0)

  let matches := checkGradient f grad_f testPoint
  let error := gradientRelativeError f grad_f testPoint

  IO.println s!"Quadratic gradient check (n={n}): {matches}"
  IO.println s!"Relative error: {error}"

/-- Test gradient of linear function: f(x) = a·x has gradient ∇f = a -/
def testLinearGradient : IO Unit := do
  IO.println "\n=== Linear Function Gradient Test ==="
  let n := 5
  let a : Vector n := ⊞ i => (i.val.toFloat + 1.0) * 2.0

  let f : Vector n → Float := fun x =>
    IndexType.foldl (fun sum i => sum + a[i] * x[i]) 0.0 (Fin n)

  let grad_f : Vector n → Vector n := fun _ => a

  let testPoint : Vector n := ⊞ i => (i.val.toFloat + 1.0) * 0.5

  let matches := checkGradient f grad_f testPoint
  let error := gradientRelativeError f grad_f testPoint

  IO.println s!"Linear gradient check: {matches}"
  IO.println s!"Relative error: {error}"

  if matches then
    IO.println "✓ Linear gradient test PASSED"
  else
    IO.println "✗ Linear gradient test FAILED"

/-- Test gradient of polynomial: f(x) = Σ xᵢ² + 3xᵢ + 2 -/
def testPolynomialGradient : IO Unit := do
  IO.println "\n=== Polynomial Gradient Test ==="
  let n := 4

  let f : Vector n → Float := fun x =>
    IndexType.foldl (fun sum i =>
      sum + x[i] * x[i] + 3.0 * x[i] + 2.0) 0.0 (Fin n)

  -- ∇f(x) = 2x + 3
  let grad_f : Vector n → Vector n := fun x =>
    ⊞ i => 2.0 * x[i] + 3.0

  let testPoint : Vector n := ⊞ i => (i.val.toFloat - 2.0)

  let matches := checkGradient f grad_f testPoint
  let error := gradientRelativeError f grad_f testPoint

  IO.println s!"Polynomial gradient check: {matches}"
  IO.println s!"Relative error: {error}"

  if matches then
    IO.println "✓ Polynomial gradient test PASSED"
  else
    IO.println "✗ Polynomial gradient test FAILED"

/-- Test gradient of product: f(x,y) = x₀·x₁ for 2D vector -/
def testProductGradient : IO Unit := do
  IO.println "\n=== Product Gradient Test ==="
  let n := 2

  let f : Vector n → Float := fun x =>
    x[⟨0, by omega⟩] * x[⟨1, by omega⟩]

  -- ∇(x₀·x₁) = (x₁, x₀)
  let grad_f : Vector n → Vector n := fun x =>
    ⊞ i =>
      if i.val == 0 then x[⟨1, by omega⟩]
      else x[⟨0, by omega⟩]

  let testPoint : Vector n := ⊞[3.0, 4.0]

  let matches := checkGradient f grad_f testPoint
  let error := gradientRelativeError f grad_f testPoint

  IO.println s!"Product gradient check: {matches}"
  IO.println s!"Relative error: {error}"

  if matches then
    IO.println "✓ Product gradient test PASSED"
  else
    IO.println "✗ Product gradient test FAILED"

/-- Run all gradient check tests -/
def runAllGradientTests : IO Unit := do
  IO.println "=========================================="
  IO.println "Running Gradient Check Tests"
  IO.println "=========================================="

  testQuadraticGradient 3
  testQuadraticGradient 10
  testLinearGradient
  testPolynomialGradient
  testProductGradient

  IO.println "\n=========================================="
  IO.println "Gradient Check Tests Complete"
  IO.println "=========================================="

end VerifiedNN.Testing.GradientCheck
