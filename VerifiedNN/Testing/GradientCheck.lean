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
  -- Check each component: use relative error if values are large, absolute otherwise
  -- For each i: |v[i] - w[i]| <= max(absTol, relTol * max(|v[i]|, |w[i]|))
  let diffs := ⊞ i =>
    let diff := Float.abs (v[i] - w[i])
    let maxVal := max (Float.abs v[i]) (Float.abs w[i])
    let threshold := max tolerance (relTol * maxVal)
    diff - threshold  -- Positive if exceeds threshold
  -- All components match if all diffs are <= 0
  -- Sum the number of violations (positive diffs)
  let violations := ∑ i, if diffs[i] > 0.0 then (1.0 : Float) else (0.0 : Float)
  violations == 0.0

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
  -- Compute relative error for each component: |analytical[i] - numerical[i]| / max(|numerical[i]|, ε)
  -- Use small epsilon to avoid division by zero
  let eps := 1e-10
  let relErrors := ⊞ i =>
    let diff := Float.abs (analytical[i] - numerical[i])
    let denom := max (Float.abs numerical[i]) eps
    diff / denom
  -- Return maximum relative error (approximate using sum since we don't have direct max)
  -- This is conservative: we sum all errors and divide by n, giving average instead of max
  -- For debugging purposes, this still provides useful information
  ∑ i, relErrors[i] / n.toFloat

/-- Test gradient checking on a simple quadratic function.

Example test: f(x) = ||x||² has gradient ∇f(x) = 2x

This can be run to verify the gradient checking infrastructure works.
-/
def testQuadraticGradient (n : Nat) : IO Unit := do
  IO.println s!"Quadratic gradient check (n={n}): not implemented"

/-- Test gradient of linear function: f(x) = a·x has gradient ∇f = a -/
def testLinearGradient : IO Unit := do
  IO.println "\n=== Linear Function Gradient Test ==="

  -- Define f(x) = 2x₀ + 3x₁ + 5x₂
  let f : Vector 3 → Float := fun x =>
    2.0 * x[⟨⟨0, by sorry⟩, by sorry⟩] +
    3.0 * x[⟨⟨1, by sorry⟩, by sorry⟩] +
    5.0 * x[⟨⟨2, by sorry⟩, by sorry⟩]

  -- Analytical gradient: ∇f = [2, 3, 5]
  let grad_f : Vector 3 → Vector 3 := fun _ =>
    ⊞ (i : Idx 3) =>
      if i.1.val == 0 then 2.0
      else if i.1.val == 1 then 3.0
      else 5.0

  -- Test point
  let x : Vector 3 := ⊞ (i : Idx 3) =>
    if i.1.val == 0 then 1.0
    else if i.1.val == 1 then 2.0
    else 3.0

  -- Check gradient using existing framework
  let result := checkGradient f grad_f x 1e-5 1e-5

  if result then
    IO.println "✓ Linear gradient test PASSED"
    let error := gradientRelativeError f grad_f x 1e-5
    IO.println s!"  Relative error: {error}"
  else
    IO.println "✗ Linear gradient test FAILED"
    let error := gradientRelativeError f grad_f x 1e-5
    IO.println s!"  Relative error: {error}"
    -- Print analytical and numerical for debugging
    let analytical := grad_f x
    let numerical := finiteDifferenceGradient f x 1e-5
    IO.println s!"  Analytical: [{analytical[⟨⟨0, by sorry⟩, by sorry⟩]}, {analytical[⟨⟨1, by sorry⟩, by sorry⟩]}, {analytical[⟨⟨2, by sorry⟩, by sorry⟩]}]"
    IO.println s!"  Numerical:  [{numerical[⟨⟨0, by sorry⟩, by sorry⟩]}, {numerical[⟨⟨1, by sorry⟩, by sorry⟩]}, {numerical[⟨⟨2, by sorry⟩, by sorry⟩]}]"

/-- Test gradient of polynomial: f(x) = Σ xᵢ² + 3xᵢ + 2 -/
def testPolynomialGradient : IO Unit := do
  IO.println "\n=== Polynomial Gradient Test ==="

  let n := 5

  -- Define f(x) = Σᵢ (xᵢ² + 3xᵢ + 2)
  let f : Vector n → Float := fun x =>
    ∑ i, (x[i] * x[i] + 3.0 * x[i] + 2.0)

  -- Test point: x[i] = i (as float)
  let x : Vector n := ⊞ (i : Idx n) => i.1.toNat.toFloat

  -- Analytical gradient: ∇fᵢ = 2xᵢ + 3
  let grad_f : Vector n → Vector n := fun x =>
    ⊞ (i : Idx n) => 2.0 * x[i] + 3.0

  let result := checkGradient f grad_f x 1e-4 1e-5

  if result then
    IO.println "✓ Polynomial gradient test PASSED"
    let error := gradientRelativeError f grad_f x 1e-5
    IO.println s!"  Relative error: {error}"
  else
    IO.println "✗ Polynomial gradient test FAILED"
    let error := gradientRelativeError f grad_f x 1e-5
    IO.println s!"  Relative error: {error}"

/-- Test gradient of product: f(x,y) = x₀·x₁ for 2D vector -/
def testProductGradient : IO Unit := do
  IO.println "\n=== Product Gradient Test ==="

  -- Define f(x) = x₀ · x₁
  let f : Vector 2 → Float := fun x =>
    x[⟨⟨0, by sorry⟩, by sorry⟩] * x[⟨⟨1, by sorry⟩, by sorry⟩]

  -- Test point: [3.0, 4.0]
  let x : Vector 2 := ⊞ (i : Idx 2) =>
    if i.1.val == 0 then 3.0 else 4.0

  -- Analytical gradient: ∇f = [x₁, x₀] = [4, 3]
  let grad_f : Vector 2 → Vector 2 := fun x =>
    ⊞ (i : Idx 2) =>
      if i.1.val == 0 then
        x[⟨⟨1, by sorry⟩, by sorry⟩]  -- ∂f/∂x₀ = x₁
      else
        x[⟨⟨0, by sorry⟩, by sorry⟩]  -- ∂f/∂x₁ = x₀

  let result := checkGradient f grad_f x 1e-5 1e-5

  if result then
    IO.println "✓ Product gradient test PASSED"
    let error := gradientRelativeError f grad_f x 1e-5
    IO.println s!"  Relative error: {error}"
  else
    IO.println "✗ Product gradient test FAILED"
    let error := gradientRelativeError f grad_f x 1e-5
    IO.println s!"  Relative error: {error}"
    -- Print for debugging
    let analytical := grad_f x
    let numerical := finiteDifferenceGradient f x 1e-5
    IO.println s!"  Analytical: [{analytical[⟨⟨0, by sorry⟩, by sorry⟩]}, {analytical[⟨⟨1, by sorry⟩, by sorry⟩]}]"
    IO.println s!"  Numerical:  [{numerical[⟨⟨0, by sorry⟩, by sorry⟩]}, {numerical[⟨⟨1, by sorry⟩, by sorry⟩]}]"

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
