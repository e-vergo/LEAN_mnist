/-
# Gradient Checking

Numerical validation of automatic differentiation using finite differences.

This module provides utilities for validating that automatic differentiation
computes correct gradients by comparing them against finite difference
approximations.

**Verification Status:** Implementation-level testing only. The finite
difference method itself is not formally verified, but serves as a numerical
sanity check for the symbolic gradient computation.
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
  ⊞ (i : Fin n) =>
    let x_plus := ⊞ (j : Fin n) => if i == j then x[j] + h else x[j]
    let x_minus := ⊞ (j : Fin n) => if i == j then x[j] - h else x[j]
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
  let maxDiff := (Fin n).foldl (init := 0.0) fun maxVal i =>
    let diff := Float.abs (v[i] - w[i])
    let scale := Float.max (Float.abs v[i]) (Float.abs w[i])
    let relativeDiff := if scale > 1.0 then diff / scale else diff
    Float.max maxVal relativeDiff
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
  (Fin n).foldl (init := 0.0) fun maxErr i =>
    let diff := Float.abs (analytical[i] - numerical[i])
    let scale := Float.max (Float.abs analytical[i]) (Float.abs numerical[i])
    let relErr := if scale > 1e-10 then diff / scale else diff
    Float.max maxErr relErr

/-- Test gradient checking on a simple quadratic function.

Example test: f(x) = ||x||² has gradient ∇f(x) = 2x

This can be run to verify the gradient checking infrastructure works.
-/
def testQuadraticGradient (n : Nat) : IO Unit := do
  let f : Vector n → Float := fun x =>
    (Fin n).foldl (init := 0.0) fun sum i => sum + x[i] * x[i]

  let grad_f : Vector n → Vector n := fun x =>
    ⊞ (i : Fin n) => 2.0 * x[i]

  let testPoint : Vector n := ⊞ (i : Fin n) => (i.val.toFloat + 1.0)

  let matches := checkGradient f grad_f testPoint
  let error := gradientRelativeError f grad_f testPoint

  IO.println s!"Quadratic gradient check (n={n}): {matches}"
  IO.println s!"Relative error: {error}"

end VerifiedNN.Testing.GradientCheck
