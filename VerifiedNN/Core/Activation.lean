import SciLean

import VerifiedNN.Core.DataTypes

/-!
# Activation Functions

Activation functions with automatic differentiation support.

## Implementation

All activation functions are implemented to work with SciLean's AD system.
- ReLU uses `if` which may have limitations for AD at the discontinuity (x = 0)
- Softmax uses SciLean's numerically stable implementation with log-sum-exp trick
- Sigmoid and tanh use Float.exp for exponential computation

## Verification Status

- **Differentiation properties:** TODO - Register with @[fun_trans] and @[fun_prop]
- **Analytical derivatives:** Provided separately for gradient checking
- **Numerical correctness:** Validated via gradient checking tests
- **Numerical stability:** Softmax implements max subtraction to prevent overflow

## Known Limitations

- ReLU gradient undefined at x=0 (convention: use 0 as subgradient)
- Tanh/sigmoid use Float.exp which may not be differentiable in SciLean yet
-/

namespace VerifiedNN.Core.Activation

open SciLean
open VerifiedNN.Core

/-- ReLU activation function: `max(0, x)`.

**Gradient at x=0:** By convention, we use 0 (subgradient choice).

**TODO:** Add differentiation properties:
- `@[fun_prop] theorem relu_differentiable` (away from 0)
- `@[fun_trans] theorem relu_fderiv` -/
@[inline]
def relu (x : Float) : Float :=
  if x > 0 then x else 0

/-- Element-wise ReLU on vectors.

Applies ReLU to each element: `y[i] = max(0, x[i])`.

**TODO:** Add differentiation properties. -/
@[inline]
def reluVec {n : Nat} (x : Vector n) : Vector n :=
  ⊞ i => relu x[i]

/-- Element-wise ReLU on batches.

Applies ReLU to each element: `y[k,i] = max(0, x[k,i])`.

**TODO:** Add differentiation properties. -/
@[inline]
def reluBatch {b n : Nat} (x : Batch b n) : Batch b n :=
  ⊞ (k, i) => relu x[k,i]

/-- Numerically stable softmax activation: `exp(x[i] - max(x)) / Σⱼ exp(x[j] - max(x))`.

Implements the log-sum-exp trick to prevent overflow:
1. Compute max value: `m = max(x)`
2. Shift inputs: `x_stable = x - m`
3. Compute: `softmax(x) = exp(x_stable) / Σ exp(x_stable)`

This is mathematically equivalent to the naive implementation but numerically stable.

**Verification Status:**
- **Numerical stability:** Implemented via max subtraction
- **Differentiation properties:** TODO - Register with @[fun_trans] and @[fun_prop]
-/
@[inline]
def softmax {n : Nat} (x : Vector n) : Vector n :=
  -- Use SciLean's built-in numerically stable softmax
  -- which implements the log-sum-exp trick with max subtraction
  DataArrayN.softmax x

/-- Sigmoid activation function: `1 / (1 + exp(-x))`.

**TODO:** Add differentiation properties.
**TODO:** Verify Float.exp is differentiable in SciLean. -/
@[inline]
def sigmoid (x : Float) : Float :=
  1 / (1 + Float.exp (-x))

/-- Element-wise sigmoid on vectors.

Applies sigmoid to each element: `y[i] = 1 / (1 + exp(-x[i]))`.

**TODO:** Add differentiation properties. -/
@[inline]
def sigmoidVec {n : Nat} (x : Vector n) : Vector n :=
  ⊞ i => sigmoid x[i]

/-- Element-wise sigmoid on batches.

Applies sigmoid to each element: `y[k,i] = 1 / (1 + exp(-x[k,i]))`.

**TODO:** Add differentiation properties. -/
@[inline]
def sigmoidBatch {b n : Nat} (x : Batch b n) : Batch b n :=
  ⊞ (k, i) => sigmoid x[k,i]

/-- Leaky ReLU activation with slope `alpha` for negative values.

**TODO:** Add differentiation properties. -/
@[inline]
def leakyRelu (alpha : Float := 0.01) (x : Float) : Float :=
  if x > 0 then x else alpha * x

/-- Element-wise Leaky ReLU on vectors.

Applies leaky ReLU to each element: `y[i] = x[i]` if `x[i] > 0` else `alpha * x[i]`.

**TODO:** Add differentiation properties. -/
@[inline]
def leakyReluVec {n : Nat} (alpha : Float := 0.01) (x : Vector n) : Vector n :=
  ⊞ i => leakyRelu alpha x[i]

/-- Tanh activation function.

**TODO:** Add differentiation properties.
**TODO:** Verify Float.exp is differentiable in SciLean. -/
@[inline]
def tanh (x : Float) : Float :=
  let expPos := Float.exp x
  let expNeg := Float.exp (-x)
  (expPos - expNeg) / (expPos + expNeg)

/-- Element-wise tanh on vectors.

Applies tanh to each element: `y[i] = tanh(x[i])`.

**TODO:** Add differentiation properties. -/
@[inline]
def tanhVec {n : Nat} (x : Vector n) : Vector n :=
  ⊞ i => tanh x[i]

/-! ## Analytical Derivatives

These are separate from the AD system and used for numerical validation via gradient checking.
Once @[fun_trans] properties are proven, SciLean will use its own AD. -/

/-- Analytical derivative of ReLU: `1` if `x > 0`, `0` otherwise.

Convention at x=0: use 0 (subgradient choice). -/
def reluDerivative (x : Float) : Float :=
  if x > 0 then 1 else 0

/-- Analytical derivative of sigmoid: `σ(x) * (1 - σ(x))`. -/
def sigmoidDerivative (x : Float) : Float :=
  let s := sigmoid x
  s * (1 - s)

/-- Analytical derivative of tanh: `1 - tanh²(x)`. -/
def tanhDerivative (x : Float) : Float :=
  let t := tanh x
  1 - t * t

/-- Analytical derivative of leaky ReLU: `1` if `x > 0`, `alpha` otherwise. -/
def leakyReluDerivative (alpha : Float := 0.01) (x : Float) : Float :=
  if x > 0 then 1 else alpha

end VerifiedNN.Core.Activation
