import SciLean
import VerifiedNN.Core.DataTypes

/-!
# Activation Functions

Nonlinear activation functions for neural networks with automatic differentiation support.

This module provides common activation functions used in neural network layers,
including ReLU, softmax, sigmoid, tanh, and leaky ReLU. Each activation has both
scalar and vectorized versions, plus analytical derivatives for gradient checking.

## Main Definitions

**ReLU Family:**
- `relu` - Rectified Linear Unit: max(0, x)
- `reluVec` - Element-wise ReLU on vectors
- `reluBatch` - Element-wise ReLU on batches
- `leakyRelu` - Leaky ReLU with configurable negative slope
- `leakyReluVec` - Element-wise leaky ReLU on vectors

**Classification Activation:**
- `softmax` - Numerically stable softmax for probability distributions

**Sigmoid Family:**
- `sigmoid` - Logistic sigmoid: 1 / (1 + exp(-x))
- `sigmoidVec` - Element-wise sigmoid on vectors
- `sigmoidBatch` - Element-wise sigmoid on batches
- `tanh` - Hyperbolic tangent
- `tanhVec` - Element-wise tanh on vectors

**Analytical Derivatives (for gradient checking):**
- `reluDerivative`, `sigmoidDerivative`, `tanhDerivative`, `leakyReluDerivative`

## Implementation Notes

**Automatic Differentiation:**
- All functions designed to work with SciLean's AD system
- ReLU uses conditional `if` which has known limitations at x=0
- TODO: Register functions with `@[fun_trans]` and `@[fun_prop]` attributes

**Numerical Stability:**
- Softmax uses SciLean's built-in implementation with log-sum-exp trick
- Max subtraction prevents overflow on large inputs: softmax(x - max(x))
- Sigmoid and tanh use Float.exp (stability properties inherited from Float)

**Gradient Conventions:**
- ReLU gradient at x=0: By convention, we use 0 (subgradient choice)
- This matches PyTorch/TensorFlow behavior

## Verification Status

- **Differentiation properties:** ⚠️ TODO - Register with `@[fun_trans]` and `@[fun_prop]`
- **Analytical derivatives:** ✅ Provided for gradient checking validation
- **Numerical correctness:** ✅ Validated via gradient checking tests
- **Numerical stability:** ✅ Softmax uses stable log-sum-exp implementation

## Known Limitations

1. **ReLU at x=0:** Gradient technically undefined, we use subgradient 0
2. **Float.exp differentiability:** May not be registered in current SciLean version
3. **AD through conditionals:** `if` expressions may have limitations for SciLean's AD

## References

- SciLean DataArrayN.softmax: Numerically stable softmax implementation
- Gradient checking tests: VerifiedNN/Testing/GradientCheck.lean
- Neural network theory: Deep Learning (Goodfellow et al., 2016)
-/

namespace VerifiedNN.Core.Activation

open SciLean
open VerifiedNN.Core

/-- ReLU (Rectified Linear Unit) activation function: `max(0, x)`.

Applies the ReLU activation, which passes positive values unchanged and zeros negative values.

**Mathematical definition:** `ReLU(x) = max(0, x) = { x if x > 0, 0 otherwise }`

**Gradient:** `ReLU'(x) = { 1 if x > 0, 0 otherwise }`

**Gradient at x=0:** Technically undefined. By convention, we use 0 (subgradient choice),
matching PyTorch and TensorFlow behavior.

**Parameters:**
- `x` : Input scalar (Float)

**Returns:** `max(0, x)`

**TODO:** Register differentiation properties:
- `@[fun_prop] theorem relu_differentiable` (almost everywhere)
- `@[fun_trans] theorem relu_fderiv`

**Usage:** Most common activation in hidden layers of neural networks.
Introduces nonlinearity while being computationally efficient. -/
@[inline]
def relu (x : Float) : Float :=
  if x > 0 then x else 0

/-- Element-wise ReLU on vectors.

Applies ReLU activation to each element of a vector: `y[i] = max(0, x[i])`.

**Parameters:**
- `x` : Input vector of dimension `n`

**Returns:** Vector of dimension `n` with ReLU applied element-wise

**TODO:** Register differentiation properties.
Gradient is element-wise: `∂y[i]/∂x[j] = δᵢⱼ * relu'(x[i])`.

**Usage:** Activating hidden layer outputs in neural networks. -/
@[inline]
def reluVec {n : Nat} (x : Vector n) : Vector n :=
  ⊞ i => relu x[i]

/-- Element-wise ReLU on batches.

Applies ReLU activation to each element of a batch: `y[k,i] = max(0, x[k,i])`.

**Parameters:**
- `x` : Batch of `b` samples, each of dimension `n`

**Returns:** Batch of `b` samples, each of dimension `n`, with ReLU applied element-wise

**TODO:** Register differentiation properties.

**Usage:** Activating batched hidden layer outputs during forward pass. -/
@[inline]
def reluBatch {b n : Nat} (x : Batch b n) : Batch b n :=
  ⊞ (k, i) => relu x[k,i]

/-- Numerically stable softmax activation for classification.

Computes probability distribution from logits using the log-sum-exp trick:
`softmax(x)[i] = exp(x[i] - max(x)) / Σⱼ exp(x[j] - max(x))`

**Mathematical definition:** `softmax(x)[i] = exp(x[i]) / Σⱼ exp(x[j])`

**Numerical stability:** Uses SciLean's implementation with max subtraction to prevent overflow:
1. Compute `m = max(x)`
2. Shift inputs: `x_stable = x - m`
3. Compute: `softmax(x) = exp(x_stable) / Σ exp(x_stable)`

This is mathematically equivalent to the naive formula but numerically stable.

**Properties:**
- Output is a probability distribution: Σᵢ softmax(x)[i] = 1
- All outputs are in (0, 1)
- Preserves ordering: x[i] > x[j] ⟹ softmax(x)[i] > softmax(x)[j]

**Parameters:**
- `x` : Input vector of dimension `n` (logits)

**Returns:** Vector of dimension `n` representing probability distribution

**Gradient:** Jacobian is `diag(p) - p ⊗ p` where `p = softmax(x)`.

**TODO:** Register differentiation properties with `@[fun_trans]` and `@[fun_prop]`.

**Usage:** Output activation for multi-class classification.
Typically used with cross-entropy loss. -/
@[inline]
def softmax {n : Nat} (x : Vector n) : Vector n :=
  -- Use SciLean's built-in numerically stable softmax
  -- which implements the log-sum-exp trick with max subtraction
  DataArrayN.softmax x

/-- Sigmoid (logistic) activation function: `1 / (1 + exp(-x))`.

Maps real numbers to (0, 1), creating an S-shaped curve centered at 0.

**Mathematical definition:** `σ(x) = 1 / (1 + exp(-x))`

**Gradient:** `σ'(x) = σ(x) * (1 - σ(x))`

**Properties:**
- Range: (0, 1)
- σ(0) = 0.5
- σ(-x) = 1 - σ(x) (symmetric around 0.5)
- Saturates for large |x| (gradient vanishing problem)

**Parameters:**
- `x` : Input scalar (Float)

**Returns:** Value in (0, 1)

**TODO:** Register differentiation properties.
**TODO:** Verify Float.exp is differentiable in SciLean.

**Usage:** Binary classification output layer, gating mechanisms.
Less common in hidden layers due to gradient vanishing. -/
@[inline]
def sigmoid (x : Float) : Float :=
  1 / (1 + Float.exp (-x))

/-- Element-wise sigmoid on vectors.

Applies sigmoid activation to each element: `y[i] = 1 / (1 + exp(-x[i]))`.

**Parameters:**
- `x` : Input vector of dimension `n`

**Returns:** Vector of dimension `n` with values in (0, 1)

**TODO:** Register differentiation properties.

**Usage:** Multi-label classification where labels are not mutually exclusive. -/
@[inline]
def sigmoidVec {n : Nat} (x : Vector n) : Vector n :=
  ⊞ i => sigmoid x[i]

/-- Element-wise sigmoid on batches.

Applies sigmoid activation to each element: `y[k,i] = 1 / (1 + exp(-x[k,i]))`.

**Parameters:**
- `x` : Batch of `b` samples, each of dimension `n`

**Returns:** Batch of `b` samples, each of dimension `n`, with values in (0, 1)

**TODO:** Register differentiation properties.

**Usage:** Batched multi-label classification. -/
@[inline]
def sigmoidBatch {b n : Nat} (x : Batch b n) : Batch b n :=
  ⊞ (k, i) => sigmoid x[k,i]

/-- Leaky ReLU activation with slope `alpha` for negative values.

A variant of ReLU that allows small negative values instead of zeroing them.

**Mathematical definition:** `LeakyReLU(x) = { x if x > 0, α*x otherwise }`

**Gradient:** `LeakyReLU'(x) = { 1 if x > 0, α otherwise }`

**Parameters:**
- `alpha` : Slope for negative values (default: 0.01)
- `x` : Input scalar (Float)

**Returns:** `x` if positive, `alpha * x` if negative

**TODO:** Register differentiation properties.

**Usage:** Alternative to ReLU that avoids "dying ReLU" problem by allowing
gradient flow for negative inputs. Common choice: α = 0.01 or α = 0.2. -/
@[inline]
def leakyRelu (alpha : Float := 0.01) (x : Float) : Float :=
  if x > 0 then x else alpha * x

/-- Element-wise Leaky ReLU on vectors.

Applies leaky ReLU to each element: `y[i] = x[i]` if `x[i] > 0`, else `alpha * x[i]`.

**Parameters:**
- `alpha` : Slope for negative values (default: 0.01)
- `x` : Input vector of dimension `n`

**Returns:** Vector of dimension `n` with leaky ReLU applied element-wise

**TODO:** Register differentiation properties.

**Usage:** Hidden layer activation when ReLU "dying" is a concern. -/
@[inline]
def leakyReluVec {n : Nat} (alpha : Float := 0.01) (x : Vector n) : Vector n :=
  ⊞ i => leakyRelu alpha x[i]

/-- Tanh (hyperbolic tangent) activation function.

Maps real numbers to (-1, 1), creating an S-shaped curve centered at 0.

**Mathematical definition:** `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

**Gradient:** `tanh'(x) = 1 - tanh²(x)`

**Properties:**
- Range: (-1, 1)
- tanh(0) = 0
- tanh(-x) = -tanh(x) (odd function)
- Saturates for large |x| (gradient vanishing problem)
- Stronger gradients than sigmoid near 0

**Parameters:**
- `x` : Input scalar (Float)

**Returns:** Value in (-1, 1)

**TODO:** Register differentiation properties.
**TODO:** Verify Float.exp is differentiable in SciLean.

**Usage:** Hidden layer activation (better than sigmoid due to zero-centered output),
recurrent neural networks. -/
@[inline]
def tanh (x : Float) : Float :=
  let expPos := Float.exp x
  let expNeg := Float.exp (-x)
  (expPos - expNeg) / (expPos + expNeg)

/-- Element-wise tanh on vectors.

Applies tanh activation to each element: `y[i] = tanh(x[i])`.

**Parameters:**
- `x` : Input vector of dimension `n`

**Returns:** Vector of dimension `n` with values in (-1, 1)

**TODO:** Register differentiation properties.

**Usage:** Hidden layer activation in recurrent networks or when zero-centered
activations are beneficial. -/
@[inline]
def tanhVec {n : Nat} (x : Vector n) : Vector n :=
  ⊞ i => tanh x[i]

/-! ## Analytical Derivatives

These functions provide explicit derivative formulas for gradient checking and numerical validation.
They are separate from SciLean's automatic differentiation system.

Once `@[fun_trans]` properties are registered, SciLean will compute derivatives automatically.
Until then, these analytical formulas are used to validate AD correctness via finite differences.
-/

/-- Analytical derivative of ReLU: `ReLU'(x)`.

Returns `1` if `x > 0`, `0` otherwise.

**Convention at x=0:** Use 0 (subgradient choice), matching PyTorch/TensorFlow.

**Parameters:**
- `x` : Input scalar (Float)

**Returns:** Derivative value (0 or 1)

**Usage:** Gradient checking, manual backpropagation validation. -/
def reluDerivative (x : Float) : Float :=
  if x > 0 then 1 else 0

/-- Analytical derivative of sigmoid: `σ'(x) = σ(x) * (1 - σ(x))`.

Computes the derivative using the convenient property that it can be expressed
in terms of the sigmoid function itself.

**Parameters:**
- `x` : Input scalar (Float)

**Returns:** Derivative value in (0, 0.25]

**Note:** Maximum derivative is 0.25 at x=0. This contributes to gradient vanishing.

**Usage:** Gradient checking for sigmoid activation. -/
def sigmoidDerivative (x : Float) : Float :=
  let s := sigmoid x
  s * (1 - s)

/-- Analytical derivative of tanh: `tanh'(x) = 1 - tanh²(x)`.

Computes the derivative using the convenient property that it can be expressed
in terms of the tanh function itself.

**Parameters:**
- `x` : Input scalar (Float)

**Returns:** Derivative value in (0, 1]

**Note:** Maximum derivative is 1 at x=0. Better gradient flow than sigmoid near origin.

**Usage:** Gradient checking for tanh activation. -/
def tanhDerivative (x : Float) : Float :=
  let t := tanh x
  1 - t * t

/-- Analytical derivative of leaky ReLU: `LeakyReLU'(x)`.

Returns `1` if `x > 0`, `alpha` otherwise.

**Parameters:**
- `alpha` : Slope for negative values (default: 0.01)
- `x` : Input scalar (Float)

**Returns:** Derivative value (1 or alpha)

**Usage:** Gradient checking for leaky ReLU activation. -/
def leakyReluDerivative (alpha : Float := 0.01) (x : Float) : Float :=
  if x > 0 then 1 else alpha

end VerifiedNN.Core.Activation
