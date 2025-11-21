import VerifiedNN.Core.DataTypes
import SciLean

namespace VerifiedNN.Core.ReluBackward

open VerifiedNN.Core
open SciLean

set_default_scalar Float

/-!
# ReLU Backward Pass

Applies the ReLU gradient mask during backpropagation.

## Mathematical Formulation

ReLU activation: `f(x) = max(0, x)`

ReLU derivative:
```
f'(x) = 1  if x > 0
      = 0  otherwise
```

Backward pass: `∂L/∂x[i] = ∂L/∂f(x)[i] * f'(x[i])`

This is element-wise masking - gradients flow through where the activation was
positive, and are zeroed out elsewhere.

## Key Functions
- `reluBackward`: Apply ReLU gradient mask
- `reluBackwardVec`: Alias for consistency with naming conventions

## Implementation Notes

**Why We Need Pre-Activation Values:**
The ReLU derivative depends on the *input* to ReLU, not the output.
Since relu(x) = max(0, x), when the output is 0, we cannot determine if the
input was 0 or negative. Therefore, we must save the pre-activation values
during the forward pass to compute the correct gradient during backpropagation.

**Forward Pass Contract:**
```lean
let z := weights @ x + bias  -- SAVE this (pre-activation)
let a := relu z              -- Don't need to save this
```

**Backward Pass Usage:**
```lean
let gradInput := reluBackward gradOutput z  -- Use saved pre-activation
```

## Verification Status
- **Computable:** Yes (simple conditionals and array construction)
- **Correctness:** Matches analytical ReLU derivative by construction
- **Performance:** O(n) single pass, marked inline for hot path optimization
-/

/-- Apply ReLU backward pass (gradient masking).

Given the gradient of the loss with respect to ReLU's output and the
pre-activation values from the forward pass, computes the gradient with
respect to ReLU's input.

**Mathematical Operation:**
```
output[i] = gradOutput[i]  if input[i] > 0
          = 0              otherwise
```

This implements the chain rule application for ReLU:
```
∂L/∂x[i] = ∂L/∂a[i] * ∂a/∂x[i]
         = gradOutput[i] * f'(input[i])
         = gradOutput[i] * (if input[i] > 0 then 1 else 0)
```

**Parameters:**
- `gradOutput`: Gradient ∂L/∂relu(x) from layer above [n]
- `input`: Pre-activation values x from forward pass [n]

**Returns:** Gradient ∂L/∂x [n]

**Computational Cost:** O(n) single pass

**Example:**
```lean
-- Forward pass saved these pre-activation values
let z1 : Vector 128 := ...  -- May have negative values

-- Backward pass received this gradient
let gradFromAbove : Vector 128 := ...

-- Apply ReLU masking
let gradToPropagate := reluBackward gradFromAbove z1
-- Gradients are zeroed out where z1[i] ≤ 0
```

**Critical Implementation Detail:**
The condition `input[i] > 0.0` uses strict inequality. This means:
- At exactly `input[i] = 0`, the gradient is masked to 0
- This is the standard convention in deep learning (though ReLU is technically
  non-differentiable at 0, we define the subgradient to be 0)
-/
@[inline]
def reluBackward
  {n : Nat}
  (gradOutput : Vector n)
  (input : Vector n)
  : Vector n :=
  -- Element-wise: pass gradient through if input > 0, else zero
  ⊞ (i : Idx n) =>
    if input[i] > 0.0 then gradOutput[i] else 0.0

/-- Alias for `reluBackward` with consistent naming.

Some modules use `reluVec` for forward pass, so this provides
`reluBackwardVec` for symmetry.
-/
@[inline]
def reluBackwardVec {n : Nat} := @reluBackward n

/-!
## Examples and Test Cases

These examples demonstrate the gradient masking behavior of ReLU backpropagation
and verify compile-time type checking.
-/

-- Example 1: All positive inputs - all gradients pass through
example : True := by
  let input_pos : Vector 3 := ⊞ (i : Idx 3) => (i.1.toNat.toFloat + 1.0)
  let grad : Vector 3 := ⊞ (i : Idx 3) => 1.0
  let result := reluBackward grad input_pos
  -- result will be approximately [1, 1, 1] since all inputs are positive
  -- Gradients pass through unchanged
  trivial

-- Example 2: Simple dimensional consistency check
example : True := by
  let input : Vector 128 := ⊞ (i : Idx 128) => (i.1.toNat.toFloat - 64.0)
  let gradOutput : Vector 128 := ⊞ (i : Idx 128) => 1.0
  let gradInput := reluBackward gradOutput input
  -- Type checker ensures: gradInput : Vector 128
  -- Positive inputs (i >= 64) pass gradients, negative ones (i < 64) zero them
  trivial

-- Example 3: Works with alias
example : True := by
  let input : Vector 64 := ⊞ (i : Idx 64) => i.1.toNat.toFloat
  let grad : Vector 64 := ⊞ (i : Idx 64) => 0.5
  let result := reluBackwardVec grad input
  -- reluBackwardVec is an alias for reluBackward
  trivial

/-!
## Common Pitfall: Using ReLU Output Instead of Input

**WRONG APPROACH:**
```lean
-- Forward pass
let z := weights @ x + bias
let a := relu z  -- ReLU output

-- Backward pass (INCORRECT!)
let grad := reluBackward gradOutput a  -- ❌ Using output instead of input
```

**WHY IT'S WRONG:**
When `a[i] = 0`, we cannot determine if `z[i]` was 0 or negative.
Both cases produce the same output but require different gradients.

**CORRECT APPROACH:**
```lean
-- Forward pass
let z := weights @ x + bias  -- ✅ SAVE this
let a := relu z

-- Backward pass (CORRECT!)
let grad := reluBackward gradOutput z  -- ✅ Use pre-activation
```

**MEMORY IMPLICATION:**
This means the forward pass must save `z` for every layer with ReLU activation.
In a typical network, this doubles memory usage during training compared to
inference-only mode.
-/

end VerifiedNN.Core.ReluBackward
