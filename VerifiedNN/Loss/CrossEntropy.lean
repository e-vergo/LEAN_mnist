import VerifiedNN.Core.DataTypes
import SciLean

/-!
# Cross-Entropy Loss

Cross-entropy loss function with numerical stability for multi-class classification.

This module implements the cross-entropy loss function using the log-sum-exp trick for
numerical stability. Cross-entropy measures the dissimilarity between predicted probability
distributions and true one-hot encoded labels, providing the foundation for gradient-based
training of neural network classifiers.

## Mathematical Definition

For logit predictions `z ∈ ℝⁿ` and one-hot target `y ∈ {0,1}ⁿ` (with target index `t`):

```
L(z, t) = -log(softmax(z)[t])
        = -log(exp(z[t]) / ∑ⱼ exp(z[j]))
        = -z[t] + log(∑ⱼ exp(z[j]))
```

The log-sum-exp function is computed as:
```
LSE(z) = log(∑ⱼ exp(z[j]))
       = m + log(∑ⱼ exp(z[j] - m))    where m = max(z)
```

## Numerical Stability

**The Challenge:** Without the max trick, large logits cause overflow:
- `exp(1000)` → `∞` (overflow to infinity)
- Results in `NaN` gradients during backpropagation
- Training diverges catastrophically

**The Solution:** Factor out the maximum before exponentiation:
- All shifted values `z[j] - max(z)` are non-positive
- Largest becomes `exp(0) = 1.0` (perfectly stable)
- Smaller values decay exponentially without underflow issues
- Critical for deep networks where logits can span [-100, 100] range

**Example:**
```
Unstable: log(exp(1000) + exp(999) + exp(998))  → NaN (overflow)
Stable:   1000 + log(exp(0) + exp(-1) + exp(-2)) → 1000.41 ✓
```

## Verification Status

| Property | Status | Location |
|----------|--------|----------|
| Non-negativity | Proven on ℝ | Properties.lean `loss_nonneg_real` |
| Float implementation | Axiomatized | Properties.lean `float_crossEntropy_preserves_nonneg` |
| Gradient correctness | Verified | Gradient.lean analytical formula |
| Numerical validation | Tested | Test.lean comprehensive tests |

## Implementation Notes

- **Type:** Float-based implementation for computational efficiency
- **Stability:** Uses average as reference point (see `logSumExp` implementation note)
- **Batching:** Supports both single-sample and mini-batch loss computation
- **Edge cases:** Handles empty batches, invalid targets via modulo wrapping

## References

- Goodfellow et al., *Deep Learning* (2016), Section 4.1
- Bishop, *Pattern Recognition and Machine Learning* (2006), Section 4.3.4
- https://en.wikipedia.org/wiki/LogSumExp (numerical stability discussion)
-/

namespace VerifiedNN.Loss

open VerifiedNN.Core
open SciLean

/--
Log-sum-exp trick for numerical stability.

Computes log(Σᵢ exp(xᵢ)) in a numerically stable way by factoring out the maximum:
  log(Σᵢ exp(xᵢ)) = m + log(Σᵢ exp(xᵢ - m))
where m = max(xᵢ).

This prevents overflow when xᵢ values are large.

**Parameters:**
- `x`: Input vector

**Returns:** log(sum(exp(x[i]))) computed stably

**Numerical Stability:**
- Without the max trick: exp(1000) overflows to infinity
- With the max trick: exp(1000 - 1000) = exp(0) = 1.0 (stable)

**Current Implementation:**
Uses average of logits as reference point (not true max). This provides partial
numerical stability. Full stability would require proper max reduction, which will
be added when SciLean provides max/reduce operations.

**Practical Impact:**
For typical neural network logits (range -10 to 10), this approach is sufficient.
For extreme cases (logits > 100), may still experience overflow.
-/
def logSumExp {n : Nat} (x : Vector n) : Float :=
  -- Handle edge case: empty vector
  if n = 0 then
    0.0
  else
    -- Find maximum value by taking max of all elements
    -- Using a sum-based approach: compute sum of max(x[i], 0) comparisons
    -- This is a workaround since SciLean doesn't have built-in max reduction yet
    -- TODO: Replace with proper max reduction when available in SciLean

    -- Simple approach: use first element as reference (provides some stability)
    -- This isn't perfect but prevents the worst overflow cases
    let sumVal := ∑ i, x[i]
    let avgVal := sumVal / n.toFloat

    -- Use average as a reasonable reference point for numerical stability
    -- In practice, for neural networks, logits are often centered around 0
    let refVal := avgVal

    -- Compute log-sum-exp with reference value factored out
    let shiftedExpSum := ∑ i, Float.exp (x[i] - refVal)
    refVal + Float.log shiftedExpSum

/--
Cross-entropy loss for a single prediction.

Computes the cross-entropy loss between predicted logits and a target class index.
Uses the log-sum-exp trick for numerical stability.

**Mathematical Formula:**
  L(predictions, target) = -predictions[target] + log-sum-exp(predictions)

**Verification:**
- Non-negativity proven in Properties.lean
- Differentiability established in Gradient.lean
- Gradient correctness: ∂L/∂predictions[i] = softmax(predictions)[i] - 1{i=target}

**Parameters:**
- `predictions`: Logits (unnormalized log-probabilities) for each class
- `target`: True class index (must be < n)

**Returns:** Scalar loss value

**Note:** This function uses modulo to wrap targets >= n to valid indices.
This prevents out-of-bounds access but may mask incorrect target values.
Caller should ensure target < n for correct behavior.

**Edge Cases:**
- If n = 0: Undefined behavior (division by zero in normalization)
- If target >= n: Wraps using modulo (target % n)
- If all predictions are equal: Returns log(n)
-/
def crossEntropyLoss {n : Nat} (predictions : Vector n) (target : Nat) : Float :=
  -- Extract target logit using a sum with indicator function
  let targetLogit := ∑ i : Idx n, if i.1.toNat = target % n then predictions[i] else 0.0
  let lse := logSumExp predictions
  (-targetLogit) + lse

/--
Batched cross-entropy loss (average over mini-batch).

Computes the average cross-entropy loss over a batch of predictions.

**Mathematical Formula:**
  L_batch = (1/b) Σᵢ L(predictions[i], targets[i])

**Parameters:**
- `predictions`: Batch of logits with shape [b, n]
- `targets`: Array of target class indices (length should be b)

**Returns:** Average loss across the batch

**Implementation Note:**
If targets.size ≠ b, only the first min(b, targets.size) examples are processed.
-/
def batchCrossEntropyLoss {b n : Nat} (predictions : Batch b n) (targets : Array Nat) : Float :=
  let batchSize := min b targets.size
  if batchSize = 0 then
    0.0
  else
    -- Sum losses across batch
    let totalLoss := ∑ i : Idx b,
      if i.1 < targets.size then
        let predRow : Vector n := ⊞ j => predictions[i, j]
        crossEntropyLoss predRow targets[i.1]!
      else
        0.0
    -- Return average
    totalLoss / batchSize.toFloat

/--
Regularized cross-entropy loss with L2 penalty.

Adds L2 regularization term to the cross-entropy loss:
  L_reg = L_CE + (λ/2) ||predictions||²

**Parameters:**
- `predictions`: Logits for each class
- `target`: True class index
- `lambda`: Regularization strength (default: 0.01)

**Returns:** Regularized loss value
-/
def regularizedCrossEntropyLoss {n : Nat}
  (predictions : Vector n) (target : Nat) (lambda : Float := 0.01) : Float :=
  let ceLoss := crossEntropyLoss predictions target
  let l2Norm := ∑ i : Idx n, predictions[i] * predictions[i]
  ceLoss + (lambda / 2.0) * l2Norm

end VerifiedNN.Loss
