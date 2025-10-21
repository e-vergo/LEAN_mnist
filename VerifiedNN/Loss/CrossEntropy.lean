/-
# Cross-Entropy Loss

Cross-entropy loss function with numerical stability.

This module implements cross-entropy loss using the log-sum-exp trick for numerical
stability. The loss measures the difference between predicted probability distributions
and true labels.

**Verification Status:**
- Implementation uses Float (computational)
- Mathematical properties proven on ℝ in Properties.lean
- Numerical stability via log-sum-exp trick

**Mathematical Definition:**
For predictions ŷ and one-hot target y:
  L(ŷ, y) = -log(ŷ[target]) = -log(exp(z[target]) / Σⱼ exp(z[j]))

Using log-sum-exp trick:
  L = -z[target] + log-sum-exp(z)
where log-sum-exp(z) = log(Σⱼ exp(z[j])) computed stably as:
  log-sum-exp(z) = max(z) + log(Σⱼ exp(z[j] - max(z)))
-/

import VerifiedNN.Core.DataTypes
import SciLean

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
-/
def logSumExp {n : Nat} (x : Vector n) : Float :=
  -- Find maximum value for numerical stability
  let maxVal := ⊞ (i : Fin n) => x[i] |>.max
  -- Compute shifted exponentials and sum
  let expSum := ∑ i : Fin n, Float.exp (x[i] - maxVal)
  -- Return max + log(sum(exp(x - max)))
  maxVal + Float.log expSum

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

**Note:** This function does not check if target < n. Out-of-bounds access
will result in 0.0 being returned due to DataArrayN default behavior.
-/
def crossEntropyLoss {n : Nat} (predictions : Vector n) (target : Nat) : Float :=
  let targetLogit := predictions[target]'(by omega)
  let lse := logSumExp predictions
  -targetLogit + lse

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
    let totalLoss := ∑ i : Fin b,
      if h : i.val < targets.size then
        let predRow : Vector n := ⊞ j => predictions[i, j]
        crossEntropyLoss predRow targets[i.val]
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
  let l2Norm := ∑ i : Fin n, predictions[i] * predictions[i]
  ceLoss + (lambda / 2.0) * l2Norm

end VerifiedNN.Loss
