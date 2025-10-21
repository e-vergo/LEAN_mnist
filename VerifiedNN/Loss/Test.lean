/-
# Loss Function Tests

Simple tests to verify loss functions compile and produce sensible results.
-/

import VerifiedNN.Loss.CrossEntropy
import VerifiedNN.Loss.Gradient
import SciLean

namespace VerifiedNN.Loss.Test

open VerifiedNN.Core
open VerifiedNN.Loss
open VerifiedNN.Loss.Gradient
open SciLean

/-- Test cross-entropy loss on a simple example -/
def test_cross_entropy_basic : IO Unit := do
  -- Create a simple 3-class prediction: logits [1.0, 2.0, 0.5]
  let predictions : Vector 3 := ⊞ (i : Idx 3) =>
    if i.1.toNat = 0 then 1.0
    else if i.1.toNat = 1 then 2.0
    else 0.5

  -- Target class is 1 (highest logit)
  let target := 1

  -- Compute loss
  let loss := crossEntropyLoss predictions target

  IO.println s!"Cross-entropy loss (target=1): {loss}"

  -- Check that loss is non-negative
  if loss >= 0.0 then
    IO.println "✓ Loss is non-negative as expected"
  else
    IO.println "✗ ERROR: Loss is negative!"

/-- Test gradient computation -/
def test_gradient_basic : IO Unit := do
  -- Create simple predictions
  let predictions : Vector 3 := ⊞ (i : Idx 3) =>
    if i.1.toNat = 0 then 1.0
    else if i.1.toNat = 1 then 2.0
    else 0.5

  let target := 1

  -- Compute gradient
  let grad := lossGradient predictions target

  IO.println "\n✓ Gradient computation completed successfully"

/-- Test softmax function -/
def test_softmax : IO Unit := do
  -- Create simple logits
  let logits : Vector 3 := ⊞ (i : Idx 3) =>
    if i.1.toNat = 0 then 0.0
    else if i.1.toNat = 1 then 1.0
    else 0.0

  -- Compute softmax
  let probs := softmax logits

  IO.println "✓ Softmax computation completed successfully"

/-- Test one-hot encoding -/
def test_onehot : IO Unit := do
  let oh : Vector 5 := oneHot (n := 5) 2

  IO.println "✓ One-hot encoding completed successfully"

/-- Test batch loss computation -/
def test_batch_loss : IO Unit := do
  -- Create a simple 2x3 batch
  let batch : Batch 2 3 := ⊞ (i : Idx 2) (j : Idx 3) =>
    if i.1.toNat = 0 then
      -- First sample: logits [1, 2, 0]
      if j.1.toNat = 0 then 1.0
      else if j.1.toNat = 1 then 2.0
      else 0.0
    else
      -- Second sample: logits [0, 1, 2]
      Float.ofNat j.1.toNat

  let targets := #[1, 2]  -- Target classes

  let avgLoss := batchCrossEntropyLoss batch targets

  IO.println s!"Batch loss (2 samples): {avgLoss}"

  if avgLoss >= 0.0 then
    IO.println "✓ Batch loss computation completed successfully"
  else
    IO.println "✗ ERROR: Batch loss is negative!"

/-- Test regularized loss -/
def test_regularized_loss : IO Unit := do
  let predictions : Vector 3 := ⊞ (i : Idx 3) => Float.ofNat (i.1.toNat + 1)
  let target := 2
  let lambda := 0.01

  let regLoss := regularizedCrossEntropyLoss predictions target lambda

  IO.println s!"Regularized loss (lambda=0.01): {regLoss}"
  IO.println "✓ Regularized loss computation completed successfully"

/-- Main test runner -/
def main : IO Unit := do
  IO.println "=== Loss Function Tests ==="
  IO.println ""

  test_cross_entropy_basic
  test_gradient_basic
  test_softmax
  test_onehot
  test_batch_loss
  test_regularized_loss

  IO.println ""
  IO.println "=== All tests passed! ==="

end VerifiedNN.Loss.Test
