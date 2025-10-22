import VerifiedNN.Loss.CrossEntropy
import VerifiedNN.Core.Activation
import VerifiedNN.Core.DataTypes
import SciLean

/-!
# Loss Function Tests

Comprehensive test suite for loss function properties and numerical stability.

## Main Tests

- `testCrossEntropyBasic`: Basic cross-entropy computation on known examples
- `testCrossEntropyNonNegative`: Validates loss ≥ 0 property
- `testCrossEntropyPerfectPrediction`: Loss = 0 when prediction matches target
- `testCrossEntropyWorstCase`: High loss for completely wrong predictions
- `testSoftmaxProperties`: Validates softmax sums to 1 and is in (0,1)
- `testLogSumExpStability`: Tests numerical stability with large values
- `testBatchLoss`: Batch loss averaging and consistency

## Implementation Notes

**Testing Strategy:** Validate mathematical properties of loss functions:
- Non-negativity: L(ŷ, y) ≥ 0 for all inputs
- Perfect prediction: L(y, y) = 0
- Worst case behavior: High loss for wrong predictions
- Numerical stability: No NaN/Inf on extreme inputs
- Batch consistency: Batch loss = average of individual losses

**Coverage Status:** Comprehensive coverage of CrossEntropy module.

**Test Framework:** IO-based assertions with detailed diagnostic output.

## Usage

```bash
# Build tests
lake build VerifiedNN.Testing.LossTests

# Run tests
lake env lean --run VerifiedNN/Testing/LossTests.lean
```

## References

- Loss.CrossEntropy: Main implementation
- Loss.Properties: Formal proofs of loss properties
-/

namespace VerifiedNN.Testing.LossTests

open VerifiedNN.Loss
open VerifiedNN.Core
open VerifiedNN.Core.Activation
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

/-! ## Cross-Entropy Loss Tests -/

/-- Test basic cross-entropy computation on known examples -/
def testCrossEntropyBasic : IO Bool := do
  IO.println "\n=== Basic Cross-Entropy Tests ==="

  let mut allPassed := true

  -- Test case 1: Simple 3-class example
  -- Logits: [2.0, 1.0, 0.1], target: class 0
  -- Expected: Small loss since class 0 has highest logit
  let logits1 : Vector 3 := ⊞[2.0, 1.0, 0.1]
  let target1 : Nat := 0
  let loss1 := crossEntropyLoss logits1 target1

  -- Loss should be positive and relatively small
  allPassed := allPassed && (← assertTrue "CE: loss ≥ 0 for correct prediction" (loss1 ≥ 0.0))
  allPassed := allPassed && (← assertTrue "CE: loss < 1 for correct prediction" (loss1 < 1.0))
  IO.println s!"  Loss for [2.0, 1.0, 0.1] with target 0: {loss1}"

  -- Test case 2: Wrong prediction
  -- Logits: [2.0, 1.0, 0.1], target: class 2
  -- Expected: Large loss since class 2 has lowest logit
  let target2 : Nat := 2
  let loss2 := crossEntropyLoss logits1 target2

  allPassed := allPassed && (← assertTrue "CE: wrong prediction has higher loss" (loss2 > loss1))
  IO.println s!"  Loss for [2.0, 1.0, 0.1] with target 2: {loss2}"

  return allPassed

/-- Test non-negativity property: L(ŷ, y) ≥ 0 -/
def testCrossEntropyNonNegative : IO Bool := do
  IO.println "\n=== Cross-Entropy Non-Negativity Tests ==="

  let mut allPassed := true

  -- Test with various logit patterns
  let testCases : List (Vector 4 × Nat) := [
    (⊞[1.0, 2.0, 3.0, 4.0], 0),
    (⊞[1.0, 2.0, 3.0, 4.0], 3),
    (⊞[-1.0, 0.0, 1.0, 2.0], 1),
    (⊞[10.0, 5.0, -5.0, -10.0], 2),
    (⊞[0.0, 0.0, 0.0, 0.0], 0)  -- All equal logits
  ]

  for (logits, target) in testCases do
    let loss := crossEntropyLoss logits target
    allPassed := allPassed && (← assertTrue s!"CE: loss ≥ 0 for target {target}" (loss ≥ 0.0))

  return allPassed

/-- Test perfect prediction: softmax probability 1 for correct class -/
def testCrossEntropyPerfectPrediction : IO Bool := do
  IO.println "\n=== Cross-Entropy Perfect Prediction Tests ==="

  let mut allPassed := true

  -- When one logit is much larger than others, softmax → 1 for that class
  -- Loss should be very close to 0
  let logits : Vector 3 := ⊞[100.0, 0.0, 0.0]
  let target : Nat := 0
  let loss := crossEntropyLoss logits target

  allPassed := allPassed && (← assertApproxEq "CE: near-perfect prediction → loss ≈ 0" loss 0.0 0.01)
  IO.println s!"  Loss for near-perfect prediction: {loss}"

  return allPassed

/-- Test worst case: prediction completely wrong -/
def testCrossEntropyWorstCase : IO Bool := do
  IO.println "\n=== Cross-Entropy Worst Case Tests ==="

  let mut allPassed := true

  -- Target class has very low logit, other classes have high logits
  let logits : Vector 3 := ⊞[100.0, 99.0, -100.0]
  let target : Nat := 2  -- Class with lowest logit
  let loss := crossEntropyLoss logits target

  -- Loss should be very large (close to the difference: 100 - (-100) = 200)
  allPassed := allPassed && (← assertTrue "CE: worst case has high loss" (loss > 100.0))
  IO.println s!"  Loss for worst case prediction: {loss}"

  return allPassed

/-! ## Softmax Property Tests -/

/-- Test softmax properties: sum to 1, values in (0, 1) -/
def testSoftmaxProperties : IO Bool := do
  IO.println "\n=== Softmax Property Tests ==="

  let mut allPassed := true

  -- Test case 1: Typical logits
  let logits1 : Vector 4 := ⊞[1.0, 2.0, 3.0, 4.0]
  let probs1 := softmax logits1

  -- Check sum to 1
  let sum1 := ∑ i, probs1[i]
  allPassed := allPassed && (← assertApproxEq "Softmax: sum = 1" sum1 1.0 1e-5)

  -- Check all values in (0, 1) using extracted values
  for i in [:4] do
    if i < 4 then
      let pi := ∑ (idx : Idx 4), if idx.1.toNat == i then probs1[idx] else 0.0
      allPassed := allPassed && (← assertTrue s!"Softmax: p[{i}] ∈ (0,1)" (pi > 0.0 && pi < 1.0))

  -- Check monotonicity: larger logit → larger probability using indicator extraction
  let p0 := ∑ i : Idx 4, if i.1.toNat == 0 then probs1[i] else 0.0
  let p1 := ∑ i : Idx 4, if i.1.toNat == 1 then probs1[i] else 0.0
  let p2 := ∑ i : Idx 4, if i.1.toNat == 2 then probs1[i] else 0.0
  let p3 := ∑ i : Idx 4, if i.1.toNat == 3 then probs1[i] else 0.0
  allPassed := allPassed && (← assertTrue "Softmax: monotonic in logits"
    (p0 < p1 && p1 < p2 && p2 < p3))

  -- Test case 2: Equal logits → equal probabilities
  let logits2 : Vector 3 := ⊞[5.0, 5.0, 5.0]
  let probs2 := softmax logits2
  let expectedProb := 1.0 / 3.0

  let p2_0 := ∑ i : Idx 3, if i.1.toNat == 0 then probs2[i] else 0.0
  let p2_1 := ∑ i : Idx 3, if i.1.toNat == 1 then probs2[i] else 0.0
  allPassed := allPassed && (← assertApproxEq "Softmax: equal logits → p = 1/n" p2_0 expectedProb 1e-5)
  allPassed := allPassed && (← assertApproxEq "Softmax: equal logits → p = 1/n" p2_1 expectedProb 1e-5)

  return allPassed

/-! ## Numerical Stability Tests -/

/-- Test log-sum-exp stability with large values -/
def testLogSumExpStability : IO Bool := do
  IO.println "\n=== Log-Sum-Exp Stability Tests ==="

  let mut allPassed := true

  -- Test with very large logits (would overflow without max trick)
  let largeLogits : Vector 3 := ⊞[1000.0, 999.0, 998.0]
  let target : Nat := 0
  let loss := crossEntropyLoss largeLogits target

  -- Loss should be finite and reasonable (not NaN or Inf)
  allPassed := allPassed && (← assertTrue "LSE: large logits don't overflow" (!loss.isNaN && !loss.isInf))
  allPassed := allPassed && (← assertTrue "LSE: loss is reasonable" (loss ≥ 0.0 && loss < 10.0))
  IO.println s!"  Loss with logits [1000, 999, 998]: {loss}"

  -- Test with very negative logits
  let negativeLogits : Vector 3 := ⊞[-1000.0, -999.0, -998.0]
  let loss2 := crossEntropyLoss negativeLogits 0

  allPassed := allPassed && (← assertTrue "LSE: negative logits don't underflow" (!loss2.isNaN && !loss2.isInf))
  IO.println s!"  Loss with logits [-1000, -999, -998]: {loss2}"

  -- Test with mixed large positive and negative
  let mixedLogits : Vector 3 := ⊞[500.0, -500.0, 0.0]
  let loss3 := crossEntropyLoss mixedLogits 1

  allPassed := allPassed && (← assertTrue "LSE: mixed large values are stable" (!loss3.isNaN && !loss3.isInf))
  IO.println s!"  Loss with logits [500, -500, 0]: {loss3}"

  return allPassed

/-! ## Batch Loss Tests -/

/-- Test batch loss computation -/
def testBatchLoss : IO Bool := do
  IO.println "\n=== Batch Loss Tests ==="

  let mut allPassed := true

  -- Create batch of 3 samples with 4 classes each
  let logitsBatch : Batch 3 4 := ⊞ (b : Idx 3) (i : Idx 4) =>
    (b.1.toNat.toFloat + i.1.toNat.toFloat)

  let targets : Array Nat := #[0, 1, 2]

  -- Compute batch loss
  let batchLossVal := batchCrossEntropyLoss logitsBatch targets

  -- Batch loss should be non-negative
  allPassed := allPassed && (← assertTrue "Batch loss: non-negative" (batchLossVal ≥ 0.0))

  -- Compute individual losses and average using indicator sums
  let mut sumIndividual : Float := 0.0
  for b in [:3] do
    if b < 3 then
      let logits_b : Vector 4 := ⊞ (i : Idx 4) => ∑ (bidx : Idx 3), if bidx.1.toNat == b then logitsBatch[bidx, i] else 0.0
      let target_b := targets.get! b  -- Note: targets[b]! doesn't work (b is Idx 3, not Nat)
      sumIndividual := sumIndividual + crossEntropyLoss logits_b target_b

  let avgIndividual := sumIndividual / 3.0

  -- Batch loss should equal average of individual losses
  allPassed := allPassed && (← assertApproxEq "Batch loss = average of individual losses"
    batchLossVal avgIndividual 1e-5)

  IO.println s!"  Batch loss: {batchLossVal}"
  IO.println s!"  Average of individual losses: {avgIndividual}"

  return allPassed

/-! ## Test Runner -/

/-- Run all loss function tests and report results -/
def runAllTests : IO Unit := do
  IO.println "=========================================="
  IO.println "Running Loss Function Tests"
  IO.println "=========================================="

  let mut totalPassed := 0
  let mut totalTests := 0

  let testSuites : List (String × IO Bool) := [
    ("Basic Cross-Entropy", testCrossEntropyBasic),
    ("Cross-Entropy Non-Negativity", testCrossEntropyNonNegative),
    ("Cross-Entropy Perfect Prediction", testCrossEntropyPerfectPrediction),
    ("Cross-Entropy Worst Case", testCrossEntropyWorstCase),
    ("Softmax Properties", testSoftmaxProperties),
    ("Log-Sum-Exp Stability", testLogSumExpStability),
    ("Batch Loss", testBatchLoss)
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
    IO.println "✓ All loss function tests passed!"
  else
    IO.println s!"✗ {totalTests - totalPassed} test suite(s) failed"

end VerifiedNN.Testing.LossTests
