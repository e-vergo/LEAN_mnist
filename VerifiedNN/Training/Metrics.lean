import VerifiedNN.Network.Architecture
import VerifiedNN.Loss.CrossEntropy
import SciLean

/-!
# Training Metrics

Evaluation metrics for neural network performance measurement.

## Main Definitions

- `getPredictedClass`: Extract predicted class index from network output (argmax)
- `isCorrectPrediction`: Check if single prediction matches true label
- `computeAccuracy`: Overall classification accuracy on dataset
- `computeAverageLoss`: Average cross-entropy loss on dataset
- `computePerClassAccuracy`: Per-class accuracy breakdown for detailed analysis
- `printMetrics`: Convenience function for console output of accuracy and loss

## Main Results

This module provides computational evaluation utilities without formal verification.
No theorems are proven. Correctness depends on the mathematical properties of
`argmax` and `crossEntropyLoss`.

## Implementation Notes

**Classification accuracy:** Defined as the fraction of correct predictions:
```
accuracy = (# correct predictions) / (# total examples)
```
For MNIST, random guessing achieves ~10% accuracy (1/10 classes).
A well-trained network should achieve >95% test accuracy. Values in [0, 1].

**Cross-entropy loss:** Measures the difference between predicted probabilities
and true labels. Lower loss indicates better calibrated predictions. Loss approaching
0 indicates perfect confidence in correct predictions. Range: [0, ∞).

**Per-class accuracy:** Computes accuracy separately for each class, useful for
identifying class imbalance or confusion patterns:
- If digit "1" has high accuracy but "8" has low accuracy, the network
  may struggle with more complex shapes
- Can guide data augmentation or model architecture decisions
- Helps detect if network is biased toward certain classes

**Edge case handling:** All functions return 0.0 for empty datasets to avoid
division by zero. Per-class accuracy returns 0.0 for classes with no examples.

**Performance:** Metrics computation is O(n) where n = dataset size. Each example
requires a forward pass through the network. For large validation sets, consider
computing metrics on a random subset for faster evaluation during training.

**Mathematical notation:**
- Accuracy: acc = (1/n) Σᵢ 1[argmax(f(xᵢ)) = yᵢ]
- Average loss: L̄ = (1/n) Σᵢ L(f(xᵢ), yᵢ)

where f is the network, L is cross-entropy loss, and 1[·] is indicator function.

## References

- Classification metrics: "Pattern Recognition and Machine Learning" (Bishop, 2006)
- Cross-entropy loss: "Deep Learning" (Goodfellow et al., 2016), Chapter 5
-/

namespace VerifiedNN.Training.Metrics

open VerifiedNN.Core
open VerifiedNN.Network
open VerifiedNN.Loss
open SciLean

/-- Get the predicted class from network output.

Returns the index of the maximum value in the output vector,
which corresponds to the predicted class label. This implements
the argmax operation: argmax(ŷ) where ŷ is the output probability vector.

**Parameters:**
- `output`: Network output probabilities (typically after softmax activation)

**Returns:** Predicted class index (0-9 for MNIST, 0 to n-1 in general)

**Mathematical definition:** argmax(ŷ) = argmax_{i ∈ {0,...,n-1}} ŷᵢ

**Usage:** After forward pass, extract most likely class:
```lean
let output := net.forward input
let prediction := getPredictedClass output  -- Index of highest probability
```

**Note:** For MNIST, output vector has 10 elements corresponding to digits 0-9.
The predicted class is the digit with highest softmax probability.
-/
def getPredictedClass {n : Nat} (output : Vector n) : Nat :=
  argmax output

/-- Check if a single prediction is correct.

Performs forward pass through the network and checks if the predicted class
(argmax of output) matches the true label. This is the fundamental unit of
accuracy computation.

**Parameters:**
- `net`: Neural network to evaluate
- `input`: Input vector (784-dimensional for MNIST)
- `trueLabel`: Ground truth class label (0-9 for MNIST)

**Returns:** True if argmax(net.forward(input)) = trueLabel, false otherwise

**Mathematical definition:** 1[argmax(f(x)) = y] where f is the network,
x is the input, y is the true label, and 1[·] is the indicator function.

**Algorithm:**
1. Compute output := net.forward(input) via forward propagation
2. Extract prediction := argmax(output)
3. Compare prediction == trueLabel

**Use case:** Building block for accuracy computation. Used in `computeAccuracy`
and `computePerClassAccuracy`.

**Complexity:** O(nParams) for forward pass, O(n) for argmax where n = 10 for MNIST
-/
def isCorrectPrediction (net : MLPArchitecture) (input : Vector 784) (trueLabel : Nat) : Bool :=
  let output := net.forward input
  let prediction := getPredictedClass output
  prediction == trueLabel

/-- Compute classification accuracy on a dataset.

Evaluates the network on all examples in the dataset and computes
the fraction of correct predictions. This is the standard metric for
classification tasks.

**Parameters:**
- `net`: Neural network to evaluate
- `testData`: Array of (input, label) pairs (typically test or validation set)

**Returns:** Accuracy as a float in [0, 1] where:
- 0.0 = all predictions incorrect (or empty dataset)
- 1.0 = all predictions correct (perfect accuracy)
- ~0.1 for MNIST = random guessing (1/10 classes)
- >0.95 for MNIST = well-trained network

**Mathematical definition:** acc = (1/n) Σᵢ₌₁ⁿ 1[argmax(f(xᵢ)) = yᵢ]
where f is the network, n is dataset size, and 1[·] is indicator function.

**Algorithm:**
1. Iterate through all examples in testData
2. For each example: check if isCorrectPrediction(net, input, label)
3. Count correct predictions
4. Return numCorrect / totalExamples

**Edge cases:**
- Empty dataset: Returns 0.0 to avoid division by zero
- All correct: Returns 1.0
- All incorrect: Returns 0.0

**Complexity:** O(n × m) where n = testData.size, m = nParams (network size)

**Use case:** Primary evaluation metric during training and final model assessment.
Complement with loss and per-class accuracy for comprehensive evaluation.
-/
def computeAccuracy
    (net : MLPArchitecture)
    (testData : Array (Vector 784 × Nat)) : Float :=
  if testData.size == 0 then
    0.0
  else
    let numCorrect := testData.foldl (fun count (input, label) =>
      if isCorrectPrediction net input label then
        count + 1
      else
        count
    ) 0
    (numCorrect.toFloat / testData.size.toFloat)

/-- Compute average loss on a dataset.

Evaluates the cross-entropy loss for all examples in the dataset
and returns the average. Loss provides a continuous measure of model
performance, complementing discrete accuracy metrics.

**Parameters:**
- `net`: Neural network to evaluate
- `testData`: Array of (input, label) pairs (typically test or validation set)

**Returns:** Average cross-entropy loss as a float in [0, ∞) where:
- 0.0 = perfect predictions (infinite confidence in correct class)
- log(numClasses) ≈ 2.3 for MNIST = random guessing
- Lower is better (indicates better calibrated predictions)

**Mathematical definition:** L̄ = (1/n) Σᵢ₌₁ⁿ L(f(xᵢ), yᵢ)
where L is cross-entropy loss, f is the network, n is dataset size.

**Cross-entropy loss:** L(ŷ, y) = -log(ŷ[y]) where ŷ is softmax output,
y is true class. Measures divergence between predicted probabilities and
true distribution.

**Algorithm:**
1. Initialize totalLoss := 0
2. For each example (input, label) in testData:
   - Compute output := net.forward(input)
   - Compute loss := crossEntropyLoss(output, label)
   - Add loss to totalLoss
3. Return totalLoss / testData.size

**Edge cases:**
- Empty dataset: Returns 0.0 to avoid division by zero
- Perfect predictions: Approaches 0.0 (log(1.0) = 0)
- Terrible predictions: Can be very large (unbounded)

**Complexity:** O(n × m) where n = testData.size, m = nParams (network size)

**Use case:** Track training progress (loss should decrease), detect overfitting
(gap between train/validation loss), assess prediction confidence calibration.
Loss is more sensitive than accuracy to small improvements.

**Note:** Depends on `crossEntropyLoss` from `VerifiedNN.Loss.CrossEntropy`.
-/
def computeAverageLoss
    (net : MLPArchitecture)
    (testData : Array (Vector 784 × Nat)) : Float :=
  if testData.size == 0 then
    0.0
  else
    let totalLoss := testData.foldl (fun sum (input, label) =>
      let output := net.forward input
      let loss := crossEntropyLoss output label
      sum + loss
    ) 0.0
    totalLoss / testData.size.toFloat

/-- Compute per-class accuracy.

Computes accuracy separately for each class, useful for identifying
which digits the network struggles with. This provides more detailed
diagnostic information than overall accuracy, revealing class-specific
performance patterns.

**Parameters:**
- `net`: Neural network to evaluate
- `testData`: Array of (input, label) pairs
- `numClasses`: Number of classes (default: 10 for MNIST)

**Returns:** Array of per-class accuracies, where result[i] = accuracy for class i

**Algorithm:**
1. For each example, increment total count for its true class
2. If prediction is correct, increment correct count for that class
3. For each class i: accuracy[i] = correct[i] / total[i]

**Mathematical definition:** For class c:
```
acc_c = (# correct predictions for class c) / (# examples of class c)
```

**Use cases:**
- Detect class imbalance in model performance
- Identify confusable classes (e.g., digits 4 and 9)
- Guide targeted improvements (data augmentation, architecture changes)
- Validate that network isn't biased toward majority classes

**Edge cases:**
- Classes with no examples return 0.0 accuracy (avoids division by zero)
- Labels ≥ numClasses are ignored (skipped in counting)

**Complexity:** O(n × m) where n = dataset size, m = nParams (forward pass cost)
-/
def computePerClassAccuracy
    (net : MLPArchitecture)
    (testData : Array (Vector 784 × Nat))
    (numClasses : Nat := 10) : Array Float :=
  -- Count correct predictions and total examples per class
  let (correct, total) := testData.foldl (fun (corrArr, totArr) (input, label) =>
    if label >= numClasses then (corrArr, totArr)
    else
      let tot := totArr.modify label (· + 1)
      let corr := if isCorrectPrediction net input label then
        corrArr.modify label (· + 1)
      else
        corrArr
      (corr, tot)
  ) (Array.replicate numClasses 0, Array.replicate numClasses 0)

  -- Compute accuracy for each class
  Array.ofFn fun (i : Fin numClasses) =>
    let t := total[i.val]!
    let c := correct[i.val]!
    if t == 0 then 0.0
    else c.toFloat / t.toFloat

/-- Print evaluation metrics to console.

Convenience function to display accuracy and loss in a readable format.
Computes both metrics and prints them with labeled output for easy interpretation.

**Parameters:**
- `net`: Neural network to evaluate
- `testData`: Test dataset (array of input-label pairs)
- `datasetName`: Name to display in output (default: "Test", also common: "Train", "Validation")

**Returns:** IO action that prints two lines:
1. "{datasetName} Accuracy: {accuracy}%"
2. "{datasetName} Loss: {loss}"

**Example output:**
```
Test Accuracy: 96.5%
Test Loss: 0.123
```

**Use case:** Quick evaluation during development and at end of training.
For production logging, consider writing to file or structured logging format.

**Implementation:** Calls `computeAccuracy` and `computeAverageLoss` sequentially,
then formats output with percentage conversion for accuracy (multiply by 100).
-/
def printMetrics
    (net : MLPArchitecture)
    (testData : Array (Vector 784 × Nat))
    (datasetName : String := "Test") : IO Unit := do
  let accuracy := computeAccuracy net testData
  let avgLoss := computeAverageLoss net testData
  let accuracyPercent := accuracy * 100.0
  IO.println s!"{datasetName} Accuracy: {accuracyPercent}%"
  IO.println s!"{datasetName} Loss: {avgLoss}"

end VerifiedNN.Training.Metrics
