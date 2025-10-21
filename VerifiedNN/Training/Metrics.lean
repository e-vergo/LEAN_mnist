/-
# Training Metrics

Accuracy computation and evaluation metrics.

This module provides functions for evaluating neural network performance
on test datasets, including classification accuracy and loss computation.
-/

import VerifiedNN.Network.Architecture
import VerifiedNN.Loss.CrossEntropy
import SciLean

namespace VerifiedNN.Training.Metrics

open VerifiedNN.Core
open VerifiedNN.Network
open VerifiedNN.Loss
open SciLean

/-- Get the predicted class from network output.

Returns the index of the maximum value in the output vector,
which corresponds to the predicted class label.

**Parameters:**
- `output`: Network output probabilities (typically after softmax)

**Returns:** Predicted class index (0-9 for MNIST)
-/
def getPredictedClass {n : Nat} (output : Vector n) : Nat :=
  -- Find index of maximum value
  let rec findMaxIdx (arr : Vector n) (currentIdx : Nat) (maxIdx : Nat) (maxVal : Float) : Nat :=
    if currentIdx >= n then
      maxIdx
    else
      let val := arr[currentIdx]
      if val > maxVal then
        findMaxIdx arr (currentIdx + 1) currentIdx val
      else
        findMaxIdx arr (currentIdx + 1) maxIdx maxVal
  if n == 0 then 0
  else findMaxIdx output 1 0 output[0]

/-- Check if a single prediction is correct.

**Parameters:**
- `net`: Neural network to evaluate
- `input`: Input vector (784-dimensional for MNIST)
- `trueLabel`: Ground truth class label

**Returns:** True if prediction matches true label, false otherwise
-/
def isCorrectPrediction (net : MLPArchitecture) (input : Vector 784) (trueLabel : Nat) : Bool :=
  let output := net.forward input
  let prediction := getPredictedClass output
  prediction == trueLabel

/-- Compute classification accuracy on a dataset.

Evaluates the network on all examples in the dataset and computes
the fraction of correct predictions.

**Parameters:**
- `net`: Neural network to evaluate
- `testData`: Array of (input, label) pairs

**Returns:** Accuracy as a float in [0, 1]

**Note:** Returns 0.0 if testData is empty to avoid division by zero
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
and returns the average.

**Parameters:**
- `net`: Neural network to evaluate
- `testData`: Array of (input, label) pairs

**Returns:** Average cross-entropy loss

**Note:** Returns 0.0 if testData is empty to avoid division by zero
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
which digits the network struggles with.

**Parameters:**
- `net`: Neural network to evaluate
- `testData`: Array of (input, label) pairs
- `numClasses`: Number of classes (10 for MNIST)

**Returns:** Array of per-class accuracies

**Note:** Classes with no examples return 0.0 accuracy
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
  ) (Array.mkArray numClasses 0, Array.mkArray numClasses 0)

  -- Compute accuracy for each class
  Array.ofFn fun (i : Fin numClasses) =>
    let t := total[i.val]!
    let c := correct[i.val]!
    if t == 0 then 0.0
    else c.toFloat / t.toFloat

/-- Print evaluation metrics to console.

Convenience function to display accuracy and loss in a readable format.

**Parameters:**
- `net`: Neural network to evaluate
- `testData`: Test dataset
- `datasetName`: Name to display (e.g., "Test" or "Validation")
-/
def printMetrics
    (net : MLPArchitecture)
    (testData : Array (Vector 784 × Nat))
    (datasetName : String := "Test") : IO Unit := do
  let accuracy := computeAccuracy net testData
  let avgLoss := computeAverageLoss net testData
  IO.println s!"{datasetName} Accuracy: {accuracy * 100.0:.2f}%"
  IO.println s!"{datasetName} Loss: {avgLoss:.4f}"

end VerifiedNN.Training.Metrics
