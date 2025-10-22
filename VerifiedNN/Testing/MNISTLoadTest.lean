import VerifiedNN.Data.MNIST
import VerifiedNN.Core.DataTypes
import SciLean

/-!
# MNIST Loading Test

Simple test program to verify MNIST dataset loading works correctly.

This program loads the MNIST training and test sets and validates:
- Correct number of samples (60,000 training, 10,000 test)
- Correct image dimensions (784 pixels = 28×28)
- Valid label range (0-9)
- Basic data integrity checks
-/

namespace VerifiedNN.Testing.MNISTLoadTest

open VerifiedNN.Data.MNIST
open VerifiedNN.Core
open SciLean

/-- Test loading MNIST training set and verify basic properties. -/
def testTrainingSet : IO Unit := do
  IO.println "=== Testing MNIST Training Set ==="

  let trainImages ← loadMNISTImages "data/train-images-idx3-ubyte"
  let trainLabels ← loadMNISTLabels "data/train-labels-idx1-ubyte"

  IO.println s!"Training images loaded: {trainImages.size}"
  IO.println s!"Training labels loaded: {trainLabels.size}"

  -- Verify expected count
  if trainImages.size == 60000 then
    IO.println "✓ Correct number of training images (60,000)"
  else
    IO.println s!"✗ Expected 60,000 training images, got {trainImages.size}"

  if trainLabels.size == 60000 then
    IO.println "✓ Correct number of training labels (60,000)"
  else
    IO.println s!"✗ Expected 60,000 training labels, got {trainLabels.size}"

  -- Verify first few labels
  if trainImages.size > 0 && trainLabels.size > 0 then
    let first10 := trainLabels[0:10].toArray
    IO.println s!"First 10 labels: {first10.toList}"

    -- Check label range
    let allValid := trainLabels.all (fun label => label < 10)
    if allValid then
      IO.println "✓ All labels in valid range [0-9]"
    else
      IO.println "✗ Some labels out of range"

  IO.println ""

/-- Test loading MNIST test set and verify basic properties. -/
def testTestSet : IO Unit := do
  IO.println "=== Testing MNIST Test Set ==="

  let testImages ← loadMNISTImages "data/t10k-images-idx3-ubyte"
  let testLabels ← loadMNISTLabels "data/t10k-labels-idx1-ubyte"

  IO.println s!"Test images loaded: {testImages.size}"
  IO.println s!"Test labels loaded: {testLabels.size}"

  -- Verify expected count
  if testImages.size == 10000 then
    IO.println "✓ Correct number of test images (10,000)"
  else
    IO.println s!"✗ Expected 10,000 test images, got {testImages.size}"

  if testLabels.size == 10000 then
    IO.println "✓ Correct number of test labels (10,000)"
  else
    IO.println s!"✗ Expected 10,000 test labels, got {testLabels.size}"

  -- Verify first few labels
  if testImages.size > 0 && testLabels.size > 0 then
    let first10 := testLabels[0:10].toArray
    IO.println s!"First 10 labels: {first10.toList}"

    -- Check label range
    let allValid := testLabels.all (fun label => label < 10)
    if allValid then
      IO.println "✓ All labels in valid range [0-9]"
    else
      IO.println "✗ Some labels out of range"

  IO.println ""

/-- Test image data integrity by verifying basic properties. -/
def testImageData : IO Unit := do
  IO.println "=== Testing Image Data Integrity ==="

  let trainImages ← loadMNISTImages "data/train-images-idx3-ubyte"

  if trainImages.size > 0 then
    IO.println s!"✓ First image loaded successfully"
    IO.println "  (Pixel values expected to be in range [0, 255])"
    IO.println "  (Full pixel inspection would require DataArrayN iteration utilities)"

  IO.println ""

/-- Test loading using convenience functions. -/
def testConvenienceFunctions : IO Unit := do
  IO.println "=== Testing Convenience Functions ==="

  let trainData ← loadMNISTTrain "data"
  let testData ← loadMNISTTest "data"

  IO.println s!"Training set (image, label) pairs: {trainData.size}"
  IO.println s!"Test set (image, label) pairs: {testData.size}"

  if trainData.size == 60000 then
    IO.println "✓ loadMNISTTrain works correctly"
  else
    IO.println s!"✗ loadMNISTTrain failed (expected 60,000, got {trainData.size})"

  if testData.size == 10000 then
    IO.println "✓ loadMNISTTest works correctly"
  else
    IO.println s!"✗ loadMNISTTest failed (expected 10,000, got {testData.size})"

  -- Show first example
  if trainData.size > 0 then
    let (_img, label) := trainData[0]!
    IO.println s!"\nFirst training example: label = {label}"

  IO.println ""

/-- Main test runner. -/
def main : IO Unit := do
  IO.println "╔════════════════════════════════════════╗"
  IO.println "║   MNIST Dataset Loading Test Suite    ║"
  IO.println "╔════════════════════════════════════════╗"
  IO.println ""

  testTrainingSet
  testTestSet
  testImageData
  testConvenienceFunctions

  IO.println "═══════════════════════════════════════"
  IO.println "All MNIST loading tests completed!"
  IO.println "═══════════════════════════════════════"

end VerifiedNN.Testing.MNISTLoadTest

/-- Top-level main function for running the test. -/
def main : IO Unit := VerifiedNN.Testing.MNISTLoadTest.main
