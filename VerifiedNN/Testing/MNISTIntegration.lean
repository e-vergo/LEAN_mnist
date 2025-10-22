import VerifiedNN.Data.MNIST
import VerifiedNN.Core.DataTypes

/-!
# MNIST Integration Test

Minimal integration test to verify MNIST data can be loaded from disk.

This test validates:
- Training images and labels load successfully
- Test images and labels load successfully
- Image and label counts match expected values (60,000 training, 10,000 test)
- All labels are in valid range [0-9]

**Note:** This is a simple smoke test. Detailed validation of pixel values
requires DataArrayN iteration utilities not yet available in SciLean.
-/

namespace VerifiedNN.Testing.MNISTIntegration

open VerifiedNN.Data.MNIST
open VerifiedNN.Core

/-- Main integration test - loads MNIST and verifies basic properties. -/
def main : IO Unit := do
  IO.println "MNIST Integration Test"
  IO.println "======================"
  IO.println ""

  -- Load training data
  IO.println "Loading training set..."
  let trainData ← loadMNISTTrain "data"

  if trainData.size != 60000 then
    IO.eprintln s!"✗ Expected 60,000 training samples, got {trainData.size}"
    IO.Process.exit 1
  else
    IO.println s!"✓ Loaded {trainData.size} training samples"

  -- Verify first few labels
  if trainData.size > 0 then
    let firstLabels := (trainData[0:10].toArray.map (fun (_, label) => label)).toList
    IO.println s!"  First 10 labels: {firstLabels}"

    -- Check all labels are valid
    let allValid := trainData.all (fun (_, label) => label < 10)
    if !allValid then
      IO.eprintln "✗ Some training labels out of range [0-9]"
      IO.Process.exit 1
    else
      IO.println "  ✓ All labels in valid range [0-9]"

  -- Load test data
  IO.println ""
  IO.println "Loading test set..."
  let testData ← loadMNISTTest "data"

  if testData.size != 10000 then
    IO.eprintln s!"✗ Expected 10,000 test samples, got {testData.size}"
    IO.Process.exit 1
  else
    IO.println s!"✓ Loaded {testData.size} test samples"

  if testData.size > 0 then
    let firstLabels := (testData[0:10].toArray.map (fun (_, label) => label)).toList
    IO.println s!"  First 10 labels: {firstLabels}"

    -- Check all labels are valid
    let allValid := testData.all (fun (_, label) => label < 10)
    if !allValid then
      IO.eprintln "✗ Some test labels out of range [0-9]"
      IO.Process.exit 1
    else
      IO.println "  ✓ All labels in valid range [0-9]"

  IO.println ""
  IO.println "======================"
  IO.println "All checks passed! ✓"
  IO.println "======================"

end VerifiedNN.Testing.MNISTIntegration
