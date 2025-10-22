import VerifiedNN.Data.MNIST
import VerifiedNN.Core.DataTypes

/-!
# MNIST Integration Test

Minimal integration smoke test to verify MNIST dataset can be loaded from disk.

## Main Tests

- `main`: Loads MNIST training and test sets, validates basic integrity

## Implementation Notes

**Test Strategy:** This is a fast smoke test (<5 seconds) that validates MNIST
data loading without extensive validation. It's designed for rapid iteration
and CI/CD pipelines where full validation would be too slow.

**Verification Status:** This is a computational validation test that checks
implementation correctness through basic integrity checks (correct counts,
valid label ranges). It complements the detailed validation in MNISTLoadTest.lean.

**Validated Properties:**
- Training set: 60,000 samples loaded successfully
- Test set: 10,000 samples loaded successfully
- All labels in valid range [0-9]
- Basic data structure integrity

**What's Not Validated:**
- Pixel value ranges (requires DataArrayN iteration utilities)
- Image content correctness (visual inspection needed)
- Data preprocessing correctness (handled separately in Preprocessing.lean)

## Usage

```bash
# Build
lake build VerifiedNN.Testing.MNISTIntegration

# Run (requires MNIST data in data/ directory)
lake env lean --run VerifiedNN/Testing/MNISTIntegration.lean

# Expected runtime: <5 seconds
```

## References

- Full validation: See VerifiedNN.Testing.MNISTLoadTest
- Data loading: See VerifiedNN.Data.MNIST
- Dataset download: Run ./scripts/download_mnist.sh
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
