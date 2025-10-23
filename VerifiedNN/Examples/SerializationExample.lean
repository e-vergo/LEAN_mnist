import VerifiedNN.Network.Architecture
import VerifiedNN.Network.Initialization
import VerifiedNN.Network.Serialization

/-!
# Model Serialization Example

Demonstrates how to save and load trained neural networks using the
human-readable Lean source file format.

This example shows:
1. Creating a network with random initialization
2. Preparing training metadata
3. Saving the model to a Lean source file
4. Instructions for loading the saved model

## Usage

Run this example to generate a sample saved model:
```bash
lake exe serializationExample
```

This will create `SavedModels/Example_MNIST.lean` containing the serialized network.

## Verification Status

- **Build status:** ✅ Compiles successfully
- **Sorries:** 0
- **Axioms:** 0 (uses axiomatized initialization but no new axioms)
-/

namespace VerifiedNN.Examples

open VerifiedNN.Network
open VerifiedNN.Network.Initialization
open VerifiedNN.Core

/-- Example: Create and save a model with random initialization -/
def exampleSaveModel : IO Unit := do
  IO.println "Creating example network with random initialization..."

  -- Initialize network with Xavier/Glorot initialization
  let net ← initializeNetwork

  IO.println "Network created successfully"
  IO.println s!"  Layer 1: 784 → 128 (weights: 784×128 = {784 * 128} parameters)"
  IO.println s!"  Layer 2: 128 → 10 (weights: 128×10 = {128 * 10} parameters)"
  IO.println s!"  Total parameters: {784 * 128 + 128 + 128 * 10 + 10}"

  -- Create metadata (simulating a trained model)
  let metadata : ModelMetadata := {
    trainedOn := "2025-10-22 Example"
    epochs := 5
    finalTrainAcc := 0.862
    finalTestAcc := 0.609
    finalLoss := 0.453
    architecture := "784→128→10 (ReLU+Softmax)"
    learningRate := 0.00001
    datasetSize := 60000
  }

  IO.println "\nMetadata:"
  IO.println s!"  Architecture: {metadata.architecture}"
  IO.println s!"  Epochs: {metadata.epochs}"
  IO.println s!"  Learning rate: {metadata.learningRate}"
  IO.println s!"  Final train accuracy: {metadata.finalTrainAcc * 100.0}%"
  IO.println s!"  Final test accuracy: {metadata.finalTestAcc * 100.0}%"
  IO.println s!"  Final loss: {metadata.finalLoss}"

  -- Save model
  let filepath := "SavedModels/Example_MNIST.lean"
  IO.println s!"\nSaving model to: {filepath}"
  IO.println "This may take a moment for large models..."

  saveModel net metadata filepath

  IO.println "\n✅ Model saved successfully!"
  IO.println "\nTo use the saved model:"
  IO.println "1. Add to lakefile.lean (if needed):"
  IO.println "   lean_lib VerifiedNN.SavedModels"
  IO.println "2. Import in your code:"
  IO.println "   import VerifiedNN.SavedModels.Example_MNIST"
  IO.println "3. Use the model:"
  IO.println "   let model := VerifiedNN.SavedModels.Example_MNIST.trainedModel"
  IO.println "   let prediction := model.predict inputImage"

end VerifiedNN.Examples
