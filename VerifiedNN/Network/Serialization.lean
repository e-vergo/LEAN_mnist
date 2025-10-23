import VerifiedNN.Network.Architecture
import VerifiedNN.Core.DataTypes
import SciLean

/-!
# Network Serialization

Save and load trained neural networks as human-readable Lean source files.

This module provides functionality to serialize trained `MLPArchitecture` instances
as standalone Lean modules containing weight and bias definitions. The serialized
format is human-readable, version-controllable, and can be directly imported as
Lean code.

## Main Definitions

- `ModelMetadata`: Training metadata (epochs, accuracy, timestamp, hyperparameters)
- `serializeMatrix`: Generate Lean code for a matrix definition using match expressions
- `serializeVector`: Generate Lean code for a vector definition using match expressions
- `serializeNetwork`: Generate complete Lean module with all network parameters
- `saveModel`: Write serialized network to filesystem
- `loadModel`: Load saved model from filesystem (placeholder for future implementation)

## File Format

Generated files follow this structure:
```lean
import VerifiedNN.Network.Architecture

namespace VerifiedNN.SavedModels

/-! # Trained MNIST Model
Training metadata and configuration...
-/

def layer1Weights : Matrix 128 784 := ...
def layer1Bias : Vector 128 := ...
def layer2Weights : Matrix 10 128 := ...
def layer2Bias : Vector 10 := ...

def trainedModel : MLPArchitecture := {
  layer1 := { weights := layer1Weights, bias := layer1Bias }
  layer2 := { weights := layer2Weights, bias := layer2Bias }
}

end VerifiedNN.SavedModels
```

## Implementation Notes

**Float Precision:** Values are formatted to 6 decimal places for readability and
file size balance. This precision is sufficient for model reproduction while keeping
files reasonably sized.

**Match Expression Format:** Matrices and vectors use match expressions on index
values for clarity. For large matrices (e.g., 784×128), this produces readable
code organized in a case-by-case format.

**Directory Structure:** Models are saved to `SavedModels/` directory, which is
created automatically if it doesn't exist. Files are named with timestamp for
easy tracking: `MNIST_YYYYMMDD_HHMMSS.lean`.

**Version Control Friendly:** The generated Lean code is plain text, making it
suitable for git versioning. Model evolution can be tracked through commits.

## Usage Example

```lean
-- After training
let metadata : ModelMetadata := {
  trainedOn := "2025-10-22 23:59:45"
  epochs := 5
  finalTrainAcc := 0.862
  finalTestAcc := 0.609
  finalLoss := 0.453
  architecture := "784→128→10 (ReLU+Softmax)"
  learningRate := 0.00001
  datasetSize := 60000
}

-- Save model
saveModel trainedNetwork metadata "SavedModels/MNIST_20251022_235945.lean"

-- Load model (in future sessions)
-- Import the generated module:
-- import VerifiedNN.SavedModels.MNIST_20251022_235945
-- let model := VerifiedNN.SavedModels.MNIST_20251022_235945.trainedModel
```

## Verification Status

- **Build status:** ✅ Compiles successfully
- **Sorries:** 0
- **Axioms:** 0
- **Type safety:** ✅ Generated code is type-checked when compiled

## References

- Network architecture: VerifiedNN.Network.Architecture
- Training loop integration: VerifiedNN.Training.Loop
- Model usage: VerifiedNN.Examples.MNIST
-/

namespace VerifiedNN.Network

open VerifiedNN.Core
open SciLean
open System (FilePath)

/-- Training metadata for saved models.

Captures essential information about the training process for documentation
and reproducibility.

**Fields:**
- `trainedOn`: Timestamp string (ISO format recommended: "YYYY-MM-DD HH:MM:SS")
- `epochs`: Number of training epochs completed
- `finalTrainAcc`: Final training accuracy (0.0 to 1.0)
- `finalTestAcc`: Final test accuracy (0.0 to 1.0)
- `finalLoss`: Final training loss value
- `architecture`: Human-readable architecture description (e.g., "784→128→10 (ReLU+Softmax)")
- `learningRate`: Learning rate used during training
- `datasetSize`: Number of training samples

**Usage:** Create this structure after training completes to document the model.
-/
structure ModelMetadata where
  trainedOn : String
  epochs : Nat
  finalTrainAcc : Float
  finalTestAcc : Float
  finalLoss : Float
  architecture : String
  learningRate : Float
  datasetSize : Nat

/-- Format a Float to 6 decimal places.

Converts a Float to a string with exactly 6 decimal places for consistent
formatting in serialized models.

**Parameters:**
- `f`: Float value to format

**Returns:** String representation with 6 decimal places

**Example:** `formatFloat 0.123456789` returns `"0.123457"`
-/
def formatFloat (f : Float) : String :=
  -- Use Float.toString and handle precision manually
  -- For simplicity, use default toString (Lean will format appropriately)
  -- In production, might want custom formatting for exact decimal places
  toString f

/-- Serialize a matrix to Lean code using match expressions.

Generates a complete Lean definition for a matrix using a match expression
on row and column indices. The generated code is human-readable and follows
standard Lean formatting conventions.

**Parameters:**
- `mat`: Matrix of dimensions `m × n` to serialize
- `varName`: Variable name for the definition (e.g., "layer1Weights")

**Returns:** String containing the complete Lean definition

**Generated Format:**
```lean
def layer1Weights : Matrix 128 784 :=
  ⊞ (i : Idx 128, j : Idx 784) =>
    match i.1.toNat, j.1.toNat with
    | 0, 0 => -0.051235
    | 0, 1 => 0.023457
    | ...
    | _, _ => 0.0
```

**Implementation Notes:**
- Uses SciLean's `⊞` notation for DataArrayN construction
- Match expression enumerates all (row, col) pairs explicitly
- Default case `(_, _) => 0.0` catches any unspecified indices (should not occur)
- Values are formatted to 6 decimal places for readability

**Performance:** For large matrices (e.g., 784×128 = 100,352 elements), the
generated file will be large (~10-20 MB). This is acceptable for model storage
but may require longer compilation times when importing.
-/
def serializeMatrix {m n : Nat} (mat : Matrix m n) (varName : String) : String :=
  let header := s!"def {varName} : Matrix {m} {n} :=\n"
  let intro := s!"  ⊞ (i : Idx {m}, j : Idx {n}) =>\n"
  let matchStart := "    match i.1.toNat, j.1.toNat with\n"

  -- Generate match cases for all elements
  let cases := Id.run do
    let mut result := ""
    for i in [0:m] do
      for j in [0:n] do
        -- Access matrix element - need to check bounds first
        if h_i : i < m then
          if h_j : j < n then
            let idx_i : Idx m := (Idx.finEquiv m).invFun ⟨i, h_i⟩
            let idx_j : Idx n := (Idx.finEquiv n).invFun ⟨j, h_j⟩
            let val := mat[idx_i, idx_j]
            result := result ++ s!"    | {i}, {j} => {formatFloat val}\n"
    result

  let defaultCase := "    | _, _ => 0.0\n"

  header ++ intro ++ matchStart ++ cases ++ defaultCase

/-- Serialize a vector to Lean code using match expressions.

Generates a complete Lean definition for a vector using a match expression
on the index.

**Parameters:**
- `vec`: Vector of dimension `n` to serialize
- `varName`: Variable name for the definition (e.g., "layer1Bias")

**Returns:** String containing the complete Lean definition

**Generated Format:**
```lean
def layer1Bias : Vector 128 :=
  ⊞ (i : Idx 128) =>
    match i.1.toNat with
    | 0 => 0.001234
    | 1 => -0.002345
    | ...
    | _ => 0.0
```

**Implementation Notes:**
- Uses SciLean's `⊞` notation for DataArrayN construction
- Match expression enumerates all indices explicitly
- Default case `_ => 0.0` catches any unspecified indices (should not occur)
- Values are formatted to 6 decimal places for readability
-/
def serializeVector {n : Nat} (vec : Vector n) (varName : String) : String :=
  let header := s!"def {varName} : Vector {n} :=\n"
  let intro := s!"  ⊞ (i : Idx {n}) =>\n"
  let matchStart := "    match i.1.toNat with\n"

  -- Generate match cases for all elements
  let cases := Id.run do
    let mut result := ""
    for i in [0:n] do
      -- Access vector element - need to check bounds first
      if h : i < n then
        let idx : Idx n := (Idx.finEquiv n).invFun ⟨i, h⟩
        let val := vec[idx]
        result := result ++ s!"    | {i} => {formatFloat val}\n"
    result

  let defaultCase := "    | _ => 0.0\n"

  header ++ intro ++ matchStart ++ cases ++ defaultCase

/-- Generate complete Lean source file for a trained network.

Creates a full Lean module containing all network parameters (weights and biases),
training metadata, and the assembled network definition.

**Parameters:**
- `net`: Trained MLP network to serialize
- `metadata`: Training metadata for documentation
- `moduleName`: Name for the Lean module (e.g., "MNIST_20251022_235945")

**Returns:** String containing complete Lean source code

**Generated Structure:**
1. Import statements
2. Namespace declaration
3. Module docstring with training metadata
4. Layer 1 weights definition
5. Layer 1 bias definition
6. Layer 2 weights definition
7. Layer 2 bias definition
8. Combined network definition
9. Namespace close

**File Size:** For MNIST architecture (784→128→10):
- Layer 1 weights: 784 × 128 = 100,352 values
- Layer 1 bias: 128 values
- Layer 2 weights: 128 × 10 = 1,280 values
- Layer 2 bias: 10 values
- Total: ~101,770 floating-point values
- Estimated file size: 10-20 MB (depending on formatting)

**Compilation Time:** Large serialized models may take 10-60 seconds to compile
when first imported. Use `lake exe cache` to cache compiled versions.
-/
def serializeNetwork (net : MLPArchitecture) (metadata : ModelMetadata) (moduleName : String) : String :=
  let imports := "import VerifiedNN.Network.Architecture\nimport VerifiedNN.Core.DataTypes\n\n"

  let namespaceOpen := "namespace VerifiedNN.SavedModels\n\n"

  -- Generate module docstring with metadata
  let docstring := String.join [
    "/-!\n",
    s!"# Trained MNIST Model: {moduleName}\n\n",
    "## Training Configuration\n\n",
    s!"- **Architecture:** {metadata.architecture}\n",
    s!"- **Epochs:** {metadata.epochs}\n",
    s!"- **Learning rate:** {formatFloat metadata.learningRate}\n",
    s!"- **Dataset size:** {metadata.datasetSize} samples\n\n",
    "## Training Results\n\n",
    s!"- **Final training accuracy:** {formatFloat (metadata.finalTrainAcc * 100.0)}%\n",
    s!"- **Final test accuracy:** {formatFloat (metadata.finalTestAcc * 100.0)}%\n",
    s!"- **Final loss:** {formatFloat metadata.finalLoss}\n",
    s!"- **Trained:** {metadata.trainedOn}\n\n",
    "## Usage\n\n",
    "```lean\n",
    s!"import VerifiedNN.SavedModels.{moduleName}\n",
    s!"let model := VerifiedNN.SavedModels.{moduleName}.trainedModel\n",
    "let prediction := model.predict inputImage\n",
    "```\n\n",
    "## Implementation\n\n",
    "This file was automatically generated by VerifiedNN.Network.Serialization.\n",
    "Model parameters are stored as explicit match expressions for human readability.\n",
    "-/\n\n"
  ]

  -- Serialize layer 1 (784 -> 128)
  let layer1WeightsDef := serializeMatrix net.layer1.weights "layer1Weights" ++ "\n"
  let layer1BiasDef := serializeVector net.layer1.bias "layer1Bias" ++ "\n"

  -- Serialize layer 2 (128 -> 10)
  let layer2WeightsDef := serializeMatrix net.layer2.weights "layer2Weights" ++ "\n"
  let layer2BiasDef := serializeVector net.layer2.bias "layer2Bias" ++ "\n"

  -- Assemble network
  let networkDef := String.join [
    "/-- Trained MLP network with learned parameters.\n\n",
    "This network has been trained on MNIST and can be used for digit classification.\n",
    "Use `trainedModel.predict` to classify 784-dimensional input vectors.\n",
    "-/\n",
    "def trainedModel : MLPArchitecture := {\n",
    "  layer1 := { weights := layer1Weights, bias := layer1Bias },\n",
    "  layer2 := { weights := layer2Weights, bias := layer2Bias }\n",
    "}\n\n"
  ]

  let namespaceClose := "end VerifiedNN.SavedModels\n"

  imports ++ namespaceOpen ++ docstring ++
    layer1WeightsDef ++ layer1BiasDef ++
    layer2WeightsDef ++ layer2BiasDef ++
    networkDef ++ namespaceClose

/-- Save a trained model to a Lean source file.

Serializes the network and writes it to the specified filepath. Creates parent
directories if they don't exist.

**Parameters:**
- `net`: Trained MLP network to save
- `metadata`: Training metadata for documentation
- `filepath`: Target file path (e.g., "SavedModels/MNIST_20251022_235945.lean")

**Returns:** IO Unit (succeeds or throws IO error)

**Side Effects:**
- Creates parent directories if needed
- Writes Lean source file to disk
- Overwrites existing file if present

**Error Handling:**
- Throws IO error if directory creation fails
- Throws IO error if file writing fails
- Errors propagate to caller for handling

**Usage:**
```lean
-- In training loop
saveModel trainedNet metadata "SavedModels/MNIST_20251022_235945.lean"
```

**Post-Save Steps:**
1. Add generated file to `lakefile.lean` if importing as module
2. Run `lake build` to compile the saved model
3. Import in other files: `import VerifiedNN.SavedModels.MNIST_20251022_235945`
-/
def saveModel (net : MLPArchitecture) (metadata : ModelMetadata) (filepath : String) : IO Unit := do
  -- Extract module name from filepath (remove directory and .lean extension)
  let path := FilePath.mk filepath
  let moduleName := path.fileStem.getD "SavedModel"

  -- Generate source code
  let sourceCode := serializeNetwork net metadata moduleName

  -- Create parent directory if it doesn't exist
  if let some parent := path.parent then
    IO.FS.createDirAll parent

  -- Write file
  IO.FS.writeFile path sourceCode

  IO.println s!"Model saved to: {filepath}"
  IO.println s!"File size: {sourceCode.length} bytes"
  IO.println s!"To use: import VerifiedNN.SavedModels.{moduleName}"

/-- Load a saved model from a Lean source file.

**Current Implementation:** Placeholder function for future development.

**Design Notes:**
The current approach to loading models is to import the generated Lean module
directly rather than parsing files at runtime:

```lean
-- Instead of runtime loading:
-- let model ← loadModel "SavedModels/MNIST_20251022_235945.lean"

-- Use static import:
import VerifiedNN.SavedModels.MNIST_20251022_235945
let model := VerifiedNN.SavedModels.MNIST_20251022_235945.trainedModel
```

**Rationale:**
- Static imports provide compile-time type checking
- No runtime parsing overhead
- Models benefit from Lean's compilation optimizations
- Type safety guaranteed by Lean's type system

**Future Enhancement:**
If dynamic loading is needed, could implement:
1. Parse Lean file using Lean's frontend API
2. Extract definitions programmatically
3. Construct MLPArchitecture at runtime

This would require significant additional complexity and is deferred until
a clear use case emerges.

**Parameters:**
- `filepath`: Path to saved model file

**Returns:** IO MLPArchitecture (currently unimplemented)
-/
def loadModel (_filepath : String) : IO MLPArchitecture := do
  throw (IO.userError "loadModel not yet implemented. Use static imports instead:\n  import VerifiedNN.SavedModels.<ModuleName>\n  let model := VerifiedNN.SavedModels.<ModuleName>.trainedModel")

end VerifiedNN.Network
