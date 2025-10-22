import VerifiedNN.Data.MNIST
import VerifiedNN.Util.ImageRenderer

/-!
# MNIST ASCII Renderer Demo

Demonstration executable showing ASCII art rendering of MNIST digits.

## Purpose

This executable demonstrates that **pure Lean visualization tools work** even when
the training executables cannot be built due to SciLean's noncomputable AD.
Since the renderer uses zero automatic differentiation, it compiles to a
standalone binary without issues.

## What This Proves

✅ **Lean can do practical I/O and visualization**
✅ **Computable functions work in executables**
✅ **Not all useful tools require noncomputable operations**
✅ **We can debug/visualize network data despite AD limitations**

## Usage

```bash
# Render first 10 test samples
lake exe renderMNIST

# Render with specific options
lake exe renderMNIST --count 5 --inverted --train
```

## Command-Line Options

- `--count N`: Number of images to render (default: 10)
- `--inverted`: Use inverted mode for light terminals
- `--train`: Use training set instead of test set
- `--help`: Display usage information

## Output

For each sample, displays:
1. Sample index and ground truth label
2. 28×28 ASCII art rendering
3. Blank line separator

This provides immediate visual feedback on:
- Whether MNIST data loaded correctly
- How the network "sees" the digits
- Quality of preprocessing

## Implementation Note

This executable is **completely computable** - it uses:
- File I/O (computable in Lean 4)
- Array operations (computable)
- String building (computable)
- Basic arithmetic (computable)

No automatic differentiation, no noncomputable operations.
-/

namespace VerifiedNN.Examples.RenderMNIST

open VerifiedNN.Core
open VerifiedNN.Data.MNIST
open VerifiedNN.Util.ImageRenderer

/-- Configuration for rendering demo. -/
structure RenderConfig where
  count : Nat := 10        -- Number of images to render
  inverted : Bool := false -- Use inverted brightness
  useTrain : Bool := false -- Use train set vs test set

/-- Parse command-line arguments.

Supports:
- `--count N`: Number of images to display
- `--inverted`: Use inverted mode for light terminals
- `--train`: Use training dataset instead of test dataset
- `--help`: Show usage information

**Returns:** IO action producing RenderConfig or exiting on error/help
-/
def parseArgs (args : List String) : IO RenderConfig := do
  let rec parseLoop (config : RenderConfig) (remaining : List String) : IO RenderConfig := do
    match remaining with
    | [] => return config
    | "--count" :: value :: rest =>
      match value.toNat? with
      | some n =>
        if n > 0 then
          parseLoop { config with count := n } rest
        else
          IO.eprintln s!"Error: --count must be positive (got {value})"
          IO.Process.exit 1
      | none =>
        IO.eprintln s!"Error: Invalid --count value: {value}"
        IO.Process.exit 1
    | "--inverted" :: rest =>
      parseLoop { config with inverted := true } rest
    | "--train" :: rest =>
      parseLoop { config with useTrain := true } rest
    | "--help" :: _ =>
      IO.println "MNIST ASCII Renderer Demo"
      IO.println ""
      IO.println "Usage: lake exe renderMNIST [OPTIONS]"
      IO.println ""
      IO.println "Options:"
      IO.println "  --count N      Number of images to render (default: 10)"
      IO.println "  --inverted     Use inverted brightness for light terminals"
      IO.println "  --train        Use training set instead of test set"
      IO.println "  --help         Show this help message"
      IO.println ""
      IO.println "Example:"
      IO.println "  lake exe renderMNIST --count 5 --inverted"
      IO.Process.exit 0
    | unknown :: _ =>
      IO.eprintln s!"Error: Unknown argument: {unknown}"
      IO.eprintln "Use --help for usage information"
      IO.Process.exit 1

  parseLoop {} args

/-- Main rendering demo.

Loads MNIST data and renders the first N samples as ASCII art to stdout.

**Algorithm:**
1. Parse command-line arguments
2. Load appropriate dataset (train or test)
3. For each of the first N samples:
   - Display sample index and ground truth label
   - Render image as ASCII art
   - Add separator line
4. Display summary statistics

**Error Handling:**
If data fails to load, prints error and exits with code 1.
Users should run `./scripts/download_mnist.sh` first.
-/
def runMain (args : List String) : IO Unit := do
  let config ← parseArgs args

  IO.println "=========================================="
  IO.println "MNIST ASCII Renderer Demo"
  IO.println "Verified Neural Network in Lean 4"
  IO.println "=========================================="
  IO.println ""

  -- Load dataset
  let datasetName := if config.useTrain then "training" else "test"
  IO.println s!"Loading {datasetName} data..."

  let data ← if config.useTrain then
    loadMNISTTrain "data"
  else
    loadMNISTTest "data"

  if data.size == 0 then
    IO.eprintln "Error: Failed to load MNIST data"
    IO.eprintln "Please run ./scripts/download_mnist.sh to download dataset"
    IO.Process.exit 1

  let countToRender := Nat.min config.count data.size
  IO.println s!"Loaded {data.size} samples"
  IO.println s!"Rendering first {countToRender} samples"
  IO.println ""

  let invertedStr := if config.inverted then " (inverted mode)" else ""
  IO.println s!"Display mode: ASCII art{invertedStr}"
  IO.println "=========================================="
  IO.println ""

  -- Render each sample
  for i in [0:countToRender] do
    let (image, label) := data[i]!

    IO.println s!"Sample {i} | Ground Truth: {label}"
    IO.println "----------------------------"

    let ascii := renderImage image config.inverted
    IO.println ascii
    IO.println ""

  -- Summary
  IO.println "=========================================="
  IO.println s!"Rendered {countToRender} images"
  IO.println "=========================================="

end VerifiedNN.Examples.RenderMNIST

-- Main entry point (must be at root level for executable)
def main (args : List String) : IO Unit :=
  VerifiedNN.Examples.RenderMNIST.runMain args
