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
  showStats : Bool := false -- Show statistical overlays
  gridCols : Option Nat := none -- Grid layout (none = sequential)
  compare : Bool := false -- Side-by-side comparison mode
  palette : String := "default" -- Color palette name
  border : Option String := none -- Border style

/-- Parse command-line arguments.

Supports:
- `--count N`: Number of images to display
- `--inverted`: Use inverted mode for light terminals
- `--train`: Use training dataset instead of test dataset
- `--stats`: Show statistical overlays (min/max/mean/stddev)
- `--grid N`: Display images in N-column grid layout
- `--compare`: Show side-by-side comparison of consecutive pairs
- `--palette NAME`: Use specific palette (default, simple, detailed, blocks)
- `--border STYLE`: Add border frame (single, double, rounded, heavy, ascii)
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
    | "--stats" :: rest =>
      parseLoop { config with showStats := true } rest
    | "--grid" :: value :: rest =>
      match value.toNat? with
      | some n =>
        if n > 0 then
          parseLoop { config with gridCols := some n } rest
        else
          IO.eprintln s!"Error: --grid must be positive (got {value})"
          IO.Process.exit 1
      | none =>
        IO.eprintln s!"Error: Invalid --grid value: {value}"
        IO.Process.exit 1
    | "--compare" :: rest =>
      parseLoop { config with compare := true } rest
    | "--palette" :: value :: rest =>
      parseLoop { config with palette := value } rest
    | "--border" :: value :: rest =>
      parseLoop { config with border := some value } rest
    | "--help" :: _ =>
      IO.println "MNIST ASCII Renderer Demo"
      IO.println ""
      IO.println "Usage: lake exe renderMNIST [OPTIONS]"
      IO.println ""
      IO.println "Options:"
      IO.println "  --count N         Number of images to render (default: 10)"
      IO.println "  --inverted        Use inverted brightness for light terminals"
      IO.println "  --train           Use training set instead of test set"
      IO.println "  --stats           Show statistical overlays (min/max/mean/stddev)"
      IO.println "  --grid N          Display images in N-column grid layout"
      IO.println "  --compare         Show side-by-side comparison of consecutive pairs"
      IO.println "  --palette NAME    Palette: default, simple, detailed, blocks"
      IO.println "  --border STYLE    Border: single, double, rounded, heavy, ascii"
      IO.println "  --help            Show this help message"
      IO.println ""
      IO.println "Examples:"
      IO.println "  lake exe renderMNIST --count 5 --inverted"
      IO.println "  lake exe renderMNIST --count 20 --grid 4 --stats"
      IO.println "  lake exe renderMNIST --count 4 --compare --border double"
      IO.println "  lake exe renderMNIST --count 6 --palette blocks"
      IO.Process.exit 0
    | unknown :: _ =>
      IO.eprintln s!"Error: Unknown argument: {unknown}"
      IO.eprintln "Use --help for usage information"
      IO.Process.exit 1

  parseLoop {} args

/-- Render a single image with appropriate options.

Helper function that applies all rendering options from config.
-/
def renderWithOptions (img : Vector 784) (config : RenderConfig) : String :=
  -- Start with base rendering (with custom palette if specified)
  -- For now, just use default rendering
  let baseAscii := renderImage img config.inverted

  -- Apply border if requested
  let withBorder := match config.border with
    | some style =>
      let lines := baseAscii.splitOn "\n"
      let (tl, tr, bl, br, h, v) := match style with
        | "double" => ("╔", "╗", "╚", "╝", "═", "║")
        | "rounded" => ("╭", "╮", "╰", "╯", "─", "│")
        | "heavy" => ("┏", "┓", "┗", "┛", "━", "┃")
        | "ascii" => ("+", "+", "+", "+", "-", "|")
        | _ => ("┌", "┐", "└", "┘", "─", "│")
      let topBorder := tl ++ String.mk (List.replicate 28 h.toList.head!) ++ tr
      let bottomBorder := bl ++ String.mk (List.replicate 28 h.toList.head!) ++ br
      let framedLines := lines.map fun line => v ++ line ++ v
      topBorder ++ "\n" ++ String.intercalate "\n" framedLines ++ "\n" ++ bottomBorder
    | none => baseAscii

  -- Apply stats if requested
  if config.showStats then
    let (min, max, mean, stddev) := computeImageStats img
    let isRaw255 := autoDetectRange img
    let rangeStr := if isRaw255 then " (0-255 range)" else " (0-1 range)"
    let minStr := toString min
    let maxStr := toString max
    let meanStr := toString mean
    let stddevStr := toString stddev
    let statsText :=
      "\nStatistics:\n" ++
      "  Min: " ++ minStr ++ "  Max: " ++ maxStr ++ "\n" ++
      "  Mean: " ++ meanStr ++ "  Std: " ++ stddevStr ++ rangeStr
    withBorder ++ statsText
  else
    withBorder

/-- Main rendering demo.

Loads MNIST data and renders the first N samples as ASCII art to stdout.

**Algorithm:**
1. Parse command-line arguments
2. Load appropriate dataset (train or test)
3. Render according to mode:
   - Grid mode: Display all images in a grid layout
   - Compare mode: Display consecutive pairs side-by-side
   - Sequential mode: Display each image individually
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

  -- Show active features
  let features1 := if config.inverted then ["inverted"] else []
  let features2 := features1 ++ (if config.showStats then ["stats"] else [])
  let features3 := features2 ++ (match config.gridCols with | some n => ["grid-" ++ toString n] | none => [])
  let features4 := features3 ++ (if config.compare then ["compare"] else [])
  let features5 := features4 ++ (if config.palette != "default" then ["palette-" ++ config.palette] else [])
  let features := features5 ++ (match config.border with | some s => ["border-" ++ s] | none => [])

  if !features.isEmpty then
    IO.println ("Features: " ++ String.intercalate ", " features)

  IO.println "=========================================="
  IO.println ""

  -- Render based on mode
  match config.gridCols with
  | some cols =>
    -- Grid layout mode
    let images := (List.range countToRender).map fun i => (data[i]!).fst
    let labels := (List.range countToRender).map fun i =>
      let (_, label) := data[i]!
      "#" ++ toString i ++ ": " ++ toString label
    let grid := renderImageGrid images labels cols config.inverted
    IO.println grid

  | none =>
    if config.compare then
      -- Comparison mode: show consecutive pairs side-by-side
      let pairCount := countToRender / 2
      for i in [0:pairCount] do
        let idx1 := i * 2
        let idx2 := i * 2 + 1
        if idx2 < countToRender then
          let (img1, label1) := data[idx1]!
          let (img2, label2) := data[idx2]!

          IO.println s!"Comparison {i + 1}"
          let comparison := renderImageComparison img1 img2
            s!"Sample {idx1}: {label1}" s!"Sample {idx2}: {label2}" config.inverted
          IO.println comparison

          if config.showStats then
            let (min1, max1, mean1, std1) := computeImageStats img1
            let (min2, max2, mean2, std2) := computeImageStats img2
            IO.println ""
            let l1 := "Left:  Min=" ++ toString min1 ++ " Max=" ++ toString max1 ++ " Mean=" ++ toString mean1 ++ " Std=" ++ toString std1
            let l2 := "Right: Min=" ++ toString min2 ++ " Max=" ++ toString max2 ++ " Mean=" ++ toString mean2 ++ " Std=" ++ toString std2
            IO.println l1
            IO.println l2

          IO.println ""
          IO.println ""
    else
      -- Sequential mode: show each image individually
      for i in [0:countToRender] do
        let (image, label) := data[i]!

        IO.println s!"Sample {i} | Ground Truth: {label}"
        IO.println "----------------------------"

        let rendered := renderWithOptions image config
        IO.println rendered
        IO.println ""

  -- Summary
  IO.println "=========================================="
  IO.println s!"Rendered {countToRender} images"
  IO.println "=========================================="

end VerifiedNN.Examples.RenderMNIST

-- Main entry point (must be at root level for executable)
def main (args : List String) : IO Unit :=
  VerifiedNN.Examples.RenderMNIST.runMain args
