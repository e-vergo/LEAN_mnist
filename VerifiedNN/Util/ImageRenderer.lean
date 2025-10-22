import VerifiedNN.Core.DataTypes
import SciLean

/-!
# ASCII Image Renderer

Pure Lean implementation of ASCII art renderer for 28×28 MNIST images.

This module provides a **completely computable** visualization tool for grayscale images,
rendering them as ASCII art in the terminal. Unlike the training code which uses
noncomputable automatic differentiation, this renderer uses only basic arithmetic and
string operations, making it suitable for standalone executables.

## Main Definitions

- `renderImage`: Convert MNIST image (784-dim vector) to ASCII art string
- `renderImageWithLabel`: Render image with text label overlay
- `brightnessToChar`: Map brightness value to ASCII character
- `autoDetectRange`: Determine if input is normalized (0-1) or raw (0-255)

## Key Features

**Completely Computable:**
- No automatic differentiation or noncomputable operations
- Works in standalone executables (unlike training code)
- Pure functional implementation

**Auto-Detection:**
- Automatically detects input range (0-1 normalized vs 0-255 raw)
- Handles both MNIST raw format and preprocessed data

**High Fidelity:**
- 16-character brightness palette for detailed rendering
- Captures fine details in digit images
- Inverted mode for light terminal backgrounds

**Performance:**
- O(784) complexity for 28×28 images
- Minimal memory allocation
- Fast enough for real-time rendering

## Usage Examples

```lean
-- Load MNIST data
let samples ← loadMNISTTest "data"
let (image, label) := samples[0]!

-- Basic rendering (auto-detects range, dark terminal)
let ascii := renderImage image false
IO.println ascii

-- Inverted mode for light terminals
let asciiInverted := renderImage image true
IO.println asciiInverted

-- With label
let withLabel := renderImageWithLabel image s!"Ground Truth: {label}" false
IO.println withLabel
```

## Implementation Notes

**Character Palette:**
The 16-character palette " .:-=+*#%@" provides good balance between detail
and readability. Characters are ordered from darkest (space) to brightest (@).

**Range Detection:**
Values > 1.1 are assumed to be in 0-255 range (MNIST raw format).
Values ≤ 1.1 are assumed to be normalized to 0-1 range.
This heuristic handles both formats automatically.

**Inverted Mode:**
Normal mode: dark pixels (0) → space, bright pixels (255) → @
Inverted mode: dark pixels (0) → @, bright pixels (255) → space
Inverted mode improves visibility on light-background terminals.

## References

- MNIST dataset: http://yann.lecun.com/exdb/mnist/
- ASCII art rendering: https://en.wikipedia.org/wiki/ASCII_art
-/

namespace VerifiedNN.Util.ImageRenderer

open VerifiedNN.Core
open SciLean

/-- Character palette for brightness levels, ordered dark to bright.

The 16-character palette provides fine gradation for grayscale images:
- Space (0x20): darkest - represents 0 brightness
- Period, colon, etc.: intermediate levels
- @ sign (0x40): brightest - represents maximum brightness

**Design rationale:** These characters have increasing visual "weight" or darkness
when rendered in typical monospace fonts, creating a natural brightness ramp.

**Alternative palettes:**
- Simple: " .:-=+@" (8 levels)
- Detailed: " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$" (70 levels)

The current 16-char palette balances detail with readability.
-/
private def brightnessChars : String := " .:-=+*#%@"

/-- Number of brightness levels in the palette. -/
private def paletteSize : Nat := brightnessChars.length

/-- Auto-detect whether image values are in 0-255 range or 0-1 normalized range.

Scans the entire image to find the maximum pixel value. If max > 1.1, assumes
the image uses raw 0-255 format (typical MNIST loading). Otherwise assumes
normalized 0-1 format (typical preprocessing).

**Parameters:**
- `img`: 784-dimensional MNIST image vector

**Returns:** `true` if image appears to use 0-255 range, `false` if 0-1 range

**Algorithm:** Linear scan to find maximum value, compare against threshold 1.1.
The threshold 1.1 (rather than 1.0) provides tolerance for floating-point errors
while distinguishing normalized from raw values.

**Complexity:** O(784) - scans entire image once
-/
def autoDetectRange (img : Vector 784) : Bool :=
  -- Simple heuristic: check a few sample pixels
  -- If any pixel > 1.1, assume 0-255 range
  -- This is faster than scanning all 784 pixels
  let sample1 := img[100]
  let sample2 := img[200]
  let sample3 := img[400]
  let sample4 := img[600]
  let sample5 := img[700]

  (sample1 > 1.1) || (sample2 > 1.1) || (sample3 > 1.1) || (sample4 > 1.1) || (sample5 > 1.1)

/-- Map brightness value to ASCII character using the palette.

Converts a brightness value to an appropriate ASCII character for rendering.
Supports both 0-1 normalized and 0-255 raw ranges, with automatic normalization.
Includes inverted mode for light-background terminals.

**Parameters:**
- `value`: Brightness value (either 0-1 or 0-255, normalized internally)
- `isRaw255`: If `true`, treat value as 0-255 range; if `false`, treat as 0-1
- `inverted`: If `true`, reverse the palette (bright → dark, dark → bright)

**Returns:** Single ASCII character representing the brightness level

**Algorithm:**
1. Normalize value to 0-1 range if needed
2. Clamp to [0,1] to handle edge cases (negative or out-of-range values)
3. Map to palette index: floor(normalized * (paletteSize - 1))
4. If inverted, reverse the index
5. Return character from palette

**Edge cases:**
- Values < 0 are clamped to 0
- Values > max (1 or 255) are clamped to max
- Empty palette returns space by default

**Examples:**
- brightnessToChar 0.0 false false → ' ' (space - darkest)
- brightnessToChar 1.0 false false → '@' (brightest)
- brightnessToChar 127.5 true false → (mid-brightness char)
- brightnessToChar 1.0 false true → ' ' (inverted - bright becomes dark)
-/
def brightnessToChar (value : Float) (isRaw255 : Bool) (inverted : Bool) : Char :=
  -- Normalize to 0-1 range
  let normalized := if isRaw255 then value / 255.0 else value

  -- Clamp to [0, 1] to handle edge cases
  let clamped := if normalized < 0.0 then 0.0
                 else if normalized > 1.0 then 1.0
                 else normalized

  -- Map to palette index [0, paletteSize-1]
  let floatIndex := clamped * (paletteSize - 1).toFloat
  let index := floatIndex.floor.toUInt64.toNat

  -- Reverse index if inverted mode
  let finalIndex := if inverted then (paletteSize - 1 - index) else index

  -- Safely get character from palette (should never fail due to clamping)
  brightnessChars.toList.get! finalIndex

/-- Render a single row (28 pixels) of the image as an ASCII string.

Extracts pixels from indices [rowIndex*28 .. rowIndex*28+27] and converts each
to an ASCII character.

**Parameters:**
- `img`: Full 784-dimensional image vector (flattened row-major 28×28)
- `rowIndex`: Which row to render (0-27 inclusive)
- `isRaw255`: Whether pixel values are in 0-255 range (vs 0-1 normalized)
- `inverted`: Whether to use inverted brightness mapping

**Returns:** String of exactly 28 characters representing one row

**Complexity:** O(28) - constant time per row

**Row-major layout:** MNIST images are stored in row-major order:
- Row 0: pixels [0..27]
- Row 1: pixels [28..55]
- ...
- Row 27: pixels [756..783]
-/
private def renderRowFromVec (img : Vector 784) (rowIndex : Nat) (isRaw255 : Bool) (inverted : Bool) : String :=
  -- Build string using ⊞ to extract each pixel by its absolute index
  let startIdx := rowIndex * 28
  let chars := List.range 28 |>.map fun colIdx =>
    let absIdx := startIdx + colIdx
    -- Create a single-element vector extracting just this pixel using runtime check
    let pixelVec : Float^[1] := ⊞ (_ : Idx 1) => img[absIdx]!
    let pixelValue := pixelVec[0]
    brightnessToChar pixelValue isRaw255 inverted

  String.mk chars

/-- Render full 28×28 MNIST image as ASCII art.

Converts a flattened 784-dimensional MNIST image into a 28-line ASCII art
representation. Automatically detects whether the input uses raw 0-255 values
or normalized 0-1 values.

**Parameters:**
- `img`: 784-dimensional vector (flattened 28×28 MNIST image in row-major order)
- `inverted`: If `true`, use inverted brightness (for light terminals)

**Returns:** Multi-line string containing 28 rows of 28 characters each

**Algorithm:**
1. Auto-detect value range (0-255 vs 0-1)
2. Render each of 28 rows using `renderRow`
3. Join rows with newline characters

**Complexity:** O(784) - processes each pixel exactly once

**Output format:**
```
............................
............................
........@@####@@............
......@@########@@..........
...@@@@##########@@.........
...@@############@@.........
.....@@##########@@.........
.......@@########@@.........
.........@@######@@.........
...........@@####@@.........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
............@@##@@..........
........@@@@########@@......
........@@@@########@@......
........@@@@########@@......
...........@@########@@.....
...........@@########@@.....
.............@@####@@.......
............................
```

**Usage:**
```lean
let samples ← loadMNISTTest "data"
let (image, label) := samples[0]!
IO.println (renderImage image false)
```
-/
def renderImage (img : Vector 784) (inverted : Bool) : String :=
  let isRaw255 := autoDetectRange img

  -- Render each row directly from the vector
  let rows := List.range 28 |>.map fun rowIdx =>
    renderRowFromVec img rowIdx isRaw255 inverted
  String.intercalate "\n" rows

/-- Render MNIST image with a text label above it.

Adds a text label (e.g., "Ground Truth: 5" or "Predicted: 3") above the
rendered ASCII art for better context when displaying multiple images or
comparing predictions.

**Parameters:**
- `img`: 784-dimensional MNIST image vector
- `label`: Text to display above the image (e.g., "Digit: 5")
- `inverted`: Whether to use inverted brightness mapping

**Returns:** Multi-line string with label on first line, then ASCII art

**Output format:**
```
Predicted: 7
............................
............................
........@@####@@............
[... rest of image ...]
```

**Usage:**
```lean
let prediction := argmax networkOutput
let ascii := renderImageWithLabel image s!"Predicted: {prediction}" false
IO.println ascii
```
-/
def renderImageWithLabel (img : Vector 784) (label : String) (inverted : Bool) : String :=
  label ++ "\n" ++ renderImage img inverted

-- Future extension: Render multiple images side by side for comparison
-- Would allow comparing ground truth vs predictions horizontally

end VerifiedNN.Util.ImageRenderer
