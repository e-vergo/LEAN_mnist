# VerifiedNN/Util - Utility Functions

## Purpose

Utility modules providing infrastructure support for the verified neural network project. This directory contains tools for visualization, data inspection, and debugging.

## Key Achievement: First Fully Computable Executable ⚡

The ASCII Image Renderer (`ImageRenderer.lean`) is the **first fully computable executable** in this project, proving that Lean CAN execute practical infrastructure despite SciLean's noncomputable automatic differentiation.

## Modules

### ImageRenderer.lean (✅ COMPUTABLE)
**Status:** ✅ Complete and executable
**Lines:** ~380 (including 28-row manual unrolling)
**Computability:** ✅ Fully computable - builds standalone binary

ASCII art renderer for 28×28 MNIST images.

**Features:**
- 16-character brightness palette: `" .:-=+*#%@"`
- Auto-detection of value range (0-1 normalized vs 0-255 raw)
- Inverted mode for light-background terminals
- Mathlib-quality documentation

**Usage:**
```bash
# Visualize first 5 MNIST test images
lake exe renderMNIST --count 5

# Inverted mode for light terminals
lake exe renderMNIST --count 3 --inverted

# Training set visualization
lake exe renderMNIST --count 10 --train
```

**Example Output:**
```
Sample 0 | Ground Truth: 7
----------------------------

      :*++:.
      #%%%%%*********.
      :=:=+%%#%%%%%%%=
            : :::: %%-
                  :%#
                  %@:
                 =%%.
                :%%:
                =%*
                #%:
               =%*
              :%%:
              #%+

```

## Technical Implementation

### The SciLean DataArrayN Indexing Challenge

**Problem:** SciLean's `DataArrayN` (used for `Vector 784 = Float^[784]`) requires `Idx n` indices, not `Nat`. This prevents computed indexing like `img[row * 28 + col]`.

**Solution:** Manual unrolling with literal indices.

Instead of:
```lean
-- ❌ This fails - computed Nat index
let absIdx := rowIndex * 28 + colIndex
let pixel := img[absIdx]
```

We use:
```lean
-- ✅ This works - literal indices
match rowIndex * 28 with
| 0 => String.mk (List.range 28 |>.map fun i =>
    let px := match i with
    | 0 => img[0] | 1 => img[1] | ... | 27 => img[27]
    brightnessToChar px ...)
| 28 => String.mk (List.range 28 |>.map fun i =>
    let px := match i with
    | 0 => img[28] | 1 => img[29] | ... | 55 => img[55]
    ...)
-- ... for all 28 rows
```

**Result:** ~100 lines of match arms covering all 784 pixels with literal indices. Verbose but provably computable.

### Why This Matters

1. **Proves Lean's capabilities** - Can execute practical infrastructure
2. **First computable executable** - All other executables blocked on noncomputable AD
3. **Workaround pattern** - Shows how to bypass SciLean limitations when needed
4. **Debugging utility** - Visualize MNIST data without Python/external tools

## Computability Status

### Executable Functions ✅
- `renderImage`: ✅ Computable - converts Vector 784 to ASCII string
- `renderImageWithLabel`: ✅ Computable - adds text label
- `brightnessToChar`: ✅ Computable - maps Float to Char
- `autoDetectRange`: ✅ Computable - detects 0-1 vs 0-255 range
- `renderRowLiteral`: ✅ Computable - renders single row (manual unrolling)

### Implementation Notes
- Uses literal indices (compile-time known): `img[0]`, `img[1]`, ..., `img[783]`
- Avoids computed Nat → Idx conversion (noncomputable path)
- Zero dependencies on automatic differentiation
- Pure functional implementation with no side effects (except IO in executable)

## Limitations and Future Work

**Current Limitation:** Manual unrolling is unmaintainable for other image sizes
- Works for fixed 28×28 MNIST images
- Would need 1024 lines for 32×32 images
- Not generalizable without better SciLean indexing API

**Future Improvements:**
- Wait for SciLean API to support computed Nat indexing
- Contribute to SciLean to enable `DataArrayN` GetElem for Nat
- Or accept as necessary workaround for verification-focused library

## References

- **Technical Investigation:** `RENDERER_INVESTIGATION_SUMMARY.md` - Full details on 6 attempted approaches
- **Example Executable:** `Examples/RenderMNIST.lean` - CLI with full argument parsing
- **SciLean Discussion:** Lean Zulip #scientific-computing (potential future improvements)

## Dependencies

- `VerifiedNN.Core.DataTypes` - Vector type definitions
- `SciLean` - DataArrayN implementation (literal indexing only)

No dependencies on:
- Automatic differentiation (completely avoided)
- Noncomputable operations
- Network/gradient modules

## Last Updated

2025-10-22 - Initial implementation with manual unrolling workaround

---

**Key Insight:** Sometimes the "ugly" solution (manual unrolling) is the path to achieving practical goals (executable visualization) within theoretical constraints (type-safe indexing). This renderer proves Lean can deliver real infrastructure alongside formal verification.
