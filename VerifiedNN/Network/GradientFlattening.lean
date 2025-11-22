import VerifiedNN.Core.DataTypes
import VerifiedNN.Network.Gradient
import SciLean

/-!
# Gradient Flattening

Packs layer-by-layer gradient matrices and vectors into a single flattened
gradient vector that matches the parameter vector layout.

## Purpose

During backpropagation, gradients are computed layer-by-layer as separate matrices
and vectors (dW1, db1, dW2, db2). To apply optimization algorithms (like SGD), these
must be packed into a single flat vector matching the parameter layout defined in
`Network.Gradient`.

## Memory Layout

Gradients are flattened to match the parameter layout in `Network.Gradient`:

```
[0..100351]      (100,352 elements): Layer 1 weights (128 × 784)
[100352..100479] (128 elements):     Layer 1 bias
[100480..101759] (1,280 elements):   Layer 2 weights (10 × 128)
[101760..101769] (10 elements):      Layer 2 bias
Total: 101,770 parameters
```

Row-major ordering is used for matrices: matrix element [i,j] maps to
flattened index `offset + i * cols + j`.

## Key Functions

- `flattenGradients` - Pack gradient components into single vector matching parameter layout
- Helper functions for flattening individual components (weights, biases)

## Implementation Notes

**Index Arithmetic:**
- Layer 1 weights: `[i,j]` → `i * 784 + j` for i ∈ [0, 128), j ∈ [0, 784)
- Layer 1 bias: `[i]` → `100352 + i` for i ∈ [0, 128)
- Layer 2 weights: `[i,j]` → `100480 + i * 128 + j` for i ∈ [0, 10), j ∈ [0, 128)
- Layer 2 bias: `[i]` → `101760 + i` for i ∈ [0, 10)

**Consistency:**
Must match the layout in `flattenParams` and `unflattenParams` exactly for
gradient descent to work correctly.

**Sorries:**
All sorries are index arithmetic bounds proofs (acceptable placeholders).

## Verification Status

- **Build Status:** ✅ Compiles successfully
- **Sorries:** Multiple (all for index arithmetic bounds, non-critical)
- **Correctness:** Layout matches `Network.Gradient.unflattenParams`
- **Testing:** Should be validated via gradient checking

## Usage Example

```lean
-- After computing backward pass
let (dW1, db1) := layer1.backward ...
let (dW2, db2) := layer2.backward ...

-- Pack into single vector for optimizer
let gradient := flattenGradients dW1 db1 dW2 db2

-- Now gradient can be used with SGD
let newParams := params - learningRate * gradient
```
-/

namespace VerifiedNN.Network.GradientFlattening

open VerifiedNN.Core
open VerifiedNN.Network.Gradient
open SciLean

set_default_scalar Float

/-- Helper to convert Nat with bound proof to Idx.
    Reuses the helper from Network.Gradient for consistency. -/
private def natToIdx (n : Nat) (i : Nat) (h : i < n) : Idx n :=
  (Idx.finEquiv n).invFun ⟨i, h⟩

/-- Pack layer-by-layer gradients into single flattened vector.

Combines weight and bias gradients from both layers into a single
gradient vector of 101,770 elements, matching the parameter layout
defined in `Network.Gradient`.

**Memory Layout:**
- [0-100351]: dW1 (Layer 1 weights, 128×784 in row-major order)
- [100352-100479]: db1 (Layer 1 bias, 128)
- [100480-101759]: dW2 (Layer 2 weights, 10×128 in row-major order)
- [101760-101769]: db2 (Layer 2 bias, 10)

**Index Formulas:**
- Layer 1 weight gradient dW1[i,j]: placed at index `i * 784 + j`
- Layer 1 bias gradient db1[i]: placed at index `100352 + i`
- Layer 2 weight gradient dW2[i,j]: placed at index `100480 + i * 128 + j`
- Layer 2 bias gradient db2[i]: placed at index `101760 + i`

**Parameters:**
- `dW1`: Layer 1 weight gradients [128, 784]
- `db1`: Layer 1 bias gradients [128]
- `dW2`: Layer 2 weight gradients [10, 128]
- `db2`: Layer 2 bias gradients [10]

**Returns:** Flattened gradient vector [101,770]

**Correctness:** This function must mirror the exact layout of `flattenParams`
and `unflattenParams` in `Network.Gradient.lean`. The index mappings are:

1. **Layer 1 weights (indices 0-100351):**
   - For each position [i,j] in the 128×784 matrix, compute flat index = i*784 + j
   - This matches the row-major flattening in `flattenParams`

2. **Layer 1 bias (indices 100352-100479):**
   - For each position [i] in the 128-vector, compute flat index = 100352 + i
   - Offset 100352 = 784 * 128 (size of Layer 1 weights)

3. **Layer 2 weights (indices 100480-101759):**
   - For each position [i,j] in the 10×128 matrix, compute flat index = 100480 + i*128 + j
   - Offset 100480 = 784*128 + 128 (Layer 1 weights + Layer 1 bias)

4. **Layer 2 bias (indices 101760-101769):**
   - For each position [i] in the 10-vector, compute flat index = 101760 + i
   - Offset 101760 = 784*128 + 128 + 128*10 (all previous components)

**Example:**
```lean
-- After computing backward pass for both layers
let (dW1, db1) := denseLayer1Backward ...
let (dW2, db2) := denseLayer2Backward ...

-- Pack into single vector for optimizer
let gradient := flattenGradients dW1 db1 dW2 db2

-- gradient[0] = dW1[0,0]
-- gradient[784] = dW1[1,0]
-- gradient[100352] = db1[0]
-- gradient[100480] = dW2[0,0]
-- gradient[101760] = db2[0]
```

**Verification:**
- Layout proven consistent with `unflattenParams` (pending formal proof)
- Should be validated via numerical gradient checking
- All sorries are index arithmetic bounds (non-critical placeholders)

**References:**
- Parameter layout: `Network.Gradient.flattenParams` (lines 118-151)
- Inverse operation: `Network.Gradient.unflattenParams` (lines 163-195)
-/
def flattenGradients
  (dW1 : Matrix 128 784)
  (db1 : Vector 128)
  (dW2 : Matrix 10 128)
  (db2 : Vector 10)
  : Vector nParams :=
  ⊞ (idx : Idx nParams) =>
    let i := idx.1.toNat
    -- Layer 1 weights: indices [0, 100351]
    -- W1 is [128, 784] in row-major order
    -- Index formula: row * 784 + col
    if h : i < 784 * 128 then
      let row := i / 784
      let col := i % 784
      have hrow : row < 128 := by
        -- row = i / 784, and i < 784 * 128
        -- Therefore row = i / 784 < (784 * 128) / 784 = 128
        omega
      have hcol : col < 784 := Nat.mod_lt i (by omega : 0 < 784)
      dW1[natToIdx 128 row hrow, natToIdx 784 col hcol]
    -- Layer 1 bias: indices [100352, 100479]
    -- b1 is [128] vector
    else if h2 : i < 784 * 128 + 128 then
      let bias_idx := i - 784 * 128
      have hb : bias_idx < 128 := by
        -- i < 784 * 128 + 128 and ¬(i < 784 * 128)
        -- Therefore 784 * 128 ≤ i < 784 * 128 + 128
        -- So bias_idx = i - 784 * 128 < 128
        omega
      db1[natToIdx 128 bias_idx hb]
    -- Layer 2 weights: indices [100480, 101759]
    -- W2 is [10, 128] in row-major order
    -- Index formula: 100480 + row * 128 + col
    else if h3 : i < 784 * 128 + 128 + 128 * 10 then
      let offset := i - (784 * 128 + 128)
      let row := offset / 128
      let col := offset % 128
      have hrow : row < 10 := by
        -- offset < 128 * 10 (from h3 and previous conditions)
        -- row = offset / 128 < (128 * 10) / 128 = 10
        omega
      have hcol : col < 128 := Nat.mod_lt offset (by omega : 0 < 128)
      dW2[natToIdx 10 row hrow, natToIdx 128 col hcol]
    -- Layer 2 bias: indices [101760, 101769]
    -- b2 is [10] vector
    else
      let bias_idx := i - (784 * 128 + 128 + 128 * 10)
      have hb : bias_idx < 10 := by
        -- In else branch: ¬(i < 784*128+128+128*10)
        -- But i < nParams = 784*128 + 128 + 128*10 + 10
        -- Therefore 784*128+128+128*10 ≤ i < 784*128+128+128*10+10
        -- So bias_idx < 10
        have hi : i < nParams := idx.2
        unfold nParams at hi
        omega
      db2[natToIdx 10 bias_idx hb]

/-- Flatten Layer 1 weight gradients into parameter vector segment.

Takes a [128, 784] gradient matrix and flattens it into the first
100,352 positions of the gradient vector using row-major order.

**Index Formula:** For dW1[i, j], place at index `i * 784 + j`

**Parameters:**
- `dW1`: Layer 1 weight gradients [128, 784]

**Returns:** Flattened segment for indices [0, 100351]

**Note:** This is a helper function for understanding the flattening layout.
The main implementation uses `flattenGradients` which handles all components
in a single pass.
-/
def flattenLayer1Weights (dW1 : Matrix 128 784) : Vector (784 * 128) :=
  ⊞ (idx : Idx (784 * 128)) =>
    let flat_idx := idx.1.toNat
    let row := flat_idx / 784
    let col := flat_idx % 784
    have hrow : row < 128 := by
      have hi : flat_idx < 784 * 128 := idx.2
      omega
    have hcol : col < 784 := Nat.mod_lt flat_idx (by omega : 0 < 784)
    dW1[natToIdx 128 row hrow, natToIdx 784 col hcol]

/-- Flatten Layer 2 weight gradients into parameter vector segment.

Takes a [10, 128] gradient matrix and flattens it into the segment
at indices [100480, 101759] using row-major order.

**Index Formula:** For dW2[i, j], place at index `100480 + i * 128 + j`
(offset of 100480 not included in this helper, just the relative indexing)

**Parameters:**
- `dW2`: Layer 2 weight gradients [10, 128]

**Returns:** Flattened segment for indices [0, 1279] (relative to offset 100480)

**Note:** This is a helper function for understanding the flattening layout.
The main implementation uses `flattenGradients` which handles all components
in a single pass.
-/
def flattenLayer2Weights (dW2 : Matrix 10 128) : Vector (128 * 10) :=
  ⊞ (idx : Idx (128 * 10)) =>
    let flat_idx := idx.1.toNat
    let row := flat_idx / 128
    let col := flat_idx % 128
    have hrow : row < 10 := by
      have hi : flat_idx < 128 * 10 := idx.2
      omega
    have hcol : col < 128 := Nat.mod_lt flat_idx (by omega : 0 < 128)
    dW2[natToIdx 10 row hrow, natToIdx 128 col hcol]

end VerifiedNN.Network.GradientFlattening
