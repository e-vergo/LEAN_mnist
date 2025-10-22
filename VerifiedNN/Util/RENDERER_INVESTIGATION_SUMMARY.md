# ASCII Image Renderer - Investigation Summary

**Date:** 2025-10-22
**Status:** Technical blocker identified - SciLean DataArrayN computability limitation

## Objective

Create a completely computable ASCII art renderer for MNIST digits (28×28 images) to demonstrate that Lean can execute practical infrastructure alongside verified neural network code.

## Implementation Completed

✅ **Full renderer logic implemented** (`VerifiedNN/Util/ImageRenderer.lean` - 330 lines)
- 16-character brightness palette: `" .:-=+*#%@"`
- Auto-detection of value range (0-1 vs 0-255)
- Inverted mode for light terminals
- All functions documented with mathlib-quality docstrings

✅ **CLI executable scaffolding** (`VerifiedNN/Examples/RenderMNIST.lean` - 145 lines)
- Full argument parsing (`--count`, `--inverted`, `--train`, `--help`)
- Error handling and usage information
- Ready to demonstrate once rendering works

✅ **Build configuration** (`lakefile.lean`)
- `renderMNIST` executable registered
- OpenBLAS linking configured

## Root Technical Blocker

**Problem:** SciLean's `DataArrayN` (used for `Vector 784 = Float^[784]`) does not support `Nat` indexing - only `Idx 784` indexing.

**Impact:** Cannot index MNIST image pixels using computed positions like `img[rowIndex * 28 + colIndex]` where indices are `Nat` values.

### What Works

```lean
-- ✅ This works - index is bound variable, automatically typed as Idx n
def normalizePixels {n : Nat} (image : Vector n) : Vector n :=
  ⊞ i => image[i] / 255.0

-- ✅ This works - literal indices (0, 1, 2, etc.)
let pixel0 := img[0]
let pixel1 := img[1]
```

### What Doesn't Work

```lean
-- ❌ Computed Nat index
let absIdx := rowIndex * 28 + colIndex
let pixel := img[absIdx]  -- Error: failed to synthesize GetElem (Vector 784) ℕ

-- ❌ Runtime bounds check
let pixel := img[absIdx]!  -- Same error

-- ❌ Idx.ofNat conversion
let idx := Idx.ofNat 784 absIdx  -- Makes executable noncomputable
let pixel := img[idx]
```

## Attempted Solutions

### 1. USize + Omega Proofs
**Approach:** Use `img[absIdx.toUSize]'h` with `omega` tactic proving bounds
**Result:** ❌ FAILED - `omega` cannot infer bounds from `List.range` context
**Error:** `omega could not prove 28*a + b < 784`

### 2. Idx.ofNat Conversion
**Approach:** Convert `Nat` to `Idx 784` using built-in function
**Result:** ❌ FAILED - Makes main noncomputable, linker error `undefined symbol: main`
**Note:** Module compiles, but executable linking fails

### 3. Unsafe Cast to Array
**Approach:** Use `unsafe def` with `unsafeCast` to convert `Vector → Array`
**Result:** ❌ FAILED - Same noncomputability propagation to main
**Code:**
```lean
private unsafe def vectorToArrayUnsafe (vec : Vector 784) : Array Float :=
  unsafeCast vec

@[implemented_by vectorToArrayUnsafe]
private def vectorToArray (vec : Vector 784) : Array Float :=
  Array.replicate 784 0.0
```

### 4. Fin Conversion with Proofs
**Approach:** `Fin 784 → Idx 784` conversion chain
**Result:** ❌ FAILED - Same proof obligation failures as omega approach

### 5. For Loop Iteration
**Approach:** Build array element by element in monadic loop
**Result:** ❌ FAILED - Loop body still requires indexing into DataArrayN

### 6. ⊞ Comprehension Nested
**Approach:** Use nested `⊞` to iterate rows and columns
**Result:** ❌ FAILED - Still need computed indices for absolute pixel position

## Why This is Hard

**Core Issue:** SciLean's `DataArrayN` type system enforces compile-time dimension safety by requiring `Idx n` indices. The `Idx n` type carries proof that the index is valid, but:

1. **No `GetElem` for Nat:** `DataArrayN` has no `GetElem` instance allowing `Nat` indices
2. **Proof obligations:** Converting `Nat → Idx n` requires proving `nat < n`, but `omega` fails when bounds come from runtime iteration
3. **Noncomputable conversions:** Functions like `Idx.ofNat` exist but make code noncomputable
4. **Unsafe contamination:** Using `unsafe` code makes the entire call chain noncomputable

## Impact on Project Goals

**User's Goals:**
1. ✅ Prove things about the NN - Not affected
2. ⚠️ Execute maximum infrastructure in Lean - Renderer blocked, but not critical

**What Still Works:**
- ✅ MNIST data loading (uses literal indexing in preprocessing)
- ✅ All verification code (mathematical proofs)
- ✅ Type safety enforcement
- ✅ Library compilation

**What's Blocked:**
- ❌ ASCII visualization executable
- ❌ Any utility requiring dynamic DataArrayN indexing

## Options Going Forward

### Option A: Post on Lean Zulip (RECOMMENDED)

**Action:** Ask SciLean community on Lean Zulip #scientific-computing:

```
Title: How to index DataArrayN with computed Nat values?

I need to render MNIST images (Vector 784 = Float^[784]) as ASCII art by accessing
pixels at computed indices: rowIndex * 28 + colIndex.

The pattern `⊞ i => img[i]` works when i is the bound variable, but I can't index
with computed Nat values. Attempted:
- img[absIdx] → GetElem synthesis error
- Idx.ofNat 784 absIdx → makes executable noncomputable
- USize + omega proofs → omega fails to prove bounds

What's the idiomatic SciLean pattern for accessing elements by computed indices?

Context: Building ASCII renderer to visualize MNIST training data
```

**Expected Response Time:** 24-48 hours (Tomáš Skřivan usually responds quickly)
**Likely Outcome:** There's a simple API we're missing, or confirmation this isn't supported yet

### Option B: Manual Unrolling (NUCLEAR OPTION)

**Approach:** Explicitly write out all 784 pixel accesses using literal indices

```lean
private def renderRow0 (img : Vector 784) ... :=
  String.mk [
    brightnessToChar img[0] ...,
    brightnessToChar img[1] ...,
    -- ... 28 times
  ]

private def renderRow1 (img : Vector 784) ... :=
  String.mk [
    brightnessToChar img[28] ...,
    brightnessToChar img[29] ...,
    -- ... 28 times
  ]

def renderImage (img : Vector 784) ... :=
  renderRow0 img ... ++ "\n" ++
  renderRow1 img ... ++ "\n" ++
  -- ... 28 times
```

**Pros:**
- ✅ Will definitely compile (literal indices work)
- ✅ Completely computable
- ✅ Zero dependencies on noncomputable operations

**Cons:**
- ❌ ~800+ lines of repetitive code
- ❌ Completely unmaintainable
- ❌ Embarrassing code quality
- ❌ Not generalizable to other image sizes

**Verdict:** Only if desperate and needed urgently

### Option C: Accept Limitation & Focus on Core Goals

**Approach:** Document the limitation and focus on neural network verification

**Rationale:**
- ASCII rendering is a nice-to-have utility
- Core NN verification work is progressing well
- SciLean API may improve in future releases
- Can visualize via Python/external tools if needed

**Next Steps:**
1. Update `README_RENDERER_STATUS.md` with findings
2. Create this summary document
3. Focus on completing gradient correctness proofs
4. Revisit renderer when SciLean API evolves

### Option D: Python/FFI Hybrid

**Approach:** Keep renderer logic in Lean but call from Python

```python
import ctypes
# Load compiled Lean library
lib = ctypes.CDLL("verifiedNN.so")
# Use Python for visualization, Lean for verification
```

**Pros:**
- ✅ Leverages both ecosystems
- ✅ Verification still in Lean
- ✅ Visualization works today

**Cons:**
- ❌ Doesn't achieve "maximum Lean execution" goal
- ❌ Adds Python dependency
- ❌ More complex build

## Lessons Learned

1. **SciLean is optimized for mathematical operations, not general systems programming**
   - Excellent for AD, derivatives, numerical computation
   - Limited support for arbitrary array indexing patterns

2. **Type safety has runtime costs**
   - `Idx n` enforces compile-time dimension checking
   - Makes dynamic indexing extremely difficult

3. **Lean executables require all functions to be computable**
   - `unsafe` contamination propagates through call chain
   - No escape hatch for "trusted but unverified" utility code

4. **lean-doc-searcher agent was valuable**
   - Discovered `Idx.ofNat` pattern
   - Confirmed GetElem instances
   - Provided SciLean source references

## Recommendations

**Immediate (Next 24-48 hours):**
1. ✅ Complete this investigation summary
2. Post on Lean Zulip #scientific-computing (Option A)
3. Continue with core verification work while waiting for response

**If Zulip provides solution:**
- Implement proper DataArrayN indexing
- Complete renderer
- Test with real MNIST data
- Document pattern for future use

**If no solution in 1 week:**
- Accept limitation (Option C)
- Document as "future work" in README
- Focus 100% on gradient correctness verification
- Revisit when SciLean v2.0 releases

## Technical References

**SciLean Source Files Examined:**
- `.lake/packages/scilean/SciLean/Data/DataArray/DataArray.lean:51-52` - GetElem instance
- `.lake/packages/scilean/SciLean/Data/Idx/Basic.lean:77-99` - Idx conversions
- `.lake/packages/scilean/SciLean/Data/ArrayOperations/Notation.lean` - GetElem for α^[ι]

**MCP Tools Used:**
- `lean-doc-searcher` - API discovery
- `lean_diagnostic_messages` - Error analysis
- `lean_goal` - Proof state inspection (attempted)

**Build Commands:**
```bash
# Module compiles successfully
lake build VerifiedNN.Util.ImageRenderer  # ✅ SUCCESS

# Executable fails at link time
lake build renderMNIST  # ❌ FAIL: undefined symbol: main
```

## Status

- **Code Quality:** ✅ Clean, formal, mathlib-standard documentation
- **Implementation:** ✅ All logic complete
- **Computability:** ❌ Blocked on SciLean API limitation
- **Project Impact:** 🟡 Low - not critical path

---

**Last Updated:** 2025-10-22
**Author:** Investigation via lean-doc-searcher and systematic exploration
**Next Action:** Await Zulip community response or pivot to Option C
