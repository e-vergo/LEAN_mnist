# ASCII Image Renderer - Implementation Status

## Current State: 90% Complete, Blocked on SciLean Indexing

### What Works ✅

1. **Module Structure** (`VerifiedNN/Util/ImageRenderer.lean`)
   - Comprehensive documentation (~300 lines of docstrings)
   - 16-character brightness palette
   - Auto-detection of value range (0-1 vs 0-255)
   - Inverted mode support
   - All logic implemented

2. **Example Executable** (`VerifiedNN/Examples/RenderMNIST.lean`)
   - Full CLI with argument parsing
   - Flags: `--count`, `--inverted`, `--train`, `--help`
   - Comprehensive documentation and usage examples

3. **Build Configuration**
   - Added to `lakefile.lean` as `renderMNIST` executable
   - Will be fully computable (no AD dependency) once indexing fixed

### What's Blocked ❌

**Single Issue:** SciLean `DataArrayN` Indexing

The blocker is at `VerifiedNN/Util/ImageRenderer.lean:210-220` (renderRowFromVec function).

**Problem:**
- MNIST images are `Vector 784` which is `Float^[784]` (SciLean's `DataArrayN`)
- `DataArrayN` uses `Idx n` type for indexing, not `Nat`
- Need to extract pixel at index `i` where `i : Nat` in range `[0, 783]`
- All attempted indexing patterns fail type checking

**Attempted Approaches:**

1. **Direct Nat indexing:** `img[pixelIdx]` → Type error (needs Idx 784)
2. **Fin conversion:** `img[⟨pixelIdx, proof⟩]` → Complex proof obligations fail
3. **⊞ reconstruction:** `⊞ (j : Idx 784) => img[j]` then `[pixelIdx]` → Cannot index result with Nat
4. **Array conversion:** Convert to `Array Float` first → Requires complex USize proofs in loop
5. **Runtime check:** `img[pixelIdx]!` → Still needs proper Idx type, not Nat

**What We Know Works:**

From `VerifiedNN/Data/Preprocessing.lean:56, 264`:
```lean
def normalizePixels {n : Nat} (image : Vector n) : Vector n :=
  ⊞ i => image[i] / 255.0
```

When `i` is bound in `⊞ i =>`, it's automatically `Idx n` and `image[i]` works perfectly.

**The Gap:**

We need to access `img[pixelIdx]` where `pixelIdx : Nat` is computed dynamically.
The `⊞` pattern only works when the index is the bound variable itself.

## Paths Forward

### Option 1: Get SciLean Community Help (Recommended)

Post on Lean Zulip #scientific-computing:

```
Title: How to index DataArrayN with computed Nat index?

I have `img : Float^[784]` and need to extract pixel at `idx : Nat` where `idx` is
computed as `rowIndex * 28 + colIndex`.

What's the idiomatic way to:
- Convert `idx : Nat` to `Idx 784` with runtime bounds check?
- Or convert `DataArrayN` to `Array` efficiently?

Context: Building ASCII renderer for MNIST visualization
```

Likely response: There's a simple idiom we're missing (e.g., a conversion function or different indexing syntax).

### Option 2: Simplify to Array Float

Create `renderImage'` that takes `Array Float` instead of `Vector 784`:

```lean
def renderImageFromArray (imgArray : Array Float) (inverted : Bool) : String :=
  -- ... existing logic, but imgArray[i]! works fine

def renderImage (img : Vector 784) (inverted : Bool) : String :=
  -- Convert Vector to Array first (one-time cost)
  let arr := (⊞ (i : Idx 784) => img[i]) -- somehow convert this...
  renderImageFromArray arr inverted
```

Still needs to solve the Vector → Array conversion, but localizes the problem.

### Option 3: Use MCP lean-lsp Tools

Use `lean_local_search` and `lean_hover_info` to find existing conversion functions:

```bash
# Search for DataArrayN to Array conversions
lean_local_search "DataArrayN.*Array"
lean_local_search "toArray"
```

May discover built-in conversion we missed.

### Option 4: Workaround with Manual Unrolling

For the specific case of 28×28:

```lean
private def renderRow0 (img : Vector 784) ... :=
  let c0 := brightnessToChar img[0] ...
  let c1 := brightnessToChar img[1] ...
  -- ... 28 times

private def renderRow1 (img : Vector 784) ... :=
  let c0 := brightnessToChar img[28] ...
  -- ...

def renderImage (img : Vector 784) ... :=
  renderRow0 img ... ++ "\n" ++
  renderRow1 img ... ++ "\n" ++
  -- ... 28 times
```

**Pros:** Will definitely compile
**Cons:** 784+ lines of repetitive code, unmaintainable

## Technical Details

**Error Message:**
```
error: failed to synthesize GetElem? (Core.Vector 784) ℕ ...
```

Translation: Lean cannot find an instance to index `Vector 784` with `Nat`.
It expects `Idx 784` instead.

**Core Issue:**
`DataArrayN` doesn't implement `GetElem` for `Nat` indices directly.
You must use `Idx n` which carries proof of bounds.

**Why This is Hard:**
In a loop/map context, proving `rowIndex * 28 + colIndex < 784` for omega
requires knowing `rowIndex < 28` and `colIndex < 28`, but those facts are
lost in the List.range context.

## Recommendation

**Immediate:** Post on Zulip #scientific-computing with the question above.

**Timeline:** Likely get answer within 24-48 hours from Tomáš Skřivan (SciLean author).

**Fallback:** If no solution in 1 week, implement Option 4 (manual unrolling) as a temporary measure, with TODO to refactor once proper idiom discovered.

## Value Proposition

Once unblocked, this renderer provides:

✅ **Completely computable** visualization (works in executables unlike training code)
✅ **Zero dependencies** on noncomputable AD operations
✅ **Debugging utility** for MNIST data loading
✅ **Proof that Lean can do practical systems tasks** despite AD limitations
✅ **Educational value** showing pure functional image processing

## Files

- Implementation: `VerifiedNN/Util/ImageRenderer.lean` (90% done, 320 lines)
- Example: `VerifiedNN/Examples/RenderMNIST.lean` (complete, 150 lines)
- Build config: `lakefile.lean` (updated)
- This status: `VerifiedNN/Util/README_RENDERER_STATUS.md`

## Next Actions

1. Post question on Zulip
2. Try `lean_local_search` for conversion functions
3. If blocked >1 week, implement manual unrolling
4. Once working, demo with: `lake exe renderMNIST --count 5`

---

**Last Updated:** 2025-10-21
**Blocker:** SciLean DataArrayN indexing API unknown
**Status:** Awaiting community input or API discovery
