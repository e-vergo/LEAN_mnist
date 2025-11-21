# File Review: Loop.lean

## Summary
Core training loop implementation with 1 noncomputable function (resumeTraining), extensive TODOs for checkpoint serialization, and active usage across 8 files. Zero axioms, zero sorries, but incomplete checkpoint feature.

## Findings

### Orphaned Code
**Partially orphaned checkpoint infrastructure:**

- **Lines 119-141:** `CheckpointConfig` structure - **Partially unused**
  - Defined but serialization never implemented
  - Only referenced in Loop.lean itself (no external usage found)
  - API exists but non-functional

- **Lines 493-507:** `saveCheckpoint` function - **Non-functional stub**
  - Returns early without saving
  - Only logs intent: "Would save to: {filename}"
  - Never actually called with non-zero `saveEveryNEpochs`

- **Lines 528-532:** `loadCheckpoint` function - **Non-functional stub**
  - Throws error immediately
  - Dead code (never successfully executes)

- **Lines 693-730:** `resumeTraining` function - **Noncomputable, unused**
  - Marked `noncomputable` (Line 693)
  - No external references found (0 usages)
  - Depends on non-functional `loadCheckpoint`

**Active code (heavily used):**
- `TrainConfig` - 8 files
- `TrainState`, `initTrainState` - 2 files
- `trainBatch`, `trainOneEpoch` - Loop.lean (internal)
- `trainEpochs`, `trainEpochsWithConfig` - 7 files (MNISTTrain.lean, SimpleExample.lean, etc.)

### Axioms (Total: 0)
**None.** Depends on verified axioms in upstream modules (Network/Gradient.lean has 2 axioms, documented elsewhere).

### Sorries (Total: 0)
**None.** No formal verification attempted in training loop itself.

### Code Correctness Issues

**Training logic appears correct:**

1. **trainBatch (Lines 366-413):**
   - ✓ Gradient accumulation: `foldl (fun accGrad ... => accGrad[i] + grad[i])`
   - ✓ Averaging: `avgGrad := gradSum[i] / batchSizeFloat`
   - ✓ SGD step application correct
   - ✓ Empty batch handling: returns state unchanged
   - ✓ Uses `networkGradientManual` (computable)

2. **trainOneEpoch (Lines 427-469):**
   - ✓ Shuffled batch creation via `createShuffledBatches`
   - ✓ Batch loop with progress tracking
   - ✓ Periodic evaluation (respects `evaluateEveryNEpochs`)
   - ✓ Correct epoch counter increment

3. **trainEpochsWithConfig (Lines 582-624):**
   - ✓ State initialization correct
   - ✓ Epoch loop structure correct
   - ✓ Checkpoint saving attempt (but saveCheckpoint is stub)
   - ✓ Final metrics computation

4. **trainEpochs (Lines 659-678):**
   - ✓ Correct wrapper around trainEpochsWithConfig
   - ✓ Extracts network from final state

**Checkpoint code is non-functional:**

5. **saveCheckpoint (Lines 493-507):**
   - ✗ **Never actually saves** (TODO on line 501)
   - ✗ Only logs "Would save to: {filename}"
   - ✗ Returns early if saveEveryNEpochs == 0 (always true in practice)

6. **loadCheckpoint (Lines 528-532):**
   - ✗ **Always throws error** (TODO on line 529)
   - ✗ Dead code—cannot succeed

7. **resumeTraining (Lines 693-730):**
   - ✗ **Noncomputable** (Line 693) - cannot execute
   - ✗ Depends on non-functional loadCheckpoint
   - ✗ No external usage found

### Hacks & Deviations

**Checkpoint serialization abandoned:**

- **Lines 481-532:** Checkpoint TODO - **Severity: moderate**
  - API defined but implementation missing
  - 3 TODOs documented (lines 481, 501, 516, 529)
  - Strategy outlined in comments (JSON serialization)
  - Non-blocking: training works without checkpoints

**Noncomputable resume function:**

- **Line 693:** `noncomputable def resumeTraining` - **Severity: minor**
  - Marked noncomputable but unused
  - Should either be removed or made computable
  - Currently dead code

**Debug logging inefficiency:**

- **Lines 191-282:** DebugUtils namespace - **Severity: minor**
  - `vectorFirstK` always returns empty array (Line 235)
  - Feature disabled for simplicity (comment acknowledges)
  - Uses norm-based logging instead (acceptable workaround)

**Magic numbers:**

- **Line 276:** Gradient explosion threshold = 10.0 - **Severity: minor**
  - Hardcoded in DebugUtils
  - Should reference GradientMonitoring thresholds (but that module is unused)

- **Line 279:** Gradient vanishing threshold = 0.0001 - **Severity: minor**
  - Hardcoded in DebugUtils
  - Duplicates GradientMonitoring.vanishingThreshold

## Statistics
- **Definitions:** 21 total
  - Core training: 8 (TrainConfig, TrainState, initTrainState, trainBatch, trainOneEpoch, trainEpochs, trainEpochsWithConfig, TrainState.init)
  - Checkpoint (non-functional): 4 (CheckpointConfig, saveCheckpoint, loadCheckpoint, resumeTraining)
  - Logging utilities: 5 (TrainingLog namespace functions)
  - Debug utilities: 4 (DebugUtils namespace functions)
- **Unused definitions:** 4 (checkpoint-related)
- **Noncomputable definitions:** 1 (resumeTraining)
- **Theorems:** 0 (computational module)
- **Axioms:** 0 (uses axioms from Network/Gradient.lean)
- **Sorries:** 0
- **Lines of code:** 732
- **Documentation quality:** Excellent (comprehensive module + function docstrings)
- **Usage:** Active in 8 files (MNISTTrain, SimpleExample, MiniTraining, TrainAndSerialize, etc.)
- **TODOs:** 6 (all checkpoint-related, lines 18, 62, 132, 481, 501, 516, 529, 564)

## Recommendations

### Priority 1: Remove or Complete Checkpoint Code

**Option A: Remove non-functional checkpoint code (recommended)**
```lean
-- Delete these definitions:
-- - CheckpointConfig (lines 119-141)
-- - saveCheckpoint (lines 493-507)
-- - loadCheckpoint (lines 528-532)
-- - resumeTraining (lines 693-730)
-- Update trainEpochsWithConfig signature to remove checkpointConfig parameter
```

**Option B: Implement checkpoint serialization**
- Use Network/Serialization.lean as reference (model saving already works)
- Implement MLPArchitecture → JSON conversion
- Add SGDState serialization
- Add tests for save/load round-trip
- Remove `noncomputable` from resumeTraining

### Priority 2: Consolidate Gradient Monitoring

**Issue:** Duplicate threshold definitions
- DebugUtils has hardcoded thresholds (10.0, 0.0001)
- GradientMonitoring.lean has same thresholds (unused module)

**Solution:** Either integrate GradientMonitoring or remove it

### Priority 3: Complete or Remove vectorFirstK

**Line 235:** `vectorFirstK` always returns empty array
- Either implement properly or remove the function
- Document why feature is disabled

### Priority 4: Document Noncomputability

**Line 693:** `resumeTraining` is noncomputable
- Add comment explaining why (references manual gradient computation? unclear)
- Consider making computable if possible
- If keeping, document when/how it should be used

## Critical Issues

**None for core training functionality.** The training loop works correctly and is actively used.

**Moderate issues:**
1. **Checkpoint code is misleading:** Functions exist but don't work (should remove or complete)
2. **Noncomputable resumeTraining:** Dead code that cannot execute (should remove)
3. **TODOs are stale:** 6 TODOs for checkpoints, no recent progress

## Production Readiness

**Core training loop: PRODUCTION-READY ✅**
- Zero correctness issues
- Active usage in 8 files
- Achieves 93% MNIST accuracy
- Complete documentation

**Checkpoint system: NOT READY ❌**
- API exists but non-functional
- Should be removed or completed
- Currently misleading (looks like it works but doesn't)
