# Documentation Update Plan - Dual Goals

## Summary

Update all project documentation to reflect the dual goals:
1. **Prove things about the neural network** (original focus)
2. **Implement and execute maximum infrastructure in Lean** (NEW emphasis)

## Files to Update

### 1. README.md (Root)

**Changes Needed:**
- Update "Core Achievement" section to include both goals:
  - PRIMARY GOAL: ✅ PROVEN - Gradient correctness
  - SECONDARY GOAL: ✅ VERIFIED - Type safety
  - **NEW: TERTIARY GOAL: ⚡ EXECUTE - Maximum infrastructure in pure Lean**

- Add new section "What Executes in Lean" showing:
  - ✅ MNIST data loading (computable)
  - ✅ Data preprocessing (computable)
  - ✅ ASCII visualization (computable - via manual unrolling workaround)
  - ❌ Training loop (blocked - noncomputable AD)
  - ❌ Gradient computation (blocked - noncomputable AD)

- Update build commands to include:
  ```bash
  # Working executables
  lake exe renderMNIST --count 5  # Visualize MNIST digits
  ```

### 2. verified-nn-spec.md (Technical Spec)

**Changes Needed:**
- Add new top-level section: "Executable Infrastructure Goals"
- Document computability boundaries:
  - What CAN execute (data loading, preprocessing, visualization)
  - What CANNOT execute (AD-dependent operations)
  - Workarounds employed (manual unrolling for renderer)

- Update success criteria:
  - Verification completeness (existing)
  - **NEW:** Infrastructure execution coverage (target: >50% of non-AD operations)

### 3. Directory READMEs (All 10)

Each README needs new section: "Computability Status"

**Template:**
```markdown
## Computability Status

### Executable Functions
- `functionName`: ✅ Computable - can run in standalone binary
- `anotherFunc`: ✅ Computable - pure Lean execution

### Noncomputable Functions
- `gradientFunc`: ❌ Noncomputable - depends on SciLean AD
- Reason: Uses `∇` operator which is noncomputable

### Workarounds
- Manual unrolling pattern for `renderRow` (literal indices)
```

**Directories to update:**
1. Core/README.md
2. Data/README.md
3. Layer/README.md
4. Loss/README.md
5. Network/README.md
6. Optimizer/README.md
7. Training/README.md
8. Verification/README.md
9. Testing/README.md
10. **Util/README.md** (NEW - document renderer achievement)

### 4. Util/README.md (Create NEW)

**Content:**
- Document the ASCII renderer
- Explain the manual unrolling workaround
- Note this as **first fully computable executable** in project
- Provide usage examples
- Document the SciLean DataArrayN indexing limitation
- Link to RENDERER_INVESTIGATION_SUMMARY.md for technical details

## Key Messages

1. **Both goals are valuable**: Verification AND execution
2. **Lean CAN execute practical infrastructure** despite limitations
3. **SciLean's noncomputable AD is a known boundary** - not a project failure
4. **Workarounds exist** - manual unrolling proves Lean's capabilities
5. **This is a research frontier** - pushing Lean's boundaries in ML

## Implementation Order

1. Create Util/README.md (document the win)
2. Update root README.md (high-level dual goals)
3. Update verified-nn-spec.md (technical details)
4. Update all 10 directory READMEs (add Computability Status sections)

---

**Status:** Ready for implementation
**Estimated Time:** 15-20 minutes total
**Impact:** High - clarifies project scope and achievements
