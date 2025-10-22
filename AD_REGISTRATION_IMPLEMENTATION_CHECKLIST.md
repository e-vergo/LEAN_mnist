# AD Registration Implementation Checklist

## Overview

This checklist tracks the implementation of `@[fun_trans]` and `@[fun_prop]` attributes across LinearAlgebra.lean and Activation.lean.

Operations are grouped by complexity and dependency order to enable incremental, testable progress.

---

## LinearAlgebra.lean - Operations to Register

### Phase 1: Foundation (Linear Vector Operations)

These operations inherit differentiability from module structure. Minimal work required.

- [ ] **vadd** - Vector addition
  - [ ] Remove TODO comment about fun_prop registration
  - [ ] Add `@[fun_prop] theorem vadd.Differentiable`
  - [ ] Proof strategy: `unfold vadd; fun_prop`

- [ ] **vsub** - Vector subtraction
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem vsub.Differentiable`
  - [ ] Proof strategy: `unfold vsub; fun_prop`

- [ ] **smul** - Scalar-vector multiplication
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem smul.Differentiable`
  - [ ] Proof strategy: `unfold smul; fun_prop`

**Testing after Phase 1:**
```bash
lake build VerifiedNN.Core.LinearAlgebra  # Should compile
# Try: #check @vadd.Differentiable
```

---

### Phase 2: Matrix Arithmetic (Linear)

Operations on matrices that are element-wise or inherited from vectors.

- [ ] **matAdd** - Matrix addition
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem matAdd.Differentiable`
  - [ ] Proof strategy: Similar to vadd

- [ ] **matSub** - Matrix subtraction
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem matSub.Differentiable`
  - [ ] Proof strategy: Similar to vsub

- [ ] **matSmul** - Scalar-matrix multiplication
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem matSmul.Differentiable`
  - [ ] Proof strategy: Similar to smul

- [ ] **transpose** - Matrix transpose
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem transpose.Differentiable`
  - [ ] Proof strategy: `unfold transpose; fun_prop` or show it's linear

**Testing after Phase 2:**
```bash
lake build VerifiedNN.Core.LinearAlgebra
```

---

### Phase 3: Bilinear Vector Operations

Operations with two vector arguments. Need separate rules for each argument.

- [ ] **vmul** - Element-wise vector multiplication
  - [ ] Remove TODO comments about fun_prop/fun_trans
  - [ ] Add `@[fun_prop] theorem vmul.arg_x.Differentiable`
  - [ ] Add `@[fun_prop] theorem vmul.arg_y.Differentiable`
  - [ ] Add `@[data_synth] theorem vmul.arg_x.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem vmul.arg_x.HasRevFDeriv`
  - [ ] Add `@[data_synth] theorem vmul.arg_y.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem vmul.arg_y.HasRevFDeriv`
  - [ ] Proof strategy: `data_synth` after unfolding

- [ ] **dot** - Vector dot product
  - [ ] Remove TODO comments
  - [ ] Add `@[fun_prop] theorem dot.arg_x.Differentiable`
  - [ ] Add `@[fun_prop] theorem dot.arg_y.Differentiable`
  - [ ] Add `@[data_synth] theorem dot.arg_x.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem dot.arg_x.HasRevFDeriv`
  - [ ] Add `@[data_synth] theorem dot.arg_y.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem dot.arg_y.HasRevFDeriv`
  - [ ] Proof strategy: `data_synth` with explicit definition unfolding

**Testing after Phase 3:**
```bash
lake build VerifiedNN.Core.LinearAlgebra
# Gradient check: vmul and dot should have working AD
```

---

### Phase 4: Norm Operations

Nonlinear operations composed from simpler operations.

- [ ] **normSq** - Squared L2 norm
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem normSq.Differentiable`
  - [ ] Add `@[data_synth] theorem normSq.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem normSq.HasRevFDeriv`
  - [ ] Proof strategy: Composition of dot(x,x); `fun_prop` for differentiability
  - [ ] Note: d/dx(dot(x,x)) = 2*x (check this formula)

- [ ] **norm** - L2 norm
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem norm.Differentiable`
  - [ ] Add `@[data_synth] theorem norm.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem norm.HasRevFDeriv`
  - [ ] Proof strategy: Composition of sqrt(normSq); need sqrt registered
  - [ ] Check: Is Float.sqrt differentiable in SciLean?

**Testing after Phase 4:**
```bash
lake build VerifiedNN.Core.LinearAlgebra
# May need to check if Float.sqrt is registered
```

---

### Phase 5: Core Matrix Operations

These are central to neural network computation. SciLean may already have some.

- [ ] **matvec** - Matrix-vector multiplication
  - [ ] FIRST: Check if SciLean already provides this
    ```bash
    grep -r "matvec" .lake/packages/SciLean/SciLean/AD/Rules/
    ```
  - If NOT in SciLean:
    - [ ] Remove TODO comments
    - [ ] Add `@[fun_prop] theorem matvec.arg_A.Differentiable`
    - [ ] Add `@[fun_prop] theorem matvec.arg_x.Differentiable`
    - [ ] Add `@[data_synth] theorem matvec.arg_A.HasFwdFDeriv`
    - [ ] Add `@[data_synth] theorem matvec.arg_A.HasRevFDeriv`
    - [ ] Add `@[data_synth] theorem matvec.arg_x.HasFwdFDeriv`
    - [ ] Add `@[data_synth] theorem matvec.arg_x.HasRevFDeriv`
  - Else: May need import statement

- [ ] **matmul** - Matrix-matrix multiplication
  - [ ] Remove TODO comments
  - [ ] Add `@[fun_prop] theorem matmul.arg_A.Differentiable`
  - [ ] Add `@[fun_prop] theorem matmul.arg_B.Differentiable`
  - [ ] Add `@[data_synth] theorem matmul.arg_A.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem matmul.arg_A.HasRevFDeriv`
  - [ ] Add `@[data_synth] theorem matmul.arg_B.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem matmul.arg_B.HasRevFDeriv`
  - [ ] Proof strategy: Composition of matvec; `data_synth`

**Testing after Phase 5:**
```bash
lake build VerifiedNN.Core.LinearAlgebra
# These are critical for network training
```

---

### Phase 6: Advanced Matrix Operations

Lower priority operations that enhance functionality.

- [ ] **outer** - Outer product
  - [ ] Remove TODO comments
  - [ ] Add `@[fun_prop] theorem outer.arg_x.Differentiable`
  - [ ] Add `@[fun_prop] theorem outer.arg_y.Differentiable`
  - [ ] Add `@[data_synth] theorem outer.arg_x.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem outer.arg_x.HasRevFDeriv`
  - [ ] Add `@[data_synth] theorem outer.arg_y.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem outer.arg_y.HasRevFDeriv`
  - [ ] Proof strategy: Direct from definition

**Testing after Phase 6:**
```bash
lake build VerifiedNN.Core.LinearAlgebra
```

---

### Phase 7: Batch Operations

May inherit from simpler operations via composition.

- [ ] **batchMatvec** - Batch matrix-vector multiplication
  - [ ] Remove TODO comments
  - [ ] Check if this can inherit from matvec
  - If not inheritable:
    - [ ] Add `@[fun_prop] theorem batchMatvec.Differentiable`
    - [ ] Add `@[data_synth]` rules as needed

- [ ] **batchAddVec** - Add vector to each row of batch
  - [ ] Remove TODO comments
  - [ ] Check if this can inherit from vadd
  - If not inheritable:
    - [ ] Add `@[fun_prop] theorem batchAddVec.Differentiable`
    - [ ] Add `@[data_synth]` rules as needed

**Testing after Phase 7:**
```bash
lake build VerifiedNN.Core.LinearAlgebra
lake build VerifiedNN.Testing.GradientCheck  # Full gradient checking
```

---

## Activation.lean - Operations to Register

### Phase A: ReLU Family

Core activation for hidden layers. ReLU is piecewise linear.

- [ ] **relu** - Rectified Linear Unit
  - [ ] Remove TODO comments (3 instances)
  - [ ] Add `@[fun_prop] theorem relu.Differentiable`
  - [ ] Add `@[data_synth] theorem relu.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem relu.HasRevFDeriv`
  - [ ] Proof strategy: `unfold relu; fun_prop` and `data_synth`
  - [ ] Note: Gradient is 1 if x > 0, else 0

- [ ] **reluVec** - Element-wise ReLU on vectors
  - [ ] Remove TODO comment
  - [ ] Check if composition automatically works
  - If yes: May just need `@[fun_prop]`
  - If no: Add `@[data_synth]` for element-wise application
  - [ ] Add `@[fun_prop] theorem reluVec.Differentiable`

- [ ] **reluBatch** - Element-wise ReLU on batches
  - [ ] Remove TODO comment
  - [ ] Similar to reluVec
  - [ ] Add `@[fun_prop] theorem reluBatch.Differentiable`

- [ ] **leakyRelu** - Leaky ReLU with parameter α
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem leakyRelu.arg_x.Differentiable`
  - [ ] Add `@[data_synth] theorem leakyRelu.arg_x.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem leakyRelu.arg_x.HasRevFDeriv`
  - [ ] Proof strategy: Similar to relu, with α instead of 0
  - [ ] Note: Gradient is α if x < 0, 1 if x > 0

- [ ] **leakyReluVec** - Element-wise leaky ReLU
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem leakyReluVec.Differentiable`

**Testing after Phase A:**
```bash
lake build VerifiedNN.Core.Activation
# ReLU is fundamental to the network
```

---

### Phase B: Sigmoid Family

Smooth activations using exp.

- [ ] **sigmoid** - Logistic sigmoid
  - [ ] Remove TODO comments (2 instances)
  - [ ] CHECK: Is Float.exp registered as differentiable?
    ```bash
    grep -r "exp.*Differentiable" .lake/packages/SciLean/SciLean/AD/Rules/
    ```
  - If Float.exp is NOT registered, may need to add it first
  - [ ] Add `@[fun_prop] theorem sigmoid.Differentiable`
  - [ ] Add `@[data_synth] theorem sigmoid.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem sigmoid.HasRevFDeriv`
  - [ ] Proof strategy: Composition of exp; `fun_prop` and `data_synth`
  - [ ] Note: σ'(x) = σ(x)(1 - σ(x))

- [ ] **sigmoidVec** - Element-wise sigmoid
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem sigmoidVec.Differentiable`

- [ ] **sigmoidBatch** - Batch sigmoid
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem sigmoidBatch.Differentiable`

- [ ] **tanh** - Hyperbolic tangent
  - [ ] Remove TODO comment
  - [ ] CHECK: Is Float.tanh or equivalent registered?
  - [ ] Add `@[fun_prop] theorem tanh.Differentiable`
  - [ ] Add `@[data_synth] theorem tanh.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem tanh.HasRevFDeriv`
  - [ ] Proof strategy: Composition or built-in; check SciLean
  - [ ] Note: tanh'(x) = 1 - tanh²(x) = sech²(x)

- [ ] **tanhVec** - Element-wise tanh
  - [ ] Remove TODO comment
  - [ ] Add `@[fun_prop] theorem tanhVec.Differentiable`

**Testing after Phase B:**
```bash
lake build VerifiedNN.Core.Activation
# Sigmoid and tanh are alternatives to ReLU
```

---

### Phase C: Softmax (Special Handling)

Most critical activation for output layer. May already be in SciLean.

- [ ] **FIRST: Check if SciLean provides softmax**
  ```bash
  grep -r "softmax" .lake/packages/SciLean/SciLean/AD/Rules/
  ```

- If YES (SciLean provides it):
  - [ ] Add import/check existing registration works
  - [ ] Remove TODO comments
  - [ ] Verify with gradient checking

- If NO (need to implement):
  - [ ] Remove TODO comments (2 instances)
  - [ ] Add `@[fun_prop] theorem softmax.Differentiable`
  - [ ] Add `@[data_synth] theorem softmax.HasFwdFDeriv`
  - [ ] Add `@[data_synth] theorem softmax.HasRevFDeriv`
  - [ ] Proof strategy: Composition of exp/sum with numerical stability
  - [ ] Note: Complex formula due to normalization
  - [ ] Reference: Jacobian is s[i]s[j] - s[i]δ[ij]

**Testing after Phase C:**
```bash
lake build VerifiedNN.Core.Activation
# Softmax is essential for classification output
```

---

## Summary Checklist

### LinearAlgebra.lean
- [ ] Phase 1: vadd, vsub, smul (3 operations)
- [ ] Phase 2: matAdd, matSub, matSmul, transpose (4 operations)
- [ ] Phase 3: vmul, dot (2 operations, 12 registrations)
- [ ] Phase 4: normSq, norm (2 operations)
- [ ] Phase 5: matvec, matmul (2 operations, ~12 registrations)
- [ ] Phase 6: outer (1 operation, 6 registrations)
- [ ] Phase 7: batchMatvec, batchAddVec (2 operations)

**Total: 18 operations**

### Activation.lean
- [ ] Phase A: relu, reluVec, reluBatch, leakyRelu, leakyReluVec (5 operations)
- [ ] Phase B: sigmoid, sigmoidVec, sigmoidBatch, tanh, tanhVec (5 operations)
- [ ] Phase C: softmax (1 operation)

**Total: 11 operations**

**Grand Total: 29 operations to register**

---

## Before Starting Implementation

### Prerequisites Checklist

- [ ] Read `AD_REGISTRATION_RESEARCH_REPORT.md` for detailed patterns
- [ ] Read `AD_REGISTRATION_QUICKSTART.md` for syntax reference
- [ ] Open `.lake/packages/SciLean/SciLean/AD/Rules/MatVecMul.lean` as template
- [ ] Open `.lake/packages/SciLean/SciLean/AD/Rules/Exp.lean` as reference
- [ ] Have `/Users/eric/LEAN_mnist/VerifiedNN/Core/LinearAlgebra.lean` open
- [ ] Have `/Users/eric/LEAN_mnist/VerifiedNN/Core/Activation.lean` open

### Critical Checks Before Each Phase

```bash
# Before implementation
lake build VerifiedNN.Core.LinearAlgebra  # Verify baseline
lake build VerifiedNN.Core.Activation

# After each phase
lake build VerifiedNN.Core.LinearAlgebra  # Should still compile
lake build VerifiedNN.Core.Activation

# Optionally run gradient checks
lake build VerifiedNN.Testing.GradientCheck
```

---

## Common Pitfalls to Avoid

1. **Forgetting to remove TODO comments** - Each TODO should be replaced with actual theorem
2. **Wrong attribute names** - Use `@[fun_prop]` not `@[prop]`, `@[data_synth]` not `@[synth]`
3. **Missing argument separation** - Use `arg_x` and `arg_y` for separate rules
4. **Incomplete registrations** - Register both forward (HasFwdFDeriv) and reverse (HasRevFDeriv) modes
5. **Forgetting to check SciLean first** - Don't reimplement what's already there (matvec, softmax, exp)
6. **Wrong scalar type** - Use `Float` not `ℝ` for computational operations
7. **Circular dependencies** - Don't use norm in norm registration, use compositions of simpler parts

---

## Success Criteria

After completing all phases:

1. **All files compile:**
   ```bash
   lake build VerifiedNN.Core.LinearAlgebra  # No errors
   lake build VerifiedNN.Core.Activation     # No errors
   ```

2. **All TODOs replaced:**
   ```bash
   grep "TODO.*fun_trans\|TODO.*fun_prop" VerifiedNN/Core/*.lean  # Empty result
   ```

3. **Gradient checking works:**
   ```bash
   lake build VerifiedNN.Testing.GradientCheck
   # Run and verify gradients match numerical approximations
   ```

4. **Training still works:**
   ```bash
   lake exe mnistTrain --epochs 1 --batch-size 32
   # Should train without errors
   ```

---

## Documentation After Completion

Once implementation is complete:

1. [ ] Update module docstrings to remove TODOs
2. [ ] Update verification status in README files
3. [ ] Add examples of derivative usage (if helpful)
4. [ ] Commit with message: "Register AD attributes for all operations"
5. [ ] Update CLEANUP_SUMMARY.md with completion status

---

## Notes Section

Use this space for tracking implementation details:

```
Phase 1 (vadd/vsub/smul):
[Your notes here]

Phase 2 (Matrix arithmetic):
[Your notes here]

Phase A (ReLU family):
[Your notes here]

etc.
```

---

## Reference Links

- **Full Research:** [AD_REGISTRATION_RESEARCH_REPORT.md](AD_REGISTRATION_RESEARCH_REPORT.md)
- **Quick Reference:** [AD_REGISTRATION_QUICKSTART.md](AD_REGISTRATION_QUICKSTART.md)
- **SciLean MatVecMul Template:** `.lake/packages/SciLean/SciLean/AD/Rules/MatVecMul.lean`
- **Project LinearAlgebra:** `VerifiedNN/Core/LinearAlgebra.lean`
- **Project Activation:** `VerifiedNN/Core/Activation.lean`

