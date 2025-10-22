# VerifiedNN Testing Guide

A comprehensive guide to writing, running, and debugging tests for the VerifiedNN project.

## Table of Contents

1. [Overview](#overview)
2. [Test Organization](#test-organization)
3. [Running Tests](#running-tests)
4. [Writing Unit Tests](#writing-unit-tests)
5. [Writing Integration Tests](#writing-integration-tests)
6. [Gradient Checking](#gradient-checking)
7. [Testing Best Practices](#testing-best-practices)
8. [Debugging Test Failures](#debugging-test-failures)
9. [Continuous Integration](#continuous-integration)

---

## Overview

### Testing Philosophy

VerifiedNN uses a multi-layered testing approach:

1. **Type-Level Verification** - Compile-time dimension checking via dependent types
2. **Unit Tests** - Component-level functional tests
3. **Integration Tests** - End-to-end pipeline validation
4. **Gradient Checking** - Numerical validation of automatic differentiation
5. **Property Tests** - Mathematical property verification (planned)

### Test Coverage Goals

| Component | Target Coverage | Current Status |
|-----------|----------------|----------------|
| Core (DataTypes, LinearAlgebra, Activation) | 90%+ | 75% (6/8 suites) |
| Optimizer (SGD, Momentum, LR schedules) | 100% | ✅ 100% (6/6 suites) |
| Layer (Dense, Composition) | 90%+ | Type-level + properties |
| Network (Architecture, Initialization) | 80%+ | Partial |
| Loss (CrossEntropy) | 90%+ | Properties proven |
| Training (Loop, Batch) | 70%+ | Partial |
| Data (MNIST loading) | 100% | ✅ 100% (validated) |

### Test Frameworks

**LSpec** - Primary test framework (Lean 4)
```lean
import LSpec

def myTests := test "description" $ ...

#eval myTests.run
```

**Manual Assertions** - For simple checks
```lean
assert! condition "error message"
```

**Gradient Checking** - Finite difference comparison
```lean
def checkGradient (f : Vector n → Float) (x : Vector n) : Bool :=
  let adGrad := (∇ x', f x') x |>.rewrite_by fun_trans
  let fdGrad := finiteDifferenceGradient f x
  vectorApproxEq adGrad fdGrad 1e-5
```

---

## Test Organization

### Directory Structure

```
VerifiedNN/Testing/
├── RunTests.lean                # Unified test runner
├── UnitTests.lean               # Core component tests
├── OptimizerTests.lean          # Optimizer validation
├── OptimizerVerification.lean   # Type-level verification
├── GradientCheck.lean           # AD numerical validation
├── Integration.lean             # End-to-end tests
├── MNISTIntegration.lean        # MNIST smoke test
├── MNISTLoadTest.lean           # MNIST data validation
├── SmokeTest.lean               # Fast CI/CD test (<10s)
└── FullIntegration.lean         # Complete integration suite
```

### Test Layers

**Layer 0: Type-Level Tests (Compile-Time)**
- Location: `OptimizerVerification.lean`
- Method: If it compiles, dimensions are correct
- Speed: Instant (part of compilation)

**Layer 1: Unit Tests (Component-Level)**
- Location: `UnitTests.lean`, `OptimizerTests.lean`
- Scope: Individual functions and modules
- Speed: Fast (<5 seconds)

**Layer 2: Integration Tests (Pipeline-Level)**
- Location: `Integration.lean`, `MNISTIntegration.lean`
- Scope: Multiple modules working together
- Speed: Medium (5-30 seconds)

**Layer 3: Full System Tests**
- Location: `FullIntegration.lean`
- Scope: Complete training pipeline
- Speed: Slow (30+ seconds)

---

## Running Tests

### Quick Start

```bash
# Run all tests via unified runner
lake build VerifiedNN.Testing.RunTests
lake env lean --run VerifiedNN/Testing/RunTests.lean

# Run individual test suites
lake env lean --run VerifiedNN/Testing/UnitTests.lean
lake env lean --run VerifiedNN/Testing/OptimizerTests.lean

# Fast smoke test for CI/CD
lake exe smokeTest
```

### Test Output

**Successful test:**
```
✓ Test: activation_relu_properties
  All checks passed
```

**Failed test:**
```
✗ Test: vectorApproxEq
  Expected: true
  Got: false
  Location: VerifiedNN/Testing/UnitTests.lean:45
```

### Running Specific Tests

```bash
# Build specific test file
lake build VerifiedNN.Testing.UnitTests

# Run with Lean interpreter
lake env lean --run VerifiedNN/Testing/UnitTests.lean

# Run compiled executable (for smoke test)
lake exe smokeTest
```

### Test Selection

Edit `RunTests.lean` to enable/disable test suites:

```lean
def main : IO Unit := do
  IO.println "Running VerifiedNN Test Suite\n"

  -- Enable/disable test suites
  runTestSuite "Unit Tests" UnitTests.allTests
  runTestSuite "Optimizer Tests" OptimizerTests.allTests
  -- runTestSuite "Integration Tests" Integration.allTests  -- Commented out
```

---

## Writing Unit Tests

### Basic Test Structure

```lean
import LSpec
import VerifiedNN.Core.Activation

namespace VerifiedNN.Testing

open LSpec
open VerifiedNN.Core.Activation

def testReLUPositive := test "ReLU on positive input" $ do
  let input : Float := 5.0
  let output := relu input
  check ("ReLU(5.0) = 5.0" :  output == 5.0)

def testReLUNegative := test "ReLU on negative input" $ do
  let input : Float := -3.0
  let output := relu input
  check ("ReLU(-3.0) = 0.0" : output == 0.0)

def testReLUZero := test "ReLU on zero" $ do
  let input : Float := 0.0
  let output := relu input
  check ("ReLU(0.0) = 0.0" : output == 0.0)

-- Combine tests into suite
def allReLUTests := [testReLUPositive, testReLUNegative, testReLUZero]

-- Run tests
#eval (allReLUTests.foldl (· ++ ·) .done).run

end VerifiedNN.Testing
```

### Testing Approximate Equality

For floating-point comparisons, use `approxEq`:

```lean
import VerifiedNN.Core.DataTypes

def testSoftmaxSumToOne := test "softmax probabilities sum to 1" $ do
  let input := ![2.0, 1.0, 0.1]
  let output := softmax input
  let sum := output[0] + output[1] + output[2]
  check ("Softmax sums to 1.0" : approxEq sum 1.0 1e-6)
```

### Testing Vector Operations

```lean
def testVectorAddition := test "vector addition" $ do
  let v1 := ![1.0, 2.0, 3.0]
  let v2 := ![4.0, 5.0, 6.0]
  let result := vadd v1 v2
  let expected := ![5.0, 7.0, 9.0]
  check ("vadd works correctly" : vectorApproxEq result expected 1e-7)
```

### Testing Matrix Operations

```lean
def testMatrixVectorMultiply := test "matrix-vector multiplication" $ do
  let A : Matrix 2 3 := ⊞ (i, j) => (i.val * 3 + j.val).toFloat
  let x : Vector 3 := ![1.0, 2.0, 3.0]
  let result := matvec A x

  -- Expected: [0*1 + 1*2 + 2*3, 3*1 + 4*2 + 5*3]
  --         = [8.0, 26.0]
  let expected := ![8.0, 26.0]
  check ("matvec correct" : vectorApproxEq result expected 1e-7)
```

### Testing Activation Functions

```lean
def testSigmoidRange := test "sigmoid output in [0,1]" $ do
  let testValues := [-100.0, -1.0, 0.0, 1.0, 100.0]
  for val in testValues do
    let output := sigmoid val
    check (s!"sigmoid({val}) ∈ [0,1]" : output ≥ 0.0 && output ≤ 1.0)
```

### Test Organization Pattern

```lean
-- 1. Define individual tests
def test1 := test "description 1" $ ...
def test2 := test "description 2" $ ...
def test3 := test "description 3" $ ...

-- 2. Group related tests
def componentATests := [test1, test2]
def componentBTests := [test3]

-- 3. Combine into suite
def allTests := componentATests ++ componentBTests

-- 4. Export for test runner
def main : IO Unit := do
  IO.println "Running MyComponent Tests"
  let results := (allTests.foldl (· ++ ·) .done).run
  IO.println results
```

---

## Writing Integration Tests

### Integration Test Structure

```lean
import VerifiedNN.Training.Loop
import VerifiedNN.Network.Architecture
import VerifiedNN.Data.MNIST

def testFullTrainingPipeline := test "end-to-end training" $ do
  -- 1. Generate or load data
  let (trainData, trainLabels) := generateSyntheticData 100 784 10

  -- 2. Initialize network
  let net ← initializeNetwork mlpArch

  -- 3. Run training
  let trainedNet ← trainEpoch net trainData trainLabels batchSize learningRate

  -- 4. Evaluate
  let (loss, acc) := evaluateFull trainedNet trainData trainLabels

  -- 5. Assert expectations
  check ("Loss decreases" : loss < 2.5)
  check ("Accuracy improves" : acc > 0.1)
```

### Data Pipeline Integration Test

```lean
def testMNISTLoadingPipeline := test "MNIST data loads correctly" $ do
  -- 1. Load data
  let trainImages ← loadMNISTImages "data/train-images-idx3-ubyte"
  let trainLabels ← loadMNISTLabels "data/train-labels-idx1-ubyte"

  -- 2. Validate dimensions
  check ("Train images count" : trainImages.size == 60000)
  check ("Train labels count" : trainLabels.size == 60000)

  -- 3. Validate ranges
  for img in trainImages.toSubarray 0 100 do
    for pixel in [0:784] do
      check ("Pixel in valid range" : img[pixel] ≥ 0.0 && img[pixel] ≤ 255.0)

  -- 4. Validate labels
  for label in trainLabels.toSubarray 0 100 do
    check ("Label in [0,9]" : label ≥ 0 && label ≤ 9)
```

### Network Initialization Test

```lean
def testNetworkInitialization := test "network initializes correctly" $ do
  let net ← initializeNetwork mlpArch

  -- Check layer dimensions
  check ("Layer 1 weights shape" :
    net.layer1.weights.size = (128, 784))
  check ("Layer 1 bias shape" :
    net.layer1.bias.size = 128)
  check ("Layer 2 weights shape" :
    net.layer2.weights.size = (10, 128))
  check ("Layer 2 bias shape" :
    net.layer2.bias.size = 10)

  -- Check initialization bounds (He initialization)
  let weightBound := Float.sqrt (2.0 / 784.0)
  for i in [0:128] do
    for j in [0:784] do
      check ("Weight initialized reasonably" :
        Float.abs net.layer1.weights[i,j] < weightBound * 10.0)
```

---

## Gradient Checking

### What is Gradient Checking?

Gradient checking validates automatic differentiation by comparing:
- **Analytical gradient** (from AD): `∇f(x)` computed symbolically
- **Numerical gradient** (finite differences): `(f(x+ε) - f(x-ε)) / 2ε`

If they match within tolerance, AD is correct.

### Finite Difference Implementation

```lean
def finiteDifferenceGradient {n : Nat}
  (f : Vector n → Float)
  (x : Vector n)
  (ε : Float := 1e-5) : Vector n := Id.run do

  let mut grad := x.copy

  for i in [0:n] do
    -- Perturb x[i] by +ε
    let mut xPlus := x.copy
    xPlus[i] := x[i] + ε

    -- Perturb x[i] by -ε
    let mut xMinus := x.copy
    xMinus[i] := x[i] - ε

    -- Central difference: (f(x+ε) - f(x-ε)) / 2ε
    grad[i] := (f xPlus - f xMinus) / (2.0 * ε)

  pure grad
```

### Basic Gradient Check

```lean
def checkGradientSimple := test "gradient of quadratic" $ do
  -- Define function: f(x) = ‖x‖² = x₀² + x₁² + x₂²
  let f (x : Vector 3) : Float := x[0]^2 + x[1]^2 + x[2]^2

  -- Test point
  let x := ![1.0, 2.0, 3.0]

  -- Analytical gradient: ∇f(x) = 2x
  let analyticalGrad := ⊞ i => 2.0 * x[i]

  -- Numerical gradient
  let numericalGrad := finiteDifferenceGradient f x

  -- Compare
  check ("Gradients match" :
    vectorApproxEq analyticalGrad numericalGrad 1e-5)
```

### Gradient Check for Network

```lean
def checkNetworkGradient := test "network gradient correctness" $ do
  let net ← initializeNetwork mlpArch
  let x := generateRandomVector 784
  let y := 5  -- Target class

  -- Define loss function
  let loss (params : Vector nParams) : Float :=
    let network := unflattenParams params
    let output := network.forwardPass x
    crossEntropyLoss output y

  -- Current parameters
  let params := flattenParams net

  -- Analytical gradient (via AD)
  let adGrad := (∇ p, loss p) params |>.rewrite_by fun_trans

  -- Numerical gradient (slow but correct)
  let fdGrad := finiteDifferenceGradient loss params

  -- Compare (looser tolerance for large networks)
  check ("Network gradient matches FD" :
    vectorApproxEq adGrad fdGrad 1e-4)
```

### Gradient Check Best Practices

**1. Use central differences, not forward differences:**
```lean
-- Good: Central difference (O(ε²) error)
let grad := (f (x + ε) - f (x - ε)) / (2 * ε)

-- Bad: Forward difference (O(ε) error)
let grad := (f (x + ε) - f x) / ε
```

**2. Choose ε carefully:**
- Too large: Approximation error dominates
- Too small: Numerical precision error dominates
- Sweet spot: `1e-5` to `1e-7` for Float

**3. Test multiple points:**
```lean
def gradientCheckMultiplePoints := test "gradient at various points" $ do
  let testPoints := [
    ![0.0, 0.0, 0.0],
    ![1.0, 1.0, 1.0],
    ![-1.0, 2.0, -3.0],
    ![100.0, -100.0, 0.0]
  ]

  for x in testPoints do
    let adGrad := ...
    let fdGrad := finiteDifferenceGradient f x
    check (s!"Gradient matches at {x}" :
      vectorApproxEq adGrad fdGrad 1e-5)
```

**4. Expect slower performance:**
- Gradient checking is O(n) function evaluations (n = parameter count)
- Use only in tests, not in training loop

---

## Testing Best Practices

### 1. Test Naming Conventions

```lean
-- Pattern: test<ComponentName><PropertyOrBehavior>
def testReLUPreservesPositive := ...
def testSoftmaxSumsToOne := ...
def testLayerOutputDimension := ...
def testOptimizerUpdateReducesLoss := ...
```

### 2. Arrange-Act-Assert Pattern

```lean
def testExample := test "example test" $ do
  -- Arrange: Set up test data
  let input := ![1.0, 2.0, 3.0]
  let expected := ![2.0, 4.0, 6.0]

  -- Act: Perform operation
  let result := smul 2.0 input

  -- Assert: Verify expectation
  check ("2 * v = 2v" : vectorApproxEq result expected 1e-7)
```

### 3. Test Independence

Each test should be self-contained:

```lean
-- Good: Self-contained
def testA := test "test A" $ do
  let data := generateTestData  -- Fresh data
  ...

def testB := test "test B" $ do
  let data := generateTestData  -- Fresh data (not reused from testA)
  ...

-- Bad: Shared mutable state
def sharedData := ...  -- Don't do this

def testA := test "test A" $ do
  modify sharedData  -- Affects testB
  ...
```

### 4. Meaningful Error Messages

```lean
-- Good: Descriptive message
check (s!"Softmax output sum should be 1.0, got {sum}" :
  approxEq sum 1.0 1e-6)

-- Bad: Generic message
check ("Test failed" : condition)
```

### 5. Edge Case Testing

Test boundary conditions:

```lean
def testReLUEdgeCases := test "ReLU edge cases" $ do
  -- Positive
  check ("ReLU(1.0) = 1.0" : relu 1.0 == 1.0)

  -- Zero (boundary)
  check ("ReLU(0.0) = 0.0" : relu 0.0 == 0.0)

  -- Negative
  check ("ReLU(-1.0) = 0.0" : relu (-1.0) == 0.0)

  -- Large values
  check ("ReLU(1e6) = 1e6" : relu 1e6 == 1e6)
  check ("ReLU(-1e6) = 0.0" : relu (-1e6) == 0.0)

  -- Very small values
  check ("ReLU(1e-7) = 1e-7" : relu 1e-7 == 1e-7)
  check ("ReLU(-1e-7) = 0.0" : relu (-1e-7) == 0.0)
```

### 6. Property-Based Testing

Test mathematical properties:

```lean
def testVectorAdditionCommutative := test "vadd is commutative" $ do
  let v1 := ![1.0, 2.0, 3.0]
  let v2 := ![4.0, 5.0, 6.0]

  let result1 := vadd v1 v2
  let result2 := vadd v2 v1

  check ("v1 + v2 = v2 + v1" :
    vectorApproxEq result1 result2 1e-10)

def testVectorAdditionAssociative := test "vadd is associative" $ do
  let v1 := ![1.0, 2.0, 3.0]
  let v2 := ![4.0, 5.0, 6.0]
  let v3 := ![7.0, 8.0, 9.0]

  let result1 := vadd (vadd v1 v2) v3
  let result2 := vadd v1 (vadd v2 v3)

  check ("(v1 + v2) + v3 = v1 + (v2 + v3)" :
    vectorApproxEq result1 result2 1e-10)
```

---

## Debugging Test Failures

### Step 1: Read the Error Message

```
✗ Test: vectorApproxEq
  Expected: true
  Got: false
  Location: VerifiedNN/Testing/UnitTests.lean:45
```

Go to line 45 and examine the test.

### Step 2: Add Debug Output

```lean
def testVectorOperation := test "vector operation" $ do
  let input := ![1.0, 2.0, 3.0]
  let result := myOperation input
  let expected := ![2.0, 4.0, 6.0]

  -- Add debug output
  IO.println s!"Input: {input}"
  IO.println s!"Result: {result}"
  IO.println s!"Expected: {expected}"

  check ("Operation correct" :
    vectorApproxEq result expected 1e-7)
```

### Step 3: Isolate the Issue

Create a minimal reproducer:

```lean
-- Original failing test
def testComplexOperation := test "complex operation" $ do
  let result := step1 |> step2 |> step3 |> step4
  check ("result correct" : ...)

-- Isolate each step
def testStep1 := test "step 1" $ do
  let result := step1
  IO.println s!"Step 1 result: {result}"
  check ("step 1 works" : ...)

def testStep2 := test "step 2" $ do
  let intermediate := step1
  let result := step2 intermediate
  IO.println s!"Step 2 result: {result}"
  check ("step 2 works" : ...)
```

### Step 4: Check Numerical Precision

```lean
-- If approximate equality fails, check the actual difference
def debugApproxEq {n : Nat} (v1 v2 : Vector n) : IO Unit := do
  let diff := ⊞ i => Float.abs (v1[i] - v2[i])
  IO.println s!"Element-wise differences: {diff}"
  let maxDiff := (∑ i, diff[i]) / n.toFloat  -- Average for now
  IO.println s!"Average absolute difference: {maxDiff}"

  if maxDiff > 1e-7 then
    IO.println "⚠️ Precision issue detected"
```

### Step 5: Verify Test Assumptions

```lean
def testWithAssumptionCheck := test "operation with checks" $ do
  let input := generateInput

  -- Check assumptions before running test
  check ("Input is valid" : isValidInput input)
  check ("Input in expected range" : allInRange input 0.0 1.0)

  -- Now run actual test
  let result := myOperation input
  check ("Result correct" : ...)
```

### Common Test Failure Patterns

**Pattern 1: Floating-point precision**
```lean
-- Symptom: check (1.0 + 1e-16 == 1.0) fails
-- Solution: Use approxEq with appropriate tolerance
check ("Values equal" : approxEq (1.0 + 1e-16) 1.0 1e-10)
```

**Pattern 2: Dimension mismatch**
```lean
-- Symptom: Type error when constructing test data
-- Solution: Verify dimensions match expectations
let v : Vector 10 := ...  -- Ensure Vector 10, not Vector 9 or 11
```

**Pattern 3: Index out of bounds**
```lean
-- Symptom: Array index out of bounds at runtime
-- Solution: Add bounds checks
for i in [0:n] do  -- Ensure i < n
  process array[i]
```

**Pattern 4: Incorrect gradient check tolerance**
```lean
-- Symptom: Gradient check fails on complex functions
-- Solution: Use looser tolerance for higher-dimensional gradients
check ("Gradient matches" :
  vectorApproxEq adGrad fdGrad 1e-4)  -- Not 1e-7 for large networks
```

---

## Continuous Integration

### Smoke Test for CI/CD

Fast test for quick feedback (<10 seconds):

```lean
-- VerifiedNN/Testing/SmokeTest.lean
def main : IO Unit := do
  IO.println "Running smoke test..."

  -- 1. Network initialization
  let net ← initializeNetwork mlpArch
  IO.println "✓ Network initializes"

  -- 2. Forward pass
  let input := generateRandomVector 784
  let output := net.forwardPass input
  assert! output.size = 10
  IO.println "✓ Forward pass works"

  -- 3. Basic prediction
  let prediction := argmax output
  assert! prediction ≥ 0 && prediction ≤ 9
  IO.println "✓ Prediction in valid range"

  IO.println "Smoke test passed! ✓"
```

Run in CI:
```bash
lake exe smokeTest || exit 1
```

### Full Test Suite for Nightly Builds

```bash
# Run all tests
lake build VerifiedNN.Testing.RunTests
lake env lean --run VerifiedNN/Testing/RunTests.lean

# Capture exit code
if [ $? -ne 0 ]; then
  echo "Tests failed"
  exit 1
fi
```

### Test Performance Benchmarks

Track test execution time:

```bash
#!/bin/bash
echo "Running performance benchmarks..."

time lake env lean --run VerifiedNN/Testing/UnitTests.lean
time lake env lean --run VerifiedNN/Testing/OptimizerTests.lean
time lake env lean --run VerifiedNN/Testing/Integration.lean
```

---

## Advanced Testing Topics

### Testing Noncomputable Code

Some code (like gradient computation) is noncomputable:

```lean
-- This won't compile to standalone binary
#eval (∇ x, f x) x0  -- Error: noncomputable

-- Instead: Test in proof mode or with computable approximations
theorem test_gradient_correct :
  fderiv ℝ f x0 = expected := by
  fun_trans
  simp [...]
```

### Testing with Random Data

```lean
def testWithRandomData := test "random data test" $ do
  -- Generate random seed
  let seed ← IO.rand 0 1000000
  IO.println s!"Using seed: {seed}"

  -- Generate random input
  let input := generateRandomVector 784 seed

  -- Test property that should hold for any input
  let output := relu input
  for i in [0:784] do
    check (s!"ReLU non-negative at index {i}" :
      output[i] ≥ 0.0)
```

### Testing IO Operations

```lean
def testDataLoading := test "MNIST data loads" $ do
  -- Use test data path
  let dataPath := "data/test-images-idx3-ubyte"

  -- Check file exists before loading
  let fileExists ← System.FilePath.pathExists dataPath
  if !fileExists then
    IO.println "⚠️ Test data not found, skipping test"
    return

  -- Load data
  let images ← loadMNISTImages dataPath

  -- Validate
  check ("Images loaded" : images.size > 0)
```

---

## Test Checklist for Contributors

Before submitting a PR, ensure:

- [ ] All new functions have unit tests
- [ ] Edge cases are tested (zero, negative, boundary values)
- [ ] Approximate equality used for floating-point comparisons
- [ ] Integration tests pass (if modifying multi-module code)
- [ ] Gradient checks pass (if modifying differentiable code)
- [ ] Smoke test passes (`lake exe smokeTest`)
- [ ] Test names are descriptive
- [ ] Error messages are meaningful
- [ ] Tests are independent (no shared mutable state)
- [ ] Documentation updated to reflect test coverage

---

## Resources

### Test Examples

- **Basic unit tests:** `VerifiedNN/Testing/UnitTests.lean`
- **Optimizer tests:** `VerifiedNN/Testing/OptimizerTests.lean`
- **Gradient checking:** `VerifiedNN/Testing/GradientCheck.lean`
- **Integration tests:** `VerifiedNN/Testing/Integration.lean`

### Documentation

- **LSpec documentation:** https://github.com/argumentcomputer/LSpec
- **Testing in Lean 4:** https://leanprover.github.io/theorem_proving_in_lean4/
- **VerifiedNN Testing README:** `VerifiedNN/Testing/README.md`

### Getting Help

- **Lean Zulip:** https://leanprover.zulipchat.com/ (#testing, #new members)
- **Project Issues:** Open an issue on GitHub
- **CLAUDE.md:** Development guidelines and conventions

---

**Last Updated:** 2025-10-22
**Maintained by:** Project contributors
**Questions?** See CONTRIBUTING.md or ask on Lean Zulip
