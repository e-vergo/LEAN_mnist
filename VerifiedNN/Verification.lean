import VerifiedNN.Verification.GradientCorrectness
import VerifiedNN.Verification.TypeSafety
import VerifiedNN.Verification.Convergence
import VerifiedNN.Verification.Tactics

/-!
# Verification Module

This module provides formal verification of neural network training correctness.

## Submodules

- **GradientCorrectness**: Proves automatic differentiation computes correct gradients
- **TypeSafety**: Proves dependent types enforce dimension correctness
- **Convergence**: Convergence theorems for SGD (axiomatized)
- **Tactics**: Custom proof tactics for verification (placeholder)

## Primary Contribution

The GradientCorrectness module establishes the core scientific contribution: formal proofs
that automatic differentiation computes mathematically correct gradients for neural network
operations.

## Verification Philosophy

Mathematical properties are proven on ℝ (real numbers), while computational implementation
uses Float (IEEE 754). The Float→ℝ gap is acknowledged—we verify symbolic correctness,
not floating-point numerics.

See the README.md in VerifiedNN/Verification/ for detailed module descriptions and
verification status.
-/
