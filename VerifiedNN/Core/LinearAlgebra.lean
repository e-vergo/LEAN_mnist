/-
# Linear Algebra Operations

Matrix and vector operations using SciLean primitives with automatic differentiation support.
-/

import VerifiedNN.Core.DataTypes
import SciLean

namespace VerifiedNN.Core.LinearAlgebra

open SciLean
open VerifiedNN.Core

/-- Matrix-vector multiplication: A * x
    Computes y[i] = Σⱼ A[i,j] * x[j] for each row i -/
def matvec {m n : Nat} (A : Matrix m n) (x : Vector n) : Vector m :=
  ⊞ i => ∑ j, A[i,j] * x[j]

/-- Matrix-matrix multiplication: A * B
    Computes C[i,k] = Σⱼ A[i,j] * B[j,k] -/
def matmul {m n p : Nat} (A : Matrix m n) (B : Matrix n p) : Matrix m p :=
  ⊞ (i, k) => ∑ j, A[i,j] * B[j,k]

/-- Vector addition: x + y
    Element-wise addition of two vectors -/
def vadd {n : Nat} (x y : Vector n) : Vector n :=
  ⊞ i => x[i] + y[i]

/-- Vector subtraction: x - y
    Element-wise subtraction of two vectors -/
def vsub {n : Nat} (x y : Vector n) : Vector n :=
  ⊞ i => x[i] - y[i]

/-- Scalar-vector multiplication: c * x
    Multiply each element of vector by scalar -/
def smul {n : Nat} (c : Float) (x : Vector n) : Vector n :=
  ⊞ i => c * x[i]

/-- Element-wise vector multiplication (Hadamard product)
    Multiply corresponding elements: z[i] = x[i] * y[i] -/
def vmul {n : Nat} (x y : Vector n) : Vector n :=
  ⊞ i => x[i] * y[i]

/-- Vector dot product: ⟨x, y⟩
    Computes Σᵢ x[i] * y[i] -/
def dot {n : Nat} (x y : Vector n) : Float :=
  ∑ i, x[i] * y[i]

/-- Vector L2 norm squared: ‖x‖²
    Computes Σᵢ x[i]² -/
def normSq {n : Nat} (x : Vector n) : Float :=
  ∑ i, x[i] * x[i]

/-- Vector L2 norm: ‖x‖
    Computes √(Σᵢ x[i]²) -/
def norm {n : Nat} (x : Vector n) : Float :=
  Float.sqrt (normSq x)

/-- Matrix transpose: Aᵀ
    Swaps rows and columns: Aᵀ[j,i] = A[i,j] -/
def transpose {m n : Nat} (A : Matrix m n) : Matrix n m :=
  ⊞ (j, i) => A[i,j]

/-- Matrix addition
    Element-wise addition: C[i,j] = A[i,j] + B[i,j] -/
def maadd {m n : Nat} (A B : Matrix m n) : Matrix m n :=
  ⊞ (i, j) => A[i,j] + B[i,j]

/-- Matrix subtraction
    Element-wise subtraction: C[i,j] = A[i,j] - B[i,j] -/
def masub {m n : Nat} (A B : Matrix m n) : Matrix m n :=
  ⊞ (i, j) => A[i,j] - B[i,j]

/-- Scalar-matrix multiplication
    Multiply each element by scalar: B[i,j] = c * A[i,j] -/
def msmul {m n : Nat} (c : Float) (A : Matrix m n) : Matrix m n :=
  ⊞ (i, j) => c * A[i,j]

/-- Batch matrix-vector multiplication: X * A^T where X is batch
    For each row in batch, compute: Y[k,i] = Σⱼ A[i,j] * X[k,j]
    This is equivalent to Y = X * A^T -/
def batchMatvec {b m n : Nat} (A : Matrix m n) (X : Batch b n) : Batch b m :=
  ⊞ (k, i) => ∑ j, A[i,j] * X[k,j]

/-- Add a vector to each row of a batch (broadcasting)
    Adds vector v to each row: Y[k,j] = X[k,j] + v[j] -/
def batchAddVec {b n : Nat} (X : Batch b n) (v : Vector n) : Batch b n :=
  ⊞ (k, j) => X[k,j] + v[j]

/-- Outer product of two vectors: x ⊗ y
    Creates matrix: A[i,j] = x[i] * y[j] -/
def outer {m n : Nat} (x : Vector m) (y : Vector n) : Matrix m n :=
  ⊞ (i, j) => x[i] * y[j]

end VerifiedNN.Core.LinearAlgebra
