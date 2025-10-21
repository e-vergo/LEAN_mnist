/-
# Activation Functions

Activation functions with automatic differentiation support.
-/

import VerifiedNN.Core.DataTypes
import SciLean

namespace VerifiedNN.Core.Activation

open SciLean
open VerifiedNN.Core

/-- ReLU activation function -/
def relu (x : Float) : Float :=
  if x > 0 then x else 0

/-- Element-wise ReLU on vectors
    Applies ReLU to each element: y[i] = max(0, x[i]) -/
def reluVec {n : Nat} (x : Vector n) : Vector n :=
  ⊞ i => relu x[i]

/-- Element-wise ReLU on batches
    Applies ReLU to each element: y[k,i] = max(0, x[k,i]) -/
def reluBatch {b n : Nat} (x : Batch b n) : Batch b n :=
  ⊞ (k, i) => relu x[k,i]

/-- Softmax activation function
    Computes exp(x[i]) / Σⱼ exp(x[j])
    TODO: Add numerical stability via max subtraction when needed -/
def softmax {n : Nat} (x : Vector n) : Vector n :=
  -- Simple implementation without numerical stability trick for now
  let expVals := ⊞ i => Float.exp x[i]
  let sumExp := ∑ i, expVals[i]
  ⊞ i => expVals[i] / sumExp

/-- Sigmoid activation function -/
def sigmoid (x : Float) : Float :=
  1 / (1 + Float.exp (-x))

/-- Element-wise sigmoid on vectors
    Applies sigmoid to each element: y[i] = 1 / (1 + exp(-x[i])) -/
def sigmoidVec {n : Nat} (x : Vector n) : Vector n :=
  ⊞ i => sigmoid x[i]

/-- Element-wise sigmoid on batches
    Applies sigmoid to each element: y[k,i] = 1 / (1 + exp(-x[k,i])) -/
def sigmoidBatch {b n : Nat} (x : Batch b n) : Batch b n :=
  ⊞ (k, i) => sigmoid x[k,i]

/-- Leaky ReLU activation with slope for negative values -/
def leakyRelu (alpha : Float := 0.01) (x : Float) : Float :=
  if x > 0 then x else alpha * x

/-- Element-wise Leaky ReLU on vectors
    Applies leaky ReLU to each element: y[i] = max(x[i], alpha * x[i]) -/
def leakyReluVec {n : Nat} (alpha : Float := 0.01) (x : Vector n) : Vector n :=
  ⊞ i => leakyRelu alpha x[i]

/-- Tanh activation function -/
def tanh (x : Float) : Float :=
  let expPos := Float.exp x
  let expNeg := Float.exp (-x)
  (expPos - expNeg) / (expPos + expNeg)

/-- Element-wise tanh on vectors
    Applies tanh to each element: y[i] = tanh(x[i]) -/
def tanhVec {n : Nat} (x : Vector n) : Vector n :=
  ⊞ i => tanh x[i]

-- Differentiation properties
-- These will be implemented once the core operations are working

/-- Analytical derivative of ReLU: 1 if x > 0, 0 if x < 0, undefined at 0 -/
def reluDerivative (x : Float) : Float :=
  if x > 0 then 1 else 0

/-- Analytical derivative of sigmoid: σ(x) * (1 - σ(x)) -/
def sigmoidDerivative (x : Float) : Float :=
  let s := sigmoid x
  s * (1 - s)

/-- Analytical derivative of tanh: 1 - tanh²(x) -/
def tanhDerivative (x : Float) : Float :=
  let t := tanh x
  1 - t * t

end VerifiedNN.Core.Activation
