import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.LinearAlgebra.Matrix.ToLin

-- Test that the key lemmas compile

theorem test_vadd {n : ℕ} (b : Fin n → ℝ) (x : Fin n → ℝ) :
    fderiv ℝ (fun v => v + b) x = ContinuousLinearMap.id ℝ (Fin n → ℝ) := by
  have h1 : DifferentiableAt ℝ (fun v => v) x := differentiable_id.differentiableAt
  have h2 : DifferentiableAt ℝ (fun _ => b) x := (differentiable_const b).differentiableAt
  rw [fderiv_add h1 h2, fderiv_id, fderiv_const]
  simp

theorem test_smul {n : ℕ} (c : ℝ) (x : Fin n → ℝ) :
    fderiv ℝ (fun v : Fin n → ℝ => c • v) x = c • ContinuousLinearMap.id ℝ (Fin n → ℝ) := by
  simp only [show (fun v => c • v) = (c • ContinuousLinearMap.id ℝ (Fin n → ℝ)) from rfl]
  exact ContinuousLinearMap.fderiv _ x

theorem test_matvec {m n : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ) :
    fderiv ℝ (fun v => A.mulVec v) x = A.toLin := by
  simp only [show (fun v => A.mulVec v) = (A.toLin : Fin n → ℝ → Fin m → ℝ) from rfl]
  exact ContinuousLinearMap.fderiv A.toLin x

#check test_vadd
#check test_smul
#check test_matvec
