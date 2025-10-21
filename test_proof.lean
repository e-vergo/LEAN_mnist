import SciLean
open SciLean

def vadd (n : Nat) (x y : Float^[n]) : Float^[n] := ⊞ i => x[i] + y[i]

-- Test how to prove array equality
theorem test_vadd_comm (n : Nat) (x y : Float^[n]) :
  vadd n x y = vadd n y x := by
  ext i
  simp [vadd]
  ring
