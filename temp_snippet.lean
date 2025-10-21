import SciLean

set_default_scalar Float

-- The fundamental insight: DataArrayN's size IS the type parameter
-- There's no runtime .size because the size is statically known

-- This is what we want to state:
-- "If v has type Float^[n], then it has n elements"
-- But this is tautological - the type already says it!

-- The correct formulation for type safety theorems:
example {n : Nat} (v : Float^[n]) : True := by
  -- The fact that v type-checks means it has exactly n elements
  -- This is enforced by the type system itself
  trivial

-- What we can prove:
-- 1. Operations preserve type-level dimensions
example {m n : Nat} (A : Float^[m, n]) (x : Float^[n]) : Float^[m] := 
  A * x  -- Matrix-vector multiply returns correctly-typed result

-- 2. The type system prevents mismatches
-- example {m n : Nat} (A : Float^[m, n]) (x : Float^[m]) : Float^[n] := 
--   A * x  -- This won't type-check!
