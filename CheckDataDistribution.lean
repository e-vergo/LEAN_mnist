import VerifiedNN.Data.MNIST

/-! Quick script to check class distribution in first 500 training samples -/

open VerifiedNN.Data.MNIST

unsafe def main : IO Unit := do
  IO.println "Checking class distribution in first 500 MNIST training samples..."
  let trainData ← loadMNISTTrain "data"

  let first500 := trainData.extract 0 (min 500 trainData.size)

  -- Count occurrences of each digit
  let mut counts : Array Nat := Array.replicate 10 0
  for (_input, label) in first500 do
    if label < 10 then
      counts := counts.modify label (· + 1)

  IO.println "\nClass distribution in first 500 samples:"
  IO.println "========================================"
  for digit in [0:10] do
    let count := counts[digit]!
    let percentage := (count.toFloat / first500.size.toFloat) * 100.0
    IO.println s!"Digit {digit}: {count} samples ({percentage.floor}%)"

  IO.println ""
  IO.println s!"Total: {first500.size} samples"
