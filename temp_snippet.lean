import VerifiedNN.Network.Architecture

example (net1 net2 : MLPArchitecture) : net1 = net2 ↔ net1.layer1 = net2.layer1 ∧ net2.layer2 = net2.layer2 := by
  constructor
  · intro h; rw [h]; exact ⟨rfl, rfl⟩
  · intro ⟨h1, h2⟩
    cases net1 with | mk l1a l2a =>
    cases net2 with | mk l1b l2b =>
    simp at h1 h2
    rw [h1, h2]