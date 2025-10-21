#!/bin/bash
cd /Users/eric/LEAN_mnist
timeout 180 lake build VerifiedNN.Verification.GradientCorrectness 2>&1 | tee /tmp/build_output.txt
echo "Exit code: $?"
