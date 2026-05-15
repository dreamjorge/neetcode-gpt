import numpy as np
from numpy.typing import NDArray


class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # --- WHAT IS SOFTMAX? ---
        #
        # Softmax is the activation function used in the OUTPUT layer of
        # classification neural networks. It converts a vector of raw scores
        # (called "logits") into a valid PROBABILITY DISTRIBUTION:
        #
        #   → Every output value is between 0 and 1
        #   → All output values sum to exactly 1.0
        #
        # This lets the network answer: "How confident am I in each class?"
        #
        # Formula:
        #                    e^(z_i)
        #   softmax(z_i) = ──────────
        #                  Σ e^(z_j)
        #
        # Example:
        #   logits z = [2.0, 1.0, 0.1]          ← raw model scores
        #   softmax  = [0.659, 0.242, 0.099]     ← probabilities
        #              ───────────────────
        #              sum = 1.0  ✓
        #
        # The highest logit becomes the highest probability,
        # but the relationship is EXPONENTIAL — not linear.
        # A small score gap becomes a large probability gap:
        #   logit diff: 2.0 - 1.0 = 1.0
        #   prob  diff: 0.659 - 0.242 = 0.417   ← amplified

        # --- STEP 1: Numerical stability shift ---
        #
        # PROBLEM: e^z explodes for large z values.
        #   z = 1000  →  e^1000 ≈ 10^434  →  float overflow → inf → NaN
        #
        # SOLUTION: subtract the maximum value before exponentiating.
        #   Mathematically, this does NOT change the result:
        #
        #     e^(z_i - max)          e^z_i × e^(-max)      e^z_i
        #   ───────────────── =  ──────────────────────── = ──────── = softmax(z_i)
        #   Σ e^(z_j - max)      Σ e^z_j × e^(-max)       Σ e^z_j
        #
        # The e^(-max) cancels out in numerator and denominator — exact same answer,
        # but now the largest value is always e^0 = 1, and all others are ≤ 1.
        #
        # Example:
        #   z = [1000, 1001, 1002]
        #   shifted = [-2, -1, 0]      ← safe range
        #   e^shifted = [0.135, 0.368, 1.0]   ← no overflow  ✓
        shifted_logits = z - np.max(z)

        # --- STEP 2: Exponentiate the shifted logits ---
        #
        # e^x is always positive → guarantees all outputs > 0.
        # Larger inputs produce larger exponentials, preserving the ranking.
        #
        # The exponential function AMPLIFIES differences:
        #   linear:      3 - 1 = 2        (difference of 2)
        #   exponential: e^3 / e^1 ≈ 7.4  (ratio of 7.4)
        exponentiated = np.exp(shifted_logits)

        # --- STEP 3: Normalize to produce a probability distribution ---
        #
        # Dividing each value by the total sum forces all values into [0, 1]
        # and guarantees they sum to exactly 1.0.
        #
        # This is the same idea as converting raw counts to percentages:
        #   counts = [3, 1, 1]  →  proportions = [0.6, 0.2, 0.2]  ✓
        probabilities = exponentiated / np.sum(exponentiated)

        return np.round(probabilities, 4)