import numpy as np
from numpy.typing import NDArray


class Solution:

    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # --- WHAT IS SIGMOID? ---
        #
        # Sigmoid is an ACTIVATION FUNCTION — a mathematical gate that decides
        # how much a neuron "fires" based on its input.
        #
        # Formula:  σ(z) = 1 / (1 + e^(-z))
        #
        # It SQUASHES any real number into the range (0, 1):
        #
        #   z = very negative  →  σ(z) ≈ 0.0   (neuron off)
        #   z = 0              →  σ(z) = 0.5   (neuron halfway)
        #   z = very positive  →  σ(z) ≈ 1.0   (neuron fully on)
        #
        # This makes sigmoid perfect for BINARY CLASSIFICATION — interpreting
        # the output as a probability: "How likely is this class 1?"
        #
        # Shape of the curve (S-shaped, hence "sigmoid"):
        #
        #   σ(z)
        #    1 │         ___________
        #      │        /
        #   .5 │-------/------------
        #      │      /
        #    0 │_____/
        #      └──────────────────→ z
        #         neg   0   pos
        #
        # `np.exp(-z)` applies e^(-z) to every element in the array at once.
        # NumPy handles the full array without a Python loop — fast and clean.

        return np.round(1 / (1 + np.exp(-z)), 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # --- WHAT IS ReLU? ---
        #
        # ReLU (Rectified Linear Unit) is the most widely used activation
        # function in modern deep learning.
        #
        # Formula:  ReLU(z) = max(0, z)
        #
        # Rule: if the input is positive, pass it through unchanged.
        #       if the input is negative, replace it with 0.
        #
        #   z = -5  →  ReLU(-5) = 0    (blocked)
        #   z =  0  →  ReLU( 0) = 0    (threshold)
        #   z =  3  →  ReLU( 3) = 3    (passed through)
        #
        # Shape of the curve (hockey-stick):
        #
        #   ReLU(z)
        #       │            /
        #       │           /
        #       │          /
        #       │         /
        #       │________/
        #       └──────────────────→ z
        #           neg   0   pos
        #
        # WHY ReLU over Sigmoid for hidden layers?
        #   Sigmoid squashes values into (0,1), which causes the "vanishing
        #   gradient" problem — gradients become so tiny that early layers
        #   stop learning in deep networks.
        #   ReLU keeps positive gradients at full strength (slope = 1),
        #   so deep networks can still learn effectively.
        #
        # `np.maximum(0, z)` compares 0 and each element of z element-wise,
        # returning the larger of the two — applied to the full array at once.

        return np.maximum(0, z)