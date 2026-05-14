import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(
        self,
        x: NDArray[np.float64],
        gamma: NDArray[np.float64],
        beta: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # --- WHAT IS LAYER NORMALIZATION? ---
        #
        # During training, the values flowing through a neural network
        # can grow very large or shrink very small — making learning
        # unstable, slow, or impossible.
        #
        # Layer Normalization solves this by RESCALING the activations
        # of each sample to have:
        #   → mean  = 0   (centered at zero)
        #   → std   = 1   (unit variance)
        #
        # Then it gives the network learnable parameters (gamma, beta)
        # to UNDO the normalization if the task requires it.
        #
        # This is used in every modern Transformer (GPT, BERT, Claude).
        # It is applied BEFORE or AFTER attention and feed-forward layers.
        #
        # Full formula:
        #
        #              x - mean(x)
        #   x_norm  =  ───────────────────
        #              √( var(x) + eps )
        #
        #   output  =  gamma × x_norm + beta
        #
        # Where:
        #   gamma → learnable SCALE     (initialized to 1 — no change at start)
        #   beta  → learnable SHIFT     (initialized to 0 — no change at start)
        #   eps   → tiny constant to prevent division by zero when var ≈ 0

        # --- STEP 1: Small constant to avoid division by zero ---
        #
        # If all values in x are identical, var(x) = 0 exactly.
        # Dividing by √0 = 0 would cause NaN (Not a Number) — fatal for training.
        # Adding eps = 1e-5 (0.00001) keeps the denominator safely above zero
        # with negligible effect on the result when var is large.
        epsilon = 1e-5

        # --- STEP 2: Compute the mean of all input values ---
        #
        # np.mean(x) averages every element in the array.
        # After subtracting this, the data will be centered at zero.
        #
        # Example: x = [2, 4, 6]  →  mean = 4.0
        mean = np.mean(x)

        # --- STEP 3: Compute the variance ---
        #
        # Variance measures how spread out the values are around the mean.
        # np.var(x) = mean of (x - mean)²
        #
        # Example: x = [2, 4, 6],  mean = 4
        #   deviations²  = [(2-4)², (4-4)², (6-4)²] = [4, 0, 4]
        #   variance     = (4 + 0 + 4) / 3 = 2.667
        variance = np.var(x)

        # --- STEP 4: Normalize — shift to zero mean, scale to unit variance ---
        #
        # Each element is:
        #   1. Shifted by subtracting the mean     → centers the data at 0
        #   2. Scaled by dividing by the std dev   → stretches/squishes to std=1
        #
        # np.sqrt(variance + epsilon) = the standard deviation (with safety eps)
        #
        # Example: x = [2, 4, 6],  mean=4,  var=2.667,  std≈1.633
        #   x_normalized = [(2-4)/1.633, (4-4)/1.633, (6-4)/1.633]
        #                = [-1.225,       0.0,          1.225]
        #   mean of result ≈ 0,  std of result ≈ 1  ✓
        x_normalized = (x - mean) / np.sqrt(variance + epsilon)

        # --- STEP 5: Apply learnable scale (gamma) and shift (beta) ---
        #
        # Pure normalization forces every layer to the same scale — too rigid.
        # gamma and beta let the NETWORK decide the best scale and offset
        # for each layer, learned through backpropagation.
        #
        # At initialization:  gamma=1, beta=0  → output = x_normalized (no change)
        # After training:     gamma and beta adjust to whatever the task needs
        #
        # This is an element-wise operation:
        #   output[i] = gamma[i] × x_normalized[i] + beta[i]
        output = gamma * x_normalized + beta

        return np.round(output, 5)