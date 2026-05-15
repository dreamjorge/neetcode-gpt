import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool
    ) -> Tuple[List[List[float]], List[float], List[float]]:
        # --- WHAT IS BATCH NORMALIZATION? ---
        #
        # Batch Norm is a stabilization technique for neural network training.
        # It normalizes activations across the BATCH dimension (axis=0),
        # meaning: for each feature, look at all samples in the batch
        # and normalize using THEIR mean and variance.
        #
        # This contrasts with Layer Norm, which normalizes across FEATURES
        # for each individual sample.
        #
        #   Batch Norm:  normalize each COLUMN across all rows (batch axis)
        #   Layer Norm:  normalize each ROW  across all columns (feature axis)
        #
        # Key difference from Layer Norm: Batch Norm has TWO modes:
        #
        #   TRAINING mode   → compute fresh stats from the current batch
        #                     update running stats for future inference
        #
        #   INFERENCE mode  → use the STORED running stats (no batch needed)
        #                     makes predictions consistent and batch-size independent
        #
        # Input x shape: (batch_size, num_features)
        # Example: 4 samples, 3 features each → shape (4, 3)

        # --- STEP 1: Convert all inputs to NumPy arrays ---
        # The function receives plain Python lists for broad compatibility,
        # but we need NumPy for vectorized math operations.
        x             = np.array(x)
        gamma         = np.array(gamma)           # learnable scale per feature
        beta          = np.array(beta)            # learnable shift per feature
        running_mean  = np.array(running_mean, dtype=np.float64)  # stored across batches
        running_var   = np.array(running_var,  dtype=np.float64)  # stored across batches

        # ══════════════════════════════════════════════════════════════
        # TRAINING MODE: compute stats from this batch, update running stats
        # ══════════════════════════════════════════════════════════════
        if training:
            # --- STEP 2a: Compute batch statistics (axis=0 = across samples) ---
            #
            # axis=0 means "look down the column" → one mean per feature.
            #
            # Example x (4 samples, 3 features):
            #   [[1, 2, 3],
            #    [4, 5, 6],
            #    [7, 8, 9],
            #    [10,11,12]]
            #
            #   batch_mean = [(1+4+7+10)/4, (2+5+8+11)/4, (3+6+9+12)/4]
            #              = [5.5, 6.5, 7.5]          ← one value per feature
            #
            #   batch_var  = variance of each column
            #              = [11.25, 11.25, 11.25]
            batch_mean = np.mean(x, axis=0)
            batch_var  = np.var(x,  axis=0)

            # --- STEP 3a: Normalize using BATCH statistics ---
            #
            # Each sample is shifted by the batch mean and scaled by batch std.
            # Adding eps prevents division by zero when a feature has zero variance.
            #
            # Result x_normalized: same shape as x, but each column now has
            # mean ≈ 0 and variance ≈ 1.
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + eps)

            # --- STEP 4a: Update running statistics with exponential moving average ---
            #
            # Running stats are a "memory" of what we've seen across many batches.
            # They are NOT used during training — only saved for inference.
            #
            # Exponential Moving Average (EMA) formula:
            #   new_running = (1 - momentum) × old_running + momentum × batch_stat
            #
            # momentum = 0.1 means: 10% new batch info, 90% old memory.
            # This smooths out noise from individual batches.
            #
            # Example with momentum=0.1, old running_mean=0.0, batch_mean=5.5:
            #   new running_mean = 0.9 × 0.0 + 0.1 × 5.5 = 0.55
            #
            # After many batches, running_mean converges to the true dataset mean.
            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            running_var  = (1 - momentum) * running_var  + momentum * batch_var

        # ══════════════════════════════════════════════════════════════
        # INFERENCE MODE: use stored running stats (no batch stats computed)
        # ══════════════════════════════════════════════════════════════
        else:
            # --- STEP 2b: Normalize using RUNNING statistics ---
            #
            # During inference we may process one sample at a time (batch_size=1).
            # We can't compute meaningful stats from a single sample, so we use
            # the running stats accumulated during training instead.
            #
            # This guarantees that the same input always produces the same output,
            # regardless of what other samples happen to be in the batch.
            x_normalized = (x - running_mean) / np.sqrt(running_var + eps)

        # --- STEP 5: Apply learnable scale (gamma) and shift (beta) ---
        #
        # Same as Layer Norm: pure normalization is too rigid.
        # gamma and beta let the network re-scale and re-center each feature
        # to whatever values best serve the task — learned via backpropagation.
        #
        # At initialization: gamma=1, beta=0 → output = x_normalized
        # After training:    gamma and beta tune each feature independently
        output = gamma * x_normalized + beta

        # --- STEP 6: Return all three results ---
        # output        → normalized + scaled activations to pass to the next layer
        # running_mean  → updated memory (saved in the model for inference)
        # running_var   → updated memory (saved in the model for inference)
        return (
            np.round(output,       4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var,  4).tolist()
        )