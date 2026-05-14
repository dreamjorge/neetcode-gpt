import torch
import torch.nn.functional as F
from torchtyping import TensorType


class Solution:

    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # --- WHAT IS RESHAPING? ---
        #
        # A tensor is a multi-dimensional array of numbers.
        # Reshaping changes HOW those numbers are organized (rows × cols)
        # WITHOUT changing the numbers themselves or their order.
        #
        # Think of it like a grid of seats in a theater:
        #   4 rows × 6 seats  =  24 people total
        #   Reshape to 12 × 2 =  still 24 people, different arrangement
        #
        # Rule: total elements must stay the same.
        #   M × N  must equal  (M*N//2) × 2
        #
        # Example:
        #   Input  shape: (4, 6)  → 24 elements
        #   Output shape: (12, 2) → 24 elements ✓
        #
        #   [[1, 2, 3, 4, 5, 6],      [[1, 2],
        #    [7, 8, 9,10,11,12],  →    [3, 4],
        #    ...]                       [5, 6],
        #                              [7, 8], ...]
        #
        # In deep learning, reshaping is used constantly to prepare
        # tensors for specific layer expectations (e.g. flatten before
        # a Linear layer, group tokens before attention).

        num_rows, num_cols = to_reshape.shape

        # New shape: half as many rows, each row has exactly 2 elements.
        # `num_rows * num_cols // 2` keeps the total element count identical.
        reshaped = torch.reshape(to_reshape, (num_rows * num_cols // 2, 2))

        return torch.round(reshaped, decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # --- AVERAGING ALONG A DIMENSION ---
        #
        # `torch.mean(tensor, dim=0)` collapses the tensor ALONG dimension 0
        # (the rows), computing the mean of each COLUMN across all rows.
        #
        # dim=0 → collapse rows    → result has shape (num_cols,)
        # dim=1 → collapse columns → result has shape (num_rows,)
        #
        # Visual example with shape (3, 4):
        #
        #   Input:           dim=0 means "average DOWN each column"
        #   [[1, 2, 3, 4],
        #    [5, 6, 7, 8],   →   mean of each column
        #    [9,10,11,12]]
        #
        #   col 0: (1+5+9)/3  = 5.0
        #   col 1: (2+6+10)/3 = 6.0
        #   col 2: (3+7+11)/3 = 7.0
        #   col 3: (4+8+12)/3 = 8.0
        #
        #   Output: [5.0, 6.0, 7.0, 8.0]  → shape (4,)
        #
        # In neural networks, averaging across the batch dimension (dim=0)
        # is used to compute mean embeddings or pooled representations.

        column_means = torch.mean(to_avg, dim=0)

        return torch.round(column_means, decimals=4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # --- CONCATENATION ALONG A DIMENSION ---
        #
        # `torch.cat((a, b), dim=1)` joins tensors SIDE BY SIDE (horizontally),
        # stacking their columns while keeping rows aligned.
        #
        # dim=0 → stack VERTICALLY   (add more rows, cols must match)
        # dim=1 → stack HORIZONTALLY (add more cols, rows must match)
        #
        # Visual example:
        #
        #   cat_one (3×2):    cat_two (3×3):    result (3×5):
        #   [[1, 2],          [[5, 6, 7],       [[1, 2, 5, 6, 7],
        #    [3, 4],    +      [8, 9,10],   →    [3, 4, 8, 9,10],
        #    [5, 6]]           [11,12,13]]        [5, 6,11,12,13]]
        #
        # Common use in deep learning: concatenating different feature vectors
        # for a single sample (e.g. image features + text features side by side).

        horizontally_joined = torch.cat((cat_one, cat_two), dim=1)

        return torch.round(horizontally_joined, decimals=4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # --- MEAN SQUARED ERROR (MSE) LOSS ---
        #
        # Loss measures how WRONG our model's predictions are.
        # MSE is the most common loss function for regression problems.
        #
        # Formula:  MSE = (1/n) × Σ (prediction_i - target_i)²
        #
        # Step by step:
        #   1. For each element: compute the error (prediction - target)
        #   2. Square each error (removes negatives, penalizes large errors more)
        #   3. Average all squared errors → one single scalar number
        #
        # Example:
        #   prediction = [3.0, 1.0, 4.0]
        #   target     = [2.0, 2.0, 3.0]
        #
        #   errors         = [1.0, -1.0,  1.0]
        #   squared errors = [1.0,  1.0,  1.0]
        #   MSE            = (1.0 + 1.0 + 1.0) / 3 = 1.0
        #
        # WHY square the errors?
        #   → Large errors are penalized disproportionately (2x error → 4x penalty)
        #   → Negative and positive errors don't cancel each other out
        #   → The result is always ≥ 0  (perfect prediction → loss = 0)
        #
        # During training, the optimizer tries to MINIMIZE this number
        # by adjusting the model's weights via backpropagation.

        mse_loss = F.mse_loss(prediction, target)

        return torch.round(mse_loss, decimals=4)