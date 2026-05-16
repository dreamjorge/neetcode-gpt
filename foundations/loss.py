import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:
        # --- WHAT IS BINARY CROSS-ENTROPY (BCE)? ---
        #
        # BCE is the loss function for BINARY CLASSIFICATION:
        # problems where each sample belongs to one of exactly TWO classes.
        #
        # Examples:
        #   → Is this email spam?      (yes=1 / no=0)
        #   → Is this tumor malignant? (yes=1 / no=0)
        #   → Will this user click?    (yes=1 / no=0)
        #
        # The model outputs a PROBABILITY p ∈ (0, 1) via Sigmoid.
        # BCE measures how wrong that probability is.
        #
        # Formula (per sample):
        #   BCE = -[ y × log(p) + (1-y) × log(1-p) ]
        #
        # Where:
        #   y = true label      (0 or 1)
        #   p = predicted prob  (value between 0 and 1)
        #
        # The formula has two terms — only ONE activates per sample:
        #
        #   When y=1 (true class IS positive):
        #     BCE = -log(p)        ← punishes low confidence in the correct class
        #     p=0.99 → loss≈0.01  (nearly right, low penalty)
        #     p=0.01 → loss≈4.60  (very wrong, high penalty)
        #
        #   When y=0 (true class IS negative):
        #     BCE = -log(1-p)      ← punishes high confidence in the wrong class
        #     p=0.01 → loss≈0.01  (nearly right, low penalty)
        #     p=0.99 → loss≈4.60  (very wrong, high penalty)
        #
        # Final loss = average BCE across all samples in the batch.

        # --- STEP 1: Clip predictions to avoid log(0) = -infinity ---
        #
        # log(0) is undefined → would produce -inf → NaN in training.
        # We clamp predictions to a tiny range away from 0 and 1:
        #   epsilon = 1e-7 = 0.0000001
        #   y_pred stays in [0.0000001,  0.9999999]
        #
        # This has negligible effect on real predictions but prevents
        # numerical catastrophe on edge cases (model is 100% confident).
        epsilon = 1e-7
        clipped_predictions = np.clip(y_pred, epsilon, 1 - epsilon)

        # --- STEP 2: Compute BCE for every sample in the batch ---
        #
        # y_true * np.log(clipped_predictions)
        #   → active when y=1: measures penalty for predicting low probability
        #
        # (1 - y_true) * np.log(1 - clipped_predictions)
        #   → active when y=0: measures penalty for predicting high probability
        #
        # Both terms together cover both cases with a single vectorized formula.
        # np.mean averages across all samples → one scalar loss value.
        per_sample_loss = y_true * np.log(clipped_predictions) + \
                          (1 - y_true) * np.log(1 - clipped_predictions)

        loss = -np.mean(per_sample_loss)

        return round(loss, 4)

    def categorical_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:
        # --- WHAT IS CATEGORICAL CROSS-ENTROPY (CCE)? ---
        #
        # CCE is the loss function for MULTI-CLASS CLASSIFICATION:
        # problems where each sample belongs to ONE of N possible classes.
        #
        # Examples:
        #   → Which digit is this? (0–9, ten classes)
        #   → What language is this text? (hundreds of classes)
        #   → Which token comes next? (vocabulary of 50,000+ tokens in LLMs)
        #
        # y_true uses ONE-HOT ENCODING:
        #   "class 2 out of 4" → [0, 0, 1, 0]
        #   Only the correct class has value 1; all others are 0.
        #
        # y_pred is a SOFTMAX output:
        #   [0.05, 0.10, 0.80, 0.05]  ← probabilities summing to 1
        #
        # Formula (per sample):
        #   CCE = -Σ y_true_i × log(y_pred_i)
        #
        # Because y_true is one-hot, only ONE term survives the sum:
        #   CCE = -log(y_pred[correct_class])
        #
        # This means CCE simply asks:
        #   "How high was the predicted probability for the correct class?"
        #   High probability → low loss.   Low probability → high loss.
        #
        # Example:
        #   y_true = [0, 0, 1, 0]       ← correct class is index 2
        #   y_pred = [0.05, 0.10, 0.80, 0.05]
        #
        #   CCE = -(0×log(0.05) + 0×log(0.10) + 1×log(0.80) + 0×log(0.05))
        #       = -log(0.80)
        #       ≈ 0.223                  ← low loss, model was mostly right ✓
        #
        #   If y_pred = [0.05, 0.10, 0.10, 0.75]  (wrong class predicted):
        #   CCE = -log(0.10) ≈ 2.303    ← high loss, model was wrong ✗

        # --- STEP 1: Clip predictions to avoid log(0) = -infinity ---
        # Same reason as BCE: prevent undefined log at the boundaries.
        epsilon = 1e-7
        clipped_predictions = np.clip(y_pred, epsilon, 1 - epsilon)

        # --- STEP 2: Compute the cross-entropy per sample ---
        #
        # y_true * np.log(clipped_predictions)
        #   → element-wise product: zeroes out all classes except the true one
        #
        # np.sum(..., axis=1)
        #   → sums across classes (axis=1) for each sample
        #   → since y_true is one-hot, this gives -log(p_correct) per sample
        #
        # Shape walkthrough:
        #   y_true:           (batch_size, num_classes)
        #   log(y_pred):      (batch_size, num_classes)
        #   element-wise ×:   (batch_size, num_classes)
        #   sum over axis=1:  (batch_size,)             ← one value per sample
        #   mean:             scalar                     ← final loss
        per_sample_loss = np.sum(y_true * np.log(clipped_predictions), axis=1)

        loss = -np.mean(per_sample_loss)

        return round(loss, 4)