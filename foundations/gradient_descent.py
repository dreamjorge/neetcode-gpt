class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # --- STRATEGY: Gradient Descent ---
        #
        # We want to find the value of x that MINIMIZES the function f(x) = x²
        #
        # Imagine f(x) = x² as a bowl-shaped curve (parabola).
        # The minimum is at x = 0, where f(0) = 0.
        #
        #         f(x)
        #          |
        #       \  |  /
        #        \ | /
        #         \|/
        #     -----+-----→ x
        #          0  ← minimum
        #
        # The algorithm: start somewhere on the curve, then repeatedly
        # take small steps in the direction that goes DOWNHILL.
        #
        # How do we know which way is "downhill"?
        # The DERIVATIVE tells us the slope at the current point:
        #   - Positive slope → we are on the RIGHT side → step LEFT (subtract)
        #   - Negative slope → we are on the LEFT side  → step RIGHT (add)
        #
        # For f(x) = x²  →  derivative f'(x) = 2x
        #
        # Update rule:
        #   x_new = x_current - learning_rate * f'(x_current)
        #         = x_current - learning_rate * (2 * x_current)

        # `current_x` is our current guess for the minimizer.
        # We start wherever the caller tells us (init).
        current_x = init

        for _ in range(iterations):
            # --- STEP 1: Compute the slope at the current position ---
            # f'(x) = 2x  →  this is how steep the curve is at current_x.
            # A large slope means we are far from the minimum.
            # A slope of 0 means we ARE at the minimum (x = 0).
            slope = 2 * current_x

            # --- STEP 2: Take one step downhill ---
            # We move AGAINST the slope direction (that's why we subtract).
            # `learning_rate` controls how big each step is:
            #   - Too large  → we might overshoot and bounce past the minimum
            #   - Too small  → we converge correctly but very slowly
            #
            # Example with current_x=10, learning_rate=0.1:
            #   slope     = 2 * 10 = 20
            #   step_size = 0.1 * 20 = 2.0
            #   current_x = 10 - 2.0 = 8.0   (moved closer to 0)
            current_x = current_x - learning_rate * slope

        # Return the final estimate of the minimizer, rounded to 5 decimal places.
        return round(current_x, 5)