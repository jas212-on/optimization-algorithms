# Optimization Techniques for Data Science: Algorithms 01 - 08

This project implements, optimizes, and visualizes the first **8 optimization algorithms** with **2D contour trajectory representations**.

---

## 1. Problem Formulation

### Algorithms 01 – 07: Anisotropic Quadratic Function (General 2D Function Optimization)
To effectively visualize trajectories and compare convergence behaviors on 2D contour maps, we formulate the test problem as an ill-conditioned quadratic bowl:
$$f(x_1, x_2) = \frac{1}{2} x_1^2 + \frac{\gamma}{2} x_2^2 \quad (\gamma = 10)$$
* **Gradient:** $\nabla f(x) = [x_1, 10 x_2]^T$
* **Hessian:** $Q = \text{diag}(1, 10)$
* **Global Minimum:** $x^* = (0, 0)$ with $f(x^*) = 0$.
* **Why this benchmark:** The condition number $\kappa = 10$ introduces an elongated valley. This cleanly exposes:
  - Zig-zagging in fixed-step gradient descent.
  - Orthogonality of consecutive search directions in exact line search ($d_k \perp d_{k+1}$).
  - Adaptive step reduction in Armijo backtracking.
  - Oscillation dampening and valley acceleration in momentum gradient descent.
  - Constant step length behavior in normalized gradient descent.

### Algorithm 08: Linear Regression Optimization via Gradient Descent
* **Model:** $\hat{y} = w x + b$
* **Loss Function (MSE):**
  $$J(w, b) = \frac{1}{2m} \sum_{i=1}^m (w x_i + b - y_i)^2$$
* **Parameter Space:** 2D space $(w, b)$, where $J(w, b)$ forms an exact 2D convex bowl.
* **Gradients:**
  $$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^m (w x_i + b - y_i) x_i, \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (w x_i + b - y_i)$$
* **Verification:** Optimal parameters $(w^*, b^*)$ are verified against the closed-form Normal Equation: $\theta^* = (X_b^T X_b)^{-1} X_b^T y$.

---

## 2. Implemented Algorithms Summary

| Script | Algorithm | Mathematical Update / Condition | Key Result |
|---|---|---|---|
| `01_gradient_descent_function_optimization.py` | Fixed-step Gradient Descent | $x_{k+1} = x_k - \alpha \nabla f(x_k)$ | Converges in 40 iterations; exhibits slow progress along the flat axis. |
| `02_gradient_descent_2d_exact_line_search.py` | GD with Exact Line Search | $\alpha_k = \frac{\|g_k\|^2}{g_k^T Q g_k}$ | Converges in 19 iterations; successive steps are strictly orthogonal ($\langle d_k, d_{k+1} \rangle \approx 0$). |
| `03_backtracking_line_search.py` | Backtracking Line Search | Shrink $\alpha \leftarrow \rho \alpha$ until $f(x_k + \alpha d_k) < f(x_k)$ | Successfully identifies valid step sizes, prevents overshooting. |
| `04_backtracking_armijo_rule.py` | Armijo Rule Backtracking | $f(x + \alpha d) \le f(x) + c \alpha \nabla f(x)^T d$ | Enforces sufficient decrease; visualizes the 1D envelope vs function slice. |
| `05_gradient_descent_armijo_backtracking.py` | GD + Armijo Backtracking | Combines GD with per-step Armijo line search | Dynamically adjusts step size per iteration; fast and stable convergence. |
| `06_momentum_gradient_descent.py` | Polyak Momentum (Heavy Ball) | $v_{k+1} = \beta v_k + \alpha \nabla f(x_k), \quad x_{k+1} = x_k - v_{k+1}$ | Smooths ravines, accelerates along valley, suppresses high-frequency bounce. |
| `07_normalized_gradient_descent.py` | Normalized Gradient Descent | $x_{k+1} = x_k - \alpha_k \frac{\nabla f(x_k)}{\|\nabla f(x_k)\|}$ | Steps depend purely on direction rather than gradient magnitude. |
| `08_gradient_descent_linear_regression.py` | GD for Linear Regression | $\theta_{k+1} = \theta_k - \alpha \nabla J(\theta_k)$ | Converges to analytical Normal Equation solution $(w^*, b^*)$. |

---

## 3. How to Run

Each script is completely self-contained and saves its generated contour plot to the `plots/` folder:

```bash
# Run individual algorithms
python 01_gradient_descent_function_optimization.py
python 02_gradient_descent_2d_exact_line_search.py
python 03_backtracking_line_search.py
python 04_backtracking_armijo_rule.py
python 05_gradient_descent_armijo_backtracking.py
python 06_momentum_gradient_descent.py
python 07_normalized_gradient_descent.py
python 08_gradient_descent_linear_regression.py

# Run master comparison script
python compare_algorithms.py
```

---

## 4. Output Plots Overview

All plots are saved in high resolution (300 DPI) inside `plots/`:
* `plots/01_gradient_descent.png`: Trajectory of standard gradient descent on the anisotropic bowl.
* `plots/02_exact_line_search.png`: Trajectory showing exact 90-degree orthogonal turns.
* `plots/03_backtracking_line_search.png`: 2D contour trajectory + 1D slice of evaluated step sizes.
* `plots/04_backtracking_armijo_rule.png`: 1D Armijo condition upper bound vs candidate points + 2D ray step.
* `plots/05_gradient_descent_armijo.png`: GD trajectory with dynamically backtracked Armijo step size history.
* `plots/06_momentum_gradient_descent.png`: Momentum path compared against standard gradient descent.
* `plots/07_normalized_gradient_descent.png`: Trajectory with normalized step lengths.
* `plots/08_linear_regression_gd.png`: 2D parameter loss contours $J(w, b)$ + fitted linear regression model.
* `plots/comparison_master.png`: Side-by-side overlay of trajectories and log-loss convergence curves.
