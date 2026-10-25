# Optimization Techniques for Data Science: Algorithms 01 – 08 & 17 – 23

This repository implements, optimizes, and visualizes classical and modern optimization algorithms with interactive **2D contour trajectory representations** and mathematical diagnostics.

---

## 1. Problem Formulations

### Algorithms 01 – 07: Anisotropic Quadratic Function (First-Order Benchmarks)
To visualize convergence behaviors and search trajectories on 2D contour maps, the benchmark problem is formulated as an ill-conditioned quadratic bowl:
$$f(x_1, x_2) = \frac{1}{2} x_1^2 + \frac{\gamma}{2} x_2^2 \quad (\gamma = 10)$$
* **Gradient:** $\nabla f(x) = [x_1, 10 x_2]^T$
* **Hessian:** $Q = \text{diag}(1, 10)$
* **Global Minimum:** $x^* = (0, 0)$ with $f(x^*) = 0$.
* **Benchmark Characteristics:** Condition number $\kappa = 10$ introduces an elongated valley exposing zig-zagging in fixed-step GD, orthogonality in exact line search ($d_k \perp d_{k+1}$), adaptive step reduction in Armijo backtracking, momentum smoothing, and constant step lengths in normalized GD.

---

### Algorithm 08: Linear Regression via Gradient Descent
* **Model:** $\hat{y} = w x + b$
* **Loss Function (MSE):**
  $$J(w, b) = \frac{1}{2m} \sum_{i=1}^m (w x_i + b - y_i)^2$$
* **Parameter Space:** 2D parameter space $(w, b)$, where $J(w, b)$ forms an exact convex quadratic bowl.
* **Verification:** Gradient descent parameters $(w^*, b^*)$ are verified against the closed-form Normal Equation: $\theta^* = (X_b^T X_b)^{-1} X_b^T y$.

---

### Algorithms 17, 18, & 20: 2D Rosenbrock Function (Newton, Damped Newton, and BFGS)
To evaluate second-order curvature and quasi-Newton approximations on non-linear geometries:
$$f(x_1, x_2) = (1 - x_1)^2 + 10 (x_2 - x_1^2)^2$$
* **Global Minimum:** $x^* = (1, 1)$ with $f(x^*) = 0$.
* **Analytical Gradient:**
  $$\nabla f(x) = \begin{bmatrix} -2(1 - x_1) - 40 x_1(x_2 - x_1^2) \\ 20(x_2 - x_1^2) \end{bmatrix}$$
* **Analytical Hessian:**
  $$\nabla^2 f(x) = \begin{bmatrix} 2 - 40 x_2 + 120 x_1^2 & -40 x_1 \\ -40 x_1 & 20 \end{bmatrix}$$
* **Benchmark Characteristics:** Parabolic curved valley testing pure Newton quadratic convergence, Damped Newton step-size stability via Armijo backtracking, and BFGS curvature building without explicit second derivatives.

---

### Algorithm 19: Quadratic Newton Invariance (One-Step Exact Jump)
* **Model:** Strictly convex quadratic function $f(x) = \frac{1}{2} x^T Q x - b^T x + c$.
* **Core Theorem:** Because the second-order Taylor expansion of a quadratic function is exact everywhere, Newton's update lands on the global minimizer in **precisely one step**:
  $$\Delta x = - [\nabla^2 f(x_0)]^{-1} \nabla f(x_0) = - Q^{-1} (Q x_0 - b) = - x_0 + x^* \implies x_1 = x^*$$

---

### Algorithms 21 & 22: Constrained Optimization (Lagrange Multipliers & KKT)
* **Equality Constraints (Alg 21):**
  $$\min_{x} \frac{1}{2} x_1^2 + 5 x_2^2 \quad \text{s.t.} \quad x_1 + 2 x_2 - 3 = 0$$
  Solved via Newton-KKT system; confirms collinearity of objective and constraint gradients: $\nabla f(x^*) = -\lambda^* \nabla g(x^*)$.
* **Inequality Constraints & KKT Conditions (Alg 22):**
  $$\min_{x} \frac{1}{2} x_1^2 + 5 x_2^2 \quad \text{s.t.} \quad 1.5 - x_1 \le 0, \quad x_2 - 0.5 \le 0, \quad -x_2 - 1.0 \le 0$$
  Solved via active-set search; validates stationarity, primal feasibility, dual feasibility ($\mu \ge 0$), and complementary slackness ($\mu_i g_i(x) = 0$).

---

### Algorithm 23: Linear Independence Constraint Qualification (LICQ)
* Evaluates active constraint gradients $\{\nabla c_i(x^*)\}_{i \in \mathcal{A}(x^*)}$ and assembles the active Jacobian matrix $J_{\mathcal{A}}$.
* Verifies full row rank condition: $\text{rank}(J_{\mathcal{A}}) = |\mathcal{A}(x^*)|$.
* Compares linearly independent constraints against degenerate/tangent constraint systems where LICQ fails.

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
| `17_newton_method_function_optimization.py` | Newton's Method | $\Delta x = - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$ | Quadratic convergence on non-linear Rosenbrock bowl in 6 iterations. |
| `18_damped_newton_method.py` | Damped Newton's Method | $x_{k+1} = x_k + \alpha_k d_k$ (Armijo line search + spectral regularization) | Guarantees global descent even when initial point is far from optimum. |
| `19_newton_one_step_quadratic.py` | One-Step Newton Quadratic | $x_1 = x_0 - Q^{-1} \nabla f(x_0) = x^*$ | Reaches analytical global minimizer in exactly 1 iteration. |
| `20_bfgs.py` | BFGS Quasi-Newton | $H_{k+1} = (I - \rho s y^T) H_k (I - \rho y s^T) + \rho s s^T$ | Rapid superlinear convergence without calculating Hessian matrices. |
| `21_lagrange_multiplier_solver.py` | Lagrange Multiplier Solver | $\nabla_x \mathcal{L} = 0, \quad g(x) = 0$ | Solves equality-constrained KKT system; verifies gradient collinearity. |
| `22_kkt_conditions_solver.py` | KKT Conditions Solver | Stationarity, Primal/Dual Feasibility, Complementary Slackness | Active-set solver identifying optimal candidate point and active constraints. |
| `23_licq_constraint_qualification_checker.py` | LICQ Checker | $\text{rank}(J_{\mathcal{A}}) = |\mathcal{A}(x^*)|$ | SVD rank diagnostics contrasting regular vs degenerate constraint systems. |

---

## 3. How to Run

Each script is self-contained and saves its generated visualization to `plots/`:

```bash
# First-order unconstrained algorithms & linear regression
python 01_gradient_descent_function_optimization.py
python 02_gradient_descent_2d_exact_line_search.py
python 03_backtracking_line_search.py
python 04_backtracking_armijo_rule.py
python 05_gradient_descent_armijo_backtracking.py
python 06_momentum_gradient_descent.py
python 07_normalized_gradient_descent.py
python 08_gradient_descent_linear_regression.py

# First-order comparison benchmark
python compare_algorithms.py

# Second-order, quasi-Newton, and constrained optimization
python 17_newton_method_function_optimization.py
python 18_damped_newton_method.py
python 19_newton_one_step_quadratic.py
python 20_bfgs.py
python 21_lagrange_multiplier_solver.py
python 22_kkt_conditions_solver.py
python 23_licq_constraint_qualification_checker.py
```

---

## 4. Output Plots Overview

All plots are rendered in high resolution (300 DPI) inside `plots/`:
* `plots/01_gradient_descent.png`: Trajectory of standard gradient descent on the anisotropic bowl.
* `plots/02_exact_line_search.png`: Trajectory showing exact 90-degree orthogonal turns.
* `plots/03_backtracking_line_search.png`: 2D contour trajectory + 1D slice of evaluated step sizes.
* `plots/04_backtracking_armijo_rule.png`: 1D Armijo condition upper bound vs candidate points + 2D ray step.
* `plots/05_gradient_descent_armijo.png`: GD trajectory with dynamically backtracked Armijo step size history.
* `plots/06_momentum_gradient_descent.png`: Momentum path compared against standard gradient descent.
* `plots/07_normalized_gradient_descent.png`: Trajectory with normalized step lengths.
* `plots/08_linear_regression_gd.png`: 2D parameter loss contours $J(w, b)$ + fitted linear regression model.
* `plots/comparison_master.png`: Side-by-side overlay of trajectories and log-loss convergence curves.
* `plots/17_newton_method.png`: Newton trajectory showcasing quadratic rate on the curved Rosenbrock valley.
* `plots/18_damped_newton_method.png`: Damped Newton trajectory and per-step damping factor $\alpha_k$ progression.
* `plots/19_newton_one_step_quadratic.png`: Single-step vector jump directly to the quadratic minimizer.
* `plots/20_bfgs.png`: Superlinear BFGS quasi-Newton path traversing the Rosenbrock valley.
* `plots/21_lagrange_multipliers.png`: Equality constraint line with aligned collinear gradient arrows at $x^*$.
* `plots/22_kkt_conditions.png`: Feasible polyhedron, active/inactive constraints, and KKT vector balance.
* `plots/23_licq_checker.png`: Side-by-side comparison of linearly independent vs degenerate (tangent) constraints.
