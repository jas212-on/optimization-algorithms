# Optimization Techniques for Data Science: Algorithms 01 – 23

This repository implements, optimizes, and visualizes classical and modern optimization algorithms with interactive **2D contour trajectory representations** and mathematical diagnostics.

---

## 1. Problem Formulations

### Algorithms 01 – 07 & 12 – 14: Anisotropic Quadratic Function (First-Order & Adaptive Benchmarks)
To visualize convergence behaviors and search trajectories on 2D contour maps, the benchmark problem is formulated as an ill-conditioned quadratic bowl:
$$f(x_1, x_2) = \frac{1}{2} x_1^2 + \frac{\gamma}{2} x_2^2 \quad (\gamma = 10)$$
* **Gradient:** $\nabla f(x) = [x_1, 10 x_2]^T$
* **Hessian:** $Q = \text{diag}(1, 10)$
* **Global Minimum:** $x^* = (0, 0)$ with $f(x^*) = 0$.
* **Benchmark Characteristics:** Condition number $\kappa = 10$ introduces an elongated valley exposing zig-zagging in fixed-step GD, orthogonality in exact line search ($d_k \perp d_{k+1}$), adaptive step reduction in Armijo backtracking, momentum smoothing, per-coordinate step adaptation in **Adagrad**, leaky exponential second-moment averaging in **RMSProp**, and combined momentum with bias-corrected second moments in **Adam**.

---

### Algorithm 08: Linear Regression via Gradient Descent
* **Model:** $\hat{y} = w x + b$
* **Loss Function (MSE):**
  $$J(w, b) = \frac{1}{2m} \sum_{i=1}^m (w x_i + b - y_i)^2$$
* **Parameter Space:** 2D parameter space $(w, b)$, where $J(w, b)$ forms an exact convex quadratic bowl.
* **Verification:** Gradient descent parameters $(w^*, b^*)$ are verified against the closed-form Normal Equation: $\theta^* = (X_b^T X_b)^{-1} X_b^T y$.

---

### Algorithm 09: Linear Regression via Stochastic Gradient Descent (SGD)
* **Model:** $\hat{y} = w x + b$
* **Per-Sample Gradient:** Instantaneous gradient on randomly drawn sample $(x_i, y_i)$:
  $$\nabla J_i(\theta) = (x_{b,i} \theta - y_i) x_{b,i}^T$$
* **Update Rule & Learning Rate Decay:**
  $$\theta_{t+1} = \theta_t - \alpha_t \nabla J_i(\theta_t), \quad \alpha_t = \frac{\alpha_0}{1 + \lambda t}$$
* **Benchmark Characteristics:** Demonstrates high-variance stochastic fluctuations and random walk exploratory behavior on the 2D contour map, contrasting sharply with the smooth trajectory of full-batch GD while converging to the closed-form Normal Equation solution $\theta^*$.

---

### Algorithm 10: Linear Regression via Mini-Batch Stochastic Gradient Descent
* **Model:** $\hat{y} = w x + b$
* **Mini-Batch Gradient:** Gradient averaged over a subset $\mathcal{B} \subset \{1, \dots, m\}$ with batch size $B = |\mathcal{B}|$:
  $$g_{\mathcal{B}}(\theta) = \frac{1}{B} X_{\mathcal{B}}^T (X_{\mathcal{B}} \theta - y_{\mathcal{B}})$$
* **Update Rule & Learning Rate Decay:**
  $$\theta_{t+1} = \theta_t - \alpha_t g_{\mathcal{B}}(\theta_t), \quad \alpha_t = \frac{\alpha_0}{1 + \lambda t}$$
* **Benchmark Characteristics:** Strikes a balance between pure SGD ($B=1$) and full-batch GD ($B=m$), providing lower gradient variance, accelerated step throughput, and stable smooth convergence to $\theta^*$.

---

### Algorithm 11: Binary Logistic Regression Optimization
* **Model:** $P(y = 1 | x) = \sigma(w x + b) = \frac{1}{1 + e^{-(w x + b)}}$
* **Loss Function (Binary Cross-Entropy / Log-Loss):**
  $$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
* **Analytical Gradient & Hessian:**
  $$\nabla J(\theta) = \frac{1}{m} X_b^T (\sigma(X_b \theta) - y), \quad \nabla^2 J(\theta) = \frac{1}{m} X_b^T S X_b \quad (S_{ii} = p_i(1 - p_i) > 0)$$
* **Decision Boundary:** $P(y=1|x) = 0.5 \implies x^* = -b / w$.
* **Benchmark Characteristics:** Asymmetric strictly convex loss bowl verifying gradient convergence from inverted initial conditions to maximum-likelihood estimates (MLE).

---

### Algorithms 12 – 14: Adaptive Learning Rate Optimizers
* **Adagrad (Alg 12):** $G_{k+1} = G_k + g_k^2, \quad x_{k+1} = x_k - \frac{\alpha}{\sqrt{G_{k+1}} + \epsilon} \odot g_k$. Scales inversely with historical gradient energy; stabilizes steep ravines.
* **RMSProp (Alg 13):** $v_{k+1} = \beta v_k + (1 - \beta) g_k^2, \quad x_{k+1} = x_k - \frac{\alpha}{\sqrt{v_{k+1}} + \epsilon} \odot g_k$. Resolves Adagrad's diminishing step size using an exponential moving average ($\beta = 0.9$).
* **Adam (Alg 14):** Combines first-moment momentum ($m_k$, $\beta_1 = 0.9$) and second-moment scaling ($v_k$, $\beta_2 = 0.999$) with initialization bias correction: $\hat{m}_k, \hat{v}_k$.

---

### Algorithm 15: Subgradient Method for Nonsmooth Function Optimization
* **Nonsmooth Objective:** $f(x_1, x_2) = |x_1| + 2|x_2|$ (rhombic/diamond level sets, non-differentiable at axes $x_1=0$ and $x_2=0$).
* **Subdifferential:** $\partial f(x_1, x_2) = \partial |x_1| \times 2\partial |x_2|$, where $\partial |u| = \{\text{sign}(u)\}$ if $u \ne 0$, else $[-1, 1]$.
* **Subgradient Update:**
  $$x_{k+1} = x_k - \alpha_k g(x_k), \quad g(x_k) \in \partial f(x_k)$$
* **Step Size Schedules & Non-Descent Dynamics:**
  - **Diminishing Step Size:** $\alpha_k = \frac{\alpha_0}{\sqrt{k+1}}$ satisfies $\sum \alpha_k = \infty, \sum \alpha_k^2 < \infty$; guarantees convergence to $x^* = (0, 0)$.
  - **Constant Step Size:** $\alpha_k = \alpha_0$ fails to converge to a point, chattering in a limit cycle of size $O(\alpha_0)$ across the crease.
  - **Best-So-Far Record:** Subgradient directions are not guaranteed descent directions ($f(x_{k+1})$ can exceed $f(x_k)$); the algorithm tracks the monotonic sequence $f_{\text{best}}^{(k)} = \min_{0 \le i \le k} f(x_i)$.

---

### Algorithm 16: Subgradient Method for Lasso Loss Optimization
* **Model & Objective Function:**
  $$J(w) = \frac{1}{2m} \|X w - y\|_2^2 + \lambda \|w\|_1$$
* **Subgradient Formulation:**
  $$g(w) = \frac{1}{m} X^T (X w - y) + \lambda \cdot s(w), \quad s(w)_j = \begin{cases} \text{sign}(w_j) & |w_j| > 0 \\ 0 & w_j = 0 \end{cases}$$
* **Sparsity Induction Mechanism:** The nonsmooth $L_1$ penalty causes rhombic corner distortions on the least-squares error ellipsoid, centering the optimum directly on the coordinate axis ($w_2^* = 0$). Diminishing step sizes drive non-informative features to zero, achieving exact feature selection in contrast with dense OLS estimates.

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
| `09_stochastic_gradient_descent_linear_regression.py` | Stochastic GD (SGD) Linear Regression | $\theta_{t+1} = \theta_t - \alpha_t \nabla J_i(\theta_t)$ | High-variance stochastic exploration converging to Normal Equation solution. |
| `10_minibatch_sgd_linear_regression.py` | Mini-Batch SGD Linear Regression | $\theta_{t+1} = \theta_t - \alpha_t g_{\mathcal{B}}(\theta_t)$ | Balances variance and throughput; smooth accelerated convergence with mini-batches. |
| `11_logistic_regression.py` | Binary Logistic Regression Optimization | $\theta_{k+1} = \theta_k - \alpha \nabla J(\theta_k)$ | Optimizes non-linear cross-entropy loss, aligns sigmoid boundary at $x^*$. |
| `12_adagrad.py` | Adagrad Optimizer | $x_{k+1} = x_k - \frac{\alpha}{\sqrt{G_{k+1}} + \epsilon} \odot g_k$ | Adapts learning rate per coordinate via cumulative squared gradients. |
| `13_rmsprop.py` | RMSProp Optimizer | $x_{k+1} = x_k - \frac{\alpha}{\sqrt{v_{k+1}} + \epsilon} \odot g_k$ | Exponentially decaying second-moment average prevents premature learning rate decay. |
| `14_adam.py` | Adam Optimizer | $x_{k+1} = x_k - \frac{\alpha_k}{\sqrt{\hat{v}_{k+1}} + \epsilon} \odot \hat{m}_{k+1}$ | Combines directional momentum with adaptive second-moment rescaling and bias correction. |
| `15_subgradient_method_function_optimization.py` | Subgradient Method | $x_{k+1} = x_k - \alpha_k g(x_k), \; g \in \partial f$ | Minimizes nonsmooth $f(x)=|x_1|+2|x_2|$; illustrates non-descent dynamics, diminishing step convergence, and constant step limit cycle chatter. |
| `16_subgradient_lasso_loss.py` | Subgradient Lasso Optimizer | $w_{k+1} = w_k - \alpha_k \left[\frac{1}{m} X^T(Xw_k - y) + \lambda s(w_k)\right]$ | Drives non-informative feature $w_2 \to 0$ to induce sparsity; benchmarked against OLS and Scikit-Learn Lasso. |
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
python 09_stochastic_gradient_descent_linear_regression.py
python 10_minibatch_sgd_linear_regression.py
python 11_logistic_regression.py

# Adaptive first-order optimizers
python 12_adagrad.py
python 13_rmsprop.py
python 14_adam.py

# Nonsmooth optimization & regularized loss
python 15_subgradient_method_function_optimization.py
python 16_subgradient_lasso_loss.py

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
* `plots/09_stochastic_gradient_descent.png`: 2D parameter loss contours $J(w, b)$ with noisy per-sample SGD updates, epoch path, Batch GD baseline, and dynamic model fit progression.
* `plots/10_minibatch_sgd_linear_regression.png`: 2D parameter loss contours $J(w, b)$ with mini-batch $(B=10)$ trajectory, Batch GD baseline, and fitted model progression.
* `plots/11_logistic_regression.png`: 2D cross-entropy loss contours $J(w, b)$ + fitted sigmoid curve with decision boundary threshold.
* `plots/12_adagrad.png`: 2D contour trajectory showing per-coordinate dampening of steep oscillations vs standard GD.
* `plots/13_rmsprop.png`: 2D contour trajectory showing steady exponential moving average gradient scaling.
* `plots/14_adam.png`: 2D contour trajectory combining directional momentum with second-moment normalization.
* `plots/15_subgradient_method.png`: 2D diamond contour trajectory comparing diminishing vs constant step size schedules and non-descent objective chatter.
* `plots/16_subgradient_lasso.png`: 2D Lasso parameter loss contours $J(w_1, w_2)$ showing rhombic corner kinks, OLS dense vs Lasso sparse minimizers, and parameter sparsity progression.
* `plots/comparison_master.png`: Side-by-side overlay of trajectories and log-loss convergence curves.
* `plots/17_newton_method.png`: Newton trajectory showcasing quadratic rate on the curved Rosenbrock valley.
* `plots/18_damped_newton_method.png`: Damped Newton trajectory and per-step damping factor $\alpha_k$ progression.
* `plots/19_newton_one_step_quadratic.png`: Single-step vector jump directly to the quadratic minimizer.
* `plots/20_bfgs.png`: Superlinear BFGS quasi-Newton path traversing the Rosenbrock valley.
* `plots/21_lagrange_multipliers.png`: Equality constraint line with aligned collinear gradient arrows at $x^*$.
* `plots/22_kkt_conditions.png`: Feasible polyhedron, active/inactive constraints, and KKT vector balance.
* `plots/23_licq_checker.png`: Side-by-side comparison of linearly independent vs degenerate (tangent) constraints.
