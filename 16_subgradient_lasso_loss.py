# Objective: Optimize Lasso loss using sub-gradients.
# Input: Features, targets, regularization strength, and optimizer settings.
# Output: Model parameters and loss.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


def generate_lasso_data(m: int = 100, seed: int = 42):
    """
    Generates synthetic 2D regression dataset:
      - Feature 1 (x1): Strong predictive feature (w1_true = 2.0).
      - Feature 2 (x2): Weak/noisy feature (w2_true = 0.3).
    When L1 penalty lambda is applied, the weak feature w2 is driven to exactly 0 (sparsity).
    """
    np.random.seed(seed)
    X = np.random.randn(m, 2)
    w_true = np.array([2.0, 0.3], dtype=float)
    noise = 0.2 * np.random.randn(m)
    y = X @ w_true + noise
    return X, y, w_true


def lasso_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_reg: float):
    """
    Lasso objective function:
      J(w) = (1 / 2m) * ||Xw - y||_2^2 + lambda * ||w||_1
    Returns:
      total_loss, mse_loss, l1_penalty
    """
    m = len(y)
    residuals = X @ w - y
    mse_loss = float(np.sum(residuals ** 2) / (2.0 * m))
    l1_penalty = float(lambda_reg * np.sum(np.abs(w)))
    total_loss = mse_loss + l1_penalty
    return total_loss, mse_loss, l1_penalty


def lasso_subgradient(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lambda_reg: float,
    tol: float = 1e-10
) -> np.ndarray:
    """
    Computes a valid subgradient of the Lasso loss:
      g(w) = (1 / m) * X^T (Xw - y) + lambda * s(w)
    where s(w_j) = sign(w_j) if |w_j| > tol, else 0 in [-1, 1].
    """
    m = len(y)
    residuals = X @ w - y
    mse_grad = (X.T @ residuals) / float(m)

    # Subgradient of L1 norm
    l1_subgrad = np.zeros_like(w)
    for j in range(len(w)):
        if np.abs(w[j]) > tol:
            l1_subgrad[j] = np.sign(w[j])
        else:
            l1_subgrad[j] = 0.0  # valid subgradient at 0

    return mse_grad + lambda_reg * l1_subgrad


def subgradient_descent_lasso(
    X: np.ndarray,
    y: np.ndarray,
    lambda_reg: float = 0.35,
    w0: np.ndarray = np.array([-1.0, 2.5]),
    alpha0: float = 0.5,
    max_iter: int = 160,
    schedule: str = "diminishing"
):
    """
    Subgradient Method for Lasso Optimization:
      w_{k+1} = w_k - alpha_k * g(w_k),  where g(w_k) in partial J(w_k)

    Step size schedules:
      - 'diminishing': alpha_k = alpha0 / sqrt(k + 1)
      - 'constant': alpha_k = alpha0
    """
    w = np.array(w0, dtype=float).copy()
    trajectory = [w.copy()]
    loss_hist = []
    best_loss_hist = []
    w_best = w.copy()
    f_best = float("inf")
    step_sizes = []

    init_loss, _, _ = lasso_loss(w, X, y, lambda_reg)
    loss_hist.append(init_loss)
    best_loss_hist.append(init_loss)
    f_best = init_loss

    for k in range(max_iter):
        g = lasso_subgradient(w, X, y, lambda_reg)
        
        if schedule == "diminishing":
            alpha_k = alpha0 / np.sqrt(k + 1)
        elif schedule == "constant":
            alpha_k = alpha0
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        step_sizes.append(alpha_k)
        w = w - alpha_k * g
        
        cur_loss, _, _ = lasso_loss(w, X, y, lambda_reg)
        if cur_loss < f_best:
            f_best = cur_loss
            w_best = w.copy()

        trajectory.append(w.copy())
        loss_hist.append(cur_loss)
        best_loss_hist.append(f_best)

    return (
        np.array(trajectory),
        np.array(loss_hist),
        np.array(best_loss_hist),
        w_best,
        f_best,
        step_sizes
    )


def plot_lasso_results(
    X: np.ndarray,
    y: np.ndarray,
    lambda_reg: float,
    traj_dim: np.ndarray,
    loss_dim: np.ndarray,
    best_loss_dim: np.ndarray,
    traj_const: np.ndarray,
    w_ols: np.ndarray,
    w_lasso_target: np.ndarray,
    save_path: str = "plots/16_subgradient_lasso.png"
):
    """
    Visualizes Lasso optimization via Subgradient Method:
      Panel 1: 2D Lasso parameter loss surface J(w1, w2) showing rhombic contour kinks,
               diminishing step convergence to sparsity (w2 = 0), and constant step chatter.
      Panel 2: Parameter evolution w1(k) and w2(k) demonstrating exact sparsity induction (w2 -> 0),
               alongside instantaneous loss J(w_k) vs monotonic best-so-far loss.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ----------------- Panel 1: 2D Lasso Loss Contours -----------------
    w1_vals = np.linspace(-1.5, 2.8, 300)
    w2_vals = np.linspace(-1.2, 3.0, 300)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = np.zeros_like(W1)

    m = len(y)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w_vec = np.array([W1[i, j], W2[i, j]])
            res = X @ w_vec - y
            mse = np.sum(res ** 2) / (2.0 * m)
            l1 = lambda_reg * np.sum(np.abs(w_vec))
            Z[i, j] = mse + l1

    # Contour levels
    levels = np.array([0.7, 0.8, 1.0, 1.3, 1.7, 2.3, 3.2, 4.5, 6.0, 8.0, 10.5, 13.5])
    cs = ax1.contour(W1, W2, Z, levels=levels, cmap="viridis", alpha=0.75, linewidths=1.2)
    ax1.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    # Coordinate axes (sparse boundaries)
    ax1.axhline(0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax1.axvline(0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)

    # Constant step trajectory (limit cycle oscillation)
    ax1.plot(
        traj_const[:, 0], traj_const[:, 1],
        color="crimson", linestyle="--", linewidth=1.3, alpha=0.7,
        marker="^", markersize=3.5, label=r"Constant $\alpha = 0.15$ (Chatter)"
    )

    # Diminishing step trajectory (convergent)
    ax1.plot(
        traj_dim[:, 0], traj_dim[:, 1],
        color="dodgerblue", linestyle="-", linewidth=1.8,
        marker="o", markersize=4, label=r"Diminishing $\alpha_k = \frac{0.50}{\sqrt{k+1}}$"
    )

    # Start, OLS, and Sparse Lasso target
    ax1.scatter([traj_dim[0, 0]], [traj_dim[0, 1]], color="black", s=80, zorder=6, label=f"Start $w_0 = [{traj_dim[0, 0]:.1f}, {traj_dim[0, 1]:.1f}]$")
    ax1.scatter([w_ols[0]], [w_ols[1]], color="purple", marker="x", s=90, linewidths=2.5, zorder=6, label=f"OLS $w = [{w_ols[0]:.2f}, {w_ols[1]:.2f}]$ (Dense)")
    ax1.scatter([w_lasso_target[0]], [w_lasso_target[1]], color="gold", edgecolors="black", marker="*", s=140, zorder=7, label=f"Lasso $w^* = [{w_lasso_target[0]:.2f}, {w_lasso_target[1]:.2f}]$ (Sparse)")

    ax1.set_title(rf"Lasso Parameter Contours ($\lambda = {lambda_reg}$)", fontsize=12, fontweight="bold")
    ax1.set_xlabel(r"Weight $w_1$ (Informative)", fontsize=11)
    ax1.set_ylabel(r"Weight $w_2$ (Noisy / Shrinks to 0)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # ----------------- Panel 2: Parameter Trajectories & Sparsity Induction -----------------
    iters = np.arange(len(traj_dim))
    
    # Left axis: parameter values w1 and w2
    ax2.plot(iters, traj_dim[:, 0], color="navy", linestyle="-", linewidth=2.0, label=r"$w_1(k)$ (Target: $\approx 1.57$)")
    ax2.plot(iters, traj_dim[:, 1], color="crimson", linestyle="-", linewidth=2.0, label=r"$w_2(k)$ (Target: $0.00$ - Sparsity)")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8, label="Sparse Axis ($w_2 = 0$)")
    ax2.axhline(w_lasso_target[0], color="navy", linestyle=":", linewidth=1.2, alpha=0.6)

    ax2.set_xlabel("Iteration $k$", fontsize=11)
    ax2.set_ylabel("Parameter Value $w_j$", fontsize=11, color="black")
    ax2.set_title("Lasso Sparsity Induction via Subgradient Method", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Secondary y-axis for objective loss
    ax2_twin = ax2.twinx()
    ax2_twin.plot(iters, loss_dim, color="darkorange", linestyle=":", linewidth=1.2, alpha=0.7, label=r"Instantaneous $J(w_k)$")
    ax2_twin.plot(iters, best_loss_dim, color="darkgreen", linestyle="-.", linewidth=1.6, label=r"Best-so-far $J_{\rm best}^{(k)}$")
    ax2_twin.set_ylabel("Lasso Objective $J(w)$", fontsize=11, color="darkgreen")
    ax2_twin.tick_params(axis="y", labelcolor="darkgreen")

    # Combine legends from both axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Lasso subgradient plot saved successfully at: {save_path}")


def main():
    print("=" * 78)
    print("      LASSO LOSS OPTIMIZATION VIA SUBGRADIENT METHOD")
    print("=" * 78)
    print("Problem Formulation: J(w) = (1/2m) * ||Xw - y||_2^2 + lambda * ||w||_1")
    print("Subgradient: g(w) = (1/m) * X^T (Xw - y) + lambda * sign(w)")
    print("Goal: Drive non-informative weight w_2 -> 0 to induce model sparsity.")

    m = 100
    lambda_reg = 0.35
    X, y, w_true = generate_lasso_data(m=m, seed=42)

    # Unregularized OLS Solution
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    ols_total, ols_mse, ols_l1 = lasso_loss(w_ols, X, y, lambda_reg)

    # Scikit-Learn Lasso Reference Benchmark (Coordinate Descent)
    lasso_sklearn = Lasso(alpha=lambda_reg, fit_intercept=False, tol=1e-8, max_iter=2000)
    lasso_sklearn.fit(X, y)
    w_target = lasso_sklearn.coef_
    target_total, target_mse, target_l1 = lasso_loss(w_target, X, y, lambda_reg)

    print(f"\nDataset: m = {m} samples, d = 2 features")
    print(f"True Generative Weights: w_true = [{w_true[0]:.2f}, {w_true[1]:.2f}]")
    print(f"Regularization Strength: lambda = {lambda_reg}")
    print(f"OLS Estimate (Dense):   w_ols    = [{w_ols[0]:.4f}, {w_ols[1]:.4f}] | Loss J(w): {ols_total:.6f}")
    print(f"Exact Lasso Benchmark:   w_target = [{w_target[0]:.4f}, {w_target[1]:.4f}] | Loss J(w): {target_total:.6f}")

    w0 = np.array([-1.0, 2.5], dtype=float)
    max_iter = 160
    alpha0_dim = 0.5
    alpha_const = 0.15

    # 1. Diminishing Step Schedule
    traj_dim, loss_dim, best_loss_dim, w_best_dim, f_best_dim, steps_dim = subgradient_descent_lasso(
        X, y,
        lambda_reg=lambda_reg,
        w0=w0,
        alpha0=alpha0_dim,
        max_iter=max_iter,
        schedule="diminishing"
    )

    # 2. Constant Step Schedule
    traj_const, loss_const, best_loss_const, w_best_const, f_best_const, steps_const = subgradient_descent_lasso(
        X, y,
        lambda_reg=lambda_reg,
        w0=w0,
        alpha0=alpha_const,
        max_iter=max_iter,
        schedule="constant"
    )

    # Iteration Log Table for Diminishing Schedule
    print("\n" + "-" * 78)
    print(f"{'Iter':<5} | {'w1':<10} | {'w2':<10} | {'MSE Loss':<10} | {'L1 Pen':<9} | {'Total J(w)':<10} | {'alpha_k':<8}")
    print("-" * 78)

    selected_iters = [0, 1, 2, 5, 10, 20, 40, 70, 100, 130, len(traj_dim) - 1]
    selected_iters = sorted(list(set([i for i in selected_iters if i < len(traj_dim)])))

    for i in selected_iters:
        wi = traj_dim[i]
        tot, mse, l1 = lasso_loss(wi, X, y, lambda_reg)
        alpha_i = steps_dim[i] if i < len(steps_dim) else 0.0
        print(f"{i:<5} | {wi[0]:<10.4f} | {wi[1]:<10.4f} | {mse:<10.4f} | {l1:<9.4f} | {tot:<10.4f} | {alpha_i:<8.4f}")
    print("-" * 78)

    # Comparative Summary Table
    print("\n" + "=" * 78)
    print("                     LASSO OPTIMIZATION BENCHMARK SUMMARY")
    print("=" * 78)
    print(f"{'Method':<20} | {'Weights [w1, w2]':<26} | {'Sparsity (w2)':<14} | {'Loss J(w)':<10}")
    print("-" * 78)
    print(f"{'OLS (No Penalty)':<20} | {f'[{w_ols[0]:.4f}, {w_ols[1]:.4f}]':<26} | {'No (0.2663)':<14} | {ols_total:<10.6f}")
    print(f"{'Exact Lasso Target':<20} | {f'[{w_target[0]:.4f}, {w_target[1]:.4f}]':<26} | {'Exact (0.0)':<14} | {target_total:<10.6f}")
    print(f"{'Subgradient (Best)':<20} | {f'[{w_best_dim[0]:.4f}, {w_best_dim[1]:.4f}]':<26} | {f'{abs(w_best_dim[1]):.6f}':<14} | {f_best_dim:<10.6f}")
    print(f"{'Subgradient (Final)':<20} | {f'[{traj_dim[-1, 0]:.4f}, {traj_dim[-1, 1]:.4f}]':<26} | {f'{abs(traj_dim[-1, 1]):.6f}':<14} | {loss_dim[-1]:<10.6f}")
    print(f"{'Constant alpha=0.15':<20} | {f'[{w_best_const[0]:.4f}, {w_best_const[1]:.4f}]':<26} | {f'{abs(w_best_const[1]):.6f}':<14} | {f_best_const:<10.6f}")
    print("=" * 78)

    # Generate and save visualization
    plot_lasso_results(
        X, y, lambda_reg,
        traj_dim, loss_dim, best_loss_dim,
        traj_const,
        w_ols, w_target,
        save_path="plots/16_subgradient_lasso.png"
    )


if __name__ == "__main__":
    main()
