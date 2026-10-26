# Objective: Optimize a nonsmooth function using the sub-gradient method.
# Input: Function, sub-gradient, initial point, and step settings.
# Output: Optimized point and objective value.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def nonsmooth_objective(x: np.ndarray) -> float:
    """
    Nonsmooth convex function: f(x_1, x_2) = |x_1| + 2*|x_2|.
    Global minimum is at x* = [0, 0] with f(x*) = 0.
    Level sets are anisotropic rhombi (diamonds).
    """
    return float(np.abs(x[0]) + 2.0 * np.abs(x[1]))


def subgradient(x: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Computes a valid subgradient g in partial f(x).
    For f(x) = |x_1| + 2|x_2|:
      partial |u| = {sign(u)} if u != 0, else [-1, 1] (we pick 0 at origin).
    """
    g1 = np.sign(x[0]) if np.abs(x[0]) > tol else 0.0
    g2 = 2.0 * np.sign(x[1]) if np.abs(x[1]) > tol else 0.0
    return np.array([g1, g2], dtype=float)


def subgradient_method(
    func,
    subgrad_func,
    x0: np.ndarray,
    alpha0: float = 0.8,
    max_iter: int = 60,
    schedule: str = "diminishing"
):
    """
    Subgradient Method for nonsmooth convex optimization.
    
    Update rule:
      x_{k+1} = x_k - alpha_k * g_k,  where g_k in partial f(x_k)

    Step size schedules:
      - 'diminishing': alpha_k = alpha0 / sqrt(k + 1)
        Satisfies sum alpha_k = inf, sum alpha_k^2 < inf (guarantees convergence to x*).
      - 'constant': alpha_k = alpha0
        Converges to an O(alpha0) ball around x*, oscillating in a limit cycle.
    
    Since the subgradient is NOT necessarily a descent direction, f(x_k) is non-monotonic.
    We track both instantaneous f(x_k) and the running best f_best^(k) = min_{i<=k} f(x_i).
    """
    x = np.array(x0, dtype=float).copy()
    trajectory = [x.copy()]
    f_history = [func(x)]
    f_best_history = [func(x)]
    x_best = x.copy()
    f_best = func(x)
    step_sizes = []

    for k in range(max_iter):
        g = subgrad_func(x)
        g_norm = np.linalg.norm(g)
        
        # Subgradient is zero only at the global minimum (0, 0)
        if g_norm < 1e-12:
            step_sizes.append(0.0)
            break

        # Compute step size based on schedule
        if schedule == "diminishing":
            alpha_k = alpha0 / np.sqrt(k + 1)
        elif schedule == "constant":
            alpha_k = alpha0
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        step_sizes.append(alpha_k)

        # Update step
        x = x - alpha_k * g
        f_val = func(x)

        if f_val < f_best:
            f_best = f_val
            x_best = x.copy()

        trajectory.append(x.copy())
        f_history.append(f_val)
        f_best_history.append(f_best)

    return (
        np.array(trajectory),
        np.array(f_history),
        np.array(f_best_history),
        x_best,
        f_best,
        step_sizes
    )


def plot_subgradient_optimization(
    traj_dim, f_dim, f_best_dim,
    traj_const, f_const, f_best_const,
    save_path: str = "plots/15_subgradient_method.png"
):
    """
    Plots:
      Panel 1: 2D Diamond Contours with Diminishing vs Constant step size trajectories.
      Panel 2: Convergence comparison showing instantaneous f(x_k) chatter vs best-so-far f_best.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ----------------- Panel 1: 2D Contour Plot -----------------
    x1_range = np.linspace(-4.5, 4.5, 400)
    x2_range = np.linspace(-3.5, 3.5, 400)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.abs(X1) + 2.0 * np.abs(X2)

    levels = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0])
    cs = ax1.contour(X1, X2, Z, levels=levels, cmap="plasma", alpha=0.75, linewidths=1.2)
    ax1.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    # Constant step trajectory (limit cycle chatter)
    ax1.plot(
        traj_const[:, 0], traj_const[:, 1],
        color="crimson", linestyle="--", linewidth=1.4, alpha=0.8,
        marker="^", markersize=4, label=r"Constant $\alpha = 0.40$ (Limit Cycle)"
    )

    # Diminishing step trajectory (convergent)
    ax1.plot(
        traj_dim[:, 0], traj_dim[:, 1],
        color="dodgerblue", linestyle="-", linewidth=1.8,
        marker="o", markersize=4.5, label=r"Diminishing $\alpha_k = \frac{0.80}{\sqrt{k+1}}$"
    )

    # Starting and optimal points
    ax1.scatter([traj_dim[0, 0]], [traj_dim[0, 1]], color="black", s=90, zorder=6, label=f"Start $x_0 = [{traj_dim[0, 0]:.1f}, {traj_dim[0, 1]:.1f}]$")
    ax1.scatter([0], [0], color="gold", edgecolors="black", s=130, marker="*", zorder=6, label=r"Minimum $x^* = (0, 0)$")

    ax1.set_title(r"Trajectory on Nonsmooth $f(x) = |x_1| + 2|x_2|$", fontsize=12, fontweight="bold")
    ax1.set_xlabel(r"$x_1$", fontsize=11)
    ax1.set_ylabel(r"$x_2$", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax1.set_xlim([-4.5, 4.5])
    ax1.set_ylim([-3.5, 3.5])

    # ----------------- Panel 2: Objective History -----------------
    iters_dim = np.arange(len(f_dim))
    iters_const = np.arange(len(f_const))

    # Instantaneous vs Best for Diminishing Step Size
    ax2.plot(iters_dim, f_dim, color="lightblue", linestyle=":", linewidth=1.2, label=r"Diminishing: Instantaneous $f(x_k)$ (Chatter)")
    ax2.plot(iters_dim, f_best_dim, color="dodgerblue", linestyle="-", linewidth=2.0, label=r"Diminishing: Best-so-far $f_{\rm best}^{(k)}$")

    # Instantaneous vs Best for Constant Step Size
    ax2.plot(iters_const, f_const, color="lightcoral", linestyle=":", linewidth=1.2, label=r"Constant: Instantaneous $f(x_k)$ (Oscillating)")
    ax2.plot(iters_const, f_best_const, color="crimson", linestyle="--", linewidth=1.8, label=r"Constant: Best-so-far $f_{\rm best}^{(k)}$")

    ax2.set_title("Subgradient Convergence: Non-Descent Dynamics", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Iteration $k$", fontsize=11)
    ax2.set_ylabel(r"Objective Value $f(x)$", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Subgradient method plot saved successfully at: {save_path}")


def main():
    print("=" * 78)
    print("      NONSMOOTH CONVEX OPTIMIZATION: SUBGRADIENT METHOD")
    print("=" * 78)
    print("Objective function: f(x_1, x_2) = |x_1| + 2*|x_2|")
    print("Subgradient: g(x) = [sign(x_1), 2*sign(x_2)]^T, with g(0, 0) = [0, 0]^T")
    print("Global minimum: x* = [0.0, 0.0], f(x*) = 0.0")

    x0 = np.array([3.7, 2.8], dtype=float)
    max_iter = 60
    alpha0_dim = 0.8
    alpha_const = 0.4

    print(f"\nInitial Point: x0 = [{x0[0]:.2f}, {x0[1]:.2f}], f(x0) = {nonsmooth_objective(x0):.4f}")

    # 1. Diminishing Step Size Schedule
    traj_dim, f_dim, f_best_dim, x_best_dim, f_best_val_dim, steps_dim = subgradient_method(
        nonsmooth_objective,
        subgradient,
        x0=x0,
        alpha0=alpha0_dim,
        max_iter=max_iter,
        schedule="diminishing"
    )

    # 2. Constant Step Size Schedule
    traj_const, f_const, f_best_const, x_best_const, f_best_val_const, steps_const = subgradient_method(
        nonsmooth_objective,
        subgradient,
        x0=x0,
        alpha0=alpha_const,
        max_iter=max_iter,
        schedule="constant"
    )

    # Formatted Iteration Table for Diminishing Schedule
    print("\n" + "-" * 78)
    print(f"{'Iter':<5} | {'x1':<10} | {'x2':<10} | {'f(x_k)':<10} | {'f_best':<10} | {'alpha_k':<9} | {'Subgrad g':<14}")
    print("-" * 78)
    
    # Print selected iterations
    selected_iters = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, len(traj_dim) - 1]
    selected_iters = sorted(list(set([i for i in selected_iters if i < len(traj_dim)])))

    for i in selected_iters:
        xi = traj_dim[i]
        fi = f_dim[i]
        fbi = f_best_dim[i]
        alpha_i = steps_dim[i] if i < len(steps_dim) else 0.0
        gi = subgradient(xi)
        g_str = f"[{gi[0]:+.0f}, {gi[1]:+.0f}]"
        print(f"{i:<5} | {xi[0]:<10.4f} | {xi[1]:<10.4f} | {fi:<10.4f} | {fbi:<10.4f} | {alpha_i:<9.4f} | {g_str:<14}")
    print("-" * 78)

    # Comparative Summary Table
    print("\n" + "=" * 78)
    print("                 SUBGRADIENT SCHEDULE COMPARISON SUMMARY")
    print("=" * 78)
    print(f"{'Schedule':<18} | {'Final x':<22} | {'Final f(x)':<12} | {'Best x':<22} | {'Best f':<10}")
    print("-" * 78)
    x_final_dim_str = f"[{traj_dim[-1, 0]:.4f}, {traj_dim[-1, 1]:.4f}]"
    x_best_dim_str = f"[{x_best_dim[0]:.4f}, {x_best_dim[1]:.4f}]"
    print(f"{'Diminishing':<18} | {x_final_dim_str:<22} | {f_dim[-1]:<12.6f} | {x_best_dim_str:<22} | {f_best_val_dim:<10.6f}")

    x_final_const_str = f"[{traj_const[-1, 0]:.4f}, {traj_const[-1, 1]:.4f}]"
    x_best_const_str = f"[{x_best_const[0]:.4f}, {x_best_const[1]:.4f}]"
    print(f"{'Constant (0.40)':<18} | {x_final_const_str:<22} | {f_const[-1]:<12.6f} | {x_best_const_str:<22} | {f_best_val_const:<10.6f}")
    print("=" * 78)
    print("\nKey Theoretical Insights:")
    print("  1. Non-Descent Property: Unlike gradient descent, the negative subgradient")
    print("     is not guaranteed to be a descent direction; f(x_k) chatters and oscillates.")
    print("  2. Best-So-Far Guarantee: The sequence f_best^(k) = min_{i<=k} f(x_i) is monotonic.")
    print("  3. Diminishing Step Size: Step alpha_k = alpha0 / sqrt(k+1) dampens oscillation,")
    print("     guaranteeing asymptotic convergence to x* = (0, 0).")
    print("  4. Constant Step Size: Remains trapped in a periodic limit cycle of size O(alpha).")

    # Generate and save visualization
    plot_subgradient_optimization(
        traj_dim, f_dim, f_best_dim,
        traj_const, f_const, f_best_const,
        save_path="plots/15_subgradient_method.png"
    )


if __name__ == "__main__":
    main()
