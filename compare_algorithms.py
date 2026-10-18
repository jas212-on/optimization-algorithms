# Master Comparison Script for Optimization Algorithms 01 to 07
# Compares convergence speed and trajectories on the Anisotropic Quadratic Bowl.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Import algorithms from individual files
from importlib import import_module

gd_01 = import_module("01_gradient_descent_function_optimization")
els_02 = import_module("02_gradient_descent_2d_exact_line_search")
armijo_05 = import_module("05_gradient_descent_armijo_backtracking")
momentum_06 = import_module("06_momentum_gradient_descent")
norm_07 = import_module("07_normalized_gradient_descent")

def run_comparison(x0=[5.0, 3.0], max_iter=30):
    func = gd_01.objective_function
    grad = gd_01.gradient
    
    results = {}
    
    # 1. Standard GD
    _, _, traj_01, loss_01 = gd_01.gradient_descent(func, grad, x0, learning_rate=0.15, max_iter=max_iter)
    results["01: Standard GD (alpha=0.15)"] = (traj_01, loss_01, "blue", "-")
    
    # 2. Exact Line Search
    _, _, traj_02, loss_02 = els_02.gradient_descent_exact_line_search(func, grad, x0, max_iter=max_iter)
    results["02: Exact Line Search"] = (traj_02, loss_02, "red", "-")
    
    # 3. Armijo Backtracking
    _, _, traj_05, loss_05, _ = armijo_05.gradient_descent_armijo(func, grad, x0, c=1e-4, rho=0.5, alpha_init=1.0, max_iter=max_iter)
    results["05: GD + Armijo Backtracking"] = (traj_05, loss_05, "green", "-")
    
    # 4. Momentum GD
    _, _, traj_06, loss_06 = momentum_06.momentum_gradient_descent(func, grad, x0, learning_rate=0.08, beta=0.8, max_iter=max_iter)
    results["06: Momentum GD (beta=0.8)"] = (traj_06, loss_06, "magenta", "-")
    
    # 5. Normalized GD
    _, _, traj_07, loss_07 = norm_07.normalized_gradient_descent(func, grad, x0, step_size=0.4, decay=True, max_iter=max_iter)
    results["07: Normalized GD (decay)"] = (traj_07, loss_07, "darkorange", "-")
    
    return results

def plot_master_comparison(results, save_path="plots/comparison_master.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Panel 1: Trajectory Comparison on Contour Map ---
    x1_vals = np.linspace(-5.5, 5.5, 400)
    x2_vals = np.linspace(-3.5, 3.5, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = 0.5 * (X1**2 + 10.0 * X2**2)
    
    levels = np.logspace(-1, 2.5, 25)
    cs = ax1.contour(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.6)
    ax1.contourf(X1, X2, Z, levels=levels, cmap="Blues", alpha=0.15)
    
    # Plot trajectories
    for name, (traj, _, color, style) in results.items():
        ax1.plot(traj[:, 0], traj[:, 1], color=color, linestyle=style, marker='o', 
                 markersize=3, linewidth=1.6, label=name)
        
    start_pt = list(results.values())[0][0][0]
    ax1.scatter([start_pt[0]], [start_pt[1]], color='black', s=120, zorder=7, label=r'Start Point $x_0$')
    ax1.scatter([0], [0], color='gold', marker='*', s=220, zorder=7, label=r'Optimum $(0, 0)$')
    
    ax1.set_title("Trajectory Comparison on Anisotropic Quadratic Bowl", fontsize=12)
    ax1.set_xlabel("$x_1$", fontsize=11)
    ax1.set_ylabel("$x_2$", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)
    
    # --- Panel 2: Convergence History (Loss vs Iterations) ---
    for name, (_, loss_hist, color, style) in results.items():
        iters = np.arange(len(loss_hist))
        ax2.semilogy(iters, loss_hist, color=color, linestyle=style, marker='s', markersize=3, label=name)
        
    ax2.set_title("Convergence Rate Comparison (Log Loss vs Iteration)", fontsize=12)
    ax2.set_xlabel("Iteration ($k$)", fontsize=11)
    ax2.set_ylabel("Objective Value $f(x_k)$ [log scale]", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Master comparison plot saved to: {save_path}")

if __name__ == "__main__":
    print("=== Running Master Benchmark & Algorithm Comparison ===")
    res = run_comparison(x0=[5.0, 3.0], max_iter=30)
    
    print("\n" + "=" * 70)
    print(f"{'Algorithm':<32}{'Final Point (x1, x2)':<24}{'Final f(x)':<14}")
    print("=" * 70)
    for name, (traj, loss_hist, _, _) in res.items():
        final_pt = traj[-1]
        final_f = loss_hist[-1]
        pt_str = f"[{final_pt[0]:.4f}, {final_pt[1]:.4f}]"
        print(f"{name:<32}{pt_str:<24}{final_f:<14.2e}")
    print("=" * 70)
    
    plot_master_comparison(res, "plots/comparison_master.png")
