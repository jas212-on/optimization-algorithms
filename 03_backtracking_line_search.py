# Objective: Find a suitable step size by backtracking.
# Input: Function, point, direction, and step settings.
# Output: Backtracked step size.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def objective_function(x):
    """
    Anisotropic Quadratic function: f(x1, x2) = 0.5 * (x1^2 + 10 * x2^2)
    """
    return 0.5 * (x[0]**2 + 10.0 * x[1]**2)

def gradient(x):
    """
    Gradient: grad f(x) = [x1, 10 * x2]
    """
    return np.array([x[0], 10.0 * x[1]], dtype=float)

def backtracking_line_search(func, x, direction, alpha_init=1.0, rho=0.5, max_backtrack=50):
    """
    Finds a suitable step size alpha by backtracking until f(x + alpha * d) < f(x).
    
    Parameters:
    - func: Objective function f(x)
    - x: Current point
    - direction: Search direction d (descent direction)
    - alpha_init: Initial step size (e.g., 1.0)
    - rho: Contraction factor (0 < rho < 1)
    - max_backtrack: Maximum backtracking iterations
    
    Returns:
    - alpha: Accepted step size
    - candidate_alphas: List of step sizes evaluated during the search
    """
    alpha = alpha_init
    candidate_alphas = [alpha]
    f_current = func(x)
    
    for step in range(max_backtrack):
        x_next = x + alpha * direction
        f_next = func(x_next)
        
        # Check simple decrease condition
        if f_next < f_current:
            return alpha, candidate_alphas
        
        alpha *= rho
        candidate_alphas.append(alpha)
        
    return alpha, candidate_alphas

def optimize_with_backtracking(func, grad_func, x0, max_iter=30, tol=1e-6):
    """
    Optimize 2D function using gradient descent equipped with backtracking line search.
    """
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    history_f = [func(x)]
    all_alphas = []
    
    print(f"{'Iter':<6}{'x1':<12}{'x2':<12}{'f(x)':<14}{'alpha':<10}{'||grad||':<12}")
    print("-" * 66)
    
    for i in range(max_iter):
        g = grad_func(x)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            print(f"Converged at iteration {i} (tolerance reached).")
            break
            
        direction = -g
        alpha, _ = backtracking_line_search(func, x, direction, alpha_init=1.0, rho=0.5)
        all_alphas.append(alpha)
        
        if i % 3 == 0 or i == max_iter - 1:
            print(f"{i:<6}{x[0]:<12.4f}{x[1]:<12.4f}{func(x):<14.6f}{alpha:<10.4f}{g_norm:<12.6f}")
            
        x = x + alpha * direction
        trajectory.append(x.copy())
        history_f.append(func(x))
        
    return x, func(x), np.array(trajectory), all_alphas

def plot_contour_and_line_search(func, trajectory, x0, direction, candidates, save_path="plots/03_backtracking_line_search.png"):
    """
    Plot 2D contour trajectory alongside a 1D visualization of the backtracking search.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 2D Contour Plot
    x1_vals = np.linspace(-5.5, 5.5, 400)
    x2_vals = np.linspace(-3.5, 3.5, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = 0.5 * (X1**2 + 10.0 * X2**2)
    
    levels = np.logspace(-1, 2.5, 25)
    cs = ax1.contour(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.7)
    ax1.clabel(cs, inline=1, fontsize=8, fmt="%.1f")
    ax1.contourf(X1, X2, Z, levels=levels, cmap="Blues", alpha=0.2)
    
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=4, linewidth=1.5, label='Backtracking GD Path')
    ax1.scatter([trajectory[0, 0]], [trajectory[0, 1]], color='green', s=100, zorder=5, label='Start Point (x0)')
    ax1.scatter([0], [0], color='red', marker='*', s=200, zorder=5, label='Global Min (0, 0)')
    
    ax1.set_title("Optimization Trajectory with Backtracking Line Search", fontsize=11)
    ax1.set_xlabel("$x_1$", fontsize=10)
    ax1.set_ylabel("$x_2$", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)
    
    # 2. 1D Line Search Slice at x0
    alphas_eval = np.linspace(0, 1.2, 200)
    f_slice = [func(x0 + a * direction) for a in alphas_eval]
    f_x0 = func(x0)
    
    ax2.plot(alphas_eval, f_slice, 'b-', linewidth=2, label=r'$\phi(\alpha) = f(x_0 + \alpha d)$')
    ax2.axhline(f_x0, color='gray', linestyle='--', label=r'Threshold $f(x_0)$')
    
    for idx, a in enumerate(candidates):
        val = func(x0 + a * direction)
        if idx == len(candidates) - 1:
            ax2.scatter([a], [val], color='green', s=90, zorder=6, label=f'Accepted step $\\alpha={a}$')
        else:
            ax2.scatter([a], [val], color='red', marker='x', s=70, zorder=5, label=f'Rejected $\\alpha={a}$' if idx == 0 else "")
            
    ax2.set_title("Step Size Selection at $x_0$ (1D Backtracking)", fontsize=11)
    ax2.set_xlabel(r"Step size $\alpha$", fontsize=10)
    ax2.set_ylabel(r"$f(x_0 + \alpha d)$", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Contour & line search plot saved to: {save_path}")

if __name__ == "__main__":
    initial_point = np.array([5.0, 3.0])
    g0 = gradient(initial_point)
    descent_direction = -g0
    
    print("=== Algorithm 03: Backtracking Line Search ===")
    print(f"Testing backtracking line search at point x0 = {initial_point}")
    print(f"Descent direction d = -grad f(x0) = {descent_direction}")
    
    # Demonstration of the single step line search
    accepted_alpha, candidate_steps = backtracking_line_search(
        objective_function, 
        initial_point, 
        descent_direction, 
        alpha_init=1.0, 
        rho=0.5
    )
    print(f"Candidate step sizes tested: {candidate_steps}")
    print(f"Accepted backtracked step size: {accepted_alpha}")
    print(f"f(x0) = {objective_function(initial_point):.4f} -> f(x0 + alpha*d) = {objective_function(initial_point + accepted_alpha * descent_direction):.4f}")
    print("-" * 66)
    
    # Full optimization using backtracking
    opt_x, opt_f, traj, alphas = optimize_with_backtracking(
        objective_function, 
        gradient, 
        initial_point, 
        max_iter=30
    )
    
    print("-" * 66)
    print(f"Optimized Point: [{opt_x[0]:.6f}, {opt_x[1]:.6f}]")
    print(f"Optimized Value: {opt_f:.6e}")
    print(f"Total Iterations: {len(traj)-1}")
    
    plot_contour_and_line_search(
        objective_function, 
        traj, 
        initial_point, 
        descent_direction, 
        candidate_steps, 
        "plots/03_backtracking_line_search.png"
    )
