# Objective: Optimize a function using Adagrad.
# Input: Function, gradient, initial point, and optimizer settings.
# Output: Optimized point and objective value.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def objective_function(x):
    """
    Anisotropic Quadratic function: f(x1, x2) = 0.5 * (x1^2 + 10 * x2^2)
    Standard benchmark exhibiting ill-conditioned curvature (kappa = 10).
    """
    return 0.5 * (x[0]**2 + 10.0 * x[1]**2)

def gradient(x):
    """
    Analytical gradient of f(x1, x2): [x1, 10 * x2]
    """
    return np.array([x[0], 10.0 * x[1]], dtype=float)

def standard_gradient_descent_trajectory(func, grad_func, x0, learning_rate=0.15, max_iter=40):
    """
    Compute baseline standard Gradient Descent trajectory for visual comparison.
    """
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    for _ in range(max_iter):
        g = grad_func(x)
        x = x - learning_rate * g
        traj.append(x.copy())
    return np.array(traj)

def adagrad_optimizer(func, grad_func, x0, learning_rate=1.5, epsilon=1e-8, max_iter=40, tol=1e-6):
    """
    Optimize objective function using Adagrad (Adaptive Gradient Algorithm).
    
    Update rules:
        g_k = grad(x_k)
        G_{k+1} = G_k + g_k^2
        x_{k+1} = x_k - (alpha / (sqrt(G_{k+1}) + eps)) * g_k
        
    Adapts learning rate per coordinate: divides by the square root of the sum of
    all historical squared gradients, damping oscillations along steep dimensions
    while allowing larger progress along flat dimensions.
    
    Inputs:
    - func: Objective function to minimize
    - grad_func: Gradient function of the objective
    - x0: Initial starting point [x1, x2]
    - learning_rate: Base step size alpha
    - epsilon: Small constant for numerical stability
    - max_iter: Maximum iterations
    - tol: Tolerance on gradient norm for stopping
    
    Outputs:
    - x: Final optimized point
    - final_f: Final objective function value
    - trajectory: Path history of x points
    - history_f: Objective values across iterations
    - history_G: History of accumulated squared gradients G
    """
    x = np.array(x0, dtype=float)
    G = np.zeros_like(x)
    trajectory = [x.copy()]
    history_f = [func(x)]
    history_G = [G.copy()]
    
    print(f"{'Iter':<6}{'x1':<12}{'x2':<12}{'f(x)':<14}{'G1':<12}{'G2':<12}{'||grad||':<12}")
    print("-" * 76)
    
    for i in range(1, max_iter + 1):
        g = grad_func(x)
        g_norm = np.linalg.norm(g)
        
        if (i - 1) % 5 == 0 or i == max_iter:
            print(f"{i-1:<6}{x[0]:<12.4f}{x[1]:<12.4f}{history_f[-1]:<14.6f}{G[0]:<12.4f}{G[1]:<12.4f}{g_norm:<12.6f}")
            
        if g_norm < tol:
            print(f"Converged at iteration {i-1} (gradient tolerance reached).")
            break
            
        # Accumulate squared gradients
        G = G + g**2
        
        # Parameter update
        effective_lr = learning_rate / (np.sqrt(G) + epsilon)
        x = x - effective_lr * g
        
        trajectory.append(x.copy())
        history_f.append(func(x))
        history_G.append(G.copy())
        
    return x, history_f[-1], np.array(trajectory), np.array(history_f), np.array(history_G)

def plot_adagrad_contour(func, trajectory, history_f, baseline_traj=None, 
                         save_path="plots/12_adagrad.png"):
    """
    Plot 2D contour map with Adagrad trajectory and log-loss convergence comparison.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Panel 1: 2D Contour Plot ---
    x1_vals = np.linspace(-5.5, 5.5, 400)
    x2_vals = np.linspace(-3.5, 3.5, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = 0.5 * (X1**2 + 10.0 * X2**2)
    
    levels = np.logspace(-1, 2.5, 25)
    cs = ax1.contour(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.7)
    ax1.clabel(cs, inline=1, fontsize=8, fmt="%.1f")
    ax1.contourf(X1, X2, Z, levels=levels, cmap="Blues", alpha=0.2)
    
    # Baseline Standard GD
    if baseline_traj is not None:
        ax1.plot(baseline_traj[:, 0], baseline_traj[:, 1], 'k--', alpha=0.5, linewidth=1.5, 
                 label='Standard GD (Fixed alpha=0.15)')
        
    # Adagrad Trajectory
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'co-', markersize=4, linewidth=1.8, 
             label=r'Adagrad ($\alpha=1.5$)')
    ax1.scatter([trajectory[0, 0]], [trajectory[0, 1]], color='green', s=110, zorder=6, 
                label=f'Start Point x0: ({trajectory[0,0]:.1f}, {trajectory[0,1]:.1f})')
    ax1.scatter([0], [0], color='gold', marker='*', s=220, zorder=7, label='Global Min (0, 0)')
    
    final_x = trajectory[-1]
    ax1.scatter([final_x[0]], [final_x[1]], color='purple', marker='X', s=120, zorder=7, 
                label=f'Adagrad Final: ({final_x[0]:.2e}, {final_x[1]:.2e})')
    
    ax1.set_title("12 - Adagrad Optimization on Anisotropic Bowl\nPer-Coordinate Step Size Adaptation", fontsize=11)
    ax1.set_xlabel("$x_1$", fontsize=10)
    ax1.set_ylabel("$x_2$", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)
    
    # --- Panel 2: Convergence History (Log Scale) ---
    iters = np.arange(len(history_f))
    ax2.semilogy(iters, history_f, 'c-o', markersize=3.5, linewidth=1.8, label='Adagrad Objective')
    
    if baseline_traj is not None:
        base_f = [func(pt) for pt in baseline_traj]
        ax2.semilogy(np.arange(len(base_f)), base_f, 'k--', linewidth=1.5, label='Standard GD Objective')
        
    ax2.set_title("Convergence Rate (Log Loss vs Iteration)", fontsize=11)
    ax2.set_xlabel("Iteration ($k$)", fontsize=10)
    ax2.set_ylabel(r"Objective Value $f(x_k)$ [log scale]", fontsize=10)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Adagrad contour plot saved to: {save_path}")

if __name__ == "__main__":
    initial_point = [5.0, 3.0]
    alpha_adagrad = 1.5
    iterations = 40
    
    print("=== Algorithm 12: Adagrad Function Optimization ===")
    opt_x, opt_f, traj, loss_hist, G_hist = adagrad_optimizer(
        objective_function, 
        gradient, 
        initial_point, 
        learning_rate=alpha_adagrad, 
        max_iter=iterations
    )
    
    # Compute baseline standard GD for comparison
    base_traj = standard_gradient_descent_trajectory(
        objective_function, 
        gradient, 
        initial_point, 
        learning_rate=0.15, 
        max_iter=iterations
    )
    
    print("-" * 76)
    print(f"Initial Point:   {initial_point}")
    print(f"Optimized Point: [{opt_x[0]:.6e}, {opt_x[1]:.6e}]")
    print(f"Optimized Value: {opt_f:.6e}")
    print(f"Accumulated G:   [{G_hist[-1, 0]:.4f}, {G_hist[-1, 1]:.4f}]")
    print(f"Total Iterations:{len(traj) - 1}")
    
    plot_adagrad_contour(objective_function, traj, loss_hist, baseline_traj=base_traj, 
                         save_path="plots/12_adagrad.png")
