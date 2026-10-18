# Objective: Optimize a function using momentum-based gradient descent.
# Input: Function, gradient, initial point, learning rate, and momentum.
# Output: Optimized point and objective value.

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

def momentum_gradient_descent(func, grad_func, x0, learning_rate=0.08, beta=0.8, max_iter=40, tol=1e-6):
    """
    Polyak's Momentum (Heavy Ball) Gradient Descent:
        v_{k+1} = beta * v_k + alpha * grad(f)(x_k)
        x_{k+1} = x_k - v_{k+1}
    """
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    trajectory = [x.copy()]
    history_f = [func(x)]
    
    print(f"{'Iter':<6}{'x1':<12}{'x2':<12}{'f(x)':<14}{'||v||':<12}{'||grad||':<12}")
    print("-" * 68)
    
    for i in range(max_iter):
        g = grad_func(x)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol and np.linalg.norm(v) < tol:
            print(f"Converged at iteration {i} (tolerance reached).")
            break
            
        # Velocity and position update
        v = beta * v + learning_rate * g
        x = x - v
        
        if i % 5 == 0 or i == max_iter - 1:
            print(f"{i:<6}{x[0]:<12.4f}{x[1]:<12.4f}{func(x):<14.6f}{np.linalg.norm(v):<12.6f}{g_norm:<12.6f}")
            
        trajectory.append(x.copy())
        history_f.append(func(x))
        
    return x, func(x), np.array(trajectory), np.array(history_f)

def standard_gradient_descent_trajectory(func, grad_func, x0, learning_rate=0.08, max_iter=40):
    """Helper to compute baseline standard GD path for visual comparison."""
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    for _ in range(max_iter):
        g = grad_func(x)
        x = x - learning_rate * g
        traj.append(x.copy())
    return np.array(traj)

def plot_contour(func, trajectory, baseline_traj=None, save_path="plots/06_momentum_gradient_descent.png"):
    """
    Plot 2D contour representation showing momentum acceleration vs standard GD.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    x1_vals = np.linspace(-5.5, 5.5, 400)
    x2_vals = np.linspace(-3.5, 3.5, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = 0.5 * (X1**2 + 10.0 * X2**2)
    
    plt.figure(figsize=(9, 6))
    levels = np.logspace(-1, 2.5, 25)
    cs = plt.contour(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.7)
    plt.clabel(cs, inline=1, fontsize=8, fmt="%.1f")
    plt.contourf(X1, X2, Z, levels=levels, cmap="Blues", alpha=0.2)
    
    # Plot baseline standard GD if available
    if baseline_traj is not None:
        plt.plot(baseline_traj[:, 0], baseline_traj[:, 1], 'k--', alpha=0.5, label='Standard GD (no momentum)')
        
    # Plot momentum trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'm-o', markersize=4, linewidth=1.8, label=r'Momentum GD ($\beta=0.8$)')
    plt.scatter([trajectory[0, 0]], [trajectory[0, 1]], color='green', s=100, zorder=5, label='Start Point (x0)')
    plt.scatter([0], [0], color='red', marker='*', s=200, zorder=5, label='Global Min (0, 0)')
    
    plt.title("06 - Momentum-Based Gradient Descent\nMomentum Smooths Oscillations & Accelerates Along Valley", fontsize=12)
    plt.xlabel("$x_1$", fontsize=11)
    plt.ylabel("$x_2$", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Contour plot saved to: {save_path}")

if __name__ == "__main__":
    initial_point = [5.0, 3.0]
    alpha_rate = 0.08
    beta_momentum = 0.8
    max_iterations = 35
    
    print("=== Algorithm 06: Momentum Gradient Descent ===")
    opt_x, opt_f, traj, loss_hist = momentum_gradient_descent(
        objective_function, 
        gradient, 
        initial_point, 
        learning_rate=alpha_rate, 
        beta=beta_momentum, 
        max_iter=max_iterations
    )
    
    print("-" * 68)
    print(f"Initial Point: {initial_point}")
    print(f"Optimized Point: [{opt_x[0]:.6f}, {opt_x[1]:.6f}]")
    print(f"Optimized Value: {opt_f:.6e}")
    print(f"Total Iterations: {len(traj)-1}")
    
    # Compute baseline standard GD for comparison
    base_traj = standard_gradient_descent_trajectory(
        objective_function, 
        gradient, 
        initial_point, 
        learning_rate=alpha_rate, 
        max_iter=max_iterations
    )
    
    plot_contour(objective_function, traj, baseline_traj=base_traj, save_path="plots/06_momentum_gradient_descent.png")
