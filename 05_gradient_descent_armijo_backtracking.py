# Objective: Optimize a function using gradient descent with Armijo backtracking.
# Input: Function, gradient, initial point, and rule parameters.
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

def armijo_line_search(func, grad_func, x, direction, alpha_init=1.0, c=1e-4, rho=0.5, max_backtrack=50):
    """
    Finds a step size alpha that satisfies the Armijo condition:
        f(x + alpha * d) <= f(x) + c * alpha * grad(f)^T d
    """
    alpha = alpha_init
    f_x = func(x)
    g = grad_func(x)
    directional_deriv = np.dot(g, direction)
    
    for _ in range(max_backtrack):
        x_next = x + alpha * direction
        if func(x_next) <= f_x + c * alpha * directional_deriv:
            return alpha
        alpha *= rho
        
    return alpha

def gradient_descent_armijo(func, grad_func, x0, c=1e-4, rho=0.5, alpha_init=1.0, max_iter=40, tol=1e-6):
    """
    Gradient Descent optimization with Armijo backtracking line search at every iteration.
    """
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    history_f = [func(x)]
    history_alpha = []
    
    print(f"{'Iter':<6}{'x1':<12}{'x2':<12}{'f(x)':<14}{'alpha':<10}{'||grad||':<12}")
    print("-" * 66)
    
    for i in range(max_iter):
        g = grad_func(x)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            print(f"Converged at iteration {i} (tolerance reached).")
            break
            
        direction = -g
        alpha = armijo_line_search(func, grad_func, x, direction, alpha_init=alpha_init, c=c, rho=rho)
        history_alpha.append(alpha)
        
        if i % 4 == 0 or i == max_iter - 1:
            print(f"{i:<6}{x[0]:<12.4f}{x[1]:<12.4f}{func(x):<14.6f}{alpha:<10.4f}{g_norm:<12.6f}")
            
        x = x + alpha * direction
        trajectory.append(x.copy())
        history_f.append(func(x))
        
    return x, func(x), np.array(trajectory), np.array(history_f), history_alpha

def plot_contour(func, trajectory, history_alpha, save_path="plots/05_gradient_descent_armijo.png"):
    """
    Plot 2D contour representation and step-size adaptation history.
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
    
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=4, linewidth=1.5, label='GD + Armijo Path')
    ax1.scatter([trajectory[0, 0]], [trajectory[0, 1]], color='green', s=100, zorder=5, label='Start Point (x0)')
    ax1.scatter([0], [0], color='red', marker='*', s=200, zorder=5, label='Global Min (0, 0)')
    
    ax1.set_title("05 - GD with Armijo Backtracking Line Search\n2D Contour Trajectory", fontsize=11)
    ax1.set_xlabel("$x_1$", fontsize=10)
    ax1.set_ylabel("$x_2$", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)
    
    # 2. Step Size Adaptation History
    ax2.plot(range(len(history_alpha)), history_alpha, 's-', color='purple', linewidth=1.5, markersize=5)
    ax2.set_title("Armijo Step Size Adaptive Selection per Iteration", fontsize=11)
    ax2.set_xlabel("Iteration", fontsize=10)
    ax2.set_ylabel(r"Step size $\alpha_k$", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Contour plot saved to: {save_path}")

if __name__ == "__main__":
    initial_point = [5.0, 3.0]
    c_param = 1e-4
    rho_param = 0.5
    alpha_0 = 1.0
    max_iterations = 35
    
    print("=== Algorithm 05: GD with Armijo Backtracking Line Search ===")
    opt_x, opt_f, traj, loss_hist, alpha_hist = gradient_descent_armijo(
        objective_function, 
        gradient, 
        initial_point, 
        c=c_param, 
        rho=rho_param, 
        alpha_init=alpha_0, 
        max_iter=max_iterations
    )
    
    print("-" * 66)
    print(f"Initial Point: {initial_point}")
    print(f"Optimized Point: [{opt_x[0]:.6f}, {opt_x[1]:.6f}]")
    print(f"Optimized Value: {opt_f:.6e}")
    print(f"Total Iterations: {len(traj)-1}")
    
    plot_contour(objective_function, traj, alpha_hist, "plots/05_gradient_descent_armijo.png")
