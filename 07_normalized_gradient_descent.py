# Objective: Optimize a function using normalized gradient descent.
# Input: Function, gradient, initial point, and step settings.
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

def normalized_gradient_descent(func, grad_func, x0, step_size=0.3, decay=True, max_iter=40, tol=1e-5, eps=1e-12):
    """
    Normalized Gradient Descent:
        x_{k+1} = x_k - alpha_k * (grad(x_k) / (||grad(x_k)|| + eps))
    
    Using unit direction ensures step length is exactly controlled by alpha_k,
    preventing explosive gradient steps in steep regions.
    """
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    history_f = [func(x)]
    
    print(f"{'Iter':<6}{'x1':<12}{'x2':<12}{'f(x)':<14}{'alpha_k':<10}{'||grad||':<12}")
    print("-" * 66)
    
    for i in range(max_iter):
        g = grad_func(x)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            print(f"Converged at iteration {i} (tolerance reached).")
            break
            
        # Optional step decay to converge smoothly to the minimum
        alpha_k = step_size / np.sqrt(i + 1) if decay else step_size
        
        if i % 5 == 0 or i == max_iter - 1:
            print(f"{i:<6}{x[0]:<12.4f}{x[1]:<12.4f}{func(x):<14.6f}{alpha_k:<10.4f}{g_norm:<12.6f}")
            
        # Normalized step update
        normalized_dir = g / (g_norm + eps)
        x = x - alpha_k * normalized_dir
        
        trajectory.append(x.copy())
        history_f.append(func(x))
        
    return x, func(x), np.array(trajectory), np.array(history_f)

def plot_contour(func, trajectory, save_path="plots/07_normalized_gradient_descent.png"):
    """
    Plot 2D contour representation showing normalized gradient descent path.
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
    
    # Plot trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'co-', markersize=4, linewidth=1.5, 
             label=r'Normalized GD ($d_k = \frac{\nabla f}{||\nabla f||}$)')
    plt.scatter([trajectory[0, 0]], [trajectory[0, 1]], color='green', s=100, zorder=5, label='Start Point (x0)')
    plt.scatter([0], [0], color='red', marker='*', s=200, zorder=5, label='Global Min (0, 0)')
    
    plt.title("07 - Normalized Gradient Descent\nSteps depend only on direction, not gradient magnitude", fontsize=12)
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
    base_step = 0.5
    use_decay = True
    max_iterations = 40
    
    print("=== Algorithm 07: Normalized Gradient Descent ===")
    opt_x, opt_f, traj, loss_hist = normalized_gradient_descent(
        objective_function, 
        gradient, 
        initial_point, 
        step_size=base_step, 
        decay=use_decay, 
        max_iter=max_iterations
    )
    
    print("-" * 66)
    print(f"Initial Point: {initial_point}")
    print(f"Optimized Point: [{opt_x[0]:.6f}, {opt_x[1]:.6f}]")
    print(f"Optimized Value: {opt_f:.6e}")
    print(f"Total Iterations: {len(traj)-1}")
    
    plot_contour(objective_function, traj, "plots/07_normalized_gradient_descent.png")
