# Objective: Optimize a two-dimensional function with exact line search.
# Input: Function, gradient, initial point, and line-search settings.
# Output: Optimized point and objective value.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

# Define symmetric positive definite matrix Q for quadratic function: f(x) = 0.5 * x^T Q x
Q = np.array([[1.0, 0.0],
              [0.0, 10.0]])

def objective_function(x):
    """
    Anisotropic Quadratic function: f(x) = 0.5 * (x1^2 + 10 * x2^2)
    In matrix form: f(x) = 0.5 * x^T Q x
    """
    return 0.5 * (x[0]**2 + 10.0 * x[1]**2)

def gradient(x):
    """
    Gradient: grad f(x) = Q x = [x1, 10 * x2]
    """
    return Q @ x

def exact_line_search(func, grad_func, x, direction, method="analytical"):
    """
    Compute exact step size alpha* = argmin_{alpha > 0} func(x + alpha * direction).
    - 'analytical': exact formula for quadratic: alpha = -(g^T d) / (d^T Q d)
    - 'numerical': 1D line minimization using scipy minimize_scalar
    """
    if method == "analytical":
        g = grad_func(x)
        denom = direction.T @ Q @ direction
        if denom <= 1e-12:
            return 1e-3
        alpha = - (g.T @ direction) / denom
        return max(alpha, 1e-8)
    else:
        # General numerical line search
        line_obj = lambda alpha: func(x + alpha * direction)
        res = minimize_scalar(line_obj, bounds=(0, 2.0), method='bounded')
        return res.x

def gradient_descent_exact_line_search(func, grad_func, x0, max_iter=25, tol=1e-6):
    """
    Perform gradient descent where step size is found via exact line search at each step.
    Notice that successive gradient directions will be orthogonal: <g_{k+1}, g_k> = 0.
    """
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    history_f = [func(x)]
    
    print(f"{'Iter':<6}{'x1':<12}{'x2':<12}{'f(x)':<14}{'alpha*':<12}{'||grad||':<12}")
    print("-" * 68)
    
    for i in range(max_iter):
        g = grad_func(x)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            print(f"Converged at iteration {i} (tolerance reached).")
            break
            
        direction = -g
        alpha = exact_line_search(func, grad_func, x, direction, method="analytical")
        
        print(f"{i:<6}{x[0]:<12.4f}{x[1]:<12.4f}{func(x):<14.6f}{alpha:<12.4f}{g_norm:<12.6f}")
        
        x = x + alpha * direction
        trajectory.append(x.copy())
        history_f.append(func(x))
        
    return x, func(x), np.array(trajectory), np.array(history_f)

def plot_contour(func, trajectory, save_path="plots/02_exact_line_search.png"):
    """
    Plot 2D contour representation showing the orthogonal steps of exact line search.
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
    
    # Plot trajectory showing right-angle steps
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=4, linewidth=1.5, 
             label='Exact Line Search Path\n(Orthogonal Steps)')
    plt.scatter([trajectory[0, 0]], [trajectory[0, 1]], color='green', s=100, zorder=5, label='Start Point (x0)')
    plt.scatter([0], [0], color='red', marker='*', s=200, zorder=5, label='Global Min (0, 0)')
    
    # Annotate orthogonality on first step
    if len(trajectory) > 2:
        plt.annotate('Orthogonal 90° Turn', 
                     xy=(trajectory[1, 0], trajectory[1, 1]), 
                     xytext=(trajectory[1, 0] + 0.8, trajectory[1, 1] + 0.5),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                     fontsize=9, fontweight='semibold')
    
    plt.title("02 - Gradient Descent with 2D Exact Line Search\nSuccessive Gradients are Orthogonal (d_k ⊥ d_{k+1})", fontsize=12)
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
    max_iterations = 25
    
    print("=== Algorithm 02: Gradient Descent with Exact Line Search ===")
    opt_x, opt_f, traj, loss_hist = gradient_descent_exact_line_search(
        objective_function, 
        gradient, 
        initial_point, 
        max_iter=max_iterations
    )
    
    print("-" * 68)
    print(f"Initial Point: {initial_point}")
    print(f"Optimized Point: [{opt_x[0]:.6f}, {opt_x[1]:.6f}]")
    print(f"Optimized Value: {opt_f:.6e}")
    print(f"Total Iterations: {len(traj)-1}")
    
    # Check orthogonality of first two step directions
    if len(traj) >= 3:
        step1 = traj[1] - traj[0]
        step2 = traj[2] - traj[1]
        dot_prod = np.dot(step1, step2)
        print(f"Dot product of step 1 and step 2: {dot_prod:.2e} (Orthogonal since ~ 0)")
        
    plot_contour(objective_function, traj, "plots/02_exact_line_search.png")
