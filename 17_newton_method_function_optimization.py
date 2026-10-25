# Objective: Optimize a function using Newton's method.
# Input: Function, gradient, Hessian, and initial point.
# Output: Optimized point and objective value.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def rosenbrock_2d(x, a=1.0, b=10.0):
    """
    2D Rosenbrock function: f(x1, x2) = (a - x1)^2 + b * (x2 - x1^2)^2
    A classic non-linear, non-quadratic optimization benchmark.
    Global minimum is at x* = (a, a^2) with f(x*) = 0.
    """
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x, a=1.0, b=10.0):
    """
    Analytical gradient of 2D Rosenbrock function.
    """
    df_dx1 = -2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0]**2)
    df_dx2 = 2.0 * b * (x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2], dtype=float)

def rosenbrock_hessian(x, a=1.0, b=10.0):
    """
    Analytical Hessian matrix of 2D Rosenbrock function.
    """
    d2f_dx1_2 = 2.0 - 4.0 * b * x[1] + 12.0 * b * x[0]**2
    d2f_dx1_dx2 = -4.0 * b * x[0]
    d2f_dx2_2 = 2.0 * b
    return np.array([
        [d2f_dx1_2, d2f_dx1_dx2],
        [d2f_dx1_dx2, d2f_dx2_2]
    ], dtype=float)

def newtons_method(func, grad_func, hess_func, x0, max_iter=20, tol=1e-8):
    """
    Standard unconstrained Newton's optimization method:
    Solve H(x_k) * delta_x = -grad(x_k)
    Update: x_{k+1} = x_k + delta_x
    """
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    history_f = [func(x)]
    
    print(f"{'Iter':<6}{'x1':<12}{'x2':<12}{'f(x)':<14}{'||grad||':<12}{'||step||':<12}")
    print("-" * 68)
    
    for i in range(max_iter):
        g = grad_func(x)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            print(f"{i:<6}{x[0]:<12.6f}{x[1]:<12.6f}{func(x):<14.6e}{g_norm:<12.6e}{0.0:<12.6e}")
            print(f"Converged at iteration {i} (gradient tolerance reached).")
            break
            
        H = hess_func(x)
        
        # Solve H * delta_x = -g
        try:
            delta_x = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            # Fallback if Hessian is singular
            delta_x = -np.linalg.pinv(H) @ g
            
        step_norm = np.linalg.norm(delta_x)
        print(f"{i:<6}{x[0]:<12.6f}{x[1]:<12.6f}{func(x):<14.6e}{g_norm:<12.6e}{step_norm:<12.6e}")
        
        x = x + delta_x
        trajectory.append(x.copy())
        history_f.append(func(x))
        
        if step_norm < tol:
            print(f"Converged at iteration {i + 1} (step size tolerance reached).")
            break
            
    return x, func(x), np.array(trajectory), np.array(history_f)

def plot_contour(func, trajectory, save_path="plots/17_newton_method.png"):
    """
    Plot 2D contour representation and Newton optimization trajectory.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    x1_vals = np.linspace(-1.5, 2.0, 400)
    x2_vals = np.linspace(-0.5, 2.5, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = (1.0 - X1)**2 + 10.0 * (X2 - X1**2)**2
    
    plt.figure(figsize=(9, 6))
    levels = np.logspace(-1, 3, 30)
    cs = plt.contour(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.7)
    plt.clabel(cs, inline=1, fontsize=8, fmt="%.1f")
    plt.contourf(X1, X2, Z, levels=levels, cmap="Blues", alpha=0.15)
    
    # Plot trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=5, linewidth=1.8, label="Newton's Path")
    plt.scatter([trajectory[0, 0]], [trajectory[0, 1]], color='green', s=100, zorder=5, label='Start Point (x0)')
    plt.scatter([1.0], [1.0], color='red', marker='*', s=200, zorder=5, label='Global Min (1, 1)')
    
    for idx, pt in enumerate(trajectory):
        plt.annotate(f"{idx}", (pt[0], pt[1]), textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight='bold', color='darkred')
        
    plt.title("17 - Newton's Method Function Optimization\nTrajectory on 2D Rosenbrock Valley (b=10)", fontsize=12)
    plt.xlabel("$x_1$", fontsize=11)
    plt.ylabel("$x_2$", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Contour plot saved to: {save_path}")

if __name__ == "__main__":
    initial_point = [-0.5, 1.5]
    
    print("=== Algorithm 17: Newton's Method Function Optimization ===")
    opt_x, opt_f, traj, loss_hist = newtons_method(
        rosenbrock_2d,
        rosenbrock_gradient,
        rosenbrock_hessian,
        initial_point,
        max_iter=20
    )
    
    print("-" * 68)
    print(f"Initial Point: {initial_point}")
    print(f"Optimized Point: [{opt_x[0]:.6f}, {opt_x[1]:.6f}]")
    print(f"Optimized Value: {opt_f:.6e}")
    print(f"Total Iterations: {len(traj)-1}")
    
    plot_contour(rosenbrock_2d, traj, "plots/17_newton_method.png")
