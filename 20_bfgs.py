# Objective: Optimize a function using the BFGS quasi-Newton method.
# Input: Function, gradient, and initial point.
# Output: Optimized point and objective value.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def rosenbrock_2d(x, a=1.0, b=10.0):
    """
    2D Rosenbrock function: f(x1, x2) = (a - x1)^2 + b * (x2 - x1^2)^2
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

def bfgs_optimization(func, grad_func, x0, c1=1e-4, rho=0.5, max_iter=50, tol=1e-8):
    """
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) Quasi-Newton algorithm:
    Iteratively estimates the inverse Hessian matrix H_k without requiring second derivatives.
    
    Update formula for inverse Hessian H_k:
    s_k = x_{k+1} - x_k
    y_k = grad(x_{k+1}) - grad(x_k)
    rho_k = 1 / (y_k^T * s_k)
    H_{k+1} = (I - rho_k * s_k * y_k^T) * H_k * (I - rho_k * y_k * s_k^T) + rho_k * s_k * s_k^T
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n, dtype=float)  # Initial inverse Hessian approximation
    
    trajectory = [x.copy()]
    history_f = [func(x)]
    
    print(f"{'Iter':<6}{'x1':<12}{'x2':<12}{'f(x)':<14}{'||grad||':<12}{'alpha':<10}")
    print("-" * 66)
    
    for i in range(max_iter):
        g = grad_func(x)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            print(f"{i:<6}{x[0]:<12.6f}{x[1]:<12.6f}{func(x):<14.6e}{g_norm:<12.6e}{'--':<10}")
            print(f"Converged at iteration {i} (gradient tolerance reached).")
            break
            
        # Quasi-Newton direction: d = - H * g
        d = - H @ g
        
        # Verify descent direction; reset to negative gradient if curvature condition fails
        directional_deriv = np.dot(g, d)
        if directional_deriv >= 0:
            H = np.eye(n, dtype=float)
            d = -g
            directional_deriv = -g_norm**2
            
        # Armijo backtracking line search
        alpha = 1.0
        f_curr = func(x)
        while func(x + alpha * d) > f_curr + c1 * alpha * directional_deriv:
            alpha *= rho
            if alpha < 1e-10:
                break
                
        print(f"{i:<6}{x[0]:<12.6f}{x[1]:<12.6f}{f_curr:<14.6e}{g_norm:<12.6e}{alpha:<10.4f}")
        
        # Step update
        s = alpha * d
        x_next = x + s
        g_next = grad_func(x_next)
        y = g_next - g
        
        # Update inverse Hessian via BFGS formula if curvature condition holds
        ys = np.dot(y, s)
        if ys > 1e-10:
            rho_k = 1.0 / ys
            I = np.eye(n)
            H = (I - rho_k * np.outer(s, y)) @ H @ (I - rho_k * np.outer(y, s)) + rho_k * np.outer(s, s)
            
        x = x_next
        trajectory.append(x.copy())
        history_f.append(func(x))
        
        if np.linalg.norm(s) < tol:
            print(f"Converged at iteration {i + 1} (step tolerance reached).")
            break
            
    return x, func(x), np.array(trajectory), np.array(history_f)

def plot_contour(func, trajectory, save_path="plots/20_bfgs.png"):
    """
    Plot 2D contour representation and BFGS quasi-Newton trajectory.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    x1_vals = np.linspace(-1.8, 1.8, 400)
    x2_vals = np.linspace(-0.5, 2.5, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = (1.0 - X1)**2 + 10.0 * (X2 - X1**2)**2
    
    plt.figure(figsize=(9, 6))
    levels = np.logspace(-1, 3.2, 30)
    cs = plt.contour(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.7)
    plt.clabel(cs, inline=1, fontsize=8, fmt="%.1f")
    plt.contourf(X1, X2, Z, levels=levels, cmap="Blues", alpha=0.15)
    
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=4, linewidth=1.6, label='BFGS Path')
    plt.scatter([trajectory[0, 0]], [trajectory[0, 1]], color='green', s=100, zorder=5, label='Start Point (x0)')
    plt.scatter([1.0], [1.0], color='red', marker='*', s=200, zorder=5, label='Global Min (1, 1)')
    
    for idx in range(0, len(trajectory), max(1, len(trajectory)//6)):
        pt = trajectory[idx]
        plt.annotate(f"{idx}", (pt[0], pt[1]), textcoords="offset points", xytext=(5, 5), fontsize=8, color='darkred')
        
    plt.title("20 - BFGS Quasi-Newton Function Optimization\nTrajectory on 2D Rosenbrock Valley (b=10)", fontsize=12)
    plt.xlabel("$x_1$", fontsize=11)
    plt.ylabel("$x_2$", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Contour plot saved to: {save_path}")

if __name__ == "__main__":
    initial_point = [-1.2, 1.0]
    
    print("=== Algorithm 20: BFGS Quasi-Newton Optimization ===")
    opt_x, opt_f, traj, loss_hist = bfgs_optimization(
        rosenbrock_2d,
        rosenbrock_gradient,
        initial_point,
        c1=1e-4,
        rho=0.5,
        max_iter=50
    )
    
    print("-" * 66)
    print(f"Initial Point: {initial_point}")
    print(f"Optimized Point: [{opt_x[0]:.6f}, {opt_x[1]:.6f}]")
    print(f"Optimized Value: {opt_f:.6e}")
    print(f"Total Iterations: {len(traj)-1}")
    
    plot_contour(rosenbrock_2d, traj, "plots/20_bfgs.png")
