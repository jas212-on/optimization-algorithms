# Objective: Optimize a function using damped Newton's method.
# Input: Function, gradient, Hessian, initial point, and damping settings.
# Output: Optimized point and objective value.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def rosenbrock_2d(x, a=1.0, b=10.0):
    """
    2D Rosenbrock function: f(x1, x2) = (a - x1)^2 + b * (x2 - x1^2)^2
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

def damped_newtons_method(func, grad_func, hess_func, x0, 
                          c=1e-4, rho=0.5, max_iter=30, tol=1e-8):
    """
    Damped Newton's optimization method with Armijo backtracking line search:
    1. Compute search direction d_k = - [H_reg(x_k)]^{-1} grad(x_k)
    2. Backtracking line search for alpha in (0, 1] satisfying:
       f(x_k + alpha * d_k) <= f(x_k) + c * alpha * grad(x_k)^T * d_k
    3. Update: x_{k+1} = x_k + alpha_k * d_k
    """
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    history_f = [func(x)]
    history_alpha = []
    
    print(f"{'Iter':<6}{'x1':<12}{'x2':<12}{'f(x)':<14}{'||grad||':<12}{'alpha':<10}")
    print("-" * 66)
    
    for i in range(max_iter):
        g = grad_func(x)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            print(f"{i:<6}{x[0]:<12.6f}{x[1]:<12.6f}{func(x):<14.6e}{g_norm:<12.6e}{'--':<10}")
            print(f"Converged at iteration {i} (gradient tolerance reached).")
            break
            
        H = hess_func(x)
        
        # Ensure positive-definiteness via spectral shift regularization if needed
        eigvals = np.linalg.eigvalsh(H)
        min_eig = np.min(eigvals)
        if min_eig <= 1e-4:
            reg = (1e-4 - min_eig) + 1e-3
            H = H + reg * np.eye(len(x))
            
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            d = -np.linalg.pinv(H) @ g
            
        # Verify descent direction
        directional_deriv = np.dot(g, d)
        if directional_deriv >= 0:
            d = -g
            directional_deriv = -g_norm**2
            
        # Armijo backtracking line search
        alpha = 1.0
        f_curr = func(x)
        while func(x + alpha * d) > f_curr + c * alpha * directional_deriv:
            alpha *= rho
            if alpha < 1e-8:
                break
                
        print(f"{i:<6}{x[0]:<12.6f}{x[1]:<12.6f}{f_curr:<14.6e}{g_norm:<12.6e}{alpha:<10.4f}")
        
        history_alpha.append(alpha)
        x = x + alpha * d
        trajectory.append(x.copy())
        history_f.append(func(x))
        
        if np.linalg.norm(alpha * d) < tol:
            print(f"Converged at iteration {i + 1} (step tolerance reached).")
            break
            
    return x, func(x), np.array(trajectory), np.array(history_f), np.array(history_alpha)

def plot_contour(func, trajectory, history_alpha, save_path="plots/18_damped_newton_method.png"):
    """
    Plot 2D contour trajectory alongside step-size alpha progression.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # --- Panel 1: 2D Contour Trajectory ---
    x1_vals = np.linspace(-1.8, 1.8, 400)
    x2_vals = np.linspace(-0.5, 2.8, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = (1.0 - X1)**2 + 10.0 * (X2 - X1**2)**2
    
    levels = np.logspace(-1, 3.2, 30)
    cs = ax1.contour(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.7)
    ax1.clabel(cs, inline=1, fontsize=8, fmt="%.1f")
    ax1.contourf(X1, X2, Z, levels=levels, cmap="Blues", alpha=0.15)
    
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=4, linewidth=1.6, label="Damped Newton Path")
    ax1.scatter([trajectory[0, 0]], [trajectory[0, 1]], color='green', s=90, zorder=5, label='Start (x0)')
    ax1.scatter([1.0], [1.0], color='red', marker='*', s=180, zorder=5, label='Global Min (1, 1)')
    
    ax1.set_title("Damped Newton Trajectory on Rosenbrock", fontsize=11)
    ax1.set_xlabel("$x_1$", fontsize=10)
    ax1.set_ylabel("$x_2$", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")
    
    # --- Panel 2: Step Size (alpha) Backtracking History ---
    iters = np.arange(len(history_alpha))
    ax2.plot(iters, history_alpha, 'bo-', linewidth=1.5, markersize=5, label=r"Damping factor $\alpha_k$")
    ax2.axhline(1.0, color='r', linestyle='--', alpha=0.6, label=r"Pure Newton ($\alpha = 1$)")
    ax2.set_title(r"Damping Factor ($\alpha_k$) per Iteration", fontsize=11)
    ax2.set_xlabel("Iteration", fontsize=10)
    ax2.set_ylabel(r"Step size $\alpha$", fontsize=10)
    ax2.set_ylim(-0.05, 1.1)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    initial_point = [-1.5, 2.0]
    
    print("=== Algorithm 18: Damped Newton's Method ===")
    opt_x, opt_f, traj, loss_hist, alpha_hist = damped_newtons_method(
        rosenbrock_2d,
        rosenbrock_gradient,
        rosenbrock_hessian,
        initial_point,
        c=1e-4,
        rho=0.5,
        max_iter=30
    )
    
    print("-" * 66)
    print(f"Initial Point: {initial_point}")
    print(f"Optimized Point: [{opt_x[0]:.6f}, {opt_x[1]:.6f}]")
    print(f"Optimized Value: {opt_f:.6e}")
    print(f"Total Iterations: {len(traj)-1}")
    
    plot_contour(rosenbrock_2d, traj, alpha_hist, "plots/18_damped_newton_method.png")
