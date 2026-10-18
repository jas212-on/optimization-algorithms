# Objective: Find a step size satisfying the Armijo condition.
# Input: Function, gradient, point, direction, and rule parameters.
# Output: Armijo-valid step size.

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

def armijo_backtracking(func, grad_func, x, direction, alpha_init=1.0, c=0.1, rho=0.5, max_backtrack=50):
    """
    Finds a step size alpha that satisfies the Armijo condition:
        f(x + alpha * d) <= f(x) + c * alpha * grad(f)^T d
        
    Parameters:
    - func: Objective function f(x)
    - grad_func: Gradient function grad f(x)
    - x: Current point
    - direction: Search direction d
    - alpha_init: Initial step size
    - c: Armijo parameter in (0, 1), usually 1e-4 to 0.1
    - rho: Contraction factor in (0, 1), usually 0.5
    - max_backtrack: Maximum number of shrink iterations
    
    Returns:
    - alpha: Valid step size satisfying Armijo condition
    - history: List of (alpha, f(x + alpha*d), armijo_bound, satisfied_bool)
    """
    alpha = alpha_init
    f_x = func(x)
    g = grad_func(x)
    directional_derivative = np.dot(g, direction)
    
    history = []
    
    for i in range(max_backtrack):
        x_next = x + alpha * direction
        f_next = func(x_next)
        armijo_bound = f_x + c * alpha * directional_derivative
        
        satisfied = (f_next <= armijo_bound)
        history.append((alpha, f_next, armijo_bound, satisfied))
        
        if satisfied:
            return alpha, history
            
        alpha *= rho
        
    return alpha, history

def plot_armijo_rule(func, grad_func, x0, direction, history, save_path="plots/04_backtracking_armijo_rule.png"):
    """
    Visualize the Armijo condition along the 1D search direction and on the 2D contour map.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Panel 1: 1D Armijo Condition Plot ---
    f_x0 = func(x0)
    g0 = grad_func(x0)
    dir_deriv = np.dot(g0, direction)
    c_val = 0.1
    
    alphas_eval = np.linspace(0, 1.2, 250)
    phi_eval = [func(x0 + a * direction) for a in alphas_eval]
    armijo_line = [f_x0 + c_val * a * dir_deriv for a in alphas_eval]
    tangent_line = [f_x0 + a * dir_deriv for a in alphas_eval]
    
    ax1.plot(alphas_eval, phi_eval, 'b-', linewidth=2, label=r'$\phi(\alpha) = f(x_0 + \alpha d)$')
    ax1.plot(alphas_eval, armijo_line, 'r--', linewidth=2, label=r'Armijo Bound: $f(x_0) + c \alpha \nabla f(x_0)^T d$')
    ax1.plot(alphas_eval, tangent_line, 'g:', linewidth=1.5, label=r'Tangent Line ($c=1.0$)')
    
    # Highlight tested step sizes
    for a, f_val, bound, sat in history:
        if sat:
            ax1.scatter([a], [f_val], color='green', s=100, zorder=6, 
                        label=f'Accepted $\\alpha={a:.4f}$ (Satisfied)')
        else:
            ax1.scatter([a], [f_val], color='red', marker='x', s=80, zorder=5, 
                        label=f'Rejected $\\alpha={a:.4f}$ (Violated)')
            
    ax1.set_title("1D Armijo Sufficient Decrease Condition", fontsize=11)
    ax1.set_xlabel(r"Step size $\alpha$", fontsize=10)
    ax1.set_ylabel(r"$f(x_0 + \alpha d)$", fontsize=10)
    ax1.set_ylim(-5, max(phi_eval) * 0.7)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)
    
    # --- Panel 2: 2D Contour Map with Search Direction ---
    x1_vals = np.linspace(-5.5, 5.5, 400)
    x2_vals = np.linspace(-3.5, 3.5, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = 0.5 * (X1**2 + 10.0 * X2**2)
    
    levels = np.logspace(-1, 2.5, 25)
    cs = ax2.contour(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.7)
    ax2.clabel(cs, inline=1, fontsize=8, fmt="%.1f")
    ax2.contourf(X1, X2, Z, levels=levels, cmap="Blues", alpha=0.2)
    
    # Plot start and global min
    ax2.scatter([x0[0]], [x0[1]], color='purple', s=120, zorder=6, label=r'Current Point $x_0$')
    ax2.scatter([0], [0], color='gold', marker='*', s=200, zorder=6, label='Global Min (0, 0)')
    
    # Plot search direction ray
    ray_pts = np.array([x0 + a * direction for a in np.linspace(0, 0.4, 50)])
    ax2.plot(ray_pts[:, 0], ray_pts[:, 1], 'k--', alpha=0.6, label='Search Direction Ray $d$')
    
    # Plot candidate steps along the ray
    for a, _, _, sat in history:
        pt = x0 + a * direction
        if sat:
            ax2.scatter([pt[0]], [pt[1]], color='green', s=100, zorder=7, label=f'Accepted $x_0 + {a:.3f}d$')
            # Draw arrow from x0 to accepted point
            ax2.annotate('', xy=pt, xytext=x0,
                         arrowprops=dict(arrowstyle="->", color="green", lw=2))
        else:
            ax2.scatter([pt[0]], [pt[1]], color='red', marker='x', s=70, zorder=6)
            
    ax2.set_title("2D Contour Representation of Armijo Step Selection", fontsize=11)
    ax2.set_xlabel("$x_1$", fontsize=10)
    ax2.set_ylabel("$x_2$", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Armijo rule plot saved to: {save_path}")

if __name__ == "__main__":
    initial_point = np.array([5.0, 3.0])
    g0 = gradient(initial_point)
    descent_direction = -g0
    c_param = 0.1
    rho_param = 0.5
    
    print("=== Algorithm 04: Backtracking with Armijo Rule ===")
    print(f"Point x0: {initial_point}")
    print(f"Direction d: {descent_direction}")
    print(f"Rule Parameters: c = {c_param}, rho = {rho_param}")
    print("-" * 65)
    print(f"{'Step':<6}{'alpha':<10}{'f(x + alpha*d)':<18}{'Armijo Bound':<18}{'Satisfied?':<12}")
    print("-" * 65)
    
    valid_alpha, search_history = armijo_backtracking(
        objective_function, 
        gradient, 
        initial_point, 
        descent_direction, 
        alpha_init=1.0, 
        c=c_param, 
        rho=rho_param
    )
    
    for i, (a, f_val, bound, sat) in enumerate(search_history):
        print(f"{i:<6}{a:<10.4f}{f_val:<18.4f}{bound:<18.4f}{str(sat):<12}")
        
    print("-" * 65)
    print(f"Armijo-Valid Step Size: {valid_alpha:.6f}")
    print(f"New Point: {initial_point + valid_alpha * descent_direction}")
    print(f"Objective Value: {objective_function(initial_point + valid_alpha * descent_direction):.6f}")
    
    plot_armijo_rule(
        objective_function, 
        gradient, 
        initial_point, 
        descent_direction, 
        search_history, 
        "plots/04_backtracking_armijo_rule.png"
    )
