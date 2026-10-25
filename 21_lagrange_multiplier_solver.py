# Objective: Solve equality-constrained optimization with Lagrange multipliers.
# Input: Objective, constraints, and initial values.
# Output: Candidate points and multipliers.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def objective(x):
    """
    Objective function: f(x1, x2) = 0.5 * x1^2 + 5 * x2^2
    """
    return 0.5 * x[0]**2 + 5.0 * x[1]**2

def objective_gradient(x):
    """
    Gradient of objective: grad f(x) = [x1, 10 * x2]
    """
    return np.array([x[0], 10.0 * x[1]], dtype=float)

def objective_hessian(x):
    """
    Hessian of objective: diag(1, 10)
    """
    return np.array([[1.0, 0.0], [0.0, 10.0]], dtype=float)

def constraint(x):
    """
    Equality constraint: g(x1, x2) = x1 + 2 * x2 - 3 = 0
    """
    return x[0] + 2.0 * x[1] - 3.0

def constraint_gradient(x):
    """
    Gradient of constraint: grad g(x) = [1, 2]
    """
    return np.array([1.0, 2.0], dtype=float)

def lagrange_multiplier_solver(obj_grad, obj_hess, constr, constr_grad, 
                               x0=[1.0, 1.0], lambda0=0.0, max_iter=20, tol=1e-8):
    """
    Solves the equality-constrained system using Newton-Raphson on the KKT/Lagrangian system:
    Lagrangian: L(x, lambda) = f(x) + lambda * g(x)
    Stationary equations F(z) = [grad_x L; g(x)] = [grad f(x) + lambda * grad g(x); g(x)] = 0
    KKT Jacobian:
    [ nabla^2_xx L    grad g(x) ] [ delta_x      ] = - [ grad_x L ]
    [ grad g(x)^T        0      ] [ delta_lambda ]     [ g(x)     ]
    """
    x = np.array(x0, dtype=float)
    lam = float(lambda0)
    
    trajectory = [x.copy()]
    history_lambda = [lam]
    
    print(f"{'Iter':<6}{'x1':<12}{'x2':<12}{'lambda':<14}{'g(x)':<14}{'||dL/dx||':<14}")
    print("-" * 72)
    
    for i in range(max_iter):
        g_val = constr(x)
        grad_f = obj_grad(x)
        grad_g = constr_grad(x)
        grad_L_x = grad_f + lam * grad_g
        
        norm_res = np.sqrt(np.sum(grad_L_x**2) + g_val**2)
        print(f"{i:<6}{x[0]:<12.6f}{x[1]:<12.6f}{lam:<14.6f}{g_val:<14.6e}{np.linalg.norm(grad_L_x):<14.6e}")
        
        if norm_res < tol:
            print(f"Converged at iteration {i} (optimality tolerance reached).")
            break
            
        # Build KKT matrix
        H_xx = obj_hess(x)  # assuming linear constraints (grad^2 g = 0)
        
        # Assemble 3x3 block system
        KKT = np.array([
            [H_xx[0, 0], H_xx[0, 1], grad_g[0]],
            [H_xx[1, 0], H_xx[1, 1], grad_g[1]],
            [grad_g[0],  grad_g[1],  0.0]
        ], dtype=float)
        
        rhs = -np.array([grad_L_x[0], grad_L_x[1], g_val], dtype=float)
        
        delta = np.linalg.solve(KKT, rhs)
        delta_x = delta[:2]
        delta_lam = delta[2]
        
        x = x + delta_x
        lam = lam + delta_lam
        
        trajectory.append(x.copy())
        history_lambda.append(lam)
        
        if np.linalg.norm(delta) < tol:
            print(f"Converged at iteration {i + 1} (step tolerance reached).")
            break
            
    return x, lam, np.array(trajectory), np.array(history_lambda)

def plot_lagrange_geometry(func, constr, x_opt, lam_opt, traj, save_path="plots/21_lagrange_multipliers.png"):
    """
    Plot 2D contour representation, constraint boundary, and gradient alignment at optimality.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    x1_vals = np.linspace(-1.0, 4.5, 400)
    x2_vals = np.linspace(-1.5, 2.5, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = 0.5 * X1**2 + 5.0 * X2**2
    
    plt.figure(figsize=(9, 6.5))
    levels = np.linspace(0.5, 20.0, 20)
    cs = plt.contour(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.7)
    plt.clabel(cs, inline=1, fontsize=8, fmt="%.1f")
    plt.contourf(X1, X2, Z, levels=levels, cmap="Blues", alpha=0.15)
    
    # Plot constraint line: x1 + 2*x2 = 3 => x2 = (3 - x1)/2
    x1_line = np.linspace(-1.0, 4.5, 200)
    x2_line = (3.0 - x1_line) / 2.0
    plt.plot(x1_line, x2_line, 'r-', linewidth=2.5, label='Constraint: $x_1 + 2x_2 = 3$')
    
    # Plot solver path
    plt.plot(traj[:, 0], traj[:, 1], 'yo--', markersize=5, linewidth=1.2, label='Newton-KKT Iterations')
    
    # Mark optimal point
    plt.scatter([x_opt[0]], [x_opt[1]], color='magenta', marker='*', s=260, zorder=6,
                label=f'Optimal Point $x^* = [{x_opt[0]:.3f}, {x_opt[1]:.3f}]$')
    
    # Quiver arrows: grad f and grad g showing collinearity: grad f = - lambda * grad g
    grad_f = objective_gradient(x_opt)
    grad_g = constraint_gradient(x_opt)
    scale = 0.15
    plt.arrow(x_opt[0], x_opt[1], scale * grad_f[0], scale * grad_f[1], 
              head_width=0.08, head_length=0.1, fc='blue', ec='blue', width=0.02,
              label=r'$\nabla f(x^*)$')
    plt.arrow(x_opt[0], x_opt[1], scale * (-lam_opt * grad_g[0]), scale * (-lam_opt * grad_g[1]), 
              head_width=0.08, head_length=0.1, fc='darkred', ec='darkred', width=0.015, linestyle=':',
              label=r'$-\lambda^* \nabla g(x^*)$')
    
    plt.title(f"21 - Lagrange Multiplier Equality-Constrained Solver\nOptimal $\\lambda^* = {lam_opt:.4f}$ (Gradients Collinear at $x^*$)", fontsize=11)
    plt.xlabel("$x_1$", fontsize=11)
    plt.ylabel("$x_2$", fontsize=11)
    plt.xlim(-1.0, 4.5)
    plt.ylim(-1.5, 2.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    initial_x = [0.0, 0.0]
    initial_lambda = 0.0
    
    print("=== Algorithm 21: Lagrange Multiplier Solver ===")
    opt_x, opt_lam, traj, lam_hist = lagrange_multiplier_solver(
        objective_gradient,
        objective_hessian,
        constraint,
        constraint_gradient,
        x0=initial_x,
        lambda0=initial_lambda
    )
    
    print("-" * 72)
    print(f"Optimal Point x*: [{opt_x[0]:.6f}, {opt_x[1]:.6f}]")
    print(f"Optimal Multiplier lambda*: {opt_lam:.6f}")
    print(f"Objective Value f(x*): {objective(opt_x):.6f}")
    print(f"Constraint Violation |g(x*)|: {abs(constraint(opt_x)):.2e}")
    
    plot_lagrange_geometry(objective, constraint, opt_x, opt_lam, traj, "plots/21_lagrange_multipliers.png")
