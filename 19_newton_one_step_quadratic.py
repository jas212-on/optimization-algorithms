# Objective: Demonstrate one-step Newton convergence for a quadratic function.
# Input: Quadratic function, gradient, Hessian, and initial point.
# Output: Exact minimizer after one Newton step.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def quadratic_function(x, Q=np.array([[1.0, 0.0], [0.0, 10.0]]), b=np.array([0.0, 0.0]), c=0.0):
    """
    Quadratic objective function: f(x) = 0.5 * x^T * Q * x - b^T * x + c
    Matches the benchmark anisotropic bowl from Algorithms 01-07 when Q = diag(1, 10), b = [0, 0].
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * float(x.T @ Q @ x) - float(b.T @ x) + c

def quadratic_gradient(x, Q=np.array([[1.0, 0.0], [0.0, 10.0]]), b=np.array([0.0, 0.0])):
    """
    Exact gradient: grad f(x) = Q * x - b
    """
    x = np.asarray(x, dtype=float)
    return Q @ x - b

def quadratic_hessian(x, Q=np.array([[1.0, 0.0], [0.0, 10.0]])):
    """
    Exact Hessian: H = Q (constant everywhere for quadratic functions)
    """
    return Q.copy()

def newton_one_step_demonstration(func, grad_func, hess_func, x0):
    """
    Executes Newton's method on a quadratic function, showing exact convergence in 1 step.
    delta_x = - [nabla^2 f(x_0)]^{-1} * nabla f(x_0)
    x_1 = x_0 + delta_x = x^*
    """
    x0 = np.array(x0, dtype=float)
    g0 = grad_func(x0)
    H0 = hess_func(x0)
    f0 = func(x0)
    
    # Single Newton step
    delta_x = np.linalg.solve(H0, -g0)
    x1 = x0 + delta_x
    g1 = grad_func(x1)
    f1 = func(x1)
    
    trajectory = np.array([x0, x1])
    
    print(f"{'Step':<8}{'x1':<14}{'x2':<14}{'f(x)':<16}{'||grad||':<16}")
    print("-" * 68)
    print(f"{'0 (Init)':<8}{x0[0]:<14.6f}{x0[1]:<14.6f}{f0:<16.6f}{np.linalg.norm(g0):<16.6f}")
    print(f"{'1 (Step)':<8}{x1[0]:<14.6f}{x1[1]:<14.6f}{f1:<16.6e}{np.linalg.norm(g1):<16.6e}")
    print("-" * 68)
    print(f"Exact minimizer reached in exactly 1 Newton step!")
    print(f"Final gradient norm: {np.linalg.norm(g1):.2e}")
    
    return x1, f1, trajectory

def plot_one_step_convergence(func, trajectory, save_path="plots/19_newton_one_step_quadratic.png"):
    """
    Plot 2D contour representation illustrating the 1-step direct jump to the minimum.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    x1_vals = np.linspace(-6.0, 6.0, 400)
    x2_vals = np.linspace(-4.0, 4.0, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = 0.5 * (X1**2 + 10.0 * X2**2)
    
    plt.figure(figsize=(9, 6))
    levels = np.logspace(-1, 2.5, 25)
    cs = plt.contour(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.7)
    plt.clabel(cs, inline=1, fontsize=8, fmt="%.1f")
    plt.contourf(X1, X2, Z, levels=levels, cmap="Blues", alpha=0.18)
    
    x0, x1 = trajectory[0], trajectory[1]
    
    # Arrow showing the direct single Newton jump
    plt.annotate(
        "", xy=(x1[0], x1[1]), xytext=(x0[0], x0[1]),
        arrowprops=dict(facecolor="crimson", edgecolor="darkred", width=2.5, headwidth=10, shrink=0.05)
    )
    
    plt.scatter([x0[0]], [x0[1]], color='green', s=120, zorder=5, label=f'Start Point $x_0 = [{x0[0]:.1f}, {x0[1]:.1f}]$')
    plt.scatter([x1[0]], [x1[1]], color='red', marker='*', s=260, zorder=6, label=f'Newton Minimizer $x_1 = [{x1[0]:.1f}, {x1[1]:.1f}]$')
    
    plt.title(r"19 - Newton's Method One-Step Convergence on Quadratic Function" + "\n" + r"Direct Jump in Exactly 1 Step: $\Delta x = -Q^{-1} \nabla f(x_0)$", fontsize=11)
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
    
    print("=== Algorithm 19: One-Step Newton Convergence for Quadratic Function ===")
    opt_x, opt_f, traj = newton_one_step_demonstration(
        quadratic_function,
        quadratic_gradient,
        quadratic_hessian,
        initial_point
    )
    
    print(f"Initial Point: {initial_point}")
    print(f"Optimized Point: [{opt_x[0]:.6f}, {opt_x[1]:.6f}]")
    print(f"Optimized Value: {opt_f:.6e}")
    
    plot_one_step_convergence(quadratic_function, traj, "plots/19_newton_one_step_quadratic.png")
