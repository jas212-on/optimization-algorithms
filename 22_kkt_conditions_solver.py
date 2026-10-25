# Objective: Solve constrained optimization using KKT conditions.
# Input: Objective, constraints, and initial values.
# Output: Candidate points and multipliers.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

def objective(x):
    """
    Objective function: f(x1, x2) = 0.5 * x1^2 + 5 * x2^2
    """
    return 0.5 * x[0]**2 + 5.0 * x[1]**2

def objective_gradient(x):
    """
    Gradient: grad f(x) = [x1, 10 * x2]
    """
    return np.array([x[0], 10.0 * x[1]], dtype=float)

def objective_hessian(x):
    """
    Hessian: diag(1, 10)
    """
    return np.array([[1.0, 0.0], [0.0, 10.0]], dtype=float)

# Linear inequality constraints g_i(x) <= 0
# Form: A * x <= b <=> A * x - b <= 0
A_ineq = np.array([
    [-1.0,  0.0],   # g1: -x1 + 1.5 <= 0  <=> x1 >= 1.5
    [ 0.0,  1.0],   # g2:  x2 - 0.5 <= 0  <=> x2 <= 0.5
    [ 0.0, -1.0]    # g3: -x2 - 1.0 <= 0  <=> x2 >= -1.0
], dtype=float)

b_ineq = np.array([-1.5, 0.5, 1.0], dtype=float)

def evaluate_constraints(x):
    """
    Evaluate g_i(x) = a_i^T * x - b_i <= 0
    """
    return A_ineq @ x - b_ineq

def solve_kkt_active_set(obj_grad, obj_hess, A, b):
    """
    Active-set KKT solver for convex QP with inequality constraints:
    Iterates through possible active sets W to find candidate points satisfying all 4 KKT conditions:
    1. Stationarity: grad f(x) + A_W^T * mu_W = 0
    2. Primal feasibility: A * x - b <= 0
    3. Dual feasibility: mu_W >= 0
    4. Complementary slackness: mu_i * (a_i^T * x - b_i) = 0
    """
    m, n = A.shape
    H = obj_hess(np.zeros(n))
    candidates = []
    
    print(f"Testing active constraint combinations (total constraints m = {m}):")
    print("-" * 72)
    
    for k in range(min(m, n) + 1):
        for active_indices in combinations(range(m), k):
            active_idx = list(active_indices)
            k_act = len(active_idx)
            
            if k_act == 0:
                # Unconstrained candidate: grad f(x) = 0 => x = H^-1 * 0
                x_cand = np.zeros(n)
                mu_cand = np.zeros(m)
            else:
                A_W = A[active_idx, :]
                b_W = b[active_idx]
                
                # Solve KKT equality block:
                # [ H    A_W^T ] [ x  ] = [ 0   ]
                # [ A_W    0   ] [ mu ]   [ b_W ]
                KKT_mat = np.zeros((n + k_act, n + k_act))
                KKT_mat[:n, :n] = H
                KKT_mat[:n, n:] = A_W.T
                KKT_mat[n:, :n] = A_W
                
                rhs = np.zeros(n + k_act)
                rhs[n:] = b_W
                
                try:
                    sol = np.linalg.solve(KKT_mat, rhs)
                    x_cand = sol[:n]
                    mu_active = sol[n:]
                    mu_cand = np.zeros(m)
                    for idx, val in zip(active_idx, mu_active):
                        mu_cand[idx] = val
                except np.linalg.LinAlgError:
                    continue
                    
            # Check KKT conditions
            g_vals = A @ x_cand - b
            primal_feasible = np.all(g_vals <= 1e-7)
            dual_feasible = np.all(mu_cand >= -1e-7)
            comp_slackness = np.all(np.abs(mu_cand * g_vals) <= 1e-7)
            
            grad_L = obj_grad(x_cand) + A.T @ mu_cand
            stationarity = np.linalg.norm(grad_L) <= 1e-7
            
            status = "Optimal (KKT Satisfied)" if (primal_feasible and dual_feasible and comp_slackness and stationarity) else "Infeasible / Non-optimal"
            print(f"Active Set {active_idx}: x = [{x_cand[0]:.3f}, {x_cand[1]:.3f}], mu = {np.round(mu_cand, 3)} -> {status}")
            
            if primal_feasible and dual_feasible and comp_slackness and stationarity:
                candidates.append((x_cand, mu_cand, active_idx))
                
    return candidates

def plot_kkt_solution(func, A, b, optimal_x, optimal_mu, active_idx, save_path="plots/22_kkt_conditions.png"):
    """
    Plot 2D contour representation, feasible polyhedron, and KKT gradient balance at optimality.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    x1_vals = np.linspace(-0.5, 4.0, 400)
    x2_vals = np.linspace(-2.0, 2.0, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = 0.5 * X1**2 + 5.0 * X2**2
    
    plt.figure(figsize=(9, 6.5))
    levels = np.linspace(0.2, 16.0, 25)
    cs = plt.contour(X1, X2, Z, levels=levels, cmap="viridis", alpha=0.7)
    plt.clabel(cs, inline=1, fontsize=8, fmt="%.1f")
    
    # Feasible region: x1 >= 1.5, -1.0 <= x2 <= 0.5
    feasible_mask = (X1 >= 1.5) & (X2 <= 0.5) & (X2 >= -1.0)
    plt.contourf(X1, X2, feasible_mask.astype(float), levels=[0.5, 1.5], colors=['limegreen'], alpha=0.25)
    
    # Draw constraint boundary lines
    plt.axvline(1.5, color='red', linewidth=2.0, label=r'$g_1(x) = 1.5 - x_1 \leq 0$ (Active)')
    plt.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, label=r'$g_2(x) = x_2 - 0.5 \leq 0$ (Inactive)')
    plt.axhline(-1.0, color='purple', linestyle='--', linewidth=1.5, label=r'$g_3(x) = -x_2 - 1.0 \leq 0$ (Inactive)')
    
    # Points
    plt.scatter([0.0], [0.0], color='blue', marker='o', s=100, zorder=5, label='Unconstrained Min $(0, 0)$')
    plt.scatter([optimal_x[0]], [optimal_x[1]], color='crimson', marker='*', s=280, zorder=6,
                label=f'KKT Optimal Point $x^* = [{optimal_x[0]:.2f}, {optimal_x[1]:.2f}]$')
    
    # Vector arrows: Stationarity nabla f(x*) + mu_1 * nabla g_1(x*) = 0
    grad_f = objective_gradient(optimal_x)
    scale = 0.25
    plt.arrow(optimal_x[0], optimal_x[1], scale * grad_f[0], scale * grad_f[1], 
              head_width=0.08, head_length=0.1, fc='blue', ec='blue', width=0.02,
              label=r'$\nabla f(x^*)$')
    plt.arrow(optimal_x[0], optimal_x[1], scale * (-optimal_mu[0] * A[0, 0]), scale * (-optimal_mu[0] * A[0, 1]), 
              head_width=0.08, head_length=0.1, fc='darkred', ec='darkred', width=0.015, linestyle=':',
              label=r'$-\mu_1^* \nabla g_1(x^*)$')
    
    plt.title("22 - Karush-Kuhn-Tucker (KKT) Constrained Optimization\nStationarity, Primal/Dual Feasibility, & Complementary Slackness", fontsize=11)
    plt.xlabel("$x_1$", fontsize=11)
    plt.ylabel("$x_2$", fontsize=11)
    plt.xlim(-0.5, 3.5)
    plt.ylim(-1.8, 1.8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    print("=== Algorithm 22: KKT Conditions Solver ===")
    candidates = solve_kkt_active_set(objective_gradient, objective_hessian, A_ineq, b_ineq)
    
    if candidates:
        opt_x, opt_mu, active_idx = candidates[0]
        print("-" * 72)
        print("Optimal Solution Verified:")
        print(f"Candidate Point x*: [{opt_x[0]:.6f}, {opt_x[1]:.6f}]")
        print(f"Lagrange Multipliers mu*: {opt_mu}")
        print(f"Active Constraints: {active_idx}")
        print(f"Objective Value f(x*): {objective(opt_x):.6f}")
        
        plot_kkt_solution(objective, A_ineq, b_ineq, opt_x, opt_mu, active_idx, "plots/22_kkt_conditions.png")
    else:
        print("No KKT point found.")
