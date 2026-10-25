# Objective: Check the Linear Independence Constraint Qualification.
# Input: Constraint functions and a candidate point.
# Output: LICQ satisfaction status.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def check_licq(constraint_funcs, constraint_grads, x_candidate, tol=1e-5):
    """
    Evaluates Linear Independence Constraint Qualification (LICQ) at candidate point x:
    
    1. Identify active constraints A(x) = {i : |c_i(x)| <= tol}
    2. Construct active constraint Jacobian matrix J_A = [grad c_i(x)^T for i in A(x)]
    3. Check if rank(J_A) == len(A(x)) (full row rank)
    """
    x = np.asarray(x_candidate, dtype=float)
    active_indices = []
    active_grads = []
    constraint_vals = []
    
    for idx, (c_func, c_grad) in enumerate(zip(constraint_funcs, constraint_grads)):
        val = c_func(x)
        constraint_vals.append(val)
        if abs(val) <= tol:
            active_indices.append(idx)
            g = np.asarray(c_grad(x), dtype=float)
            active_grads.append(g)
            
    m_active = len(active_indices)
    
    if m_active == 0:
        # No active constraints -> LICQ trivially satisfied
        return {
            "licq_satisfied": True,
            "m_active": 0,
            "active_indices": [],
            "rank": 0,
            "jacobian": np.empty((0, len(x))),
            "singular_values": np.array([]),
            "condition_number": 1.0,
            "status_message": "Trivially satisfied (no active constraints at candidate point)."
        }
        
    J_A = np.vstack(active_grads)  # shape (m_active, n)
    singular_values = np.linalg.svd(J_A, compute_uv=False)
    rank = int(np.linalg.matrix_rank(J_A, tol=tol))
    licq_satisfied = (rank == m_active)
    
    cond_num = float(singular_values[0] / singular_values[-1]) if (len(singular_values) > 0 and singular_values[-1] > 1e-12) else np.inf
    
    if licq_satisfied:
        status = f"LICQ SATISFIED: All {m_active} active constraint gradients are linearly independent (rank = {rank})."
    else:
        status = f"LICQ VIOLATED: Active gradients are linearly dependent (rank = {rank} < {m_active} active constraints)."
        
    return {
        "licq_satisfied": licq_satisfied,
        "m_active": m_active,
        "active_indices": active_indices,
        "rank": rank,
        "jacobian": J_A,
        "singular_values": singular_values,
        "condition_number": cond_num,
        "status_message": status
    }

def plot_licq_comparison(save_path="plots/23_licq_checker.png"):
    """
    Visualizes two pedagogical cases:
    Left: LICQ Satisfied (independent active constraint normals)
    Right: LICQ Violated (tangent constraints yielding linearly dependent gradients)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ---------------- Case A: LICQ Satisfied ----------------
    # Constraints: c1(x) = x1 - 1 = 0, c2(x) = x2 - 1 = 0
    pt_A = np.array([1.0, 1.0])
    
    x1_grid = np.linspace(-0.5, 2.5, 200)
    x2_grid = np.linspace(-0.5, 2.5, 200)
    
    ax1.axvline(1.0, color='crimson', linewidth=2, label=r'$c_1(x) = x_1 - 1 = 0$')
    ax1.axhline(1.0, color='dodgerblue', linewidth=2, label=r'$c_2(x) = x_2 - 1 = 0$')
    
    # Gradient arrows: [1, 0] and [0, 1]
    ax1.arrow(pt_A[0], pt_A[1], 0.6, 0.0, head_width=0.08, head_length=0.1, fc='crimson', ec='crimson', width=0.02)
    ax1.arrow(pt_A[0], pt_A[1], 0.0, 0.6, head_width=0.08, head_length=0.1, fc='dodgerblue', ec='dodgerblue', width=0.02)
    ax1.scatter([pt_A[0]], [pt_A[1]], color='darkgreen', s=140, zorder=5, label=r'Candidate $x^* = (1, 1)$')
    
    ax1.text(pt_A[0] + 0.65, pt_A[1], r'$\nabla c_1 = [1, 0]^T$', fontsize=10, color='crimson', verticalalignment='center')
    ax1.text(pt_A[0], pt_A[1] + 0.75, r'$\nabla c_2 = [0, 1]^T$', fontsize=10, color='dodgerblue', horizontalalignment='center')
    
    ax1.set_title("Case A: LICQ Satisfied\nOrthogonal Active Gradients (Rank = 2 = Active Count)", fontsize=11)
    ax1.set_xlabel("$x_1$", fontsize=10)
    ax1.set_ylabel("$x_2$", fontsize=10)
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")
    
    # ---------------- Case B: LICQ Violated ----------------
    # Tangent constraints: c1(x) = x2 - x1^2 = 0, c2(x) = -x2 = 0 at (0, 0)
    pt_B = np.array([0.0, 0.0])
    
    x1_vals = np.linspace(-1.5, 1.5, 300)
    x2_parabola = x1_vals**2
    
    ax2.plot(x1_vals, x2_parabola, color='crimson', linewidth=2, label=r'$c_1(x) = x_2 - x_1^2 = 0$')
    ax2.axhline(0.0, color='purple', linewidth=2, label=r'$c_2(x) = -x_2 = 0$')
    
    # Gradients at (0, 0): grad c1 = [0, 1], grad c2 = [0, -1] -> collinear!
    ax2.arrow(pt_B[0], pt_B[1], 0.0, 0.5, head_width=0.08, head_length=0.08, fc='crimson', ec='crimson', width=0.02)
    ax2.arrow(pt_B[0], pt_B[1], 0.0, -0.5, head_width=0.08, head_length=0.08, fc='purple', ec='purple', width=0.02)
    ax2.scatter([pt_B[0]], [pt_B[1]], color='darkred', marker='x', s=160, linewidth=3, zorder=5, label=r'Candidate $x^* = (0, 0)$')
    
    ax2.text(0.1, 0.45, r'$\nabla c_1(x^*) = [0, 1]^T$', fontsize=10, color='crimson')
    ax2.text(0.1, -0.45, r'$\nabla c_2(x^*) = [0, -1]^T$', fontsize=10, color='purple')
    
    ax2.set_title("Case B: LICQ Violated (Degenerate / Tangency)\nCollinear Active Gradients (Rank = 1 < 2 Active Constraints)", fontsize=11)
    ax2.set_xlabel("$x_1$", fontsize=10)
    ax2.set_ylabel("$x_2$", fontsize=10)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.0, 1.5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    print("=== Algorithm 23: LICQ Constraint Qualification Checker ===")
    
    # ---------------- TEST CASE 1: LICQ Satisfied ----------------
    print("\n--- Test Case 1: Independent Linear Constraints at (1, 1) ---")
    funcs_1 = [
        lambda x: x[0] - 1.0,           # c1(x) = x1 - 1
        lambda x: x[1] - 1.0            # c2(x) = x2 - 1
    ]
    grads_1 = [
        lambda x: np.array([1.0, 0.0]),
        lambda x: np.array([0.0, 1.0])
    ]
    cand_1 = [1.0, 1.0]
    res_1 = check_licq(funcs_1, grads_1, cand_1)
    print(f"Candidate Point: {cand_1}")
    print(f"Active Constraints: {res_1['active_indices']}")
    print(f"Active Jacobian J_A:\n{res_1['jacobian']}")
    print(f"Matrix Rank: {res_1['rank']} / {res_1['m_active']}")
    print(f"Singular Values: {np.round(res_1['singular_values'], 4)}")
    print(f"Status: {res_1['status_message']}")
    
    # ---------------- TEST CASE 2: LICQ Violated ----------------
    print("\n--- Test Case 2: Tangent Constraints at (0, 0) ---")
    funcs_2 = [
        lambda x: x[1] - x[0]**2,        # c1(x) = x2 - x1^2
        lambda x: -x[1]                  # c2(x) = -x2
    ]
    grads_2 = [
        lambda x: np.array([-2.0 * x[0], 1.0]),
        lambda x: np.array([0.0, -1.0])
    ]
    cand_2 = [0.0, 0.0]
    res_2 = check_licq(funcs_2, grads_2, cand_2)
    print(f"Candidate Point: {cand_2}")
    print(f"Active Constraints: {res_2['active_indices']}")
    print(f"Active Jacobian J_A:\n{res_2['jacobian']}")
    print(f"Matrix Rank: {res_2['rank']} / {res_2['m_active']}")
    print(f"Singular Values: {np.round(res_2['singular_values'], 4)}")
    print(f"Status: {res_2['status_message']}")
    
    plot_licq_comparison("plots/23_licq_checker.png")
