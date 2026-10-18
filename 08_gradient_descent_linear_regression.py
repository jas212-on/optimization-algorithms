# Objective: Train a linear regression model using gradient descent.
# Input: Features, targets, learning rate, and iteration settings.
# Output: Model parameters and loss.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_synthetic_data(n_samples=50, true_w=2.5, true_b=1.0, noise_std=0.8, random_state=42):
    """
    Generate synthetic 1D regression dataset: y = true_w * x + true_b + noise.
    """
    np.random.seed(random_state)
    x = np.random.uniform(-2, 3, size=(n_samples, 1))
    noise = np.random.normal(0, noise_std, size=(n_samples, 1))
    y = true_w * x + true_b + noise
    return x, y

def compute_loss(X_b, y, theta):
    """
    Mean Squared Error (MSE) loss: J(theta) = 1/(2m) * ||X_b * theta - y||^2
    """
    m = len(y)
    predictions = X_b @ theta
    errors = predictions - y
    return (1.0 / (2.0 * m)) * np.sum(errors ** 2)

def compute_gradient(X_b, y, theta):
    """
    Gradient of MSE with respect to theta: grad = 1/m * X_b^T * (X_b * theta - y)
    """
    m = len(y)
    predictions = X_b @ theta
    errors = predictions - y
    return (1.0 / m) * (X_b.T @ errors)

def train_linear_regression_gd(X, y, learning_rate=0.1, max_iter=60, tol=1e-6):
    """
    Train Linear Regression using Gradient Descent on parameters theta = [w, b]^T.
    
    Inputs:
    - X: Feature array of shape (m, 1)
    - y: Target array of shape (m, 1)
    - learning_rate: Step size alpha
    - max_iter: Maximum iterations
    
    Outputs:
    - theta: Optimized parameter vector [w, b]
    - final_loss: MSE loss at convergence
    - trajectory: History of theta vectors
    - loss_history: History of MSE loss
    """
    m = len(y)
    # Add intercept column (bias feature x0 = 1)
    X_b = np.hstack([X, np.ones((m, 1))])
    
    # Initialize parameters theta = [w, b]^T far from optimum
    theta = np.array([[-1.5], [-2.0]], dtype=float)
    trajectory = [theta.flatten().copy()]
    loss_history = [compute_loss(X_b, y, theta)]
    
    # Analytical solution (Normal Equation) for verification: theta* = (X_b^T X_b)^(-1) X_b^T y
    theta_star = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    print(f"{'Iter':<6}{'Weight (w)':<14}{'Bias (b)':<14}{'MSE Loss':<14}{'||grad||':<12}")
    print("-" * 62)
    
    for i in range(max_iter):
        g = compute_gradient(X_b, y, theta)
        g_norm = np.linalg.norm(g)
        
        if i % 10 == 0 or i == max_iter - 1:
            print(f"{i:<6}{theta[0, 0]:<14.4f}{theta[1, 0]:<14.4f}{loss_history[-1]:<14.6f}{g_norm:<12.6f}")
            
        if g_norm < tol:
            print(f"Converged at iteration {i} (tolerance reached).")
            break
            
        theta = theta - learning_rate * g
        trajectory.append(theta.flatten().copy())
        loss_history.append(compute_loss(X_b, y, theta))
        
    return theta.flatten(), loss_history[-1], np.array(trajectory), np.array(loss_history), theta_star.flatten(), X_b

def plot_linear_regression_contour(X, y, X_b, trajectory, theta_star, save_path="plots/08_linear_regression_gd.png"):
    """
    Plot 2D contour representation of the parameter loss surface J(w, b) alongside the fitted line.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Panel 1: Parameter Space Contour Plot J(w, b) ---
    w_vals = np.linspace(-2.5, 4.5, 200)
    b_vals = np.linspace(-3.0, 3.5, 200)
    W, B = np.meshgrid(w_vals, b_vals)
    Z = np.zeros_like(W)
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            th = np.array([[W[i, j]], [B[i, j]]])
            Z[i, j] = compute_loss(X_b, y, th)
            
    levels = np.logspace(np.log10(Z.min() + 1e-2), np.log10(Z.max()), 25)
    cs = ax1.contour(W, B, Z, levels=levels, cmap="viridis", alpha=0.7)
    ax1.clabel(cs, inline=1, fontsize=8, fmt="%.1f")
    ax1.contourf(W, B, Z, levels=levels, cmap="Blues", alpha=0.2)
    
    # Trajectory of (w, b)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=4, linewidth=1.5, label='GD Trajectory (w, b)')
    ax1.scatter([trajectory[0, 0]], [trajectory[0, 1]], color='green', s=100, zorder=5, label=f'Init: ({trajectory[0,0]:.1f}, {trajectory[0,1]:.1f})')
    ax1.scatter([theta_star[0]], [theta_star[1]], color='gold', marker='*', s=200, zorder=6, 
                label=f'Optimal: ({theta_star[0]:.2f}, {theta_star[1]:.2f})')
    
    ax1.set_title("08 - Loss Surface Contours $J(w, b)$ & Optimization Path", fontsize=11)
    ax1.set_xlabel("Weight $w$", fontsize=10)
    ax1.set_ylabel("Bias (Intercept) $b$", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8)
    
    # --- Panel 2: Fitted Regression Line in Data Space ---
    ax2.scatter(X, y, color='dodgerblue', alpha=0.7, edgecolors='k', label='Observed Data')
    
    x_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 100)
    
    # Plot initial and a couple intermediate fits
    ax2.plot(x_line, trajectory[0, 0] * x_line + trajectory[0, 1], 'g--', alpha=0.5, label='Initial Model')
    if len(trajectory) > 5:
        mid_idx = len(trajectory) // 4
        ax2.plot(x_line, trajectory[mid_idx, 0] * x_line + trajectory[mid_idx, 1], 'orange', linestyle=':', alpha=0.7, label=f'Iter {mid_idx}')
        
    final_w, final_b = trajectory[-1, 0], trajectory[-1, 1]
    ax2.plot(x_line, final_w * x_line + final_b, 'r-', linewidth=2, label=f'Final Model: y = {final_w:.2f}x + {final_b:.2f}')
    
    ax2.set_title("Fitted Linear Regression Model", fontsize=11)
    ax2.set_xlabel("Feature $x$", fontsize=10)
    ax2.set_ylabel("Target $y$", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Linear regression contour plot saved to: {save_path}")

if __name__ == "__main__":
    print("=== Algorithm 08: Gradient Descent Linear Regression ===")
    X, y = generate_synthetic_data(n_samples=50, true_w=2.5, true_b=1.0)
    
    alpha = 0.1
    iterations = 60
    
    opt_params, opt_loss, traj, loss_hist, theta_exact, X_b = train_linear_regression_gd(
        X, y, learning_rate=alpha, max_iter=iterations
    )
    
    print("-" * 62)
    print(f"Optimal Parameters (Normal Eq): w = {theta_exact[0]:.4f}, b = {theta_exact[1]:.4f}")
    print(f"GD Estimated Parameters:       w = {opt_params[0]:.4f}, b = {opt_params[1]:.4f}")
    print(f"Final MSE Loss:                 {opt_loss:.6f}")
    print(f"Total Iterations:               {len(traj)-1}")
    
    plot_linear_regression_contour(X, y, X_b, traj, theta_exact, "plots/08_linear_regression_gd.png")
