# Objective: Train a linear regression model using mini-batch SGD.
# Input: Features, targets, batch size, learning rate, and iterations.
# Output: Model parameters and loss.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_synthetic_data(n_samples=50, true_w=2.5, true_b=1.0, noise_std=0.8, random_state=42):
    """
    Generate synthetic 1D regression dataset: y = true_w * x + true_b + noise.
    Matches the benchmark dataset in Algorithms 08 and 09.
    """
    np.random.seed(random_state)
    x = np.random.uniform(-2, 3, size=(n_samples, 1))
    noise = np.random.normal(0, noise_std, size=(n_samples, 1))
    y = true_w * x + true_b + noise
    return x, y

def compute_loss(X_b, y, theta):
    """
    Mean Squared Error (MSE) loss across all samples:
    J(theta) = 1/(2m) * ||X_b * theta - y||^2
    """
    m = len(y)
    predictions = X_b @ theta
    errors = predictions - y
    return (1.0 / (2.0 * m)) * np.sum(errors ** 2)

def compute_minibatch_gradient(X_batch, y_batch, theta):
    """
    Compute gradient of MSE loss on a mini-batch of samples B:
    grad_B = 1/|B| * X_B^T * (X_B * theta - y_B)
    
    Inputs:
    - X_batch: Feature matrix of shape (|B|, 2)
    - y_batch: Target vector of shape (|B|, 1)
    - theta: Parameter vector of shape (2, 1)
    
    Output:
    - grad_B: Mini-batch gradient vector of shape (2, 1)
    """
    b_size = len(y_batch)
    predictions = X_batch @ theta
    errors = predictions - y_batch
    return (1.0 / b_size) * (X_batch.T @ errors)

def compute_batch_gd_baseline(X_b, y, theta_init, learning_rate=0.1, max_iter=60):
    """
    Compute full-batch Gradient Descent trajectory for baseline comparison.
    """
    theta = theta_init.copy()
    traj = [theta.flatten().copy()]
    m = len(y)
    for _ in range(max_iter):
        g = (1.0 / m) * (X_b.T @ (X_b @ theta - y))
        theta = theta - learning_rate * g
        traj.append(theta.flatten().copy())
    return np.array(traj)

def train_linear_regression_minibatch_sgd(X, y, batch_size=10, initial_learning_rate=0.08, 
                                          n_epochs=30, lr_decay=0.005, random_state=42):
    """
    Train Linear Regression using Mini-Batch Stochastic Gradient Descent on theta = [w, b]^T.
    
    Partitions the dataset into small subsets (mini-batches of size B), balancing the
    high-variance noise of single-sample SGD with the computational overhead of full Batch GD.
    
    Learning rate schedule:
        alpha_t = alpha_0 / (1 + decay * t)
    
    Inputs:
    - X: Feature array of shape (m, 1)
    - y: Target array of shape (m, 1)
    - batch_size: Mini-batch size B (1 < B < m)
    - initial_learning_rate: Base step size alpha_0
    - n_epochs: Number of complete passes over the training dataset
    - lr_decay: Step-size decay factor
    - random_state: Random seed for shuffling
    
    Outputs:
    - theta: Final optimized parameter vector [w, b]
    - final_loss: Final full-dataset MSE loss
    - step_trajectory: History of theta vectors at every mini-batch update
    - epoch_trajectory: History of theta vectors at the end of each epoch
    - epoch_loss_history: History of full MSE loss after each epoch
    - theta_star: Analytical solution via Normal Equation
    - X_b: Design matrix with intercept column
    """
    np.random.seed(random_state)
    m = len(y)
    X_b = np.hstack([X, np.ones((m, 1))])
    
    # Initialize parameters theta = [w, b]^T far from optimum (matching Algs 08 and 09)
    theta = np.array([[-1.5], [-2.0]], dtype=float)
    step_trajectory = [theta.flatten().copy()]
    epoch_trajectory = [theta.flatten().copy()]
    epoch_loss_history = [compute_loss(X_b, y, theta)]
    
    # Analytical solution (Normal Equation) for verification: theta* = (X_b^T X_b)^(-1) X_b^T y
    theta_star = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    total_steps = 0
    num_batches_per_epoch = int(np.ceil(m / batch_size))
    
    print(f"{'Epoch':<8}{'Step':<8}{'Weight (w)':<14}{'Bias (b)':<14}{'MSE Loss':<14}{'Learning Rate':<14}")
    print("-" * 72)
    print(f"{'Init':<8}{0:<8}{theta[0, 0]:<14.4f}{theta[1, 0]:<14.4f}{epoch_loss_history[0]:<14.6f}{initial_learning_rate:<14.6f}")
    
    for epoch in range(1, n_epochs + 1):
        # Randomly shuffle data indices at each epoch
        indices = np.random.permutation(m)
        
        for start_idx in range(0, m, batch_size):
            total_steps += 1
            batch_idx = indices[start_idx:start_idx + batch_size]
            X_batch = X_b[batch_idx]
            y_batch = y[batch_idx]
            
            # Decaying learning rate
            lr_t = initial_learning_rate / (1.0 + lr_decay * total_steps)
            
            # Mini-batch gradient
            g_b = compute_minibatch_gradient(X_batch, y_batch, theta)
            
            # Parameter update
            theta = theta - lr_t * g_b
            step_trajectory.append(theta.flatten().copy())
            
        epoch_loss = compute_loss(X_b, y, theta)
        epoch_trajectory.append(theta.flatten().copy())
        epoch_loss_history.append(epoch_loss)
        
        if epoch % 5 == 0 or epoch == n_epochs:
            current_lr = initial_learning_rate / (1.0 + lr_decay * total_steps)
            print(f"{epoch:<8}{total_steps:<8}{theta[0, 0]:<14.4f}{theta[1, 0]:<14.4f}{epoch_loss:<14.6f}{current_lr:<14.6f}")
            
    final_theta = theta.flatten()
    final_loss = epoch_loss_history[-1]
    
    return (final_theta, final_loss, np.array(step_trajectory), 
            np.array(epoch_trajectory), np.array(epoch_loss_history), 
            theta_star.flatten(), X_b)

def plot_minibatch_linear_regression_contour(X, y, X_b, step_trajectory, epoch_trajectory, 
                                             epoch_loss_history, theta_star, batch_size=10, 
                                             baseline_traj=None, 
                                             save_path="plots/10_minibatch_sgd_linear_regression.png"):
    """
    Plot 2D contour representation of the parameter loss surface J(w, b) alongside the fitted line.
    Includes Mini-Batch SGD step trajectory, epoch checkpoints, batch GD baseline, and loss convergence inset.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Panel 1: Parameter Space Contour Plot J(w, b) & Mini-Batch SGD Trajectory ---
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
    
    # Batch GD baseline for direct visual contrast
    if baseline_traj is not None:
        ax1.plot(baseline_traj[:, 0], baseline_traj[:, 1], 'k--', alpha=0.5, linewidth=1.5, 
                 label='Batch GD Baseline (Alg 08)')
        
    # Mini-batch step trajectory (showing reduced variance compared to pure SGD)
    ax1.plot(step_trajectory[:, 0], step_trajectory[:, 1], color='darkorange', alpha=0.7, 
             linewidth=1.2, marker='.', markersize=3, label=f'Mini-Batch Steps (B={batch_size})')
    
    # Mini-batch epoch-level checkpoints
    ax1.plot(epoch_trajectory[:, 0], epoch_trajectory[:, 1], 'ro-', markersize=4, 
             linewidth=1.8, label='Epoch Checkpoints')
    
    # Initial point
    ax1.scatter([step_trajectory[0, 0]], [step_trajectory[0, 1]], color='green', s=110, zorder=6, 
                label=f'Init: ({step_trajectory[0,0]:.1f}, {step_trajectory[0,1]:.1f})')
    
    # Analytical optimum
    ax1.scatter([theta_star[0]], [theta_star[1]], color='gold', marker='*', s=220, zorder=7, 
                label=f'Optimal: ({theta_star[0]:.2f}, {theta_star[1]:.2f})')
    
    # Final Mini-Batch SGD point
    final_w, final_b = step_trajectory[-1, 0], step_trajectory[-1, 1]
    ax1.scatter([final_w], [final_b], color='purple', marker='X', s=130, zorder=7, 
                label=f'Mini-Batch Final: ({final_w:.2f}, {final_b:.2f})')
    
    ax1.set_title(f"10 - Loss Surface Contours $J(w, b)$ & Mini-Batch SGD (B={batch_size})", fontsize=11)
    ax1.set_xlabel("Weight $w$", fontsize=10)
    ax1.set_ylabel("Bias (Intercept) $b$", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8)
    
    # --- Panel 2: Fitted Regression Line in Data Space ---
    ax2.scatter(X, y, color='dodgerblue', alpha=0.75, edgecolors='k', s=40, label='Observed Data')
    
    x_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 100)
    
    # Initial model (t = 0)
    ax2.plot(x_line, step_trajectory[0, 0] * x_line + step_trajectory[0, 1], 'g--', alpha=0.6, 
             linewidth=1.5, label=f'Initial: y = {step_trajectory[0,0]:.1f}x + {step_trajectory[0,1]:.1f}')
    
    # Intermediate step models illustrating rapid convergence
    if len(step_trajectory) > 5:
        w_s2, b_s2 = step_trajectory[2, 0], step_trajectory[2, 1]
        ax2.plot(x_line, w_s2 * x_line + b_s2, color='darkorange', linestyle=':', alpha=0.75, 
                 label=f'Batch Step 2: y = {w_s2:.2f}x + {b_s2:.2f}')
        
    if len(epoch_trajectory) > 1:
        w_ep1, b_ep1 = epoch_trajectory[1, 0], epoch_trajectory[1, 1]
        ax2.plot(x_line, w_ep1 * x_line + b_ep1, color='teal', linestyle='-.', alpha=0.7, 
                 label=f'Epoch 1: y = {w_ep1:.2f}x + {b_ep1:.2f}')
        
    # Final Mini-Batch SGD model
    ax2.plot(x_line, final_w * x_line + final_b, 'r-', linewidth=2.2, 
             label=f'Final Mini-Batch: y = {final_w:.2f}x + {final_b:.2f}')
    
    # Optimal model (Normal Equation)
    ax2.plot(x_line, theta_star[0] * x_line + theta_star[1], 'k--', linewidth=1.6, alpha=0.7, 
             label=f'Optimal (Norm Eq): y = {theta_star[0]:.2f}x + {theta_star[1]:.2f}')
    
    ax2.set_title("Fitted Linear Regression Model & Dynamic Evolution", fontsize=11)
    ax2.set_xlabel("Feature $x$", fontsize=10)
    ax2.set_ylabel("Target $y$", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=7.5)
    
    # Inset Axes: MSE Loss vs Epoch on Log Scale
    inset_ax = ax2.inset_axes([0.55, 0.10, 0.40, 0.32])
    inset_ax.set_facecolor('#f9f9f9')
    epochs = np.arange(len(epoch_loss_history))
    inset_ax.semilogy(epochs, epoch_loss_history, 'r-o', markersize=3, linewidth=1.3)
    inset_ax.set_title("MSE Loss vs Epoch (Log Scale)", fontsize=8)
    inset_ax.set_xlabel("Epoch", fontsize=7)
    inset_ax.set_ylabel("MSE (log)", fontsize=7)
    inset_ax.grid(True, which="both", linestyle="--", alpha=0.4)
    inset_ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    # Also save secondary alias copy
    alt_save_path = "plots/10_minibatch_sgd.png"
    if save_path != alt_save_path:
        plt.savefig(alt_save_path, dpi=300)
        
    plt.close()
    print(f"Mini-batch SGD contour plot saved to: {save_path}")
    if save_path != alt_save_path:
        print(f"Also saved secondary copy to: {alt_save_path}")

if __name__ == "__main__":
    print("=== Algorithm 10: Mini-Batch Stochastic Gradient Descent Linear Regression ===")
    X, y = generate_synthetic_data(n_samples=50, true_w=2.5, true_b=1.0)
    
    batch_size = 10
    alpha_0 = 0.08
    decay = 0.005
    epochs = 30
    
    opt_params, opt_loss, step_traj, epoch_traj, epoch_losses, theta_exact, X_b = train_linear_regression_minibatch_sgd(
        X, y, batch_size=batch_size, initial_learning_rate=alpha_0, n_epochs=epochs, lr_decay=decay, random_state=42
    )
    
    # Compute Batch GD baseline starting from identical initial point
    theta_init = np.array([[-1.5], [-2.0]], dtype=float)
    batch_gd_traj = compute_batch_gd_baseline(X_b, y, theta_init, learning_rate=0.1, max_iter=60)
    
    print("-" * 72)
    print(f"Optimal Parameters (Normal Eq): w = {theta_exact[0]:.4f}, b = {theta_exact[1]:.4f}")
    print(f"Mini-Batch SGD Parameters:      w = {opt_params[0]:.4f}, b = {opt_params[1]:.4f}")
    print(f"Final MSE Loss:                 {opt_loss:.6f}")
    print(f"Optimal MSE Loss:               {compute_loss(X_b, y, theta_exact.reshape(-1, 1)):.6f}")
    print(f"Total Epochs:                   {epochs} ({len(step_traj)-1} mini-batch updates)")
    
    plot_minibatch_linear_regression_contour(
        X, y, X_b, step_traj, epoch_traj, epoch_losses, theta_exact, 
        batch_size=batch_size, 
        baseline_traj=batch_gd_traj, 
        save_path="plots/10_minibatch_sgd_linear_regression.png"
    )
