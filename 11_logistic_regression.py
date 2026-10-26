# Objective: Train a binary logistic regression model.
# Input: Features, labels, learning rate, and iteration settings.
# Output: Model parameters and predictions.

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

def generate_synthetic_classification_data(n_samples=80, true_w=2.0, true_b=-0.5, random_state=42):
    """
    Generate synthetic 1D binary classification dataset:
        z = true_w * x + true_b
        P(y = 1 | x) = sigma(z)
        y ~ Bernoulli(P)
    """
    np.random.seed(random_state)
    x = np.random.uniform(-3.0, 3.0, size=(n_samples, 1))
    z = true_w * x + true_b
    prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    y = (np.random.rand(n_samples, 1) < prob).astype(float)
    return x, y

def sigmoid(z):
    """
    Numerically stable sigmoid activation function: sigma(z) = 1 / (1 + exp(-z)).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def compute_loss(X_b, y, theta, eps=1e-15):
    """
    Binary Cross-Entropy (BCE / Log-Loss):
    J(theta) = -1/m * sum [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
    """
    p = sigmoid(X_b @ theta)
    # Clip probabilities to prevent log(0) numerical instability
    p = np.clip(p, eps, 1.0 - eps)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

def compute_gradient(X_b, y, theta):
    """
    Analytical gradient of Binary Cross-Entropy with respect to theta:
    grad = 1/m * X_b^T * (sigma(X_b * theta) - y)
    """
    m = len(y)
    p = sigmoid(X_b @ theta)
    return (1.0 / m) * (X_b.T @ (p - y))

def compute_hessian(X_b, theta):
    """
    Analytical Hessian of BCE with respect to theta:
    H = 1/m * X_b^T * S * X_b, where S = diag(p * (1 - p))
    """
    m = len(y)
    p = sigmoid(X_b @ theta)
    s = (p * (1.0 - p)).flatten()
    return (1.0 / m) * (X_b.T @ (s[:, np.newaxis] * X_b))

def predict_probability(X_b, theta):
    """
    Compute predicted posterior probabilities P(y = 1 | x).
    """
    return sigmoid(X_b @ theta)

def predict_classes(X_b, theta, threshold=0.5):
    """
    Classify inputs into binary labels {0, 1} based on decision threshold.
    """
    return (predict_probability(X_b, theta) >= threshold).astype(int)

def compute_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy percentage.
    """
    return np.mean(y_true.flatten() == y_pred.flatten())

def train_logistic_regression_gd(X, y, learning_rate=1.2, max_iter=100, tol=1e-6):
    """
    Train Binary Logistic Regression using Gradient Descent on parameters theta = [w, b]^T.
    
    Inputs:
    - X: Feature array of shape (m, 1)
    - y: Target binary labels of shape (m, 1)
    - learning_rate: Step size alpha
    - max_iter: Maximum iterations
    - tol: Convergence tolerance on gradient norm
    
    Outputs:
    - theta: Optimized parameter vector [w, b]
    - final_loss: Binary cross-entropy loss at convergence
    - trajectory: History of theta vectors
    - loss_history: History of BCE loss
    - accuracy_history: Classification accuracy progression
    - theta_star: Reference benchmark solution via numerical optimization
    - X_b: Augmented design matrix [X, 1]
    """
    m = len(y)
    X_b = np.hstack([X, np.ones((m, 1))])
    
    # Initialize parameters theta = [w, b]^T far from optimum (predicting inverse slope)
    theta = np.array([[-1.5], [2.0]], dtype=float)
    trajectory = [theta.flatten().copy()]
    loss_history = [compute_loss(X_b, y, theta)]
    
    init_preds = predict_classes(X_b, theta)
    accuracy_history = [compute_accuracy(y, init_preds)]
    
    # Reference optimum using high-precision L-BFGS-B optimization
    def cost_func(th):
        return compute_loss(X_b, y, th.reshape(-1, 1))
    def grad_func(th):
        return compute_gradient(X_b, y, th.reshape(-1, 1)).flatten()
        
    res = minimize(cost_func, [0.0, 0.0], jac=grad_func, method='L-BFGS-B', tol=1e-8)
    theta_star = res.x
    
    print(f"{'Iter':<6}{'Weight (w)':<14}{'Bias (b)':<14}{'BCE Loss':<14}{'||grad||':<12}{'Accuracy':<10}")
    print("-" * 70)
    
    for i in range(max_iter):
        g = compute_gradient(X_b, y, theta)
        g_norm = np.linalg.norm(g)
        
        preds = predict_classes(X_b, theta)
        acc = compute_accuracy(y, preds)
        
        if i % 10 == 0 or i == max_iter - 1:
            print(f"{i:<6}{theta[0, 0]:<14.4f}{theta[1, 0]:<14.4f}{loss_history[-1]:<14.6f}{g_norm:<12.6f}{acc * 100:<9.1f}%")
            
        if g_norm < tol:
            print(f"Converged at iteration {i} (tolerance reached: ||grad|| < {tol}).")
            break
            
        theta = theta - learning_rate * g
        trajectory.append(theta.flatten().copy())
        loss_history.append(compute_loss(X_b, y, theta))
        accuracy_history.append(acc)
        
    final_preds = predict_classes(X_b, theta)
    final_acc = compute_accuracy(y, final_preds)
    accuracy_history[-1] = final_acc
    
    return (theta.flatten(), loss_history[-1], np.array(trajectory), 
            np.array(loss_history), np.array(accuracy_history), theta_star, X_b)

def plot_logistic_regression_contour(X, y, X_b, trajectory, loss_history, accuracy_history, 
                                     theta_star, save_path="plots/11_logistic_regression.png"):
    """
    Plot 2D contour representation of the parameter loss surface J(w, b) alongside the fitted sigmoid probability curve.
    Includes decision boundary threshold and loss/accuracy convergence insets.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Panel 1: Parameter Space Contour Plot J(w, b) & Optimization Path ---
    w_vals = np.linspace(-2.5, 3.5, 200)
    b_vals = np.linspace(-2.5, 3.5, 200)
    W, B = np.meshgrid(w_vals, b_vals)
    Z = np.zeros_like(W)
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            th = np.array([[W[i, j]], [B[i, j]]])
            Z[i, j] = compute_loss(X_b, y, th)
            
    levels = np.logspace(np.log10(Z.min() + 1e-3), np.log10(Z.max()), 25)
    cs = ax1.contour(W, B, Z, levels=levels, cmap="viridis", alpha=0.7)
    ax1.clabel(cs, inline=1, fontsize=8, fmt="%.2f")
    ax1.contourf(W, B, Z, levels=levels, cmap="Blues", alpha=0.2)
    
    # GD Trajectory of (w, b)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=3.5, linewidth=1.6, label='GD Trajectory (w, b)')
    ax1.scatter([trajectory[0, 0]], [trajectory[0, 1]], color='green', s=110, zorder=6, 
                label=f'Init: ({trajectory[0,0]:.1f}, {trajectory[0,1]:.1f})')
    ax1.scatter([theta_star[0]], [theta_star[1]], color='gold', marker='*', s=220, zorder=7, 
                label=f'Optimal (MLE): ({theta_star[0]:.2f}, {theta_star[1]:.2f})')
    
    final_w, final_b = trajectory[-1, 0], trajectory[-1, 1]
    ax1.scatter([final_w], [final_b], color='purple', marker='X', s=130, zorder=7, 
                label=f'Final GD: ({final_w:.2f}, {final_b:.2f})')
    
    ax1.set_title("11 - Cross-Entropy Loss Contours $J(w, b)$ & Optimization Path", fontsize=11)
    ax1.set_xlabel("Weight $w$", fontsize=10)
    ax1.set_ylabel("Bias (Intercept) $b$", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8)
    
    # --- Panel 2: Fitted Sigmoid Probability & Decision Boundary in Data Space ---
    class_0 = (y.flatten() == 0)
    class_1 = (y.flatten() == 1)
    ax2.scatter(X[class_0], y[class_0], color='crimson', edgecolors='k', s=45, alpha=0.8, label='Class 0 (y = 0)')
    ax2.scatter(X[class_1], y[class_1], color='dodgerblue', edgecolors='k', s=45, alpha=0.8, label='Class 1 (y = 1)')
    
    x_grid = np.linspace(-3.5, 3.5, 300)
    
    # Initial model curve (t = 0)
    p_init = sigmoid(trajectory[0, 0] * x_grid + trajectory[0, 1])
    ax2.plot(x_grid, p_init, 'g--', alpha=0.6, linewidth=1.5, 
             label=f'Initial: sigma({trajectory[0,0]:.1f}x + {trajectory[0,1]:.1f})')
    
    # Intermediate iteration curves
    inter_iters = [5, 15]
    colors_inter = ['darkorange', 'teal']
    for it, c in zip(inter_iters, colors_inter):
        if it < len(trajectory):
            p_it = sigmoid(trajectory[it, 0] * x_grid + trajectory[it, 1])
            ax2.plot(x_grid, p_it, color=c, linestyle=':', alpha=0.75, 
                     label=f'Iter {it}: sigma({trajectory[it,0]:.2f}x + {trajectory[it,1]:.2f})')
            
    # Final fitted sigmoid
    p_final = sigmoid(final_w * x_grid + final_b)
    ax2.plot(x_grid, p_final, 'b-', linewidth=2.2, 
             label=f'Final GD: sigma({final_w:.2f}x + {final_b:.2f})')
    
    # Reference optimal sigmoid
    p_opt = sigmoid(theta_star[0] * x_grid + theta_star[1])
    ax2.plot(x_grid, p_opt, 'k--', linewidth=1.5, alpha=0.6, 
             label=f'Optimal (MLE): sigma({theta_star[0]:.2f}x + {theta_star[1]:.2f})')
    
    # Decision boundary line (P = 0.5 threshold => x_db = -b / w)
    x_db = -final_b / final_w
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Threshold P=0.5')
    ax2.axvline(x_db, color='purple', linestyle='-.', linewidth=1.6, alpha=0.8, 
                label=f'Decision Boundary: x = {x_db:.2f}')
    
    ax2.set_title("Fitted Sigmoid Probability Curve $P(y=1|x)$ & Classification", fontsize=11)
    ax2.set_xlabel("Feature $x$", fontsize=10)
    ax2.set_ylabel("Predicted Probability $P(y=1)$", fontsize=10)
    ax2.set_ylim(-0.08, 1.08)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=7.5)
    
    # Inset Axes: BCE Loss vs Iteration on Log Scale
    inset_ax = ax2.inset_axes([0.55, 0.12, 0.40, 0.32])
    inset_ax.set_facecolor('#f9f9f9')
    iters = np.arange(len(loss_history))
    inset_ax.semilogy(iters, loss_history, 'r-', linewidth=1.5, label='BCE Loss')
    inset_ax.set_title("BCE Loss Convergence", fontsize=8)
    inset_ax.set_xlabel("Iteration", fontsize=7)
    inset_ax.set_ylabel("Loss", fontsize=7)
    inset_ax.grid(True, which="both", linestyle="--", alpha=0.4)
    inset_ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Logistic regression contour plot saved to: {save_path}")

if __name__ == "__main__":
    print("=== Algorithm 11: Binary Logistic Regression Optimization ===")
    X, y = generate_synthetic_classification_data(n_samples=80, true_w=2.0, true_b=-0.5, random_state=42)
    
    alpha = 1.2
    max_iterations = 100
    
    opt_params, opt_loss, traj, loss_hist, acc_hist, theta_exact, X_b = train_logistic_regression_gd(
        X, y, learning_rate=alpha, max_iter=max_iterations, tol=1e-5
    )
    
    final_preds = predict_classes(X_b, opt_params.reshape(-1, 1))
    final_acc = compute_accuracy(y, final_preds)
    decision_boundary = -opt_params[1] / opt_params[0]
    
    print("-" * 70)
    print(f"Reference Parameters (MLE):    w = {theta_exact[0]:.4f}, b = {theta_exact[1]:.4f}")
    print(f"GD Estimated Parameters:       w = {opt_params[0]:.4f}, b = {opt_params[1]:.4f}")
    print(f"Decision Boundary:              x = -b/w = {decision_boundary:.4f}")
    print(f"Final BCE Loss:                 {opt_loss:.6f}")
    print(f"Optimal Reference Loss:         {compute_loss(X_b, y, theta_exact.reshape(-1, 1)):.6f}")
    print(f"Final Classification Accuracy:  {final_acc * 100:.2f}%")
    print(f"Total Iterations:               {len(traj) - 1}")
    
    plot_logistic_regression_contour(
        X, y, X_b, traj, loss_hist, acc_hist, theta_exact, 
        save_path="plots/11_logistic_regression.png"
    )
