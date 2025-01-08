import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the adaptive smooth L1 loss function
def l1_loss(network_output, gt, beta=0.168, alpha=0.52, gamma=1.0, epsilon=1e-6):
    """
    Adaptive Smooth L1 Loss function with dynamic weights and improved stability.

    Parameters:
    - network_output: Tensor, predicted values.
    - gt: Tensor, target values.
    - beta: float, controls the point where the loss changes from L2 to L1.
    - alpha: float, base weight for the L2 region (small differences).
    - gamma: float, base weight for the L1 region (large differences).
    - epsilon: float, small constant to improve numerical stability.

    Returns:
    - loss: Tensor, adaptive smooth L1 loss.
    """
    diff = torch.abs(network_output - gt) + epsilon  # Add epsilon for stability

    # Dynamically adjust alpha and gamma based on the magnitude of the error
    dynamic_alpha = alpha / (1 + torch.log(1 + diff))  # Reduce alpha for large differences
    dynamic_gamma = gamma * (1 - torch.exp(-diff))    # Increase gamma for large differences

    # Calculate loss with weighted L2 and L1 regions
    loss = torch.where(
        diff < beta,
        dynamic_alpha * (diff ** 2) / (2 * beta),  # Adaptive L2 region
        dynamic_gamma * (diff - 0.5 * beta)       # Adaptive L1 region
    )
    
    return loss

# Generate test data
x_values = torch.linspace(-1, 1, 500)  # Predicted values
y_target = torch.tensor(0.0)           # Target value (ground truth)
loss_values = l1_loss(x_values, y_target).numpy()

# Plot the loss function
plt.figure(figsize=(8, 6))
plt.plot(x_values.numpy(), loss_values, label='Adaptive Smooth L1 Loss', color='blue')
plt.axvline(x=0, color='red', linestyle='--', label='Target Value (y=0)')
plt.title("Adaptive Smooth L1 Loss Function")
plt.xlabel("Network Output")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("adaptive_smooth_l1_loss.png", dpi=300)  # You can specify the file name and resolution (dpi)
plt.show()
