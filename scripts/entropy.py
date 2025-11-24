import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy
from scipy.stats import gaussian_kde


# Function to compute Shannon entropy with choice of method
def compute_shannon_entropy(data, method='scipy', bins=50):
    """
    Compute Shannon entropy of a feature vector.

    Parameters:
    - data  : Input feature vector (e.g., Z or Z_T)
    - method: 'kde' for kernel density estimation or 'scipy' for histogram-based entropy
    - bins  : Number of bins for histogram (if method='scipy')

    Returns:
    - entropy_value: Shannon entropy in bits
    """
    if method == 'kde':
        kde    = gaussian_kde(data)
        x_grid = np.linspace(min(data), max(data), 1000)
        pdf    = kde(x_grid)
        pdf    = pdf / np.sum(pdf)  # Normalize to sum to 1
        entropy_value = -np.sum(pdf * np.log2(pdf + 1e-10))  # Avoid log(0)

    elif method == 'scipy':
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        entropy_value   = scipy_entropy(hist + 1e-10, base=2)  # Add small constant to avoid log(0)
    else:
        raise ValueError("Method must be 'kde' or 'scipy'")
    return entropy_value

# Function to compute Mutual Information
def compute_mutual_information(Z, Z_T, bins=50):
    """
    Compute Mutual Information between original (Z) and protected (Z_T) templates.

    Parameters:
    - Z  : Original feature vector
    - Z_T: Protected template
    - bins: Number of bins for histogram-based MI estimation

    Returns:
    - mi: Mutual Information in bits
    """
    # Joint histogram
    hist_2d, x_edges, y_edges = np.histogram2d(Z, Z_T, bins=bins, density=True)
    joint_prob                = hist_2d / np.sum(hist_2d)

    # Marginal histograms
    prob_Z   = np.sum(joint_prob, axis=1)
    prob_Z_T = np.sum(joint_prob, axis=0)

    # Mutual Information: sum(p(x,y) * log2(p(x,y) / (p(x) * p(y))))
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if joint_prob[i, j] > 0 and prob_Z[i] > 0 and prob_Z_T[j] > 0:
                mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (prob_Z[i] * prob_Z_T[j] + 1e-10))
    return mi

# Function to compute Information Loss
def compute_information_loss(Z, Z_T):
    """
    Compute Information Loss between original (Z) and protected (Z_T) templates.

    Parameters:
    - Z  : Original feature vector
    - Z_T: Protected template

    Returns:
    - info_loss: Normalized Information Loss
    """
    mutual_info = np.abs(np.corrcoef(Z, Z_T)[0, 1])  # Proxy for MI (simplified)
    info_loss   = 1 - mutual_info
    return info_loss

# Function to compute Information Preservation
def compute_information_preservation(Z, Z_T):
    """
    Compute Information Preservation between original (Z) and protected (Z_T) templates.

    Parameters:
    - Z  : Original feature vector
    - Z_T: Protected template

    Returns:
    - info_preserve: Normalized Information Preservation
    """
    mse           = np.mean((Z - Z_T) ** 2)
    info_preserve = 1 / (1 + mse)
    return info_preserve

# Function to evaluate and plot metrics in a single interactive plot
def evaluate_btp_metrics(Z, Z_T, entropy_method='scipy', bins=50):
    """
    Evaluate and visualize BTP metrics in a single interactive plot.

    Parameters:
    - Z  : Original feature vector
    - Z_T: Protected template
    - entropy_method: 'kde' or 'scipy' for entropy calculation
    - bins: Number of bins for histogram-based calculations
    """
    # Compute metrics
    entropy_Z     = compute_shannon_entropy(Z, method=entropy_method, bins=bins)
    entropy_Z_T   = compute_shannon_entropy(Z_T, method=entropy_method, bins=bins)
    mutual_info   = compute_mutual_information(Z, Z_T, bins=bins)
    info_loss     = compute_information_loss(Z, Z_T)
    info_preserve = compute_information_preservation(Z, Z_T)
    
    return entropy_Z, entropy_Z_T, mutual_info, info_loss, info_preserve



