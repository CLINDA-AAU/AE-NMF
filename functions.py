import numpy as np
from scipy.optimize import linear_sum_assignment, nnls
import scipy.spatial as sp
import matplotlib.pyplot as plt
import torch

def kl_divergence_poisson(P, Q, mse=True):
    """
    Computes the KL divergence for Poisson-distributed data.
    
    Parameters:
    - P: First probability distribution (PyTorch Tensor or NumPy array).
    - Q: Second probability distribution (PyTorch Tensor or NumPy array).
    - mse: If True, returns the mean KL divergence, otherwise returns the sum.
    
    Returns:
    - KL divergence value (scalar or tensor/array depending on input type).
    """
    epsilon = 1e-10  # Small constant to avoid division by zero or log(0)
    
    # Check if inputs are PyTorch tensors
    if isinstance(P, torch.Tensor) and isinstance(Q, torch.Tensor):
        # Ensure P and Q are non-zero by clamping to epsilon
        P = torch.maximum(P, torch.tensor(epsilon, dtype=P.dtype, device=P.device))
        Q = torch.maximum(Q, torch.tensor(epsilon, dtype=Q.dtype, device=Q.device))
        
        # Compute KL divergence for Poisson distribution
        kl_tensor = P * torch.log(P / Q) - P + Q
        
        # Return mean if mse=True, otherwise return sum
        return torch.mean(kl_tensor) if mse else torch.sum(kl_tensor)
    
    # Check if inputs are NumPy arrays
    elif isinstance(P, np.ndarray) and isinstance(Q, np.ndarray):
        # Ensure P and Q are non-zero by clamping to epsilon
        P = np.maximum(P, epsilon)
        Q = np.maximum(Q, epsilon)
        
        # Compute KL divergence for Poisson distribution
        kl_tensor = P * np.log(P / Q) - P + Q
        
        # Return mean if mse=True, otherwise return sum
        return np.mean(kl_tensor) if mse else np.sum(kl_tensor)
    
    else:
        raise TypeError("Input types must be either both torch.Tensor or both np.ndarray.")
        
def refit(data, signatures, method='nnls', mean=True):
    """
    Refits the given data using the provided signatures.
    
    Parameters:
    - data: Input data (Pandas DataFrame or Numpy array).
    - signatures: Signature matrix (Numpy array).
    - method: 'nnls' for non-negative least squares, 'kl' for KL divergence optimization.
    - mean: Whether to normalize the output error by the data size.
    
    Returns:
    - out_error: Computed error based on the selected method.
    """
    
    if isinstance(signatures, torch.Tensor):
        signatures = signatures.numpy()
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    elif hasattr(data, 'to_numpy'):  # For pandas DataFrame
        data = data.to_numpy()
    
    rank = signatures.shape[1]
    max_iter = 30 * rank

    if method == 'nnls':
        exposures = data.apply(lambda x: nnls(A=signatures, b=x, maxiter=max_iter)[0], axis=0)
        rec = exposures.T @ signatures.T
        denominator = np.prod(data.shape) if mean else 1
        out_error = np.linalg.norm(data.to_numpy() - rec.T)**2 / denominator

    elif method == 'kl':
        n = data.shape[1]
        G = np.random.rand(rank, n)
        F = signatures
        
        for _ in range(max_iter):
            FG = np.maximum(F @ G, 1e-50)
            G *= (F.T @ np.divide(data, FG)) / np.sum(F, axis=0, keepdims=True).T
        
        exposures = G
        out_error = kl_divergence_poisson(data, F @ G, mse=mean)

    else:
        raise ValueError("Invalid method. Choose 'nnls' or 'kl'.")

    return out_error


def cosine_HA(est_set, ref_set):
    """
    Performs optimal signature matching using the Hungarian algorithm based on cosine similarity.

    Parameters:
    est_set (numpy.ndarray): Estimated signatures.
    ref_set (numpy.ndarray): Reference signatures.

    Returns:
    numpy.ndarray: Matched cosine similarity matrix.
    numpy.ndarray: Indices representing the matched signatures.
    """
    sim = 1 - sp.distance.cdist(est_set, ref_set, 'cosine')
    if np.any(np.isinf(sim) | np.isnan(sim)):
      raise ValueError("The cosine similarity matrix contains invalid values (infinity or NaN). Please check your input data.")
    _, col_ind  = linear_sum_assignment(-sim.T)
    return((sim.T[:,col_ind]).T, col_ind)

