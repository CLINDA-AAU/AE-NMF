import numpy as np
from functions import kl_divergence_poisson

def NMF_mult_tol(X, rank, tol=1e-3, relative_tol=True, mse=False, max_iter=int(1e8), objective_type="frobenius", F_0 = None, G_0 = None):
    """
    Performs Non-negative Matrix Factorization with a given tolerance level.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (p, n).
    rank (int): Rank of the factorization.
    tol (float): Tolerance level for convergence (default is 1e-3).
    relative_tol (bool): If True, the tolerance is relative, otherwise absolute (default is True).
    mse (bool): If True, returns Mean Squared Error, otherwise Frobenius Norm (default is False).
    max_iter (int): Maximum number of iterations before analysis is disrupted (default is 10e8).
    objective_type (str): 'frobenius' for Frobenius norm-based objective, 'kl' for KL divergence
    F_0 (numpy.ndarray): If given initial basis matrix of shape (p, rank) (default is None).
    G_0 (numpy.ndarray): If given initial weight matrix of shape (rank, n) (default is None).

    
    Returns:
    F (numpy.ndarray): Basis matrix of shape (p, rank).
    G (numpy.ndarray): Weight matrix of shape (rank, n).
    loss (list): List of loss values at each iteration.
    F_0 (numpy.ndarray): Initial basis matrix.
    G_0 (numpy.ndarray): Initial weight matrix.
    n_iter (int): Number of iterations performed.
    """
    p, n = X.shape
    F_0 = F_0 if F_0 is not None else np.random.rand(p, rank)   # Initialize F - signatures - E
    G_0 = G_0 if G_0 is not None else np.random.rand(rank, n)  # Initialize G - exposures - P
    
    F, G = F_0, G_0
    denominator_mse = (n * p) if mse else 1
    
    if objective_type == "kl":
        # Initial loss for KL divergence
        loss_0 = kl_divergence_poisson(X, F @ G, mse = mse)
    else:
        # Initial loss for Frobenius norm
        curr_loss = np.linalg.norm(X - F @ G) ** 2
        loss_0 = curr_loss / denominator_mse
    
    loss = [loss_0]
    rel_diff = 200  # Initialize with a large difference
    n_iter = 0

    while rel_diff > tol and n_iter < max_iter:
        n_iter += 1
        
        # Update rules for multiplicative updates
        if objective_type == "kl":
            FG = np.maximum(F@G, 1e-50)
            F *= (np.divide(X,FG)@G.T)/np.sum(G, axis=1, keepdims=True).T
            FG =np.maximum(F@G, 1e-50)
            G *= (F.T @ np.divide(X,FG))/ np.sum(F, axis=0, keepdims=True).T
            #G *= np.dot(F.T, X / (F @ G)) / np.sum(G, axis=1, keepdims=True)
            #F *= np.dot(X / (F @ G), G.T) / np.sum(F, axis=0, keepdims=True)

            loss_val = kl_divergence_poisson(X, F @ G, mse = mse)
            
        else:
            G *= np.divide(F.T @ X, (F.T @ F @ G))
            F *= np.divide(X @ G.T, (F @ G @ G.T))
            # Calculate new Frobenius norm loss
            curr_loss = np.linalg.norm(X - F @ G) ** 2
            loss_val = curr_loss / denominator_mse
        
        loss.append(loss_val)
        
        # Compute relative difference for convergence check
        denominator = loss[-2] if relative_tol else 1
        rel_diff = abs((loss[-1] - loss[-2]) / denominator)
    
    return F, G, loss, F_0, G_0, n_iter
