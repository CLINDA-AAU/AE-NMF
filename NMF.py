import numpy as np


def NMF_mult_tol(X, rank, tol = 1e-3, relative_tol = True, mse = False, max_iter = 10e8, F_0 = None, G_0 = None):
    """
    Performs Non-negative Matrix Factorization with a given tolerance level.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (p, n).
    rank (int): Rank of the factorization.
    tol (float): Tolerance level for convergence (default is 1e-3).
    relative_tol (bool): If True, the tolerance is relative, otherwise absolute (default is True).
    mse (bool): If True, returns Mean Squared Error, otherwise Frobenius Norm (default is False).
    max_iter (int): Maximum number of iterations before analysis is disrupted (default is 10e8).
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
    
    p,n = X.shape
    if F_0 is None:
       F_0 = np.random.rand(p, rank)
    if G_0 is None:  
       G_0 = np.random.rand(rank,n)
        
    F = F_0
    G = G_0

    frob_norm = np.linalg.norm(X - F@G)**2
    denominator_mse = (n*p) if mse else 1
    loss_0 = frob_norm/denominator_mse
    loss = [loss_0]
    rel_diff = float('inf')
    n_iter = 0
    while(rel_diff>tol):
        n_iter += 1 
        
        G = G*(np.divide(F.T@X, F.T@F@G))
        F = F*(np.divide(X@(G.T), F@G@(G.T)))
        
        frob_norm =  np.linalg.norm(X - F@G)**2
        loss_val = frob_norm/(n*p) if mse else frob_norm
        loss.append(loss_val)
        denominator = loss[-2] if relative_tol else 1
        rel_diff = abs((loss[-1] - loss[-2]))/denominator 
        if n_iter >= max_iter:
            break

    return(F, G, loss, F_0, G_0, n_iter)
    