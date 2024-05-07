import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def cvx_update(XtX, G, W):
    """
    Computes a single convex NMF update.

    Parameters:
    XtX (numpy.ndarray): Precomputed X.T @ X matrix.
    G (numpy.ndarray): Weight matrix of shape (n, rank).
    W (numpy.ndarray): Encoding matrix of shape (n, rank).

    Returns:
    G (numpy.ndarray): Updated weight matrix.
    W (numpy.ndarray): Updated encoding matrix.
    """
    XtXW = XtX@W
    XtXG = XtX@G
    GWtXtXW = G@W.T@XtXW
    XtXWGtG = XtXW@G.T@G
    G = G * np.sqrt(np.divide(XtXW,GWtXtXW))
    W = W * np.sqrt(np.divide(XtXG,XtXWGtG))
    return(G, W)
    
def cvx_nmf_init(X, rank, init = "random"):
    """
    Initializes the encoding and weight matrices for Convex NMF.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (p, n).
    rank (int): Rank of the factorization.
    init (str): Initialization method, either "random" or "kmeans" (default is "random").

    Returns:
    G_0 (numpy.ndarray): Initial encoding matrix.
    W_0 (numpy.ndarray): Initial weight matrix.
    """
    _,n = X.shape
    if init == "kmeans":
        kmeans = KMeans(n_clusters=rank).fit(X.T)

        H = pd.get_dummies(kmeans.labels_) 
        n_vec = H.sum(axis = 0)
        G = H + 0.2*np.ones((n, rank)) 
        W = G@np.diag(1/np.array(n_vec))  
        W_0 = W.to_numpy()
        G_0 = G.to_numpy()
    if init == "random":
        # estimation matrices the size of the feature space of the data matrix
        W_0 = np.random.rand(n, rank)
        G_0 = np.random.rand(n, rank)
    return(G_0, W_0)


def convex_nmf_tol(X, rank, init = "random", tol = 1e-3, relative_tol = True, mse = False, max_iter = 10e8):
    """
    Performs Convex Non-negative Matrix Factorization with a given tolerance level.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (p, n).
    rank (int): Rank of the factorization.
    init (str): Initialization method, either "random" or "kmeans" (default is "random").
    tol (float): Tolerance level for convergence (default is 1e-3).
    relative_tol (bool): If True, the tolerance is relative, otherwise absolute (default is True).
    mse (bool): If True, returns Mean Squared Error, otherwise Frobenius Norm (default is False).
    max_iter (int): Maximum number of iterations (default is 10e8).

    Returns:
    G (numpy.ndarray): Encoding matrix of shape (n, rank).
    W (numpy.ndarray): Weight matrix of shape (n, rank).
    loss (list): List of loss values at each iteration.
    G_0 (numpy.ndarray): Initial encoding matrix.
    W_0 (numpy.ndarray): Initial weight matrix.
    n_iter (int): Number of iterations performed.
    """
    p,n = X.shape
    G_0, W_0 = cvx_nmf_init(X, rank, init)

        
    XtX = X.T@X
    G = G_0
    W = W_0
            
    frob_norm = (np.linalg.norm(X - X@W@G.T)**2)
    denominator = (p*n) if mse else 1
    loss_0 = frob_norm/denominator
    loss = [loss_0]
    rel_diff = float('inf')
    n_iter = 0
    while(rel_diff>tol):
        n_iter = n_iter + 1
        G, W = cvx_update(XtX, G, W)
        frob_norm =  (np.linalg.norm(X - X@W@G.T)**2)#np.sum(((X - X@W@G.T)**2))
        loss_val = frob_norm/denominator
        loss.append(loss_val)
        rel_diff = abs((loss[-1] - loss[-2]))/loss[-2] if relative_tol else abs(loss[-1] - loss[-2])#NMF tager relativ diff mht første loss
        if n_iter >= max_iter:
            break

    return(G, W, loss, G_0, W_0, n_iter)
