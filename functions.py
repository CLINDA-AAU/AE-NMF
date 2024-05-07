import numpy as np
from scipy.optimize import linear_sum_assignment, nnls
import scipy.spatial as sp
import matplotlib.pyplot as plt

def refit(data, signatures, mean = True):
    """
    Refits the given data using the provided signatures.

    Parameters:
    data (DataFrame): Input data.
    signatures (numpy.ndarray): Signatures matrix.
    mean (bool): If True, computes mean squared error, else computes total squared error (default is True).

    Returns:
    float: Reconstruction error.
    """
    max_iter = 30*signatures.shape[1]
    exposures = data.apply(lambda x: nnls(A = signatures, b = x, maxiter = max_iter)[0], axis = 0)
    rec = exposures.T@signatures.T
    denominator = np.prod(data.shape) if mean else 1
    out_error = np.linalg.norm(data.to_numpy() - rec.T)**2/denominator  
    return(out_error)



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

