import numpy as np

def normalize(X):
    """
    Parameters
    ----------
    X
        MxG matrix of raw counts where M is the number of samples
        and G is the number of genes

    """
    # Total counts. This is an MxG matrix where each row repeats
    # the total counts for the given row's sample.
    N = np.full(
        (X.shape[1], X.shape[0]), 
        np.sum(X, axis=1)
    ).T

    # Compute \pi_j = \frac{\sum_i y_ij}{\sum_i n_i}
    pi = np.sum(X, axis=0) / np.sum(N)

    # Compute the matrix M where element M_{i,j} = n_i\pi_j
    M = np.outer(np.sum(X, axis=1), pi)

    # Compute residual deviances
    R = np.sign(X - M) * np.sqrt(
        np.nan_to_num(2*X*np.log(X/M), copy=True, nan=0.0) \
        + 2*(N-X)*np.log((N-X)/(N-M))
    )

    return R
