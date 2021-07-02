import numpy as np

def normalize(X, umi_counts=None):
    """
    Parameters
    ----------
    X
        MxG matrix of raw counts where M is the number of samples
        and G is the number of genes
    umi_counts
        Total counts. This is an M-length array of total UMI counts
        for each sample
    """
    if umi_counts is None:
        umi_counts = np.sum(X, axis=1)

    N = np.full(
        (X.shape[1], X.shape[0]),
        umi_counts
    ).T

    # Compute \pi_j = \frac{\sum_i y_ij}{\sum_i n_i}
    pi = np.sum(X, axis=0) / np.sum(N, axis=0)

    # Compute the matrix M where element M_{i,j} = n_i\pi_j
    M = np.outer(umi_counts, pi)

    square = np.nan_to_num(2*X*np.log(X/M), copy=True, nan=0.0) \
        + 2*(N-X)*np.log((N-X)/(N-M))

    for x in square.flatten():
        # If this number is negative, make sure it's just underflow
        if x < 0:
            assert abs(x) < 0.01, f"Term value {x} is negative"

    R = np.nan_to_num(np.sign(X - M) * np.sqrt(square), copy=True, nan=0.0)

    return R


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('../../spatial_transcriptomics/simulate_data/simulated_datasets/g_CALML5.CALML5_sigma_10_cs_0.0/7.tsv', sep='\t')

    umi_counts = np.array(df['size_factors'])
    X = np.array(df[['count_gene_1', 'count_gene_2']])
    print(X)
    X_res = normalize(X, umi_counts=umi_counts)
    print(np.sum(np.array(X_res).flatten()))

