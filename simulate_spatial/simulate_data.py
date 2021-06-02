import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import importlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import scanpy as sc
from anndata import AnnData
from collections import defaultdict

from . import poisson_lognormal
import warnings

def simulate_pairwise_from_dataset(
        adata,
        z_mean,
        gene_1,
        gene_2,
        row_key='array_row',
        col_key='array_col',
        sigma=10,
        cov_strength=5,
        poisson=False,
        size_factors=None,
        mean_g1=None,
        mean_g2=None,
        var_g1=None,
        var_g2=None
    ):
    """
    Simulate pairwise expression with spatially varying correlation.

    Parameters
    ----------
    adata
        The AnnData object storing the spatial expression data.
    z_mean
        The mean correlation across the slide.
    gene_1
        The name of a gene on which to base the simulated expression
        values for the first gene. The simulated data's first gene will
        have the same mean and variance as this gene.
    gene_2
        The name of a gene on which to base the simulated expression
        values for the second gene. The simulated data's second gene will
        have the same mean and variance as this gene.
    row_key
        The name of the column in `adata.obs` that stores the row
        coordinates of each spot.
    col_key
        The name of the column in `adata.obs` that stores the column 
        coordinates of each spot.
    sigma
        The size of the Radial Basis Function's kernel bandwidth that
        is used to generate patterns of spatially varying correlation.
    cov_strength
        This parameter sets the strength of correlation. Higher values
        lead to larger magnitudes for the correlation of each gene.
    poisson
        If True, sample counts instead of normally distributed values.
    size_factors
        A list containing the total UMI count for each spot. If `poisson` 
        is set to true, then this argument must be provided.
    """
    if poisson:
        adata = adata.copy()

    # Estimate means and variances from real data for the multivariate
    # normal distribution
    if mean_g1 is None or mean_g2 is None or var_g1 is None \
        or var_g2 is None:
        if poisson:
            means_1, vars_1 = poisson_lognormal.fit(
                adata.obs_vector(gene_1),
                size_factors.T[0]
            )
            mean_g1 = np.mean(means_1.squeeze())
            var_g1 = np.mean(vars_1.squeeze())**2

            means_2, vars_2 = poisson_lognormal.fit(
                adata.obs_vector(gene_2),
                size_factors.T[1]
            )
            mean_g2 = np.mean(means_2.squeeze())
            var_g2 = np.mean(vars_2.squeeze())**2
        else:
            mean_g1 = np.mean(adata.obs_vector(gene_1)) 
            mean_g2 = np.mean(adata.obs_vector(gene_2))
            var_g1 = np.var(adata.obs_vector(gene_1))
            var_g2 = np.var(adata.obs_vector(gene_2))

    coords = np.array(adata.obs[[
        row_key,
        col_key
    ]])

    z_means = np.full(adata.shape[0], z_mean)
    means_g1 = np.full(
        adata.shape[0],
        mean_g1
    )
    means_g2 = np.full(
        adata.shape[0],
        mean_g2,
    )
    vars_g1 = np.full(
        adata.shape[0],
        var_g1,
    )
    vars_g2 = np.full(
        adata.shape[0],
        var_g2,
    )

    corrs, covs, sample = simulate_expression_guassian_cov(
        z_means,
        means_g1,
        means_g2,
        vars_g1,
        vars_g2,
        coords,
        sigma=sigma,
        cov_strength=cov_strength,
        poisson=poisson,
        size_factors=size_factors
    )
    adata_sim = AnnData(
        X=sample.T,
        obs=adata.obs
    )
    return corrs, covs, adata_sim


def simulate_pairwise_condition_covariate_from_dataset(
        adata,
        clust_to_z_mean,
        clust_to_gene_stat_1,
        clust_to_gene_stat_2,
        clust_to_sigma,
        clust_to_cov_strength,
        clust_key,
        row_key='array_row',
        col_key='array_col',
        poisson=False,
        size_factors=None
    ):
    """
    Simulate pairwise expression with spatially varying correlation.

    Parameters
    ----------
    adata
        The AnnData object storing the spatial expression data.
    clust_to_z_mean
        The mean correlation across the slide.
    gene_1
        The name of a gene on which to base the simulated expression
        values for the first gene. The simulated data's first gene will
        have the same mean and variance as this gene.
    gene_2
        The name of a gene on which to base the simulated expression
        values for the second gene. The simulated data's second gene will
        have the same mean and variance as this gene.
    row_key
        The name of the column in `adata.obs` that stores the row
        coordinates of each spot.
    col_key
        The name of the column in `adata.obs` that stores the column
        coordinates of each spot.
    clust_to_sigma : dictionary
        Map each cluster to the size of the radial Basis Function's kernel 
        bandwidth that is used to generate patterns of spatially varying 
        correlation within spots of that cluster.
    clust_to_cov_strength : dictionary
        Map each cluster to the size of the the strength of correlation. 
        Higher values lead to larger magnitudes for the correlation of 
        each gene
    clust_key
        The name of the column in `adata.obs` that stores the cluster
        ID of each spot.
    poisson
        If True, sample counts instead of normally distributed values.
    size_factors
        A list containing the total UMI count for each spot. If `poisson`
        is set to True, then this argument must be provided.
    """
    if poisson:
        adata = adata.copy()
    
    if poisson:
        gene_to_mean = {}
        gene_to_var = {}
        all_genes = set(clust_to_gene_stat_1.values()) | set(clust_to_gene_stat_2.values())
        for gene in all_genes:
            means, varss = poisson_lognormal.fit(
                adata.obs_vector(gene),
                size_factors.T[0]
            )
            mean = np.mean(means.squeeze())
            varr = np.mean(varss.squeeze())**2
            gene_to_mean[gene] = mean
            gene_to_var[gene] = varr
    else:
        for gene in all_genes:
            gene_to_mean[gene] = np.mean(adata.obs_vector(gene))
            gene_to_var[gene] = np.var(adata.obs_vector(gene))

    coords = np.array(adata.obs[[
        row_key,
        col_key
    ]])

    z_means = np.array([
        clust_to_z_mean[clust]
        for clust in adata.obs[clust_key]
    ])
    means_g1 = np.array([
        gene_to_mean[clust_to_gene_stat_1[clust]]
        for clust in adata.obs[clust_key]
    ])
    print(means_g1)
    print(set(means_g1))
    means_g2 = np.array([
        gene_to_mean[clust_to_gene_stat_2[clust]]
        for clust in adata.obs[clust_key]
    ])
    vars_g1 = np.array([
        gene_to_var[clust_to_gene_stat_1[clust]]
        for clust in adata.obs[clust_key]
    ])
    vars_g2 = np.array([
        gene_to_var[clust_to_gene_stat_2[clust]]
        for clust in adata.obs[clust_key]
    ])

    # Map each cluster to the indices of spots belonging to that
    # cluster
    clust_to_indices = defaultdict(lambda: [])
    for ind, clust in enumerate(adata.obs[clust_key]):
        clust_to_indices[clust].append(ind)

    corrs, covs, sample = simulate_expression_guassian_cov_condition(
        z_means,
        means_g1,
        means_g2,
        vars_g1,
        vars_g2,
        coords,
        clust_to_indices,
        clust_to_sigma,
        clust_to_cov_strength,
        poisson=poisson,
        size_factors=size_factors
    )
    adata_sim = AnnData(
        X=sample.T,
        obs=adata.obs
    )
    return corrs, covs, adata_sim


def simulate_expression_guassian_cov(
        z_means, 
        x_means_g1, 
        x_means_g2, 
        x_vars_g1, 
        x_vars_g2,
        coords, 
        sigma=10, 
        cov_strength=5,
        poisson=False,
        size_factors=None
    ):
    """
    z_means: the spot-wise means of the Fisher transformed correlations
        between Gene 1 and Gene 2
    x_vars_g1: the spot-wise variances for Gene 1
    x_vars_g2: the spot-wise variances for Gene 2
    """
    # Compute the covariance matrix for the Fisher-transformed correlations
    # using a Gaussian kernel
    dist_matrix = euclidean_distances(coords)
    kernel_matrix = np.exp(-1 * np.power(dist_matrix,2) / sigma**2)
    z_cov = kernel_matrix * cov_strength

    # Sample the Fisher-transformed correlations
    z_corrs = np.random.multivariate_normal(z_means, z_cov)
   
    # Transform to correlation via inverse of arctanh
    corrs = np.tanh(z_corrs)

    # Sample expression values at each spot for the two
    # genes
    sample = []
    covs = [] # The true pairwise covariance at each spot
    for s_i in range(len(coords)):
        # Means at each spot
        spot_means = np.array([
            x_means_g1[s_i],
            x_means_g2[s_i]
        ])

        # Form the covariance matrix
        spot_vars = [
            x_vars_g1,
            x_vars_g2
        ]
        cov_g12 = corrs[s_i] * np.sqrt(x_vars_g1[s_i] * x_vars_g2[s_i])
        covs.append(cov_g12)
        cov_mat = np.array([
            [x_vars_g1[s_i], cov_g12],
            [cov_g12, x_vars_g2[s_i]]
        ])

        # Sample
        lamb_s = np.random.multivariate_normal(spot_means, cov_mat)
        if poisson:
            poiss_mean = np.exp(lamb_s) * size_factors[s_i]
            x_s = np.random.poisson(poiss_mean)
        else:
            x_s = lamb_s
        sample.append(x_s)
    sample = np.array(sample).T
    covs = np.array(covs)
    return corrs, covs, sample






def simulate_expression_guassian_cov_mult_genes(
        x_means,
        x_vars,
        coords,
        sigma=10,
        cov_strength=5,
        poisson=False,
        size_factors=None
    ):
    """
    z_means: the spot-wise means of the Fisher transformed correlations
        between Gene 1 and Gene 2
    x_vars_g1: the spot-wise variances for Gene 1
    x_vars_g2: the spot-wise variances for Gene 2
    """
    warnings.filterwarnings('error')

    # Number of genes
    G = len(x_means)

    # Compute the covariance matrix for the Fisher-transformed correlations
    # using a Gaussian kernel
    dist_matrix = euclidean_distances(coords)
    K= np.exp(-1 * np.power(dist_matrix,2) / sigma**2)

    #noise = 0.1

    #B = np.random.rand(3, 3)*2-1 * 0.001
    #np.fill_diagonal(B, np.ones(len(B)))

    # B matrix
    B = np.identity(G)

    # Base covariance matrix
    U = np.zeros((G,G))
    np.fill_diagonal(U, x_vars)

    Z = np.array([
        np.random.multivariate_normal(np.zeros(len(K)), K * cov_strength)
        for i in range(len(B))
    ])
    
    covs = []
    samples = []
    for si in range(len(coords)):
        z = Z.T[si]
        cov = np.outer(z, z)
        mean = x_means.T[si]
        
        cov += U
        #print(cov)
        covs.append(cov)
        #print(is_pos_def(cov))

        lamb_s = np.random.multivariate_normal(mean, cov)
        #if poisson:
        #    #poiss_mean = np.exp(lamb_s) + np.log(size_factors[s_i])
        #    poiss_mean = np.exp(lamb_s) * size_factors[s_i]
        #    x_s = np.random.poisson(poiss_mean)
        #else:
        x_s = lamb_s
        samples.append(x_s)
    samples = np.array(samples).T        

    corrs = [
        cov / np.sqrt(np.outer(np.diagonal(cov), np.diagonal(cov)))
        for cov in covs
    ]
    return corrs, covs, samples


def simulate_expression_guassian_cov_condition(
        z_means,
        x_means_g1,
        x_means_g2,
        x_vars_g1,
        x_vars_g2,
        coords,
        clust_to_indices,
        clust_to_sigma,
        clust_to_cov_strength,
        poisson=True,
        size_factors=None
    ):
    """
    z_means: the spot-wise means of the Fisher transformed correlations
        between Gene 1 and Gene 2
    x_vars_g1: the spot-wise variances for Gene 1
    x_vars_g2: the spot-wise variances for Gene 2
    clust_to_indices: a dictionary mapping the name of a cell type or cluster
        to the indices in coords corresponding to that cluster/cell type
    clust_to_sigma: a dictionary mapping name of a cell type or cluster to 
        the bandwidth parameter used to simulate correlations.
    clust_to_cov_strenth: a dictionary mapping name of a cell type or cluster
        to the constant used to multiply the kernel matrix in order to yield
        the covariance matrix.
    """

    # Map index to cluster
    index_to_clust = {}
    for clust, indices in clust_to_indices.items():
        for index in indices:
            index_to_clust[index] = clust
    clusts = [
        index_to_clust[i]
        for i in range(len(coords))
    ]

    dist_matrix = euclidean_distances(coords)

    z_corrs = np.zeros(len(coords))
    z_cov = np.zeros(dist_matrix.shape) 
    for clust, indices in clust_to_indices.items():
        # Compute the covariance matrix for the Fisher-transformed correlations
        # using a Gaussian kernel
        if clust in clust_to_sigma:
            dist_matrix = euclidean_distances(coords[indices])

            sigma = clust_to_sigma[clust]
            cov_strength = clust_to_cov_strength[clust] 
            kernel_matrix_clust = np.exp(-1 * np.power(dist_matrix,2) / sigma**2)
            
            z_cov_clust = kernel_matrix_clust * cov_strength

            # Sample the Fisher-transformed correlations
            z_corrs_clust = np.random.multivariate_normal(z_means[indices], z_cov_clust)
            for index, z_corr in zip(indices, z_corrs_clust):
                z_corrs[index] = z_corr

    # Transform to correlation via inverse of arctanh
    corrs = np.tanh(z_corrs)

    # Sample expression values at each spot for the two
    # genes
    sample = []
    covs = [] # The true pairwise covariance at each spot
    for s_i in range(len(coords)):
        # Means at each spot
        spot_means = np.array([
            x_means_g1[s_i],
            x_means_g2[s_i]
        ])

        # Form the covariance matrix
        spot_vars = [
            x_vars_g1,
            x_vars_g2
        ]
        cov_g12 = corrs[s_i] * np.sqrt(x_vars_g1[s_i] * x_vars_g2[s_i])
        covs.append(cov_g12)
        cov_mat = np.array([
            [x_vars_g1[s_i], cov_g12],
            [cov_g12, x_vars_g2[s_i]]
        ])

        #print(cov_mat)

        # Sample
        lamb_s = np.random.multivariate_normal(spot_means, cov_mat)
        if poisson:
            poiss_mean = np.exp(lamb_s) * size_factors[s_i]
            x_s = np.random.poisson(poiss_mean)
        else:
            # Let the expression simply be the draw
            # from the distribution
            x_s = lamb_s

        sample.append(x_s)
    sample = np.array(sample).T
    covs = np.array(covs)
    return corrs, covs, sample






def simulate_expression_guassian_no_spatial(
        x_means_g1,
        x_means_g2,
        x_vars_g1,
        x_vars_g2,
        x_covs,
        coords,
        poisson=False,
        size_factors=None
    ):
    """
    x_means_g1: the spot-wise means for Gene 1
    x_means_g2: the spot-wise means for gene 2
    x_vars_g1: the spot-wise variances for Gene 1
    x_vars_g2: the spot-wise variances for Gene 2
    x_covs: the spot-wise covariances
    coords: TODO
    """
    # Sample expression values at each spot for the two
    # genes
    sample = []
    covs = [] # The true pairwise covariance at each spot
    for s_i in range(len(coords)):
        # Means at each spot
        spot_means = np.array([
            x_means_g1[s_i],
            x_means_g2[s_i]
        ])

        cov_mat = np.array([
            [x_vars_g1[s_i], x_covs[s_i]],
            [x_covs[s_i], x_vars_g2[s_i]]
        ])

        #print(cov_mat)

        corrs = x_covs / np.sqrt(x_vars_g1 * x_vars_g2)

        # Sample
        lamb_s = np.random.multivariate_normal(spot_means, cov_mat)
        if poisson:
            x_s = np.random.poisson(
                np.exp(lamb_s + np.log(size_factors[s_i]))
            )
        else:
            x_s = lamb_s
        sample.append(x_s)
    sample = np.array(sample).T
    return corrs, sample


def plot_simulated_data(
        df, 
        expr_1, 
        expr_2, 
        corrs, 
        covs, 
        row_key='row', 
        col_key='col', 
        dot_size=30
    ):
    figure, axarr = plt.subplots(
        2,
        2,
        figsize = (10,10)
    )

    y = -1 * np.array(df[row_key])
    x = df[col_key]
    color = expr_1
    axarr[0][0].scatter(x,y,c=color, cmap='viridis', s=dot_size)
    axarr[0][0].set_title('Gene 1 Expression')

    y = -1 * np.array(df[row_key])
    x = df[col_key]
    color = expr_2
    axarr[0][1].scatter(x,y,c=color, cmap='viridis', s=dot_size)
    axarr[0][1].set_title('Gene 2 Expression')

    y = -1 * np.array(df[row_key])
    x = df[col_key]
    color = np.abs(corrs)
    im = axarr[1][0].scatter(x,y,c=color, cmap='viridis', s=dot_size, vmin=0, vmax=1)
    figure.colorbar(im, ax=axarr[1][0])
    axarr[1][0].set_title('Absolute value of correlation')

    y = -1 * np.array(df[row_key])
    x = df[col_key]
    color = corrs
    im = axarr[1][1].scatter(x,y,c=color, cmap='RdBu_r', s=dot_size, vmin=-1, vmax=1)
    figure.colorbar(im, ax=axarr[1][1])
    axarr[1][1].set_title('Underlying Correlation')

    plt.show()
