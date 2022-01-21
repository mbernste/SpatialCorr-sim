"""
The core API behind SpatialCorr-sim for simulating spatial transcriptomics data
with spatially varying correlation structure.

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from anndata import AnnData
from collections import defaultdict

from . import poisson_lognormal

################################### The API ########################################

def simulate_pairwise_from_dataset( # TODO DEPRECATE???
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
    Simulate the expression of two genes with spatially varying correlation.

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

    fisher_corr_means = np.full(adata.shape[0], z_mean)
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
        fisher_corr_means,
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


def simulate_gene_pair_within_region_varying_correlation(
        adata,
        gene_1,
        gene_2,
        clust_to_fisher_corr_mean=None,
        fisher_corr_mean=None,
        clust_to_bandwidth=None,
        bandwidth=None,
        clust_to_cov_strength=None,
        cov_strength=None,
        clust_key='cluster',
        row_key='row',
        col_key='col',
        poisson=False,
        size_factors=None,
        gene_to_clust_to_mean=None,
        gene_to_clust_to_var=None
    ):
    """
    Simulate pairwise expression with non-varying correlation within each
    region, but differing correlation between regions.

    Parameters
    ----------
    adata : AnnData
        The AnnData object storing the spatial expression data that is
        to be used to seed the simulation.
    gene_1 : string
        The name of a gene on which to base the simulated expression
        values for the first gene. The simulated data's first gene will
        have the same mean and variance as this gene.
    gene_2 : string
        The name of a gene on which to base the simulated expression
        values for the second gene. The simulated data's second gene will
        have the same mean and variance as this gene.
    clust_to_fisher_corr_mean : dictionary, optional (default : None)
        Map each cluster to the mean Fisher correlation between the two genes
        within that cluster. The correlation within each cluster will vary, but
        the Fisher-transformed correlations will vary around this mean.
    fisher_corr_mean : float, optional (default : None)
        The mean Fisher correlation between the two genes within every cluster.
        If `clust_to_fisher_corr_mean` is not provided, this value will be used for all 
        clusters/regions. Otherwise, this argument will be over-ridden by the values in 
        `clust_to_fisher_corr_mean`.
    clust_to_bandwidth : dictionary, optional (default : None)
        Map each cluster/region to the bandwidth parameter used in the Gaussian kernel
        used to sample correlations within that cluster. Larger bandwidth parameters
        will produce coarser patterns of correlation.
    bandwidth : float, optional (deault : None)
        The bandwidth parameter to use for all clusters. If `clust_to_bandwidth` is not
        provided, this value will be used for all clusters/regions. Otherwise, this 
        argument will be over-ridden by the values in `clust_to_bandwidth`.
    clust_to_cov_strength : dictionary, optional (default : None)
        Map each cluster/region to the size of the "correlation strength" (i.e., the
        scalar that scales the covariance matrix used in the Guassian process to generate
        a spatial pattern of correlation). Higher values lead to larger magnitudes for the 
        correlation of each gene.
    cov_strength : float, optional (default : None)
        The size of the "correlation strength" (i.e., the scalar that scales the covariance 
        matrix used in the Guassian process to generate a spatial pattern of correlation). 
        Higher values lead to larger magnitudes for the correlation of each gene. If 
        `clust_to_cov_strength` is not provided, this value will be used for all clusters/regions. 
        Otherwise, this argument will be over-ridden by the values in `clust_to_cov_strength`.
    clust_key : string, optional (default : 'cluster')
        The name of the column in `adata.obs` that stores the cluster/region
        ID of each spot.
    row_key : string (default : 'row')
        The name of the column in `adata.obs` that stores the row
        coordinates of each spot.
    col_key : string (default : 'col')
        The name of the column in `adata.obs` that stores the column
        coordinates of each spot.
    poisson : boolean, optional (default : False)
        If False, sample expression values from a multivariate lognormal
        distribution at each spot. If True, these expression values are used
        to construct the mean counts, and counts are sampled from a
        Poisson-lognormal distribution.
    size_factors : ndarray, optional (default : None)
        A N-length array, where N is the number of spots containing the
        size-factor (i.e., library size) for each spot. If `poisson`
        is set to True, then this argument must be provided.
    gene_to_clust_to_mean : dictionary, optional (default : None)
        A dictionary mapping each simulated gene to a dictionary that
        maps each cluster to the latent expression mean. If not provided,
        they will be estimated from the expression data in `adata` via
        a hierarchical Bayesian model.
    gene_to_clust_to_var : dictionary, optional (default : None)
        A dictionary mapping each simulated gene to a dictionary that
        maps each cluster to the latent expression variance. If not provided,
        they will be estimated from the expression data in `adata` via a
        hierarchical Bayesian model.
        
    Returns
    -------
    corrs : ndarray
        An N-length array, where N is the number of spots, storing the latent 
        correlation used to generate expression at each spot. 
    covs : ndarray
        An N-length array, where N is the number of spots, storing the latent 
        covariance used to generate expression at each spot.
    adata_sim : AnnData
        A Nx2 simulated dataset for N spots and two genes.
    """
    if poisson:
        adata = adata.copy()
 
    # Determine Gaussian process parameters from arguments
    if clust_to_fisher_corr_mean is None:
        if fisher_corr_mean is None:
            print("Warning. Mean Fisher correlations were not provided. Defaulting to zero.")
            clust_to_fisher_corr_mean = {
                ct: 0
                for ct in set(adata.obs[clust_key])
            }
        else:
            clust_to_fisher_corr_mean = {
                ct: fisher_corr_mean
                for ct in set(adata.obs[clust_key])
            }
    if clust_to_bandwidth is None:
        if bandwidth is None:
            print("Warning. The kernel bandwidth parameters were not provided. Defaulting to 5.")
            clust_to_bandwidth = {
                ct: 5
                for ct in set(adata.obs[clust_key])
            }
        else:
            clust_to_bandwidth = {
                ct: bandwidth
                for ct in set(adata.obs[clust_key])
            }
    if clust_to_cov_strength is None:
        if cov_strength is None:
            print("Warning. The covariance-strength parameters were not provided. Defaulting to 0.25.")
            clust_to_cov_strength = {
                ct: 0.25
                for ct in set(adata.obs[clust_key])
            }
        else:
            clust_to_cov_strength = {
                ct: cov_strength
                for ct in set(adata.obs[clust_key])
            }

    # Map clusters to indices
    clust_to_inds = defaultdict(lambda: [])
    for ind, ct in enumerate(adata.obs[clust_key]):
        clust_to_inds[ct].append(ind)

    # If the latent means and variances of each gene within each region are
    # not provided, we must infer them using the hierarchical model.
    all_genes = set([gene_1, gene_2])
    if gene_to_clust_to_mean is None or gene_to_clust_to_var is None:
        gene_to_clust_to_mean = defaultdict(lambda: {})
        gene_to_clust_to_var = defaultdict(lambda: {})
        if poisson:
            for gene in all_genes:
                for clust in sorted(set(adata.obs[clust_key])):
                    inds = clust_to_inds[clust]
                    means, varss = poisson_lognormal.fit(
                        adata.obs_vector(gene)[inds],
                        size_factors.T[0][inds]
                    )
                    mean = np.mean(means.squeeze())
                    varr = np.mean(varss.squeeze())**2
                    gene_to_clust_to_mean[gene][clust] = mean
                    gene_to_clust_to_var[gene][clust] = varr
        else:
            for gene in all_genes:
                gene_to_clust_to_mean[gene] = np.mean(adata.obs_vector(gene)[inds])
                gene_to_clust_to_var[gene] = np.var(adata.obs_vector(gene)[inds])
    else:
        assert frozenset(gene_to_clust_to_mean.keys()) == frozenset(all_genes)
        assert frozenset(gene_to_clust_to_var.keys()) == frozenset(all_genes)

    # Reformat the spatial coordinates
    coords = np.array(adata.obs[[
        row_key,
        col_key
    ]])

    # Reformat spotwise simulation model parameters
    fisher_corr_means = np.array([
        clust_to_fisher_corr_mean[clust]
        for clust in adata.obs[clust_key]
    ])
    means_g1 = np.array([
        gene_to_clust_to_mean[gene_1][clust]
        for clust in adata.obs[clust_key]
    ])
    means_g2 = np.array([
        gene_to_clust_to_mean[gene_2][clust]
        for clust in adata.obs[clust_key]
    ])
    vars_g1 = np.array([
        gene_to_clust_to_var[gene_1][clust]
        for clust in adata.obs[clust_key]
    ])
    vars_g2 = np.array([
        gene_to_clust_to_var[gene_2][clust]
        for clust in adata.obs[clust_key]
    ])

    # Map each cluster to the indices of spots belonging to that
    # cluster
    clust_to_indices = defaultdict(lambda: [])
    for ind, clust in enumerate(adata.obs[clust_key]):
        clust_to_indices[clust].append(ind)

    # Generate the simulated dataset
    corrs, covs, sample = generate_gene_pair_expression_within_region_varying_corr(
        fisher_corr_means,
        means_g1,
        means_g2,
        vars_g1,
        vars_g2,
        coords,
        clust_to_indices,
        clust_to_bandwidth,
        clust_to_cov_strength,
        poisson=poisson,
        size_factors=size_factors
    )
    adata_sim = AnnData(
        X=sample.T,
        obs=adata.obs
    )
    return corrs, covs, adata_sim


def simulate_gene_pair_region_specific(
        adata,
        clust_to_corr,
        gene_1,
        gene_2,
        clust_key,
        row_key='row',
        col_key='col',
        poisson=False,
        size_factors=None,
        gene_to_clust_to_mean=None,
        gene_to_clust_to_var=None
    ):
    """
    Simulate pairwise expression with non-varying correlation within each
    region, but differing correlation between regions.

    Parameters
    ----------
    adata : AnnData
        The AnnData object storing the spatial expression data that is
        to be used to seed the simulation.
    clust_to_corr
        Map each cluster to the correlation between the two genes
        within that cluster.
    gene_1 : string
        The name of a gene on which to base the simulated expression
        values for the first gene. The simulated data's first gene will
        have the same mean and variance as this gene.
    gene_2 : string
        The name of a gene on which to base the simulated expression
        values for the second gene. The simulated data's second gene will
        have the same mean and variance as this gene.
    clust_key : string
        The name of the column in `adata.obs` that stores the cluster/region
        ID of each spot.
    row_key : string (default : 'row')
        The name of the column in `adata.obs` that stores the row
        coordinates of each spot.
    col_key : string (default : 'col')
        The name of the column in `adata.obs` that stores the column
        coordinates of each spot.
    poisson : boolean, optional (default : False)
        If False, sample expression values from a multivariate lognormal
        distribution at each spot. If True, these expression values are used
        to construct the mean counts, and counts are sampled from a 
        Poisson-lognormal distribution.
    size_factors : ndarray, optional (default : None)
        A N-length array, where N is the number of spots containing the 
        size-factor (i.e., library size) for each spot. If `poisson` 
        is set to True, then this argument must be provided.
    gene_to_clust_to_mean : dictionary, optional (default : None)
        A dictionary mapping each simulated gene to a dictionary that
        maps each cluster to the latent expression mean. If not provided,
        they will be estimated from the expression data in `adata` via
        a hierarchical Bayesian model.
    gene_to_clust_to_var : dictionary, optional (default : None)
        A dictionary mapping each simulated gene to a dictionary that
        maps each cluster to the latent expression variance. If not provided,
        they will be estimated from the expression data in `adata` via a 
        hierarchical Bayesian model.

    Returns
    -------
    corrs : ndarray
        An N-length array, where N is the number of spots, storing the latent
        correlation used to generate expression at each spot.
    covs : ndarray
        An N-length array, where N is the number of spots, storing the latent
        covariance used to generate expression at each spot.
    adata_sim : AnnData
        A Nx2 simulated dataset for N spots and two genes.
    """
    if poisson:
        adata = adata.copy()

    # Map clusters to indices
    clust_to_inds = defaultdict(lambda: [])
    for ind, ct in enumerate(adata.obs[clust_key]):
        clust_to_inds[ct].append(ind)

    # If latent mean and variances are not provided, they will be estimated
    # from the provided gene expression data
    all_genes = set([gene_1, gene_2])
    if gene_to_clust_to_mean is None or gene_to_clust_to_var is None:
        gene_to_clust_to_mean = defaultdict(lambda: {})
        gene_to_clust_to_var = defaultdict(lambda: {})
        if poisson:
            for gene in all_genes:
                for clust in sorted(set(adata.obs[clust_key])):
                    inds = clust_to_inds[clust]
                    means, varss = poisson_lognormal.fit(
                        adata.obs_vector(gene)[inds],
                        size_factors.T[0][inds]
                    )
                    mean = np.mean(means.squeeze())
                    varr = np.mean(varss.squeeze())**2
                    gene_to_clust_to_mean[gene][clust] = mean
                    gene_to_clust_to_var[gene][clust] = varr
        else:
            for gene in all_genes:
                gene_to_clust_to_mean[gene] = np.mean(adata.obs_vector(gene)[inds])
                gene_to_clust_to_var[gene] = np.var(adata.obs_vector(gene)[inds])
    else:
        assert frozenset(gene_to_clust_to_mean.keys()) == frozenset(all_genes)
        assert frozenset(gene_to_clust_to_var.keys()) == frozenset(all_genes)

    # Reformat coordinates
    coords = np.array(adata.obs[[
        row_key,
        col_key
    ]])

    # Specify each spot's latent mean and variance
    means_g1 = np.array([
        gene_to_clust_to_mean[gene_1][clust]
        for clust in adata.obs[clust_key]
    ])
    means_g2 = np.array([
        gene_to_clust_to_mean[gene_2][clust]
        for clust in adata.obs[clust_key]
    ])
    vars_g1 = np.array([
        gene_to_clust_to_var[gene_1][clust]
        for clust in adata.obs[clust_key]
    ])
    vars_g2 = np.array([
        gene_to_clust_to_var[gene_2][clust]
        for clust in adata.obs[clust_key]
    ])

    # Specify each spot's latent correlation
    corrs = np.array([
        clust_to_corr[clust]
        for clust in adata.obs[clust_key]
    ])

    # Generate counts
    covs, sample = generate_gene_pair_expression_prespecified_corr(
        means_g1,
        means_g2,
        vars_g1,
        vars_g2,
        corrs,
        poisson=poisson,
        size_factors=size_factors
    )
    adata_sim = AnnData(
        X=sample.T,
        obs=adata.obs
    )
    return corrs, covs, adata_sim


def simulate_gene_set_within_region_varying_correlation(
        adata,
        genes,
        row_key='array_row',
        col_key='array_col',
        clust_to_bandwidth=None,
        bandwidth=None,
        clust_to_cov_strength=None,
        cov_strength=None,
        clust_key='cluster',
        poisson=False,
        size_factors=None,
        gene_to_clust_to_mean=None,
        gene_to_clust_to_var=None
    ):
    """
    Simulate spatial gene expression for a set of genes for which the correlation
    matrix varies smoothly within each region.

    Parameters
    ----------
    adata : AnnData
        The AnnData object storing the spatial expression data that is
        to be used to seed the simulation.
    genes : list
        A G-length list of gene names for  which to base the simulated expression
        values. Within each region, the simulated datas' genes will have similar 
        means and variances as these genes.
    clust_to_bandwidth : dictionary, optional (default : None)
        Map each cluster/region to the bandwidth parameter used in the Gaussian kernel
        used to sample correlations within that cluster. Larger bandwidth parameters
        will produce coarser patterns of correlation.
    bandwidth : float, optional (deault : None)
        The bandwidth parameter to use for all clusters. If `clust_to_bandwidth` is not
        provided, this value will be used for all clusters/regions. Otherwise, this 
        argument will be over-ridden by the values in `clust_to_bandwidth`.
    clust_to_cov_strength : dictionary, optional (default : None)
        Map each cluster/region to the size of the "correlation strength" (i.e., the
        scalar that scales the covariance matrix used in the Guassian process to generate
        a spatial pattern of correlation). Higher values lead to larger magnitudes for the 
        correlation of each gene.
    cov_strength : float, optional (default : None)
        The size of the "correlation strength" (i.e., the scalar that scales the covariance 
        matrix used in the Guassian process to generate a spatial pattern of correlation). 
        Higher values lead to larger magnitudes for the correlation of each gene. If 
        `clust_to_cov_strength` is not provided, this value will be used for all clusters/regions. 
        Otherwise, this argument will be over-ridden by the values in `clust_to_cov_strength`.
    clust_key : string
        The name of the column in `adata.obs` that stores the cluster/region
        ID of each spot.
    row_key : string (default : 'row')
        The name of the column in `adata.obs` that stores the row
        coordinates of each spot.
    col_key : string (default : 'col')
        The name of the column in `adata.obs` that stores the column
        coordinates of each spot.
    poisson : boolean, optional (default : False)
        If False, sample expression values from a multivariate lognormal
        distribution at each spot. If True, these expression values are used
        to construct the mean counts, and counts are sampled from a
        Poisson-lognormal distribution.
    size_factors : ndarray, optional (default : None)
        A N-length array, where N is the number of spots containing the
        size-factor (i.e., library size) for each spot. If `poisson`
        is set to True, then this argument must be provided.
    gene_to_clust_to_mean : dictionary, optional (default : None)
        A dictionary mapping each simulated gene to a dictionary that
        maps each cluster to the latent expression mean. If not provided,
        they will be estimated from the expression data in `adata` via
        a hierarchical Bayesian model.
    gene_to_clust_to_var : dictionary, optional (default : None)
        A dictionary mapping each simulated gene to a dictionary that
        maps each cluster to the latent expression variance. If not provided,
        they will be estimated from the expression data in `adata` via a
        hierarchical Bayesian model.
        
    Returns
    -------
    corrs : ndarray
        An NxGxG sized array, where N is the number of spots, and G is the number
        of genes, storing the latent correlation matrix used to generate expression 
        at each spot.
    covs : ndarray
        An NxGxG sized array, where N is the number of spots, and G is the number
        of genes, storing the latent covariance matrix used to generate expression 
        at each spot.
    adata_sim : AnnData
        A NxG simulated dataset for N spots and G genes.
    """
    if poisson:
        adata = adata.copy()

    # Determine Gaussian process parameters from arguments
    if clust_to_bandwidth is None:
        if bandwidth is None:
            print("Warning. The kernel bandwidth parameters were not provided. Defaulting to 5.")
            clust_to_bandwidth = {
                ct: 5
                for ct in set(adata.obs[clust_key])
            }
        else:
            clust_to_bandwidth = {
                ct: bandwidth
                for ct in set(adata.obs[clust_key])
            }
    if clust_to_cov_strength is None:
        if cov_strength is None:
            print("Warning. The covariance-strength parameters were not provided. Defaulting to 0.25.")
            clust_to_cov_strength = {
                ct: 0.25
                for ct in set(adata.obs[clust_key])
            }
        else:
            clust_to_cov_strength = {
                ct: cov_strength
                for ct in set(adata.obs[clust_key])
            }

    # Map clusters to indices
    clust_to_inds = defaultdict(lambda: [])
    for ind, ct in enumerate(adata.obs[clust_key]):
        clust_to_inds[ct].append(ind)

    # If the latent mean and variance parameters are not provided, we must infer
    # them from the input data. We do so using Bayesian inference on a hierarchical
    # model.
    all_genes = genes
    if gene_to_clust_to_mean is None or gene_to_clust_to_var is None:
        gene_to_clust_to_mean = defaultdict(lambda: {})
        gene_to_clust_to_var = defaultdict(lambda: {})
        if poisson:
            for gene in all_genes:
                for clust in sorted(set(adata.obs[clust_key])):
                    inds = clust_to_inds[clust]
                    means, varss = poisson_lognormal.fit(
                        adata.obs_vector(gene)[inds],
                        size_factors.T[0][inds]
                    )
                    mean = np.mean(means.squeeze())
                    varr = np.mean(varss.squeeze())**2
                    gene_to_clust_to_mean[gene][clust] = mean
                    gene_to_clust_to_var[gene][clust] = varr
        else:
            for gene in all_genes:
                gene_to_clust_to_mean[gene] = np.mean(adata.obs_vector(gene)[inds])
                gene_to_clust_to_var[gene] = np.var(adata.obs_vector(gene)[inds])
    else:
        assert frozenset(gene_to_clust_to_mean.keys()) == frozenset(all_genes)
        assert frozenset(gene_to_clust_to_var.keys()) == frozenset(all_genes)

    # Create GxN matrices where G is number of genes
    # in the set and N is number of spots
    spot_means = np.array([
        [
            gene_to_clust_to_mean[gene][clust]
            for clust in adata.obs[clust_key]
        ]
        for gene in genes
    ])
    spot_vars = np.array([
        [
            gene_to_clust_to_var[gene][clust]
            for clust in adata.obs[clust_key]
        ]
        for gene in genes
    ])

    coords = np.array(adata.obs[[
        row_key,
        col_key
    ]])

    corrs, covs, sample = generate_gene_set_expression_varying_cov(
        spot_means,
        spot_vars,
        coords,
        clust_to_inds,
        clust_to_bandwidth,
        clust_to_cov_strength,
        poisson=poisson,
        size_factors=size_factors
    )
    adata_sim = AnnData(
        X=sample.T,
        obs=adata.obs
    )
    return corrs, covs, adata_sim


def simulate_gene_set_region_specific(
        adata,
        all_genes,
        spot_covs,
        clust_to_corr_mat,
        clust_key='cluster',
        row_key='row',
        col_key='col',
        poisson=False,
        size_factors=None,
        gene_to_clust_to_mean=None,
        gene_to_clust_to_var=None
    ):
    """
    Simulate gene expression for a set of genes with non-varying correlation 
    within each region, but differing correlation matrices between regions.

    Parameters
    ----------
    adata : AnnData
        The AnnData object storing the spatial expression data that is
        to be used to seed the simulation.
    all_genes : list
        A G-length list of gene names for  which to base the simulated expression
        values. Within each region, the simulated datas' genes will have similar 
        means and variances as these genes.
    clust_key : string
        The name of the column in `adata.obs` that stores the cluster/region
        ID of each spot.
    row_key : string (default : 'row')
        The name of the column in `adata.obs` that stores the row
        coordinates of each spot.
    col_key : string (default : 'col')
        The name of the column in `adata.obs` that stores the column
        coordinates of each spot.
    poisson : boolean, optional (default : False)
        If False, sample expression values from a multivariate lognormal
        distribution at each spot. If True, these expression values are used
        to construct the mean counts, and counts are sampled from a
        Poisson-lognormal distribution.
    size_factors : ndarray, optional (default : None)
        A N-length array, where N is the number of spots containing the
        size-factor (i.e., library size) for each spot. If `poisson`
        is set to True, then this argument must be provided.
    gene_to_clust_to_mean : dictionary, optional (default : None)
        A dictionary mapping each simulated gene to a dictionary that
        maps each cluster to the latent expression mean. If not provided,
        they will be estimated from the expression data in `adata` via
        a hierarchical Bayesian model.
    gene_to_clust_to_var : dictionary, optional (default : None)
        A dictionary mapping each simulated gene to a dictionary that
        maps each cluster to the latent expression variance. If not provided,
        they will be estimated from the expression data in `adata` via a
        hierarchical Bayesian model.

    Returns
    -------
    corrs : ndarray
        An NxGxG sized array, where N is the number of spots, and G is the number
        of genes, storing the latent correlation matrix used to generate expression 
        at each spot.
    covs : ndarray
        An NxGxG sized array, where N is the number of spots, and G is the number
        of genes, storing the latent covariance matrix used to generate expression 
        at each spot.
    adata_sim : AnnData
        A NxG simulated dataset for N spots and G genes.
    """
    if poisson:
        adata = adata.copy()

    # Map clusters to indices
    clust_to_inds = defaultdict(lambda: [])
    for ind, ct in enumerate(adata.obs[clust_key]):
        clust_to_inds[ct].append(ind)

    if gene_to_clust_to_mean is None or gene_to_clust_to_var is None:
        gene_to_clust_to_mean = defaultdict(lambda: {})
        gene_to_clust_to_var = defaultdict(lambda: {})
        if poisson:
            for gene in all_genes:
                for clust in sorted(set(adata.obs[clust_key])):
                    inds = clust_to_inds[clust]
                    means, varss = poisson_lognormal.fit(
                        adata.obs_vector(gene)[inds],
                        size_factors.T[0][inds]
                    )
                    mean = np.mean(means.squeeze())
                    varr = np.mean(varss.squeeze())**2
                    gene_to_clust_to_mean[gene][clust] = mean
                    gene_to_clust_to_var[gene][clust] = varr
        else:
            for gene in all_genes:
                gene_to_clust_to_mean[gene] = np.mean(adata.obs_vector(gene)[inds])
                gene_to_clust_to_var[gene] = np.var(adata.obs_vector(gene)[inds])
    else:
        assert frozenset(gene_to_clust_to_mean.keys()) == frozenset(all_genes)
        assert frozenset(gene_to_clust_to_var.keys()) == frozenset(all_genes)

    # Create NxG matrix of spotwise gene means where G is number 
    # of genes in the set and N is number of spots
    spot_means = np.array([
        [
            gene_to_clust_to_mean[g][clust]
            for g in all_genes
        ]
        for clust in adata.obs[clust_key]
    ])

    coords = np.array(adata.obs[[
        row_key,
        col_key
    ]])

    corrs, covs, sample = generate_gene_set_expression_prespecified_corr(
        spot_means,
        spot_covs,
        coords,
        poisson=poisson,
        size_factors=size_factors
    )
    adata_sim = AnnData(
        X=sample.T,
        obs=adata.obs
    )
    return corrs, covs, adata_sim


################ Lower-level functions for generateing simulations #################


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


def generate_gene_set_expression_varying_cov(
        spot_means,
        spot_vars,
        coords,
        clust_to_indices,
        clust_to_bandwidth,
        clust_to_cov_strength,
        poisson=False,
        size_factors=None
    ):
    """
    A low-level function for generating expression values (either Poisson counts
    or Gaussian "latent" correlation values) for a set of genes for a set of
    spots. This function will generate a random pattern of spatially varying
    correlation within each region using a Gaussian process model and then, given
    this pattern of correlation, will simulate expression at each spot using the
    Poisson-lognormal model.

    Parameters
    ----------
    spot_means : ndarray
         NxG sized array, where N is the number of spots and G is the number
         of genes, storing the per-spot mean values for each gene.
    spot_vars : ndarray
        NxG sized array, where N is the number of spots and G is the number
         of genes, storing the per-spot variances for each gene.
    coords : ndarray
        Nx2 sized array, where N is the number of spots, storing each spot's
        x-y coordinates.
    clust_to_indices : dictionary
        Maps each cluster/region to the indices of spots in `spot_means`, 
        `spot_vars`, `coords`, and `size_factors` that belong to that
        cluster/region.
    clust_to_bandwidth : dictionary
        Map each cluster/region to the bandwidth parameter used in the Gaussian kernel
        used to sample correlations within that cluster. Larger bandwidth parameters
        will produce coarser patterns of correlation.
    clust_to_cov_strength : dictionary
        Map each cluster/region to the size of the the strength of correlation.
        Higher values lead to larger magnitudes for the correlation of
        each gene.
    poisson : boolean, optional (default : False)
        If False, sample expression values from a multivariate lognormal
        distribution at each spot. If True, these expression values are used
        to construct the mean counts, and counts are sampled from a
        Poisson-lognormal distribution.
    size_factors : ndarray, optional (default : None)
        A N-length array, where N is the number of spots containing the
        size-factor (i.e., library size) for each spot. If `poisson`
        is set to True, then this argument must be provided.

    Returns
    -------
    corrs : ndarray
        NxGxG array, where N is the number of spots and G is the number of
        genes, storing the correlation matrices at each spot used to generate 
        the expression values.
    covs : ndarray
        NxGxG array, where N is the number of spots and G is the number of
        genes, storing the covariance matrices at each spot used to generate 
        the expression values.
    sample : ndarray
        NxG size array, where N is the number of spots and G is the number of
        genes, of the simulated expression values of the set of genes at each 
        spot.
    """
    # Number of genes
    G = spot_means.shape[0]

    # Create an empty list that will ultimately store
    # the spot-wise latent correlation factors
    Z = [None for i in range(len(coords))]

    for clust, indices in clust_to_indices.items():
        if clust in clust_to_bandwidth:
            # Cluster-specific parameters
            bandwidth = clust_to_bandwidth[clust]
            cov_strength = clust_to_cov_strength[clust]

            # Compute the kernel for the Gaussian process
            dist_matrix = euclidean_distances(coords[indices])
            K= np.exp(-1 * np.power(dist_matrix,2) / bandwidth**2)
            
            # Compute the latent correlation factor at each spot.
            # This is a NxG matrix where for each spot, we store the
            # latent correlation factor
            Z_clust = np.array([
                np.random.multivariate_normal(
                    np.zeros(len(K)), 
                    K * cov_strength
                )
                for i in range(G)
            ]).T

            # Store in global array containing all spots
            for index, z in zip(indices, Z_clust):
                Z[index] = z
    Z = np.array(Z)
    assert Z.shape[0] == len(coords)
    assert Z.shape[1] == G

    covs = []
    samples = []
    for si in range(len(coords)):
        z = Z[si]
        cov = np.outer(z, z)
        mean = spot_means.T[si]
       
        # Base covariance matrix
        U = np.zeros((G,G)) 
        np.fill_diagonal(U, spot_vars.T[si])
        cov += U
        covs.append(cov)

        # Sample the expression values at this spot
        lamb_s = np.random.multivariate_normal(mean, cov)
        if poisson:
            poiss_mean = np.exp(lamb_s) * size_factors[si]
            x_s = np.random.poisson(poiss_mean)
        else:
            x_s = lamb_s
        samples.append(x_s)
    samples = np.array(samples).T        

    corrs = [
        cov / np.sqrt(np.outer(np.diagonal(cov), np.diagonal(cov)))
        for cov in covs
    ]
    return corrs, covs, samples

def generate_gene_set_expression_prespecified_corr(
        spot_means,
        spot_covs,
        coords,
        poisson=False,
        size_factors=None
    ):
    """
    A low-level function for generating expression values (either Poisson counts
    or Gaussian "latent" correlation values) for a set of genes with pre-specified
    covariance matrix at each spot.
    
    Parameters
    ----------
    spot_means : ndarray
         NxG sized array, where N is the number of spots and G is the number
         of genes, storing the per-spot mean values for each gene.
    spot_vars : ndarray
        NxG sized array, where N is the number of spots and G is the number
         of genes, storing the per-spot variances for each gene.
    coords : ndarray
        Nx2 sized array, where N is the number of spots, storing each spot's
        x-y coordinates.
    poisson : boolean, optional (default : False)
        If False, sample expression values from a multivariate lognormal
        distribution at each spot. If True, these expression values are used
        to construct the mean counts, and counts are sampled from a
        Poisson-lognormal distribution.
    size_factors : ndarray, optional (default : None)
        A N-length array, where N is the number of spots containing the
        size-factor (i.e., library size) for each spot. If `poisson`
        is set to True, then this argument must be provided.

    Returns
    -------
    spot_covs : ndarray
        N-length array, where N is the number of spots, of the covariance
        between the two genes used to generate the expression values.
    sample : ndarray
        Nx2 size array, where N is the number of spots, of the simulated
        expression values of the two genes at each spot.
    """
    samples = []
    covs = spot_covs # Rename for consistency with other methods
    for si in range(len(coords)):
        cov = covs[si]
        mean = spot_means[si]

        lamb_s = np.random.multivariate_normal(mean, cov)
        if poisson:
            poiss_mean = np.exp(lamb_s) * size_factors[si]
            x_s = np.random.poisson(poiss_mean)
        else:
            x_s = lamb_s
        samples.append(x_s)
    samples = np.array(samples).T

    corrs = [
        cov / np.sqrt(np.outer(np.diagonal(cov), np.diagonal(cov)))
        for cov in covs
    ]
    return corrs, covs, samples


def generate_gene_pair_expression_within_region_varying_corr(
        fisher_corr_means,
        spot_means_g1,
        spot_means_g2,
        spot_vars_g1,
        spot_vars_g2,
        coords,
        clust_to_indices,
        clust_to_bandwidth,
        clust_to_cov_strength,
        poisson=True,
        size_factors=None
    ):
    """
    A low-level function for generating expression values (either Poisson counts
    or Gaussian "latent" correlation values) for a pair of genes with spatially 
    varying correlation within each region.

    Parameters
    ----------
    fisher_corr_means : ndarray
        An N-length array storing the mean of the Fisher correlation between the 
        two genes at each of N spots. This mean is used in the Gaussian process 
        to generate a latent correlation value at each spot.
    spot_means_g1 : ndarray
         N-length array, where N is the number of spots, of the per-spot
         mean values for the first gene.
    spot_means_g2 : ndarray
        N-length array, where N is the number of spots, of the per-spot
         mean values for the second gene.
    spot_vars_g1 : ndarray
        N-length array, where N is the number of spots, of the per-spot
        variances for the first gene.
    spot_vars_g2 : ndarray
        N-length array, where N is the number of spots, of the per-spot
        variances for the second gene.
    coords : ndarray
        Nx2 sized array, where N is the number of spots, storing each spot's
        x-y coordinates.
    clust_to_indices : dictionary
        Maps each cluster/region to the indices of spots in `spot_means`,
        `spot_vars`, `coords`, and `size_factors` that belong to that
        cluster/region.
    clust_to_bandwidth : dictionary
        Map each cluster/region to the bandwidth parameter used in the Gaussian kernel
        used to sample correlations within that cluster. Larger bandwidth parameters
        will produce coarser patterns of correlation.
    clust_to_cov_strength : dictionary
        Map each cluster/region to the size of the the strength of correlation.
        Higher values lead to larger magnitudes for the correlation of
        each gene.
    poisson : boolean, optional (default : False)
        If False, sample expression values from a multivariate lognormal
        distribution at each spot. If True, these expression values are used
        to construct the mean counts, and counts are sampled from a
        Poisson-lognormal distribution.
    size_factors : ndarray, optional (default : None)
        A N-length array, where N is the number of spots containing the
        size-factor (i.e., library size) for each spot. If `poisson`
        is set to True, then this argument must be provided.

    Returns
    -------
    spot_covs : ndarray
        N-length array, where N is the number of spots, of the covariance
        between the two genes used to generate the expression values.
    sample : ndarray
        Nx2 size array, where N is the number of spots, of the simulated
        expression values of the two genes at each spot.
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

    # Build distance matrix
    dist_matrix = euclidean_distances(coords)

    z_corrs = np.zeros(len(coords))
    z_cov = np.zeros(dist_matrix.shape) 
    for clust, indices in clust_to_indices.items():
        # Compute the covariance matrix for the Fisher-transformed correlations
        # using a Gaussian kernel
        if clust in clust_to_bandwidth:
            dist_matrix = euclidean_distances(coords[indices])

            bandwidth = clust_to_bandwidth[clust]
            cov_strength = clust_to_cov_strength[clust]
            kernel_matrix_clust = np.exp(-1 * np.power(dist_matrix,2) / bandwidth**2)
            
            z_cov_clust = kernel_matrix_clust * cov_strength

            # Sample the Fisher-transformed correlations
            z_corrs_clust = np.random.multivariate_normal(fisher_corr_means[indices], z_cov_clust)
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
            spot_means_g1[s_i],
            spot_means_g2[s_i]
        ])

        # Form the covariance matrix
        spot_vars = [
            spot_vars_g1,
            spot_vars_g2
        ]
        cov_g12 = corrs[s_i] * np.sqrt(spot_vars_g1[s_i] * spot_vars_g2[s_i])
        covs.append(cov_g12)
        cov_mat = np.array([
            [spot_vars_g1[s_i], cov_g12],
            [cov_g12, spot_vars_g2[s_i]]
        ])

        # Sample expression values at the given spot
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


def generate_gene_pair_expression_prespecified_corr(
        spot_means_g1,
        spot_means_g2,
        spot_vars_g1,
        spot_vars_g2,
        spot_corrs,
        poisson=False,
        size_factors=None
    ):
    """
    A low-level function for generating expression values (either Poisson counts
    or Gaussian "latent" correlation values) for a pair of genes for a set of 
    spots given a pre-specified mean, variance, and correlation for the two genes 
    at each spot.

    Parameters
    ----------
    spot_means_g1 : ndarray 
         N-length array, where N is the number of spots, of the per-spot 
         mean values for the first gene. 
    spot_means_g2 : ndarray 
        N-length array, where N is the number of spots, of the per-spot 
         mean values for the second gene.
    spot_vars_g1 : ndarray
        N-length array, where N is the number of spots, of the per-spot 
        variances for the first gene.
    spot_vars_g2 : ndarray
        N-length array, where N is the number of spots, of the per-spot 
        variances for the second gene.
    spot_corrs : ndarray
        N-length array, where N is the number of spots, of the correlation 
        at each spot between the two genes.
    poisson : boolean, optional (default : False)
        If False, sample expression values from a multivariate lognormal
        distribution at each spot. If True, these expression values are used
        to construct the mean counts, and counts are sampled from a
        Poisson-lognormal distribution.
    size_factors : ndarray, optional (default : None)
        A N-length array, where N is the number of spots containing the
        size-factor (i.e., library size) for each spot. If `poisson`
        is set to True, then this argument must be provided.

    Returns
    -------
    spot_covs : ndarray
        N-length array, where N is the number of spots, of the covariance
        between the two genes used to generate the expression values.
    sample : ndarray
        Nx2 size array, where N is the number of spots, of the simulated
        expression values of the two genes at each spot.
    """
    # Sample expression values at each spot for the two
    # genes
    sample = []

    # Calculate covariance at each spot
    spot_covs = spot_corrs * np.sqrt(spot_vars_g1 * spot_vars_g2)

    for s_i in range(len(spot_means_g1)):
        # Means at each spot
        spot_means = np.array([
            spot_means_g1[s_i],
            spot_means_g2[s_i]
        ])

        # Covariance matrix at each spot
        cov_mat = np.array([
            [spot_vars_g1[s_i], spot_covs[s_i]],
            [spot_covs[s_i], spot_vars_g2[s_i]]
        ])

        # Sample the expression values at the given spot
        lamb_s = np.random.multivariate_normal(spot_means, cov_mat)
        if poisson:
            poiss_mean = np.exp(lamb_s) * size_factors[s_i]
            x_s = np.random.poisson(poiss_mean)
        else:
            x_s = lamb_s
        sample.append(x_s)
    sample = np.array(sample).T
    return spot_covs, sample


