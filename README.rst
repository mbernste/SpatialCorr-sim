======================================================
SpatialCorr-Sim: Simulate spatial transcriptomics data
======================================================

SpatialCorr-sim is a set a Python framework for simulating spatial transcriptomics data with spatially varying correlation across the slide. Specifically, this framework takes as input a spatial transcriptomics dataset and uses this dataset to "seed" simulated datasets. That is, simulated datasets are generated such that the marginal distribution of UMI counts match those in the input dataset, but the correlation between genes can be specified by the user.Moreover, this package contains a framework for generating random patterns of spatially varying correlation among a set of genes.  Altogether, SpatialCorr-sim can be used for the evaluation of computational methods that analyze the correlation among genes in spatial transcriptomics data. 

For instructions on installing and running SpatialCorr-sim, see the GitHub repository: https://github.com/mbernste/SpatialCorr-sim 

