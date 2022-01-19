.. SpatialDC documentation master file, created by
   sphinx-quickstart on Mon Sep 20 18:47:44 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SpatialCorr-sim -- Simulate spatial transcriptomics data with spatially varying correlation
===========================================================================================

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   api


SpatialCorr-sim is a set a Python framework for simulating spatial transcriptomics data with spatially varying correlation across the slide. Specifically, this framework takes as input a spatial transcriptomics dataset and uses this dataset to "seed" simulated datasets. That is, simulated datasets are generated such that the marginal distribution of UMI counts match those in the input dataset, but the correlation between genes can be specified by the user.Moreover, this package contains a framework for generating random patterns of spatially varying correlation among a set of genes.

Altogether, SpatialCorr-sim can be used for the evaluation of computational methods that analyze the correlation among genes in spatial transcriptomics data.

.. raw:: html

    <p align="center">
    <img src="https://raw.githubusercontent.com/mbernste/SpatialCorr-sim/master/imgs/SpatialCorr_sim_figure.png"/>
    </p>

