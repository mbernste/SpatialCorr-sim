"""
Perform inference of latent mean and variance using a Bayesian inference on a
hierarchical Poisson-lognormal model.

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

#import stan
import numpy as np
import os
import stan

# The Poisson-lognormal model used to generate spatial UMI counts
POISS_LOGNORMAL = """
data{
   int<lower=0> N;
   int y[N];
   real log_s[N];
}
 
parameters {
   real mu;
   real<lower=0> sigma;
   vector[N] log_lambda;
}
 
model {
    sigma ~ cauchy(0, 5);
    mu ~ normal(0, 20);
    log_lambda ~ normal(mu, sigma);
    for (n in 1:N)
        target += poisson_log_lpmf(y[n] | log_s[n] + log_lambda[n]);    
}
"""

def fit(y, size_factors, verbose=0):
    data = {
        "N": len(y),
        "y": y.astype(int),
        "log_s": np.log(size_factors)
    }
    posterior = stan.build(POISS_LOGNORMAL, data=data)
    fit = posterior.sample(
        num_chains=4, 
        num_samples=1000
    )
    sigma = fit["sigma"]
    mu = fit["mu"]

    if verbose > 0:
        print(f"Mu: {mu}")
        print(f"Sigma: {sigma}")

    return mu, sigma

if __name__ == '__main__':
    main()
