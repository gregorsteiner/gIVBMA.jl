# gIVBMA.jl

[![Build Status](https://github.com/gregorsteiner/IVBMA.jl/workflows/CI/badge.svg)](https://github.com/gregorsteiner/IVBMA.jl/actions)

Bayesian Model Averaging in Instrumental Variable Models. This package implements the gIVBMA method proposed in Steiner and Steel (2025).

## Installation

The package is not yet available from the general registry, but can be installed directly from GitHub:
```julia
using Pkg; Pkg.add(url="https://github.com/gregorsteiner/gIVBMA.jl.git")
```

## Usage

The main function is `givbma` which requires an outcome vector `y`, a matrix of endogenous variables `X`, a matrix of potential instruments `Z`, and a matrix of potential (exogenous) covariates `W`:
```julia
fit = givbma(y, X, Z, W)
```
will return a `GIVBMA` object containing a posterior sample of the model parameters averaged over the outcome and treatment models, the visited outcome and treatment models, and the input data. Alternatively, one can only specify a matrix `Z` of potential instruments and covariates:
```julia
fit = givbma(y, X, Z)
```
which allows all columns of `Z` to be included in the outcome and treatment model. Note that an intercept is automatically included and there is no need to include it in either `X`, `W`, or `Z`. The optional keyword arguments are:
* `iter`: the number of iterations.
* `burn`: the number of iterations discarded as burn-in; the function returns `iter-burn` posterior samples.
* `dist`: a vector of distributions of the outcome and the endogenous variables (defaults to Gaussian). Currently, we support `"Gaussian"`, `"PLN"` (Poisson-Log-Normal), and `"BL"` (Beta-Logistic).
* `g_prior`: the choice of the g hyperparameter. Currently, we support `"BRIC"` (default) and `"hyper-g/n"`.
* `two_comp`: a Boolean indicating whether the two-component g-prior should be used in the treatment model (defaults to `false`). The two-component prior can only be used with a single endogenous variable (i.e. `X` only has a single column).
* `ν`: the degrees of freedom parameter for the inverse Wishart prior on the covariance matrix. If not specified, this defaults to an Exponential hyperprior.
* `m`: the prior mean model size. If not specified, it defaults to `k/2` in the outcome model and `(k+p)/2` in the treatment model, where `k` is the number of exogenous covariates and `p` is the number of instruments.
* `r_prior`: a `Distribution` object specifying the prior on additional parameters for any non-Gaussian distributions involved. Currently, this only includes the dispersion parameter of the Beta-Logistic distribution (which defaults to an Exponential with scale 1).

A useful function is `rbw`:
```julia
posterior = rbw(fit)
```
which returns a vector of `Distribution` objects containing Rao-Blackwellized estimates of the marginal posterior distributions of each component of the treatment effect vector. This can be used to extract summary statistics or plot the marginal posterior distribution:
```julia
# Extract a vector of posterior means or medians
map(mean, posterior)
map(median, posterior)

# Plot the posterior distribution corresponding to the first column of X
using CairoMakie
lines(posterior[1])
```

## References

Steiner, G. and Steel, M. (2025). Bayesian Model Averaging in Causal Instrumental Variable Models. arXiv:2504.13520
