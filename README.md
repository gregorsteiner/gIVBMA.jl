# gIVBMA.jl

[![Build Status](https://github.com/gregorsteiner/IVBMA.jl/workflows/CI/badge.svg)](https://github.com/gregorsteiner/IVBMA.jl/actions)

Bayesian Model Averaging in instrumental variable models.

## Usage

The main function is `givbma` which requires an outcome vector `y`, a matrix of endogenous variables `X`, a matrix of potential instruments `Z`, and a matrix of potential covariates `W`:
```
fit = givbma(y, X, Z, W)
```
will return a `GIVBMA` object containing a posterior sample of the model parameters averaged over the outcome and treatment models, the visited outcome and treatment models, and the input data. Alternatively, one can only specify a matrix `Z` of potential instruments and covariates:
```
fit = givbma(y, X, Z)
```
allows all columns of `Z` to be included in the outcome and treatment model.


