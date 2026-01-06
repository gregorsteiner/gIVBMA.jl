module gIVBMA

using LinearAlgebra, Distributions, Statistics, Random
using InvertedIndices, SpecialFunctions, LogExpFunctions
using CSV, DataFrames

export givbma, lps, rbw

include("priors.jl")
include("posterior_ml.jl")
include("non_gaussian.jl")
include("mcmc.jl")
include("lps.jl")
include("rao_blackwell.jl")


"""
    Bayesian model averaging with instrumental variables: Sample from the joint posterior distribution of all parameters and models.

    There are currently two ways to use this function: One can either provide a matrix of potential instruments Z and a matrix of potential covariates W or just one of them.
    If both are given, Z can only be included in the treatment model, while W can be included in both models. If only one matrix of potential instruments and covariates
    is specified all of them can be included in both models. The two-component g-prior can only be used in the first case.

    # Arguments
    - `y::AbstractVector{<:Real}` a vector containing the outcome
    - `X::AbstractVecOrMat{<:Real}` a vector containing the endogenous variables
    - `Z::AbstractMatrix{<:Real}` a matrix of potential instruments
    - `W::AbstractMatrix{<:Real}` a matrix of exogenous control variates (optional)
    - `iter::Integer = 2000` the number of iterations of the Gibbs sampler
    - `burn::Integer = 1000` the number of initial iteratios to discard as burn-in (should be less than `iter`)
    - `dist::Vector{String} = repeat(["Gaussian"], size(X, 2) + 1)` a vector of strings containing the distributions of the outcome and all endogenous variabes; currently "Gaussian" (default), "PLN" (Poisson-Log-Normal), and "BL" (Beta-Logistic) are implemented
    - `two_comp::Bool = false` a Boolean indicating whether the two-component g-prior should be used for the treatment parameters. This is currently only implemented for a single endogenous variable.
    - `g_prior = "BRIC"` the prior choice of g; currently "BRIC" (g = max(n, p^2)) and a "hyper-g/n" prior are implemented
    - `m::Union{AbstractVector, Nothing} = nothing` the prior mean model size (defaults to k/2 or (k+p)/2 where k is the number of covariates and p is the number of instruments)
    - `cov_prior::String = "IW"` the covariance prior structure; the inverse Wishart ("IW") prior and Cholesky-based ("Cholesky") priors are available
    - `ν = size(X::AbstractVector{<:Real}, 2) + 2` the covariance degrees of freedom ν
    - `ω_a::Number = 1.0` the prior variance on the scaled residual covariance (only relevant for the Cholesky prior)
    - `r_prior::Distribution = Exponential(1)` the prior on the dispersion parameter r (only relevant for Beta-Logistic model)
"""
function givbma(
    y::AbstractVector{<:Real},
    X::AbstractVecOrMat{<:Real},
    Z::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    dist::Vector{String} = repeat(["Gaussian"], size(X, 2) + 1),
    two_comp::Bool = false,
    g_prior::String = "BRIC",
    m::Union{AbstractVector, Nothing} = nothing,
    cov_prior::String = "IW",
    ν::Union{Nothing, Number} = nothing,
    ω_a::Number = 1.0,
    r_prior::Distribution = Exponential(1)
)
    # if X is a vector turn it into an nx1 matrix
    if ndims(X) == 1
        X = permutedims(X)'
    end

    # Use default prior mean model size if not specified
    k = size(W, 2)
    p = size(Z, 2)
    if isnothing(m)
        m = [k/2, (k+p)/2]
    end

    res = givbma_mcmc(y, X, Z, W, iter, burn, dist, two_comp, g_prior, m, cov_prior, ν, ω_a, r_prior)

    return res
end

function givbma(
    y::AbstractVector{<:Real},
    X::AbstractVecOrMat{<:Real},
    Z::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    dist::Vector{String} = repeat(["Gaussian"], size(X, 2) + 1),
    two_comp = false,
    g_prior::String = "BRIC",
    m::Union{AbstractVector, Nothing} = nothing,
    cov_prior::String = "IW",
    ν::Union{Nothing, Number} = nothing,
    ω_a::Number = 1.0,
    r_prior::Distribution = Exponential(1)
)
    # if X is a vector turn it into an nx1 matrix
    if ndims(X) == 1
        X = permutedims(X)'
    end

    # Use default prior mean model size if not specified
    n = length(y)
    p = size(Z, 2)
    if isnothing(m)
        m = [p/2, p/2]
    end

    res = givbma_mcmc(y, X, Matrix{Float64}(undef, n, 0), Z, iter, burn, dist, two_comp, g_prior, m, cov_prior, ν, ω_a, r_prior)

    return res
end


end
