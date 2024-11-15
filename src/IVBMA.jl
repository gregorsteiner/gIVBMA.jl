module IVBMA

using LinearAlgebra, Distributions, Statistics, Random
using InvertedIndices, SpecialFunctions

export ivbma, lps

include("priors.jl")
include("posterior_ml.jl")
include("non_gaussian.jl")
include("mcmc.jl")
include("lps.jl")


"""
    Bayesian model averaging with instrumental variables: Sample from the joint posterior function of all parameters and models.

    There are currently two ways to use this function: One can either provide a matrix of potential instruments Z and a matrix of potential covariates W or just one of them.
    If both are given, Z can only be included in the treatment model, while W can be included in both models. If only one matrix of potential instruments and covariates
    is specified all of them can be included in both models. The two-component g-prior can only be used in the first case.

    # Arguments
    - `y::AbstractVector{<:Real}` a vector containing the outcome
    - `X::AbstractVecOrMat{<:Real}` a vector containing the endogenous variables
    - `Z::AbstractMatrix{<:Real}` a matrix of potential instruments
    - `W::AbstractMatrix{<:Real}` a matrix of exogenous control variates
    - `iter::Integer = 2000` the number of iterations of the Gibbs sampler
    - `burn::Integer = 1000` the number of initial iteratios to discard as burn-in (should be less than `iter`)
    - `dist::String = repeat(["Gaussian"], size(X, 2))` a vector of strings containing the distribution s of all endogenous variabes; currently "Gaussian" (default), "PLN" (Poisson-Log-Normal), and "BL" (Beta-Logistic) are implemented
    - `ν = size(X::AbstractVector{<:Real}, 2) + 2` the covariance degrees of freedom ν
    - `g_prior = "BRIC"` the prior choice of g; currently BRIC (g = max(n, p^2)) and a hyper-g/n prior are implemnted
    - `m::Union{AbstractVector, Nothing} = nothing` the prior mean model size (defaults to k/2 where k is the number of covariates)
    - `r_prior::Distribution = Exponential(1)` the prior on the dispersion parameter r (only relevant for Beta-Logistic model)
"""
function ivbma(
    y::AbstractVector{<:Real},
    X::AbstractVecOrMat{<:Real},
    Z::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    dist::Vector{String} = repeat(["Gaussian"], size(X, 2)),
    ν::Union{Nothing, Number} = nothing,
    g_prior::String = "BRIC",
    m::Union{AbstractVector, Nothing} = nothing,
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

    res = ivbma_mcmc(y, X, Z, W, dist, iter, burn, ν, m, g_prior, r_prior)

    return res
end

function ivbma(
    y::AbstractVector{<:Real},
    X::AbstractVecOrMat{<:Real},
    Z::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    dist::Vector{String} = repeat(["Gaussian"], size(X, 2)),
    ν::Union{Nothing, Number} = nothing,
    m::Union{AbstractVector, Nothing} = nothing,
    g_prior::String = "BRIC",
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

    res = ivbma_mcmc(y, X, Matrix{Float64}(undef, n, 0), Z, dist, iter, burn, ν, m, g_prior, r_prior)

    return res
end


end
