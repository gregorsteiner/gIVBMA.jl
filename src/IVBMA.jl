module IVBMA

using LinearAlgebra, Distributions, Statistics, Random
using InvertedIndices, SpecialFunctions
using PrettyTables

export ivbma, describe, lps, posterior_predictive, plot


include("ivbma_priors.jl")
include("posterior_ml.jl")
include("ivbma_fit.jl")
include("ivbma_2c.jl")
include("ivbma_ng.jl")
include("ivbma_tools.jl")

"""
    Bayesian model averaging with instrumental variables: Sample from the joint posterior function of all parameters and models.

    There are currently two ways to use this function: One can either provide a matrix of potential instruments Z and a matrix of potential covariates W or just one of them.
    If both are given, Z can only be included in the treatment model, while W can be included in both models. If only one matrix of potential instruments and covariates
    is specified all of them can be included in both models. The two-component g-prior can only be used in the first case.

    # Arguments
    - `y::AbstractVector{<:Real}` a vector containing the outcome
    - `x::AbstractVector{<:Real}` a vector containing the endogenous variable or treatment
    - `Z::AbstractMatrix{<:Real}` a matrix of potential instruments
    - `W::AbstractMatrix{<:Real}` a matrix of exogenous control variates
    - `iter::Integer = 2000` the number of iterations of the Gibbs sampler
    - `burn::Integer = 1000` the number of initial iteratios to discard as burn-in (should be less than `iter`)
    - `two_comp::Bool = false` if true the two-componentn g-prior is used for the treatment parameters
    - `dist::String = "Gaussian"` the distribution of the treatment; currently "Gaussian" (default) and "PLN" (Poisson-Log-Normal) are implemented
    - `κ2` the prior variance on the intercept (only relevant for the Poisson Log-Normal model)
    - `ν_prior::Function = ν -> log(jp_ν(ν, size(Z, 2) + size(W, 2) + 3))` the hyperprior on the covariance degrees of freedom ν
    - `g_L_prior` the hyperprior on g for the outcome model (defaults to the hyper-g/n prior with a = 3)
    - `g_M_prior` the hyperprior on g for the treatment model (defaults to the hyper-g/n prior with a = 3)
    - `g_l_prior` the hyperprior on the larger g in the two-component prior 
    - `g_s_prior` the hyperprior on the smaller g in the two-component prior 
    - `m::Union{AbstractVector, Nothing} = nothing` the prior mean model size (defaults to k/2 where k is the number of covariates)

"""
function ivbma(
    y::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    two_comp::Bool = false,
    dist::String = "Gaussian",
    κ2::Number = 100,
    ν_prior::Function = ν -> log(jp_ν(ν, size(Z, 2) + size(W, 2) + 3)),
    g_L_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_M_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_l_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_s_prior::Function = g -> log(hyper_g_n(g; a = 4, n = length(y))),
    m::Union{AbstractVector, Nothing} = nothing,
    r_prior::Distribution = Exponential(1)
)

    # Check that x is not constant
    if length(unique(x)) == 1
        error("The treatment x is constant!")
    end

    # centre regressors (only if the treatment is Gaussian)
    if dist == "Gaussian"
        x = x .- mean(x)
    end
    Z = Z .- mean(Z; dims = 1)
    W = W .- mean(W; dims = 1)

    # Use default prior mean model size of not specified
    k = size(W, 2)
    p = size(Z, 2)
    if isnothing(m)
        m = [k/2, (k+p)/2]
    end

    # Fit models
    if dist == "Gaussian" && !two_comp
        res = ivbma_mcmc(y, x, Z, W, iter, burn, ν_prior, g_L_prior, g_M_prior, m)
    elseif dist == "Gaussian" && two_comp
        res = ivbma_mcmc_2c(y, x, Z, W, iter, burn, ν_prior, g_L_prior, g_l_prior, g_s_prior, m)
    elseif dist !== "Gaussian" && !two_comp
        res = ivbma_mcmc_ng(y, x, Z, W, iter, burn, κ2, ν_prior,  g_L_prior, g_M_prior, m, dist, r_prior)
    elseif dist !== "Gaussian" && two_comp
        res = ivbma_mcmc_ng_2c(y, x, Z, W, iter, burn, κ2, ν_prior,  g_L_prior, g_l_prior, g_s_prior, m, dist, r_prior)
    end

    return res
end

function ivbma(
    y::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    two_comp::Bool = false,
    dist::String = "Gaussian",
    κ2::Number = 100,
    ν_prior::Function = ν -> log(jp_ν(ν, size(Z, 2) + 3)),
    g_L_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_M_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    m::Union{AbstractVector, Nothing} = nothing,
    r_prior::Distribution = Exponential(1)
)

    # Check that x is not constant
    if length(unique(x)) == 1
        error("The treatment x is constant!")
    end
    
    # Add error for two-component prior (which only makes sense if one provides instruments and exogenous covariates)
    if two_comp
        error("The two-component prior cannot be used without exogenous covariates.")
    end

    # centre regressors (only centre treatment if Gaussian)
    if dist == "Gaussian"
        x = x .- mean(x)
    end
    Z = Z .- mean(Z; dims = 1)

    # Use default prior mean model size of not specified
    n = length(y)
    p = size(Z, 2)
    if isnothing(m)
        m = [p/2, p/2]
    end

    # Fit models
    if dist == "Gaussian"
        res = ivbma_mcmc(y, x, Matrix{Float64}(undef, n, 0), Z, iter, burn, ν_prior, g_L_prior, g_M_prior, m)
    elseif dist !== "Gaussian"
        res = ivbma_mcmc_ng(y, x, Matrix{Float64}(undef, n, 0), Z, iter, burn, κ2, ν_prior,  g_L_prior, g_M_prior, m, dist, r_prior)
    end

    return res
end


end
