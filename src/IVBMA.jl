module IVBMA

using LinearAlgebra, Distributions, Statistics
using InvertedIndices, SpecialFunctions
using StatsPlots, Infiltrator

export ivbma, lpd, posterior_predictive, plot


include("ivbma_priors.jl")
include("posterior_ml.jl")
include("ivbma_fit.jl")
include("ivbma_2c.jl")
include("ivbma_pln.jl")
include("ivbma_tools.jl")

"""
    Main wrapper function.
"""
function ivbma(
    y::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    two_comp::Bool = false,
    pln::Bool = false,
    κ2::Number = 100,
    ν_prior::Function = ν -> log(jp_ν(ν, size(Z, 2) + size(W, 2) + 3)),
    g_L_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_M_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_l_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_s_prior::Function = g -> log(hyper_g_n(g; a = 4, n = length(y))),
    m::Union{AbstractVector, Nothing} = nothing
)

    # centre regressors (don't centre treatment if the treatment model is a Poisson Log-normal)
    if !pln
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
    if !pln && !two_comp
        res = ivbma_mcmc(y, x, Z, W, iter, burn, ν_prior, g_L_prior, g_M_prior, m)
    elseif !pln && two_comp
        res = ivbma_mcmc_2c(y, x, Z, W, iter, burn, ν_prior, g_L_prior, g_l_prior, g_s_prior, m)
    elseif pln && !two_comp
        res = ivbma_mcmc_pln(y, x, Z, W, iter, burn, κ2, ν_prior,  g_L_prior, g_M_prior, m)
    elseif pln && two_comp
        res = ivbma_mcmc_pln_2c(y, x, Z, W, iter, burn, κ2, ν_prior,  g_L_prior, g_l_prior, g_s_prior, m)
    end

    return res
end

"""
    A second method with potentially invalid instruments, i.e. the user only provides one matrix of potential instruments,
    all of which could be included in both the outcome and treatment model.
"""
function ivbma(
    y::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    two_comp::Bool = false,
    pln::Bool = false,
    κ2::Number = 100,
    ν_prior::Function = ν -> log(jp_ν(ν, size(Z, 2) + 3)),
    g_L_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_M_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    m::Union{AbstractVector, Nothing} = nothing
)

    # Add error for two-component prior (which only makes sense if one provides instruments and exogenous covariates)
    if two_comp
        error("The two-component prior cannot be used without exogenous covariates.")
    end

    # centre regressors (don't centre treatment if the treatment model is a Poisson Log-normal)
    if !pln
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
    if !pln
        res = ivbma_mcmc(y, x, Matrix{Float64}(undef, n, 0), Z, iter, burn, ν_prior, g_L_prior, g_M_prior, m)
    elseif pln
        res = ivbma_mcmc_pln(y, x, Matrix{Float64}(undef, n, 0), Z, iter, burn, κ2, ν_prior,  g_L_prior, g_M_prior, m)
    end

    return res
end


end
