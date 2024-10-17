

"""
    Implement the iv procedure without model steps
"""
struct PostSampleIV
    α::Vector{Float64}
    τ::Vector{Float64}
    β::Matrix{Float64}
    γ::Vector{Float64}
    δ::Matrix{Float64}
    Σ::Array{Matrix{Float64}}
    g_L::Vector{Float64}
    g_M::Vector{Float64}
    ν::Vector{Float64}
end

function iv_fit(
    y::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    ν = 10,
    g_L_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_M_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
)

    # centre all regressors
    x = x .- mean(x)
    Z = Z .- mean(Z; dims = 1)
    
    n = size(Z, 1)
    p = size(Z, 2)
    k = size(W, 2)

    α_store = zeros(iter)
    τ_store = zeros(iter)
    β_store = zeros(iter, p)
    γ_store = zeros(iter)
    δ_store = zeros(iter, p)
    Σ_store = Array{Matrix{Float64}}(undef, iter)
    Σ_store[1] = [1.0 0.0; 0.0 1.0]

    g_L_store = zeros(iter); g_L_store[1] = n
    propVar_g_L = 1/2; acc_g_L = 0
    g_M_store = zeros(iter); g_M_store[1] = n
    propVar_g_M = 1/2; acc_g_M = 0

    # Some precomputations
    U = [x W]
    U_t_U = U'U

    V = [Z W]
    V_t_V = V'V

    η = x - (γ_store[1]*ones(n) + Z * δ_store[1,:])

    for i in 2:iter

        # Step 1.1: Draw g_l
        curr = g_L_store[i-1]
        prop = rand(LogNormal(log(curr), propVar_g_L))

        post_prop = marginal_likelihood_outcome(y, U, U_t_U, η, Σ_store[i-1], prop) + g_L_prior(prop) - logpdf(LogNormal(log(curr), propVar_g_L), prop)
        post_curr = marginal_likelihood_outcome(y, U, U_t_U, η, Σ_store[i-1], curr) + g_L_prior(curr) - logpdf(LogNormal(log(prop), propVar_g_L), curr)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            g_L_store[i] = prop
            acc_g_L += 1
        else
            g_L_store[i] = curr
        end

        propVar_g_L = adjust_variance(propVar_g_L, acc_g_L, i)
        
        
        # Step 1.1: Update outcome parameters
        draw = post_sample_outcome(y, U, U_t_U, η, Σ_store[i-1], g_L_store[i])
        α_store[i] = draw.α
        τ_store[i] = draw.τ
        if k > 0
            β_store[i, :] = draw.β
        end

        # Step 2.0: Precompute residuals
        ϵ = y - (α_store[i] * ones(n) + τ_store[i] * x + Z * β_store[i,:])

        # Step 2.1: Update g_M
        curr = g_M_store[i-1]
        prop = rand(LogNormal(log(curr), propVar_g_M))

        post_prop = marginal_likelihood_treatment(x, V, V_t_V, ϵ, Σ_store[i-1], prop) + g_M_prior(prop) - logpdf(LogNormal(log(curr), propVar_g_M), prop)
        post_curr = marginal_likelihood_treatment(x, V, V_t_V, ϵ, Σ_store[i-1], curr) + g_M_prior(curr) - logpdf(LogNormal(log(prop), propVar_g_M), curr)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            g_M_store[i] = prop
            acc_g_M += 1
        else
            g_M_store[i] = curr
        end

        propVar_g_M = adjust_variance(propVar_g_M, acc_g_M, i)

        # Step 2.2: Update treatment parameters
        draw = post_sample_treatment(x, V, V_t_V, ϵ, Σ_store[i-1], g_M_store[i])
        γ_store[i] = draw.γ
        δ_store[i, :] = draw.δ

        # Step 3: Update covariance Matrix
        η = x - (γ_store[i] * ones(n) + Z * δ_store[i,:])
        Σ_store[i] = post_sample_cov(ϵ, η, ν)

    end

    return PostSampleIV(
        α_store[(burn+1):end],
        τ_store[(burn+1):end],
        β_store[(burn+1):end,:],
        γ_store[(burn+1):end],
        δ_store[(burn+1):end,:],
        Σ_store[(burn+1):end],
        g_L_store[(burn+1):end],
        g_M_store[(burn+1):end],
        []
    )
end

"""
    Define a second method with only instruments but no covariates.
"""
function iv_fit(
    y::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    ν = 10,
    g_L_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_M_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
)
    return iv_fit(
        y,
        x,
        Z,
        Matrix{Float64}(undef, length(y), 0);
        iter = iter,
        burn = burn,
        ν = ν,
        g_L_prior = g_L_prior,
        g_M_prior = g_M_prior,
    )
end
