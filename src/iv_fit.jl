


"""
    Functions to sample from the conditional posteriors and compute marginal likelihoods.
"""
function post_sample_outcome(y, U, U_t_U, η, Σ, g)
    n = length(y)
    ψ = calc_psi(Σ)

    y_bar = Statistics.mean(y)
    η_bar = Statistics.mean(η)

    if (rank(U_t_U) < size(U_t_U, 1))
        error("Non-full rank model!")
    end

    B = g/(g+1) * inv(U_t_U)

    α = rand(Normal(y_bar - Σ[1,2]/Σ[2,2] * η_bar, ψ^2/n))

    Mean = B * U' * (y - Σ[1,2]/Σ[2,2] * η)
    β_tilde = rand(MvNormal(Mean, Symmetric(ψ^2 * B)))
    τ = β_tilde[1]
    β = β_tilde[2:end]    
    
    return (α = α, τ = τ, β = β)
end

function post_sample_treatment(x, V, V_t_V, ϵ, Σ, g)
    n = length(x)

    if (rank(V_t_V) < size(V_t_V, 1))
        error("Non-full rank model!")
    end

    ψ = calc_psi(Σ)
    ϵ_bar = Statistics.mean(ϵ)

    a = Σ[1,2]^2/(Σ[2,2] * ψ^2) + 1
    A = (g / (a*g + 1)) * inv(V_t_V)

    γ = rand(Normal(-Σ[1,2]/a * ϵ_bar, Σ[2,2]/(a*n))) 

    δ = rand(MvNormal(a * A * V' * (x - (Σ[1,2]/Σ[1,1]) * ϵ), Σ[2,2] * Symmetric(A)))
    
    return (γ = γ, δ = δ)
end

function post_sample_cov(ϵ, η, ν)
    n = length(ϵ)

    Q = [ϵ η]' * [ϵ η]
    if any(map(!isfinite, Q))
        error("Infinite sample covariance: Try increasing ν!")
    end
    Σ = rand(InverseWishart(ν + n, I + Q))
    return (Σ = Σ)
end

function marginal_likelihood_outcome(y, U, U_t_U, η, Σ, g)
    n = length(y)
    k = size(U, 2)

    if (rank(U_t_U) < size(U_t_U, 1))
        error("Non-full rank model!")
    end
    
    ψ = calc_psi(Σ)
    y_bar = Statistics.mean(y)
    η_bar = Statistics.mean(η)

    y_tilde = y - Σ[1,2]/Σ[2,2] * η 
    s = y_tilde' * (I - g/(g+1) * (U * inv(U_t_U) * U')) * y_tilde - n * (y_bar - Σ[1,2]/Σ[2,2] * η_bar)^2
    
    log_ml =  (-(k)/2)*log(g+1) - s/(2*ψ^2)
    return log_ml
end

function marginal_likelihood_treatment(x, V, V_t_V, ϵ, Σ, g)
    n = length(x)
    k = size(V, 2)

    if (rank(V_t_V) < size(V_t_V, 1))
        error("Non-full rank model!")
    end

    ψ = calc_psi(Σ)
    ϵ_bar = Statistics.mean(ϵ)

    a = Σ[1,2]^2/(Σ[2,2] * ψ^2) + 1

    x_tilde = (x - (Σ[1,2]/Σ[1,1]) * ϵ)
    t = (Σ[2,2]/Σ[1,1]) * ϵ'ϵ + x'x - 2 * (Σ[1,2]/Σ[1,1]) * ϵ'x - n * (Σ[1,2]^2/a^2) * ϵ_bar^2 - (a*g / (a*g+1)) * (x_tilde' * (V * inv(V_t_V) * V') * x_tilde)
    
    log_ml = (-k/2)*log(1 + g*a) - a*t/(2*Σ[2,2])
    return log_ml
end

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
    ν = 3,
    g_L_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_M_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
)

    # centre all regressors
    x = x .- mean(x)
    Z = Z .- mean(Z; dims = 1)
    W = W .- mean(W; dims = 1)
    
    n = size(Z, 1)
    p = size(Z, 2)
    k = size(W, 2)

    α_store = zeros(iter)
    τ_store = zeros(iter)
    β_store = zeros(iter, k)
    γ_store = zeros(iter)
    δ_store = zeros(iter, k+p)
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

    η = x - (γ_store[1]*ones(n) + V * δ_store[1,:])

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
        ϵ = y - (α_store[i] * ones(n) + τ_store[i] * x + W * β_store[i,:])

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
        η = x - (γ_store[i] * ones(n) + V * δ_store[i,:])
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
