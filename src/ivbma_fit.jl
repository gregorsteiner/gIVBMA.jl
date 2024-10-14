

"""
    A type to store the posterior sample in
"""
struct PostSample
    α::Vector{Float64}
    τ::Vector{Float64}
    β::Matrix{Float64}
    γ::Vector{Float64}
    δ::Matrix{Float64}
    Σ::Array{Matrix{Float64}}
    L::Matrix{Bool}
    M::Matrix{Bool}
    g_L::Vector{Float64}
    g_M::Vector{Float64}
    ν::Vector{Float64}
end



"""
    Functions to sample from the conditional posteriors and compute marginal likelihoods.
"""
function post_sample_outcome(y, U, U_t_U, η, Σ, g)
    n = length(y)
    ψ = calc_psi(Σ)

    y_bar = Statistics.mean(y)
    η_bar = Statistics.mean(η)

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
    ψ = calc_psi(Σ)
    ϵ_bar = Statistics.mean(ϵ)

    a = Σ[1,2]^2/(Σ[2,2] * ψ^2) + 1

    x_tilde = (x - (Σ[1,2]/Σ[1,1]) * ϵ)
    t = (Σ[2,2]/Σ[1,1]) * ϵ'ϵ + x'x - 2 * (Σ[1,2]/Σ[1,1]) * ϵ'x - n * (Σ[1,2]^2/a^2) * ϵ_bar^2 - (a*g / (a*g+1)) * (x_tilde' * (V * inv(V_t_V) * V') * x_tilde)
    
    log_ml = (-k/2)*log(1 + g*a) - a*t/(2*Σ[2,2])
    return log_ml
end

"""
    Fit ivbma models with a regular g-prior and hyperpriors on g and ν.
"""
function ivbma(
    y::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    ν_prior::Function = ν -> log(jp_ν(ν, size(Z, 2) + size(W, 2) + 3)),
    g_L_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_M_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    m::Union{AbstractVector, Nothing} = nothing
)

    # centre all regressors
    x = x .- mean(x)
    Z = Z .- mean(Z; dims = 1)
    W = W .- mean(W; dims = 1)
    
    n = size(W, 1)
    k = size(W, 2)
    p = size(Z, 2)
 
    if isnothing(m)
        m_o = k/2
        m_t = (k+p)/2
    else
        m_o = m[1]
        m_t = m[2]
    end

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

    ν_store = zeros(iter); ν_store[1] = 10
    propVar_ν = 1/2; acc_ν = 0

    W_L = W
    W_M = [Z W]

    M_incl = Matrix{Bool}(undef, iter, k+p)
    M_incl[1,:] = sample([true, false], k+p, replace = true)
    L_incl = Matrix{Bool}(undef, iter, k)
    L_incl[1,:] = sample([true, false], k, replace = true)

    # Some precomputations
    U = [x W_L[:,L_incl[1,:]]]
    U_t_U = U'U

    V = W_M[:,M_incl[1,:]]
    V_t_V = V'V

    η = x - (γ_store[1]*ones(n) + W_M * δ_store[1,:])

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
        
        # Step 1.2: Update outcome model
        curr = L_incl[i-1,:]
        prop = L_incl[i-1,:]
        ind = sample((1:k))
        prop[ind] = !prop[ind]

        U_prop = [x W_L[:,prop]]
        U_t_U_prop = U_prop'U_prop

        post_prop = marginal_likelihood_outcome(y, U_prop, U_t_U_prop, η, Σ_store[i-1], g_L_store[i]) + model_prior(prop, k, 1, m_o)
        post_curr = marginal_likelihood_outcome(y, U, U_t_U, η, Σ_store[i-1], g_L_store[i]) + model_prior(curr, k, 1, m_o)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            L_incl[i,:] = prop
            U = U_prop
            U_t_U = U_t_U_prop
        else
            L_incl[i,:] = curr
        end

        # Step 1.3: Update outcome parameters
        draw = post_sample_outcome(y, U, U_t_U, η, Σ_store[i-1], g_L_store[i])
        α_store[i] = draw.α
        τ_store[i] = draw.τ
        β_store[i, L_incl[i,:]] = draw.β
        

        # Step 2.0: Precompute residuals
        ϵ = y - (α_store[i] * ones(n) + τ_store[i] * x + W_L * β_store[i,:])

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

        # Step 2.2: Update treatment model
        curr = M_incl[i-1,:]
        prop = M_incl[i-1,:]
        ind = sample(1:(k+p))
        prop[ind] = !prop[ind]

        V_prop = W_M[:, prop]
        V_t_V_prop = V_prop'V_prop

        post_prop = marginal_likelihood_treatment(x, V_prop, V_t_V_prop, ϵ, Σ_store[i-1], g_M_store[i]) + model_prior(prop, k+p, 1, m_t)
        post_curr = marginal_likelihood_treatment(x, V, V_t_V, ϵ, Σ_store[i-1], g_M_store[i]) + model_prior(curr, k+p, 1, m_t)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            M_incl[i,:] = prop
            V = V_prop
            V_t_V = V_t_V_prop
        else
            M_incl[i,:] = curr
        end

        # Step 2.3: Update treatment parameters
        draw = post_sample_treatment(x, V, V_t_V, ϵ, Σ_store[i-1], g_M_store[i])
        γ_store[i] = draw.γ
        δ_store[i, M_incl[i,:]] = draw.δ


        # Step 3: Update covariance Matrix
        η = x - (γ_store[i] * ones(n) + W_M * δ_store[i,:])
        Σ_store[i] = post_sample_cov(ϵ, η, ν_store[i-1]) 

        # Step 4: Update ν
        curr = ν_store[i-1]
        prop = rand(Truncated(Normal(curr, propVar_ν), 1, Inf))

        post_prop = logpdf(InverseWishart(prop, [1 0; 0 1]), Σ_store[i]) + ν_prior(prop) - logpdf(Truncated(Normal(curr, propVar_ν), 1, Inf), prop)
        post_curr = logpdf(InverseWishart(curr, [1 0; 0 1]), Σ_store[i]) + ν_prior(curr) - logpdf(Truncated(Normal(prop, propVar_ν), 1, Inf), curr)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            ν_store[i] = prop
            acc_ν += 1
        else
            ν_store[i] = curr
        end

        propVar_ν = adjust_variance(propVar_ν, acc_ν, i)
    end

    return PostSample(
        α_store[(burn+1):end],
        τ_store[(burn+1):end],
        β_store[(burn+1):end,:],
        γ_store[(burn+1):end],
        δ_store[(burn+1):end,:],
        Σ_store[(burn+1):end],
        L_incl[(burn+1):end,:],
        M_incl[(burn+1):end,:],
        g_L_store[(burn+1):end],
        g_M_store[(burn+1):end],
        ν_store[(burn+1):end]
    )
end


"""
    Fit IVBMA with potentially invalid instruments. In this method we only consider a matrix Z that includes both potential instruments
    and exogenous covariates. We still consider hyperpriors on g, but ν is ifxed now.
"""
function ivbma(
    y::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    ν = 10,
    g_L_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_M_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    m::Union{AbstractVector, Nothing} = nothing
)

    # centre all regressors
    x = x .- mean(x)
    Z = Z .- mean(Z; dims = 1)
    
    n = size(Z, 1)
    p = size(Z, 2)
 
    if isnothing(m)
        m_o = p/2
        m_t = p/2
    else
        m_o = m[1]
        m_t = m[2]
    end

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

    M_incl = Matrix{Bool}(undef, iter, p)
    M_incl[1,:] = sample([true, false], p, replace = true)
    L_incl = Matrix{Bool}(undef, iter, p)
    L_incl[1,:] = sample([true, false], p, replace = true)

    # Some precomputations
    U = [x Z[:,L_incl[1,:]]]
    U_t_U = U'U

    V = Z[:,M_incl[1,:]]
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
        
        # Step 1.2: Update outcome model
        curr = L_incl[i-1,:]
        prop = L_incl[i-1,:]
        ind = sample((1:p))
        prop[ind] = !prop[ind]

        U_prop = [x Z[:,prop]]
        U_t_U_prop = U_prop'U_prop

        post_prop = marginal_likelihood_outcome(y, U_prop, U_t_U_prop, η, Σ_store[i-1], g_L_store[i]) + model_prior(prop, p, 1, m_o)
        post_curr = marginal_likelihood_outcome(y, U, U_t_U, η, Σ_store[i-1], g_L_store[i]) + model_prior(curr, p, 1, m_o)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            L_incl[i,:] = prop
            U = U_prop
            U_t_U = U_t_U_prop
        else
            L_incl[i,:] = curr
        end

        # Step 1.3: Update outcome parameters
        draw = post_sample_outcome(y, U, U_t_U, η, Σ_store[i-1], g_L_store[i])
        α_store[i] = draw.α
        τ_store[i] = draw.τ
        β_store[i, L_incl[i,:]] = draw.β
        

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

        # Step 2.2: Update treatment model
        curr = M_incl[i-1,:]
        prop = M_incl[i-1,:]
        ind = sample(1:(p))
        prop[ind] = !prop[ind]

        V_prop = Z[:, prop]
        V_t_V_prop = V_prop'V_prop

        post_prop = marginal_likelihood_treatment(x, V_prop, V_t_V_prop, ϵ, Σ_store[i-1], g_M_store[i]) + model_prior(prop, p, 1, m_t)
        post_curr = marginal_likelihood_treatment(x, V, V_t_V, ϵ, Σ_store[i-1], g_M_store[i]) + model_prior(curr, p, 1, m_t)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            M_incl[i,:] = prop
            V = V_prop
            V_t_V = V_t_V_prop
        else
            M_incl[i,:] = curr
        end

        # Step 2.3: Update treatment parameters
        draw = post_sample_treatment(x, V, V_t_V, ϵ, Σ_store[i-1], g_M_store[i])
        γ_store[i] = draw.γ
        δ_store[i, M_incl[i,:]] = draw.δ


        # Step 3: Update covariance Matrix
        η = x - (γ_store[i] * ones(n) + Z * δ_store[i,:])
        Σ_store[i] = post_sample_cov(ϵ, η, ν)

        # Step 3.1: If there is no instrument, sample Σ from prior instead
        if !has_instrument(L_incl[i,:], M_incl[i,:])
            Σ_store[i] = rand(InverseWishart(ν, [1 0; 0 1]))
        end

    end

    return PostSample(
        α_store[(burn+1):end],
        τ_store[(burn+1):end],
        β_store[(burn+1):end,:],
        γ_store[(burn+1):end],
        δ_store[(burn+1):end,:],
        Σ_store[(burn+1):end],
        L_incl[(burn+1):end,:],
        M_incl[(burn+1):end,:],
        g_L_store[(burn+1):end],
        g_M_store[(burn+1):end],
        []
    )
end

