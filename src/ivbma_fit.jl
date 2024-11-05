

"""
    A type to store the posterior sample in
"""
struct PostSample
    α::Vector{Float64} # Outcome intercept
    τ::Vector{Float64} # Treatment effect of the endogenous variable
    β::Matrix{Float64} # Outcome remaining slope coefficients
    γ::Vector{Float64} # Treatment intercept
    δ::Matrix{Float64} # Treatment slope coefficients
    Σ::Array{Matrix{Float64}} # Residual covariance matrix
    L::Matrix{Bool} # Outcome models
    M::Matrix{Bool} # Treatment models
    g::Matrix{Float64} # matrix of g
    ν::Vector{Float64} # prior degrees of freedome of Σ
    q::Matrix{Float64} # latent Gaussians (only relevant if the treatment is not Gaussian, otherwise empty)
    r::Vector{Float64} # latent treatment dispersion parameters (only relevant if the treatment is not Gaussian, otherwise empty)
end

"""
    Propose a new model by permuting one inclusion index (MC3)
"""
function mc3_proposal(curr)
    prop = copy(curr)
    ind = sample((1:length(curr)))
    prop[ind] = !prop[ind]
    return prop
end

"""
    Fit ivbma models with a regular g-prior and hyperpriors on g and ν.
"""
function ivbma_mcmc(
    y::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real},
    iter::Integer,
    burn::Integer,
    ν_prior::Function,
    g_L_prior::Function,
    g_M_prior::Function,
    m::AbstractVector
)

    n = size(W, 1)
    k = size(W, 2)
    p = size(Z, 2)

    m_o = m[1]; m_t = m[2]

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

        post_prop = marginal_likelihood_outcome(y, U, U_t_U, η, Σ_store[i-1], prop) + g_L_prior(prop) + log(prop)
        post_curr = marginal_likelihood_outcome(y, U, U_t_U, η, Σ_store[i-1], curr) + g_L_prior(curr) + log(curr)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            g_L_store[i] = prop
            acc_g_L += 1
        else
            g_L_store[i] = curr
        end

        propVar_g_L = adjust_variance(propVar_g_L, acc_g_L, i)
        
        # Step 1.2: Update outcome model
        curr = copy(L_incl[i-1,:])
        prop = mc3_proposal(L_incl[i-1,:])

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

        post_prop = marginal_likelihood_treatment(x, V, V_t_V, ϵ, Σ_store[i-1], prop) + g_M_prior(prop) + log(prop)
        post_curr = marginal_likelihood_treatment(x, V, V_t_V, ϵ, Σ_store[i-1], curr) + g_M_prior(curr) + log(curr)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            g_M_store[i] = prop
            acc_g_M += 1
        else
            g_M_store[i] = curr
        end

        propVar_g_M = adjust_variance(propVar_g_M, acc_g_M, i)

        # Step 2.2: Update treatment model
        curr = copy(M_incl[i-1,:])
        prop = mc3_proposal(M_incl[i-1,:])

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
        prop = rand(truncated(Normal(curr, propVar_ν), 1, Inf))

        post_prop = logpdf(InverseWishart(prop, [1 0; 0 1]), Σ_store[i]) + ν_prior(prop) - logpdf(truncated(Normal(curr, propVar_ν), 1, Inf), prop)
        post_curr = logpdf(InverseWishart(curr, [1 0; 0 1]), Σ_store[i]) + ν_prior(curr) - logpdf(truncated(Normal(prop, propVar_ν), 1, Inf), curr)
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
        [g_L_store[(burn+1):end] g_M_store[(burn+1):end]],
        ν_store[(burn+1):end],
        Matrix(undef, 0, 0),
        Vector(undef, 0)
    )
end


