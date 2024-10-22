
"""
    A structure to store the posterior sample in
"""
struct PostSampleIVBMA2C
    α::Vector{Float64}
    τ::Vector{Float64}
    β::Matrix{Float64}
    γ::Vector{Float64}
    δ::Matrix{Float64}
    Σ::Array{Matrix{Float64}}
    L::Matrix{Bool}
    M::Matrix{Bool}
    g_L::Vector{Float64}
    g_l::Vector{Float64}
    g_s::Vector{Float64}
    ν::Vector{Float64}
end


"""
    Modified functions for the treatment posterior and marginal likelihood based on the two-component prior.
"""
function post_sample_treatment_2c(x, V, V_t_V, ϵ, Σ, G)
    n = length(x)

    if (rank(V_t_V) < size(V_t_V, 1))
        error("Non-full rank model!")
    end

    ψ = calc_psi(Σ)
    a = Σ[1,2]^2/(Σ[2,2] * ψ^2) + 1
    ϵ_bar = Statistics.mean(ϵ)

    A = inv(a * V_t_V + inv(G) * V_t_V * inv(G))

    γ = rand(Normal(-Σ[1,2]/a * ϵ_bar, Σ[2,2]/(a*n))) 
    δ = rand(MvNormal(a * A * V' * (x - (Σ[1,2]/Σ[1,1]) * ϵ), Σ[2,2] * Symmetric(A)))

    return (γ = γ, δ = δ)
end


function marginal_likelihood_treatment_2c(x, V, V_t_V, ϵ, Σ, G)
    n = length(x)

    if (rank(V_t_V) < size(V_t_V, 1))
        error("Non-full rank model!")
    end

    ψ = calc_psi(Σ)
    a = Σ[1,2]^2/(Σ[2,2] * ψ^2) + 1
    ϵ_bar = Statistics.mean(ϵ)

    A = inv(a * V_t_V + inv(G) * V_t_V * inv(G))
    
    x_tilde = (x - (Σ[1,2]/Σ[1,1]) * ϵ)
    t = (Σ[2,2]/Σ[1,1]) * ϵ'ϵ + x'x - 2 * (Σ[1,2]/Σ[1,1]) * ϵ'x - n * (Σ[1,2]^2/a^2) * ϵ_bar^2 - (x_tilde' * V * A * V' * x_tilde)

    log_ml = (1/2)*(log(det(A)) - log(det(G* inv(V_t_V) * G))) - a*t/(2*Σ[2,2])
    return log_ml
end

"""
    Helper function to construct the G matrix.
"""
function G_constr(g_s, g_l, ind, p, k)
    return Diagonal([repeat([sqrt(g_s)], p); repeat([sqrt(g_l)], k)])[ind, ind]
end


"""
    Fit IVBMA with the two-component prior and hyperpriors on g and ν.
"""
function ivbma_2c(
    y::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real};
    iter::Integer = 2000,
    burn::Integer = 1000,
    ν_prior::Function = ν -> log(jp_ν(ν, size(Z, 2) + size(W, 2) + 3)),
    g_L_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_l_prior::Function = g -> log(hyper_g_n(g; a = 3, n = length(y))),
    g_s_prior::Function = g -> log(hyper_g_n(g; a = 4, n = length(y))),
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

    g_L_store = zeros(iter); g_L_store[1] = n;
    propVar_g_L = 1/2; acc_g_L = 0
    g_l_store = zeros(iter); g_l_store[1] = n;
    propVar_g_l = 1/2; acc_g_l = 0
    g_s_store = zeros(iter); g_s_store[1] = n;
    propVar_g_s = 1/2; acc_g_s = 0

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

    η =x - (γ_store[1] * ones(n) + W_M * δ_store[1,:])

    for i in 2:iter
        
        # Step 1.1: Draw g_L
        curr = g_L_store[i-1]
        prop = rand(Truncated(LogNormal(log(curr), propVar_g_L), 1, Inf))

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

        # Step 2.1: Update g_l
        curr = g_l_store[i-1]
        prop = rand(Truncated(LogNormal(log(curr), propVar_g_l), 1, Inf))

        post_prop = marginal_likelihood_treatment_2c(x, V, V_t_V, ϵ, Σ_store[i-1], G_constr(g_s_store[i-1], prop, M_incl[i-1,:], p, k)) + g_l_prior(prop) - logpdf(LogNormal(log(curr), propVar_g_l), prop)
        post_curr = marginal_likelihood_treatment_2c(x, V, V_t_V, ϵ, Σ_store[i-1], G_constr(g_s_store[i-1], curr, M_incl[i-1,:], p, k)) + g_l_prior(curr) - logpdf(LogNormal(log(prop), propVar_g_l), curr)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            g_l_store[i] = prop
            acc_g_l += 1
        else
            g_l_store[i] = curr
        end

        propVar_g_l = adjust_variance(propVar_g_l, acc_g_l, i)

        # Step 2.2: Update g_s
        curr = g_s_store[i-1]
        prop = rand(Truncated(LogNormal(log(curr), propVar_g_s), 1, Inf))

        post_prop = marginal_likelihood_treatment_2c(x, V, V_t_V, ϵ, Σ_store[i-1], G_constr(prop, g_l_store[i], M_incl[i-1,:], p, k)) + g_s_prior(prop) - logpdf(LogNormal(log(curr), propVar_g_s), prop)
        post_curr = marginal_likelihood_treatment_2c(x, V, V_t_V, ϵ, Σ_store[i-1], G_constr(curr, g_l_store[i], M_incl[i-1,:], p, k)) + g_s_prior(curr) - logpdf(LogNormal(log(prop), propVar_g_s), curr)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            g_s_store[i] = prop
            acc_g_s += 1
        else
            g_s_store[i] = curr
        end

        propVar_g_s = adjust_variance(propVar_g_s, acc_g_s, i)


        # Step 2.3: Update treatment model
        curr = copy(M_incl[i-1,:])
        prop = mc3_proposal(M_incl[i-1,:])

        V_prop = W_M[:, prop]
        V_t_V_prop = V_prop'V_prop

        post_prop = marginal_likelihood_treatment_2c(x, V_prop, V_t_V_prop, ϵ, Σ_store[i-1], G_constr(g_s_store[i], g_l_store[i], prop, p, k)) + model_prior(prop, k+p, 1, m_t)
        post_curr = marginal_likelihood_treatment_2c(x, V, V_t_V, ϵ, Σ_store[i-1], G_constr(g_s_store[i], g_l_store[i], curr, p, k)) + model_prior(curr, k+p, 1, m_t)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            M_incl[i,:] = prop
            V = V_prop
            V_t_V = V_t_V_prop
        else
            M_incl[i,:] = curr
        end

        # Step 2.4: Update treatment parameters
        draw = post_sample_treatment_2c(x, V, V_t_V, ϵ, Σ_store[i-1], G_constr(g_s_store[i], g_l_store[i], M_incl[i,:], p, k))
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

    return PostSampleIVBMA2C(
        α_store[(burn+1):end],
        τ_store[(burn+1):end],
        β_store[(burn+1):end,:],
        γ_store[(burn+1):end],
        δ_store[(burn+1):end,:],
        Σ_store[(burn+1):end],
        L_incl[(burn+1):end,:],
        M_incl[(burn+1):end,:],
        g_L_store[(burn+1):end],
        g_l_store[(burn+1):end],
        g_s_store[(burn+1):end],
        ν_store[(burn+1):end]
    )
end