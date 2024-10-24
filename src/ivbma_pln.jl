

"""
    Outcome posterior and marginal likelihood with a proper prior on the intercept.
    We put a Normal(0, κ²) prior on the intercept α (pi = proper intercept).
"""
function post_sample_outcome_pi(y_tilde, U, b, B, ψ, g)
    n = length(y_tilde)
    ι = ones(n)

    β_tilde = rand(MvNormal(
        B * U' * (I - b * (ι * inv(ι'ι) * ι')) * y_tilde,
        Symmetric(ψ^2 * B))
    )
    τ = β_tilde[1]
    β = β_tilde[2:end]

    α = rand(Normal(ι' * (y_tilde - U * β_tilde) / (n*b), b * ψ^2 / n))
    
    return (α = α, τ = τ, β = β)
end

function marginal_likelihood_outcome_pi(y_tilde, U, b, B, ψ, g)
    n = length(y_tilde)
    ι = ones(n)

    IbP = (I - b * (ι * inv(ι'ι) * ι'))
    s = y_tilde' * IbP * y_tilde - y_tilde' * IbP * U * B * U' * IbP * y_tilde
    
    log_ml =  (1/2) * (log(det(B)) - log(det(g * inv(U'U)))) - s/(2*ψ^2)
    return log_ml
end

"""
    Auxiliary function that computes b and B (see paper).
"""
function calc_bB(U, Σ, κ2, g)
    n = size(U, 1)
    ψ = calc_psi(Σ)
    b = 1 / (1 + ψ^2 / (n * κ2))

    ι = ones(n)
    B = inv( U' * ((g+1)/g * I - b * (ι * inv(ι'ι) * ι')) * U)

    return (b, B)
end

"""
    Barker proposal, see Zens & Steel (2024) and Livingstone & Zanella (2022)
"""
function barker_proposal(x, q, GradCurr, PropVar)
    n = length(x)
    
    Qi = [rand(Normal(0, PropVar[j])) for j in 1:n]
    bi = 2 * (rand(n) .< (1 ./ (1 .+ exp.(-GradCurr .* Qi)))) .- 1

    q_prop = q .+ Qi .* bi
    return q_prop
end

function barker_correction_term(curr, prop, GradCurr, GradProp)
    beta1 = -GradProp .* (curr .- prop)
    beta2 = -GradCurr .* (prop .- curr)

    result = -(max.(beta1, zeros(length(curr))) .+ log1p.(exp.(-abs.(beta1)))) .+
             (max.(beta2, zeros(length(curr))) .+ log1p.(exp.(-abs.(beta2))))

    return result
end


"""
    Fit Gaussian-PLN (Gaussian outcome, Poisson treatment) ivbma models with a regular g-prior (+proper prior on the intercept) and hyperpriors on g and ν.
"""
function ivbma_mcmc_pln(
    y::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real},
    iter::Integer,
    burn::Integer,
    κ2,
    ν_prior::Function,
    g_L_prior::Function,
    g_M_prior::Function,
    m::AbstractVector
)

    n = size(W, 1)
    k = size(W, 2)
    p = size(Z, 2)
 
    m_o = m[1]; m_t = m[2]

    q_store = Matrix{Float64}(undef, iter, n)
    q_store[1,:] = rand(Normal(0, 1), n)

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

    propVar_q = ones(n) * 0.1

    W_L = W
    W_M = [Z W]

    M_incl = Matrix{Bool}(undef, iter, k+p)
    M_incl[1,:] = sample([true, false], k+p, replace = true)
    L_incl = Matrix{Bool}(undef, iter, k)
    L_incl[1,:] = sample([true, false], k, replace = true)

    # Some precomputations
    U = [x W_L[:,L_incl[1,:]]]
    

    V = W_M[:,M_incl[1,:]]
    V_t_V = V'V

    η = q_store[1,:] - (γ_store[1]*ones(n) + W_M * δ_store[1,:])

    for i in 2:iter

        # Step 0.0: Precomputations
        ψ = calc_psi(Σ_store[i-1])
        cov_ratio = (Σ_store[i-1][1,2] / Σ_store[i-1][2,2])

        # Step 0.1: Draw latent Gaussian q
        q_curr = q_store[i-1,:]
        GradCurr = (x - exp.(q_curr)) - η / Σ_store[i-1][2, 2]
        q_prop = barker_proposal(x, q_curr, GradCurr, propVar_q)
        η_prop = q_prop - (γ_store[i-1]*ones(n) + W_M * δ_store[i-1,:])
        GradProp = (x - exp.(q_prop)) - η_prop / Σ_store[i-1][2, 2]

        Mean_y = α_store[i-1] * ones(n) + [x W_L] * [τ_store[i-1]; β_store[i-1, :]]
        Mean_prior = γ_store[i-1] * ones(n) + W_M * δ_store[i-1, :]

        post_curr = [(
            logpdf(Normal(Mean_y[j] + cov_ratio * η[j], ψ^2), y[j])
            + logpdf(Poisson(exp(q_curr[j])), x[j])
            + logpdf(Normal(Mean_prior[j], Σ_store[i-1][2,2]), q_curr[j])
        ) for j in eachindex(q_curr)]

        post_prop = [(
            logpdf(Normal(Mean_y[j] + cov_ratio * η_prop[j], ψ^2), y[j])
            + logpdf(Poisson(exp(q_prop[j])), x[j])
            + logpdf(Normal(Mean_prior[j], Σ_store[i-1][2,2]), q_prop[j])
        ) for j in eachindex(q_prop)]

        corr_term = barker_correction_term(q_curr, q_prop, GradCurr, GradProp)
        acc_prob = min.(ones(n), exp.(post_prop - post_curr + corr_term))
        acc = rand(n) .< acc_prob

        q_store[i, acc] = q_prop[acc]
        q_store[i, .!acc] = q_curr[.!acc]

        # global scale adaptation
        l_propVar_q2 = log.(propVar_q.^2) .+ (i^(-0.6)) .* (acc_prob .- 0.57)
        propVar_q = sqrt.(exp.(l_propVar_q2))

        # Update 'residuals'
        η[acc] = η_prop[acc]
        y_tilde = y - cov_ratio * η

        # Step 1.1: Draw g_l
        curr = g_L_store[i-1]
        prop = rand(LogNormal(log(curr), propVar_g_L))

        @infiltrate
        b, B = calc_bB(U, Σ_store[i-1], κ2, curr)
        b, B_prop = calc_bB(U, Σ_store[i-1], κ2, prop)
        post_prop = marginal_likelihood_outcome_pi(y_tilde, U, b, B_prop, ψ, prop) + g_L_prior(prop) + log(prop)
        post_curr = marginal_likelihood_outcome_pi(y_tilde, U, b, B, ψ, curr) + g_L_prior(curr) + log(curr)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            g_L_store[i] = prop
            acc_g_L += 1
            B = B_prop
        else
            g_L_store[i] = curr
        end

        propVar_g_L = adjust_variance(propVar_g_L, acc_g_L, i)
        
        # Step 1.2: Update outcome model
        @infiltrate
        curr = copy(L_incl[i-1,:])
        prop = mc3_proposal(L_incl[i-1,:])

        U_prop = [x W_L[:,prop]]
        b, B_prop = calc_bB(U_prop, Σ_store[i-1], κ2, g_L_store[i])

        post_prop = marginal_likelihood_outcome_pi(y_tilde, U_prop, b, B_prop, ψ, g_L_store[i]) + model_prior(prop, k, 1, m_o)
        post_curr = marginal_likelihood_outcome_pi(y_tilde, U, b, B, ψ, g_L_store[i]) + model_prior(curr, k, 1, m_o)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            L_incl[i,:] = prop
            U = U_prop
            B = B_prop
        else
            L_incl[i,:] = curr
        end

        # Step 1.3: Update outcome parameters
        draw = post_sample_outcome_pi(y_tilde, U, b, B, ψ, g_L_store[i])
        α_store[i] = draw.α
        τ_store[i] = draw.τ
        β_store[i, L_incl[i,:]] = draw.β
        

        # Step 2.0: Precompute residuals
        ϵ = y - (α_store[i] * ones(n) + τ_store[i] * x + W_L * β_store[i,:])

        # Step 2.1: Update g_M
        curr = g_M_store[i-1]
        prop = rand(LogNormal(log(curr), propVar_g_M))

        post_prop = marginal_likelihood_treatment(q_store[i,:], V, V_t_V, ϵ, Σ_store[i-1], prop) + g_M_prior(prop) + log(prop)
        post_curr = marginal_likelihood_treatment(q_store[i,:], V, V_t_V, ϵ, Σ_store[i-1], curr) + g_M_prior(curr) + log(curr)
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

        post_prop = marginal_likelihood_treatment(q_store[i,:], V_prop, V_t_V_prop, ϵ, Σ_store[i-1], g_M_store[i]) + model_prior(prop, k+p, 1, m_t)
        post_curr = marginal_likelihood_treatment(q_store[i,:], V, V_t_V, ϵ, Σ_store[i-1], g_M_store[i]) + model_prior(curr, k+p, 1, m_t)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            M_incl[i,:] = prop
            V = V_prop
            V_t_V = V_t_V_prop
        else
            M_incl[i,:] = curr
        end

        # Step 2.3: Update treatment parameters
        draw = post_sample_treatment(q_store[i,:], V, V_t_V, ϵ, Σ_store[i-1], g_M_store[i])
        γ_store[i] = draw.γ
        δ_store[i, M_incl[i,:]] = draw.δ


        # Step 3: Update covariance Matrix
        #@infiltrate
        η = q_store[i,:] - (γ_store[i] * ones(n) + W_M * δ_store[i,:])
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
        [g_L_store[(burn+1):end] g_M_store[(burn+1):end]],
        ν_store[(burn+1):end],
        q_store[(burn+1):end,:]
    )
end


"""
    Fit Gaussian-PLN (Gaussian outcome, Poisson treatment) ivbma models with a regular g-prior (+proper prior on the intercept) and hyperpriors on g and ν.
"""
function ivbma_mcmc_pln_2c(
    y::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    Z::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real},
    iter::Integer,
    burn::Integer,
    κ2,
    ν_prior::Function,
    g_L_prior::Function,
    g_l_prior::Function,
    g_s_prior::Function,
    m::AbstractVector
)

    n = size(W, 1)
    k = size(W, 2)
    p = size(Z, 2)
 
    m_o = m[1]; m_t = m[2]

    q_store = Matrix{Float64}(undef, iter, n)
    q_store[1,:] = rand(Normal(0, 1), n)

    α_store = zeros(iter)
    τ_store = zeros(iter)
    β_store = zeros(iter, k)
    γ_store = zeros(iter)
    δ_store = zeros(iter, k+p)
    Σ_store = Array{Matrix{Float64}}(undef, iter)
    Σ_store[1] = [1.0 0.0; 0.0 1.0]

    g_L_store = zeros(iter); g_L_store[1] = n
    propVar_g_L = 1/2; acc_g_L = 0
    g_l_store = zeros(iter); g_l_store[1] = n;
    propVar_g_l = 1/2; acc_g_l = 0
    g_s_store = zeros(iter); g_s_store[1] = n;
    propVar_g_s = 1/2; acc_g_s = 0

    ν_store = zeros(iter); ν_store[1] = 10
    propVar_ν = 1/2; acc_ν = 0

    propVar_q = ones(n) * 0.1

    W_L = W
    W_M = [Z W]

    M_incl = Matrix{Bool}(undef, iter, k+p)
    M_incl[1,:] = sample([true, false], k+p, replace = true)
    L_incl = Matrix{Bool}(undef, iter, k)
    L_incl[1,:] = sample([true, false], k, replace = true)

    # Some precomputations
    U = [x W_L[:,L_incl[1,:]]]
    

    V = W_M[:,M_incl[1,:]]
    V_t_V = V'V

    η = q_store[1,:] - (γ_store[1]*ones(n) + W_M * δ_store[1,:])

    for i in 2:iter

        # Step 0.0: Precomputations
        ψ = calc_psi(Σ_store[i-1])
        cov_ratio = (Σ_store[i-1][1,2] / Σ_store[i-1][2,2])

        # Step 0.1: Draw latent Gaussian q
        q_curr = q_store[i-1,:]
        GradCurr = (x - exp.(q_curr)) - η / Σ_store[i-1][2, 2]
        q_prop = barker_proposal(x, q_curr, GradCurr, propVar_q)
        η_prop = q_prop - (γ_store[i-1]*ones(n) + W_M * δ_store[i-1,:])
        GradProp = (x - exp.(q_prop)) - η_prop / Σ_store[i-1][2, 2]

        Mean_y = α_store[i-1] * ones(n) + [x W_L] * [τ_store[i-1]; β_store[i-1, :]]
        Mean_prior = γ_store[i-1] * ones(n) + W_M * δ_store[i-1, :]

        post_curr = [(
            logpdf(Normal(Mean_y[j] + cov_ratio * η[j], ψ^2), y[j])
            + logpdf(Poisson(exp(q_curr[j])), x[j])
            + logpdf(Normal(Mean_prior[j], Σ_store[i-1][2,2]), q_curr[j])
        ) for j in eachindex(q_curr)]

        post_prop = [(
            logpdf(Normal(Mean_y[j] + cov_ratio * η_prop[j], ψ^2), y[j])
            + logpdf(Poisson(exp(q_prop[j])), x[j])
            + logpdf(Normal(Mean_prior[j], Σ_store[i-1][2,2]), q_prop[j])
        ) for j in eachindex(q_prop)]

        corr_term = barker_correction_term(q_curr, q_prop, GradCurr, GradProp)
        acc_prob = min.(ones(n), exp.(post_prop - post_curr + corr_term))
        acc = rand(n) .< acc_prob

        q_store[i, acc] = q_prop[acc]
        q_store[i, .!acc] = q_curr[.!acc]

        # global scale adaptation
        l_propVar_q2 = log.(propVar_q.^2) .+ (i^(-0.6)) .* (acc_prob .- 0.57)
        propVar_q = sqrt.(exp.(l_propVar_q2))

        # Update 'residuals'
        η[acc] = η_prop[acc]
        y_tilde = y - cov_ratio * η

        # Step 1.1: Draw g_l
        curr = g_L_store[i-1]
        prop = rand(LogNormal(log(curr), propVar_g_L))

        b, B = calc_bB(U, Σ_store[i-1], κ2, curr)
        b, B_prop = calc_bB(U, Σ_store[i-1], κ2, prop)
        post_prop = marginal_likelihood_outcome_pi(y_tilde, U, b, B_prop, ψ, prop) + g_L_prior(prop) + log(prop)
        post_curr = marginal_likelihood_outcome_pi(y_tilde, U, b, B, ψ, curr) + g_L_prior(curr) + log(curr)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            g_L_store[i] = prop
            acc_g_L += 1
            B = B_prop
        else
            g_L_store[i] = curr
        end

        propVar_g_L = adjust_variance(propVar_g_L, acc_g_L, i)
        
        # Step 1.2: Update outcome model
        curr = copy(L_incl[i-1,:])
        prop = mc3_proposal(L_incl[i-1,:])

        U_prop = [x W_L[:,prop]]
        b, B_prop = calc_bB(U_prop, Σ_store[i-1], κ2, g_L_store[i])

        post_prop = marginal_likelihood_outcome_pi(y_tilde, U_prop, b, B_prop, ψ, g_L_store[i]) + model_prior(prop, k, 1, m_o)
        post_curr = marginal_likelihood_outcome_pi(y_tilde, U, b, B, ψ, g_L_store[i]) + model_prior(curr, k, 1, m_o)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            L_incl[i,:] = prop
            U = U_prop
            B = B_prop
        else
            L_incl[i,:] = curr
        end

        # Step 1.3: Update outcome parameters
        draw = post_sample_outcome_pi(y_tilde, U, b, B, ψ, g_L_store[i])
        α_store[i] = draw.α
        τ_store[i] = draw.τ
        β_store[i, L_incl[i,:]] = draw.β
        

        # Step 2.0: Precompute residuals
        ϵ = y - (α_store[i] * ones(n) + τ_store[i] * x + W_L * β_store[i,:])

        # Step 2.1: Update g_l
        curr = g_l_store[i-1]
        prop = rand(LogNormal(log(curr), propVar_g_l))

        post_prop = marginal_likelihood_treatment_2c(q_store[i,:], V, V_t_V, ϵ, Σ_store[i-1], G_constr(g_s_store[i-1], prop, M_incl[i-1,:], p, k)) + g_l_prior(prop) + log(prop)
        post_curr = marginal_likelihood_treatment_2c(q_store[i,:], V, V_t_V, ϵ, Σ_store[i-1], G_constr(g_s_store[i-1], curr, M_incl[i-1,:], p, k)) + g_l_prior(curr) + log(curr)
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
        prop = rand(LogNormal(log(curr), propVar_g_s))

        post_prop = marginal_likelihood_treatment_2c(q_store[i,:], V, V_t_V, ϵ, Σ_store[i-1], G_constr(prop, g_l_store[i], M_incl[i-1,:], p, k)) + g_s_prior(prop) + log(prop)
        post_curr = marginal_likelihood_treatment_2c(q_store[i,:], V, V_t_V, ϵ, Σ_store[i-1], G_constr(curr, g_l_store[i], M_incl[i-1,:], p, k)) + g_s_prior(curr) + log(curr)
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

        post_prop = marginal_likelihood_treatment_2c(q_store[i,:], V_prop, V_t_V_prop, ϵ, Σ_store[i-1], G_constr(g_s_store[i], g_l_store[i], prop, p, k)) + model_prior(prop, k+p, 1, m_t)
        post_curr = marginal_likelihood_treatment_2c(q_store[i,:], V, V_t_V, ϵ, Σ_store[i-1], G_constr(g_s_store[i], g_l_store[i], curr, p, k)) + model_prior(curr, k+p, 1, m_t)
        acc = exp(post_prop - post_curr)
        
        if rand() < min(acc, 1)
            M_incl[i,:] = prop
            V = V_prop
            V_t_V = V_t_V_prop
        else
            M_incl[i,:] = curr
        end

        # Step 2.4: Update treatment parameters
        draw = post_sample_treatment_2c(q_store[i,:], V, V_t_V, ϵ, Σ_store[i-1], G_constr(g_s_store[i], g_l_store[i], M_incl[i,:], p, k))
        γ_store[i] = draw.γ
        δ_store[i, M_incl[i,:]] = draw.δ

        # Step 3: Update covariance Matrix
        η = q_store[i,:] - (γ_store[i] * ones(n) + W_M * δ_store[i,:])
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
        [g_L_store[(burn+1):end] g_l_store[(burn+1):end] g_s_store[(burn+1):end]],
        ν_store[(burn+1):end],
        q_store[(burn+1):end,:]
    )
end
