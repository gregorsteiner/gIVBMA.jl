


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
    This is a helper function to adaptively tune the proposal variances of the MH steps.
"""
function adjust_variance(propVar, n_acc, iter)
    acc_rate = n_acc / iter
    if acc_rate < 0.2
        return propVar * 0.99
    elseif acc_rate > 0.4
        return propVar * 1.01
    end
    return propVar
end



"""
    The main MCMC function that returns posterior samples.
"""
function ivbma_mcmc(y, X, Z, W, iter, burn, ν, m, g_prior)
    # dimensions
    n, l = size(X)
    p = size(Z, 2); k = size(W, 2)

    # centre instruments and covariates
    Z = Z .- mean(Z; dims = 1)
    W = W .- mean(W; dims = 1)

    # g prior
    if g_prior == "BRIC"
        random_g = false
    elseif g_prior == "hyper-g/n"
        random_g = true
    end

    # starting values
    α, τ, β = (0, zeros(l), zeros(k))
    Γ, Δ = (zeros(l), zeros(k+p, l))
    Σ = Diagonal(ones(l+1))
    L = sample([true, false], k, replace = true)
    M = sample([true, false], k+p, replace = true)

    g_L, g_M = (max(n, k^2), max(n, (k+p)^2))
    if random_g
        proposal_variance_g_L, proposal_variance_g_M = (1, 1)
        acc_g_L, acc_g_M = (0, 0)
    end

    # storage objects
    nsave = iter - burn
    α_samples = zeros(nsave)
    τ_samples = zeros(nsave, l)
    β_samples = zeros(nsave, k)
    Γ_samples = zeros(nsave, l)
    Δ_samples = zeros(nsave, k + p, l)
    Σ_samples = Array{Matrix{Float64}}(undef, nsave)
    L_samples = zeros(Bool, nsave, k)
    M_samples = zeros(Bool, nsave, k + p)
    G_samples = zeros(nsave, 2)

    # Some precomputations
    ι = ones(n)
    U = [X W[:, L]]
    A = calc_A(U, g_L)

    V = [Z W][:, M]
    H = X - (ι * Γ' + V * Δ[M, :])

    # Gibbs sampler
    for i in 1:iter

        # Some precomputations
        (σ_y_x, Σ_yx, Σ_xx) = variances(Σ)
        B = calc_B_Σ(σ_y_x, Σ_yx, Σ_xx)
        
        # Update y_tilde
        y_tilde = y - H * inv(Σ_xx) * Σ_yx

        # Step 1: Outcome model
        # Update model
        prop = mc3_proposal(L)
        U_prop = [X W[:, prop]]
        A_prop = calc_A(U_prop, g_L)
        
        acc = min(1, exp(
            marginal_likelihood_outcome(y_tilde, A_prop, U_prop, σ_y_x, g_L) + model_prior(prop, k, 1, m[1]) - 
            (marginal_likelihood_outcome(y_tilde, A, U, σ_y_x, g_L) + model_prior(L, k, 1, m[1]))
        ))
        if rand() < acc
            L, U, A = (prop, U_prop, A_prop)
        end

        # Update g_L
        if random_g
            prop = exp(rand(truncated(Normal(log(g_L), sqrt(proposal_variance_g_L)), 0, Inf)))
            A_prop = calc_A(U, prop)

            acc = min(1, exp(
                marginal_likelihood_outcome(y_tilde, A_prop, U, σ_y_x, prop) + hyper_g_n(prop; a = 3, n = n) + log(prop) - 
                (marginal_likelihood_outcome(y_tilde, A, U, σ_y_x, g_L) + hyper_g_n(g_L; a = 3, n = n) + log(g_L))
            ))
            if rand() < acc
                g_L, A = (prop, A_prop)
                acc_g_L += 1
            end
            proposal_variance_g_L = adjust_variance(proposal_variance_g_L, acc_g_L, i)
        end

        # Update parameters
        α, τ, β = post_sample_outcome(y_tilde, X, A, U, σ_y_x)

        # Update residuals
        ϵ = y - (α * ι + U * [τ; β])
        X_tilde = X - (1/σ_y_x) * ϵ * Σ_yx' * inv(B)'

        # Step 2: Treatment model

        # Update model
        prop = mc3_proposal(M)
        V_prop = [Z W][:, prop]

        acc = min(1, exp(
            marginal_likelihood_treatment(X_tilde, B, V_prop, Σ_xx, g_M) + model_prior(prop, k+p, 1, m[2]) -
            (marginal_likelihood_treatment(X_tilde, B, V, Σ_xx, g_M) + model_prior(M, k+p, 1, m[2]))
        ))
        if rand() < acc
            M, V = (prop, V_prop)
        end

        # Update g_M
        if random_g
            prop = exp(rand(truncated(Normal(log(g_M), sqrt(proposal_variance_g_M)), 0, Inf)))
            acc = min(1, exp(
                marginal_likelihood_treatment(X_tilde, B, V, Σ_xx, prop) + hyper_g_n(prop; a = 3, n = n) + log(prop) -
                (marginal_likelihood_treatment(X_tilde, B, V, Σ_xx, g_M) + hyper_g_n(g_M; a = 3, n = n) + log(g_M))
            ))
            if rand() < acc
                g_M = prop
                acc_g_M += 1
            end
            proposal_variance_g_M = adjust_variance(proposal_variance_g_M, acc_g_M, i)
        end

        # Update parameters
        Γ, Δ = post_sample_treatment(X_tilde, B, V, Σ_xx, g_M)

        # Update residuals
        H = X - (ι * Γ' + V * Δ)

        # Step 3: Update covariance
        Σ = post_sample_cov(ϵ, H, ν)

        # Step 4: Store sampled values after burn in
        if i > burn
            α_samples[i - burn] = α
            τ_samples[i - burn, :] = τ
            β_samples[i - burn, L] = β
            Γ_samples[i - burn, :] = Γ
            Δ_samples[i - burn, M, :] = Δ
            Σ_samples[i - burn] = Σ
            L_samples[i - burn, :] = L
            M_samples[i - burn, :] = M
            G_samples[i - burn, :] = [g_L, g_M]
        end

    end

    
    return (
        α = α_samples,
        τ = τ_samples,
        β = β_samples,
        Γ = Γ_samples,
        Δ = Δ_samples,
        Σ = Σ_samples,
        L = L_samples,
        M = M_samples,
        G = G_samples
    )

end
