


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
    Adapt the proposal scale based on a desired acceptance probability.
"""
function adjust_variance(curr_variance, acc_prob, desired_acc_prob, iter)
    log_variance = log(curr_variance) + iter^(-0.6) * (acc_prob - desired_acc_prob)
    return exp(log_variance)
end


"""
    The main MCMC function that returns posterior samples.
"""
function ivbma_mcmc(y, X, Z, W, dist, iter, burn, ν, m, g_prior, r_prior)
    # dimensions
    n, l = size(X)
    p = size(Z, 2); k = size(W, 2)

    # centre instruments and covariates
    Z_c = Z .- mean(Z; dims = 1)
    W_c = W .- mean(W; dims = 1)

    if length(dist) != l
        error("`dist` must have an element for each column of X")
    end

    # g prior
    if g_prior == "BRIC"
        random_g = false
    elseif g_prior == "hyper-g/n"
        random_g = true
    end

    # starting values
    L = sample([true, false], k, replace = true)
    M = sample([true, false], k+p, replace = true)

    α, τ, β = (0, zeros(l), zeros(k)[L])
    Γ, Δ = (zeros(l), zeros(k+p, l)[M, :])
    Σ = Diagonal(ones(l+1))
    
    g_L, g_M = (max(n, k^2), max(n, (k+p)^2))
    if random_g
        proposal_variance_g_L, proposal_variance_g_M = (0.01, 0.01)
    end

    Q = copy(X)
    proposal_variance_Q = ones(n, l) * 0.1
    r = ones(l) # dispersion parameter (only relevant for Beta-Logistic)
    proposal_variance_r = ones(l) * 0.1
    
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
    Q_samples = zeros(nsave, n, l)

    # Some precomputations
    ι = ones(n)
    U = [ι X W_c[:, L]]

    V = [ι [Z_c W_c][:, M]]
    H = Q - V * [Γ'; Δ]

    # Gibbs sampler
    for i in 1:iter

        # Some precomputations
        (σ_y_x, Σ_yx, Σ_xx) = variances(Σ)
        B = calc_B_Σ(σ_y_x, Σ_yx, Σ_xx)

        # Step 0.1: Draw latent Gaussian Q
        Mean_y = U * [α; τ; β]
        Mean_Q = V * [Γ'; Δ]

        for (idx_d, d) in enumerate(dist)
            if d != "Gaussian"

                # Use set values for 0 and 1 observations in the Beta case
                if d == "BL"
                    X[:, idx_d] = set_values_0_1(X[:, idx_d], Q[:, idx_d], r[idx_d])
                end

                q_curr = Q[:, idx_d]
                GradCurr = gradient(y, X, Q, Mean_y, Mean_Q, σ_y_x, Σ_yx, Σ_xx, d, idx_d, r[idx_d])
                q_prop = barker_proposal(q_curr, GradCurr, proposal_variance_Q[:, idx_d])
                Q_prop = hcat(Q[:, 1:idx_d-1], q_prop, Q[:, idx_d+1:end])
                GradProp = gradient(y, X, Q_prop, Mean_y, Mean_Q, σ_y_x, Σ_yx, Σ_xx, d, idx_d, r[idx_d])

                post_prop = posterior_q(y, X, Q_prop, Mean_y, Mean_Q, σ_y_x, Σ_yx, Σ_xx, d, idx_d, r[idx_d])
                post_curr = posterior_q(y, X, Q, Mean_y, Mean_Q, σ_y_x, Σ_yx, Σ_xx, d, idx_d, r[idx_d])
                corr_term = barker_correction_term(q_curr, q_prop, GradCurr, GradProp)

                acc_prob = min.(ones(n), exp.(post_prop - post_curr + corr_term))
                acc = rand(n) .< acc_prob
                Q[acc, idx_d] = q_prop[acc]

                proposal_variance_Q[:, idx_d] = adjust_variance.(proposal_variance_Q[:, idx_d], acc_prob, 0.57, i)

                # update r (only for BL model)
                if d == "BL"
                    prop = rand(LogNormal(log(r[idx_d]), proposal_variance_r[idx_d]))
                    acc = min(1, exp(
                        post_r(prop, X[:, idx_d], Q[:, idx_d], r_prior) + log(prop) -
                        (post_r(r[idx_d], X[:, idx_d], Q[:, idx_d], r_prior) + log(r[idx_d]))
                    ))
                    if rand() < acc
                       r[idx_d] = prop
                    end
                    proposal_variance_r[idx_d] = adjust_variance(proposal_variance_r[idx_d], acc, 0.234, i)
                end
            end
        end

        # Update residuals
        H = Q - V * [Γ'; Δ]
        y_tilde = y - H * inv(Σ_xx) * Σ_yx

        # Step 1: Outcome model
        # Update model
        prop = mc3_proposal(L)
        U_prop = [ι X W_c[:, prop]]
        
        acc = min(1, exp(
            marginal_likelihood_outcome(y_tilde, U_prop, σ_y_x, g_L) + model_prior(prop, k, 1, m[1]) - 
            (marginal_likelihood_outcome(y_tilde, U, σ_y_x, g_L) + model_prior(L, k, 1, m[1]))
        ))
        if rand() < acc
            L, U = (prop, U_prop)
        end

        # Update g_L
        if random_g
            prop = rand(LogNormal(log(g_L), sqrt(proposal_variance_g_L)))

            acc = min(1, exp(
                marginal_likelihood_outcome(y_tilde, U, σ_y_x, prop) + log(hyper_g_n(prop; a = 3, n = n)) + log(prop) - 
                (marginal_likelihood_outcome(y_tilde, U, σ_y_x, g_L) + log(hyper_g_n(g_L; a = 3, n = n)) + log(g_L))
            ))
            if rand() < acc
                g_L = prop
            end
            proposal_variance_g_L = adjust_variance(proposal_variance_g_L, acc, 0.234, i)
        end

        # Update parameters
        α, τ, β = post_sample_outcome(y_tilde, X, U, σ_y_x, g_L)

        # Update residuals
        ϵ = y - U * [α; τ; β]
        Q_tilde = Q - (1/σ_y_x) * ϵ * Σ_yx' * inv(B)'

        # Step 2: Treatment model

        # Update model
        prop = mc3_proposal(M)
        V_prop = [ι [Z_c W_c][:, prop]]

        acc = min(1, exp(
            marginal_likelihood_treatment(Q_tilde, B, V_prop, Σ_xx, g_M) + model_prior(prop, k+p, 1, m[2]) -
            (marginal_likelihood_treatment(Q_tilde, B, V, Σ_xx, g_M) + model_prior(M, k+p, 1, m[2]))
        ))
        if rand() < acc
            M, V = (prop, V_prop)
        end

        # Update g_M
        if random_g
            prop = rand(LogNormal(log(g_M), sqrt(proposal_variance_g_M)))
            acc = min(1, exp(
                marginal_likelihood_treatment(Q_tilde, B, V, Σ_xx, prop) + log(hyper_g_n(prop; a = 3, n = n)) + log(prop) -
                (marginal_likelihood_treatment(Q_tilde, B, V, Σ_xx, g_M) + log(hyper_g_n(g_M; a = 3, n = n)) + log(g_M))
            ))
            if rand() < acc
                g_M = prop
            end
            proposal_variance_g_M = adjust_variance(proposal_variance_g_M, acc, 0.234, i)
        end

        # Update parameters
        Γ, Δ = post_sample_treatment(Q_tilde, B, V, Σ_xx, g_M)

        # Update residuals
        H = Q - V * [Γ'; Δ]

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
            Q_samples[i - burn, :, :] = Q
        end

    end

    return (
        y = y,
        X = X,
        Z = Z,
        W = W,
        α = α_samples,
        τ = τ_samples,
        β = β_samples,
        Γ = Γ_samples,
        Δ = Δ_samples,
        Σ = Σ_samples,
        L = L_samples,
        M = M_samples,
        G = G_samples,
        Q = Q_samples
    )

end
