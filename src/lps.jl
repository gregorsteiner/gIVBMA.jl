
"""
    Compute the log-predictive score on a holdout dataset.
"""
function lps(ivbma::GIVBMA, y_h, X_h, Z_h, W_h)
    # if X is a vector turn it into an nx1 matrix
    if ndims(X_h) == 1
        X_h = permutedims(X_h)'
    end

    n_h, l = size(X_h)
    n_post = length(ivbma.α)
    scores = Matrix{Float64}(undef, n_h, n_post)

    # demean holdout sample using the mean over the training sample
    Z_h = Z_h .- mean(ivbma.Z; dims = 1)
    W_h = W_h .- mean(ivbma.W; dims = 1)

    for i in 1:n_post
        σ_y_x, Σ_yx, Σ_xx = variances(ivbma.Σ[i])

        # Draw latent Q_x
        Mean_Q_x = (ones(n_h) * ivbma.Γ[i:i,:] + [Z_h W_h] * ivbma.Δ[i, :, :])
        Q_x = rand(MatrixNormal(Mean_Q_x, Diagonal(ones(n_h)), Σ_xx))

        # Use X instead of Q if Gaussian
        for (idx, d) in enumerate(ivbma.dist[2:end])
            if d == "Gaussian"
                Q_x[:, idx] = X_h[:, idx]
            end
        end

        # Draw latent q_y
        H = Q_x - Mean_Q_x
        mean_q_y = ones(n_h) * ivbma.α[i] + X_h * ivbma.τ[i, :] + W_h * ivbma.β[i, :] + H * inv(Σ_xx) * Σ_yx
        q_y = rand(MvNormal(mean_q_y, σ_y_x * I))

        if ivbma.dist[1] == "Gaussian"
            scores[:, i] = [pdf(Normal(mean_q_y[j], sqrt(σ_y_x)), y_h[j]) for j in eachindex(y_h)]
        elseif ivbma.dist[1] == "PLN"
            scores[:, i] = [pdf(Poisson(exp(q_y[j])), y_h[j]) for j in eachindex(y_h)]
        elseif ivbma.dist[1] == "BL"
            μ, r = (logit.(q_y), ivbma.r[i, 1])
            B_α, B_β = (μ * r, r * (1 .- μ))
            scores[:, i] .+= [pdf(Beta(B_α[j], B_β[j]), y_h[j]) for j in eachindex(y_h)]
        end
    end

    scores_avg = mean(scores; dims = 2)
    lps = -mean(log.(scores_avg))
    return lps
end

function lps(ivbma::GIVBMA, y_h, X_h, Z_h)
    # if X is a vector turn it into an nx1 matrix
    if ndims(X_h) == 1
        X_h = permutedims(X_h)'
    end

    n_h, l = size(X_h)
    n_post = length(ivbma.α)
    scores = Matrix{Float64}(undef, n_h, n_post)

    # demean holdout sample using the mean over the training sample
    Z_h = Z_h .- mean(ivbma.W; dims = 1)

    for i in 1:n_post
        σ_y_x, Σ_yx, Σ_xx = variances(ivbma.Σ[i])

        # Draw latent Q_x
        Mean_Q_x = (ones(n_h) * ivbma.Γ[i:i,:] + Z_h * ivbma.Δ[i, :, :])
        Q_x = rand(MatrixNormal(Mean_Q_x, Diagonal(ones(n_h)), Σ_xx))

        # Use X instead of Q if Gaussian
        for (idx, d) in enumerate(ivbma.dist[2:end])
            if d == "Gaussian"
                Q_x[:, idx] = X_h[:, idx]
            end
        end

        # Draw latent q_y
        H = Q_x - Mean_Q_x
        mean_q_y = ones(n_h) * ivbma.α[i] + X_h * ivbma.τ[i, :] + Z_h * ivbma.β[i, :] + H * inv(Σ_xx) * Σ_yx
        q_y = rand(MvNormal(mean_q_y, σ_y_x * I))

        if ivbma.dist[1] == "Gaussian"
            scores[:, i] = [pdf(Normal(mean_q_y[j], sqrt(σ_y_x)), y_h[j]) for j in eachindex(y_h)]
        elseif ivbma.dist[1] == "PLN"
            scores[:, i] = [pdf(Poisson(exp(q_y[j])), y_h[j]) for j in eachindex(y_h)]
        elseif ivbma.dist[1] == "BL"
            μ, r = (logit.(q_y), ivbma.r[i, 1])
            B_α, B_β = (μ * r, r * (1 .- μ))
            scores[:, i] .+= [pdf(Beta(B_α[j], B_β[j]), y_h[j]) for j in eachindex(y_h)]
        end
    end

    scores_avg = mean(scores; dims = 2)
    lps = -mean(log.(scores_avg))
    return lps
end

"""
    Return the posterior predictive distribution for a single observation.
"""

function posterior_predictive(ivbma::GIVBMA, x_h, z_h, w_h)
    n_post = length(ivbma.α)
    
    # demean holdout sample using the mean over the training sample
    w_c = w_h - mean(ivbma.W; dims = 1)[1, :]
    z_c = z_h - mean(ivbma.Z; dims = 1)[1, :]

    mean_q_y, var_q_y = (Vector(undef, n_post), Vector(undef, n_post))
    for i in 1:n_post
        σ_y_x, Σ_yx, Σ_xx = variances(ivbma.Σ[i])

        # Draw latent Q_x
        Mean_Q_x = vec(ivbma.Γ[i:i,:] + z_c' * ivbma.Δ[i, :, :])
        Q_x = rand(MvNormal(Mean_Q_x, Σ_xx))

        # Use X instead of Q if Gaussian
        for (idx, d) in enumerate(ivbma.dist[2:end])
            if d == "Gaussian"
                Q_x[idx] = x_h[idx]
            end
        end

        mean_q_y[i] = ivbma.α[i] + x_h' * ivbma.τ[i,:] + w_c' * ivbma.β[i,:]
        var_q_y[i] = σ_y_x
    end

    if ivbma.dist[1] == "Gaussian"
        ds = map((μ, σ) -> Normal(μ, sqrt(σ)), mean_q_y, var_q_y)
    elseif ivbma.dist[1] == "PLN"
        q_y = map((μ, σ) -> rand(Normal(μ, sqrt(σ))), mean_q_y, var_q_y)
        ds = map(q -> Poisson(exp(q)), q_y)
    elseif ivbma.dist[1] == "BL"
        q_y = map((μ, σ) -> rand(Normal(μ, sqrt(σ))), mean_q_y, var_q_y)
        μ, r = (logit.(q_y), ivbma.r[i, 1])
        B_α, B_β = (μ * r, r * (1 .- μ))
        ds = map((b_α, b_β) -> Beta(b_α, b_β), B_α, B_β)
    end

    d = MixtureModel(ds)
    return d
end

function posterior_predictive(ivbma::GIVBMA, x_h, z_h)
    n_post = length(ivbma.α)
    
    # demean holdout sample using the mean over the training sample
    z_c = z_h - mean(ivbma.W; dims = 1)[1, :]

    mean_q_y, var_q_y = (Vector(undef, n_post), Vector(undef, n_post))
    for i in 1:n_post
        σ_y_x, Σ_yx, Σ_xx = variances(ivbma.Σ[i])

        # Draw latent Q_x
        Mean_Q_x = vec(ivbma.Γ[i:i,:] + z_c' * ivbma.Δ[i, :, :])
        Q_x = rand(MvNormal(Mean_Q_x, Σ_xx))

        # Use X instead of Q if Gaussian
        for (idx, d) in enumerate(ivbma.dist[2:end])
            if d == "Gaussian"
                Q_x[idx] = x_h[idx]
            end
        end

        mean_q_y[i] = ivbma.α[i] + x_h' * ivbma.τ[i,:] + z_c' * ivbma.β[i,:]
        var_q_y[i] = σ_y_x
    end

    if ivbma.dist[1] == "Gaussian"
        ds = map((μ, σ) -> Normal(μ, sqrt(σ)), mean_q_y, var_q_y)
    elseif ivbma.dist[1] == "PLN"
        q_y = map((μ, σ) -> rand(Normal(μ, sqrt(σ))), mean_q_y, var_q_y)
        ds = map(q -> Poisson(exp(q)), q_y)
    elseif ivbma.dist[1] == "BL"
        q_y = map((μ, σ) -> rand(Normal(μ, sqrt(σ))), mean_q_y, var_q_y)
        μ, r = (logit.(q_y), ivbma.r[i, 1])
        B_α, B_β = (μ * r, r * (1 .- μ))
        ds = map((b_α, b_β) -> Beta(b_α, b_β), B_α, B_β)
    end

    d = MixtureModel(ds)
    return d
end
