
"""
    Compute the log-predictive score on a holdout dataset.
"""
function lps(ivbma, y_h, X_h, Z_h, W_h)
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

function lps(ivbma, y_h, X_h, Z_h)
    # if X is a vector turn it into an nx1 matrix
    if ndims(X_h) == 1
        X_h = permutedims(X_h)'
    end

    n_h, l = size(X_h)
    n_post = length(ivbma.α)
    scores = Matrix{Float64}(undef, n_h, n_post)

    # demean holdout sample using the mean over the training sample
    Z_h = Z_h .- mean(ivbma.Z; dims = 1)

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