
"""
    Compute the log-predictive score on a holdout dataset.
"""
function lps(ivbma, y_h, X_h, Z_h, W_h)
    n_h, l = size(X_h)
    n_post = length(ivbma.α)
    scores = Matrix{Float64}(undef, n_h, n_post)

    gauss = map(idx -> (length(unique(ivbma.Q[:, 1, idx])) == 1), 1:l)
    Q = Matrix(undef, n_h, l)

    # demean holdout sample using the mean over the training sample
    Z_h = Z_h .- mean(ivbma.Z; dims = 1)
    W_h = W_h .- mean(ivbma.W; dims = 1)

    for i in 1:n_post
        σ_y_x, Σ_yx, Σ_xx = variances(ivbma.Σ[i])

        Mean_Q = (ones(n_h) * ivbma.Γ[i:i,:] + [Z_h W_h] * ivbma.Δ[i, :, :])
        Q = rand(MatrixNormal(Mean_Q, Diagonal(ones(n_h)), Σ_xx))

        # Use X instead of Q if Gaussian
        for (idx, d) in enumerate(gauss)
            if d
                Q[:, idx] = X_h[:, idx]
            end
        end

        H = Q - Mean_Q
        Mean_y = ivbma.α[i] * ones(n_h) + X_h * ivbma.τ[i,:] + W_h * ivbma.β[i,:] + H * inv(Σ_xx) * Σ_yx

        scores[:, i] = [pdf(Normal(Mean_y[j], σ_y_x), y_h[j]) for j in eachindex(y_h)]
    end

    scores_avg = mean(scores; dims = 2)
    lps = -mean(log.(scores_avg))
    return lps
end

function lps(ivbma, y_h, X_h, Z_h)
    n_h, l = size(X_h)
    n_post = length(ivbma.α)
    scores = Matrix{Float64}(undef, n_h, n_post)

    gauss = map(idx -> (length(unique(ivbma.Q[:, 1, idx])) == 1), 1:l)
    Q = Matrix(undef, n_h, l)

    # demean holdout sample using the mean over the training sample
    Z_h = Z_h .- mean(ivbma.W; dims = 1)

    for i in 1:n_post
        σ_y_x, Σ_yx, Σ_xx = variances(ivbma.Σ[i])

        Mean_Q = (ones(n_h) * ivbma.Γ[i:i,:] + Z_h * ivbma.Δ[i, :, :])
        Q = rand(MatrixNormal(Mean_Q, Diagonal(ones(n_h)), Σ_xx))

        # Use X instead of Q if Gaussian
        for (idx, d) in enumerate(gauss)
            if d
                Q[:, idx] = X_h[:, idx]
            end
        end

        H = Q - Mean_Q
        Mean_y = ivbma.α[i] * ones(n_h) + X_h * ivbma.τ[i,:] + Z_h * ivbma.β[i,:] + H * inv(Σ_xx) * Σ_yx

        scores[:, i] = [pdf(Normal(Mean_y[j], σ_y_x), y_h[j]) for j in eachindex(y_h)]
    end

    scores_avg = mean(scores; dims = 2)
    lps = -mean(log.(scores_avg))
    return lps
end