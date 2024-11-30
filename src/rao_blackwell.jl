
"""
    This file adds functions to rao-blackwellise point estimates from the posterior sample.
"""
function rbw(sample; ci = 0.95)
    n, l = size(sample.X)
    Z_c, W_c = (sample.Z .- mean(sample.Z; dims = 1), sample.W .- mean(sample.W; dims = 1))
    V = [ones(n) Z_c W_c]

    alpha = 1 - ci
    
    means_tau = Matrix(undef, length(sample.α), l)
    vars_tau = Matrix(undef, length(sample.α), l)

    for i in eachindex(sample.α)
        (σ_y_x, Σ_yx, Σ_xx) = variances(sample.Σ[i])
        H = sample.Q[i, :, 2:end] - V * [(sample.Γ[i, :])'; sample.Δ[i, :, :]]
        y_tilde = sample.Q[i, :, 1] - H * inv(Σ_xx) * Σ_yx
        L = sample.L[i, :]
        U = [ones(n) sample.X sample.W[:, L]]
        sf = sample.G[i, 1] / (1 + sample.G[i, 1])

        rho_mean = sf * inv(U'U) * U' * y_tilde
        rho_cov = sf * σ_y_x * inv(U'U)
        
        means_tau[i, :] = rho_mean[2:(l+1)]
        vars_tau[i, :] = Diagonal(rho_cov)[2:(l+1)]
    end

    tau, lower, upper = (Vector(undef, l), Vector(undef, l), Vector(undef, l))
    for j in 1:l
        d = MixtureModel(map((μ, σ) -> Normal(μ, sqrt(σ)), means_tau[:, j], vars_tau[:, j]))
        tau[j] = mean(d)
        lower[j], upper[j] =  map(Base.Fix1(quantile, d), [alpha/2, 1 - alpha/2])
    end

    return (mean = tau, ci = [lower upper])
end




