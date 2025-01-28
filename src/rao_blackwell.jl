
"""
    Rao-blackwellise the marginal posterior of τ.
"""
function rbw(sample::GIVBMA)
    n, l = size(sample.X)
    Z_c, W_c = (sample.Z .- mean(sample.Z; dims = 1), sample.W .- mean(sample.W; dims = 1))
    V = [ones(n) Z_c W_c]
    
    means_tau = Matrix(undef, length(sample.α), l)
    vars_tau = Matrix(undef, length(sample.α), l)

    for i in eachindex(sample.α)
        (σ_y_x, Σ_yx, Σ_xx) = variances(sample.Σ[i])
        H = sample.Q[i, :, 2:end] - V * [(sample.Γ[i, :])'; sample.Δ[i, :, :]]
        y_tilde = sample.Q[i, :, 1] - H * inv(Σ_xx) * Σ_yx

        U = [ones(n) sample.X sample.W[:, sample.L[i, :]]]
        sf = sample.G[i, 1] / (1 + sample.G[i, 1])

        rho_mean = sf * inv(U'U) * U' * y_tilde
        rho_cov = sf * σ_y_x * inv(U'U)
        
        means_tau[i, :] = rho_mean[2:(l+1)]
        vars_tau[i, :] = diag(rho_cov)[2:(l+1)]
    end

    d = [MixtureModel(map((μ, σ) -> Normal(μ, sqrt(σ)), means_tau[:, j], vars_tau[:, j])) for j in 1:l]

    return d
end




