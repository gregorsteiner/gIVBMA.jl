
"""
    This file adds functions to rao-blackwellise point estimates from the posterior sample.
"""
function rbw_posterior_mean(sample)
    n, l = size(sample.X)
    Z_c, W_c = (sample.Z .- mean(sample.Z; dims = 1), sample.W .- mean(sample.W; dims = 1))
    V = [ones(n) Z_c W_c]
    
    tau_store = Matrix(undef, length(sample.α), l)

    for i in eachindex(sample.α)
        (σ_y_x, Σ_yx, Σ_xx) = variances(sample.Σ[i])
        H = sample.Q[i, :, 2:end] - V * [(sample.Γ[i, :])'; sample.Δ[i, :, :]]
        y_tilde = sample.Q[i, :, 1] - H * inv(Σ_xx) * Σ_yx
        L = sample.L[i, :]
        U = [ones(n) sample.X sample.W[:, L]]
        sf = sample.G[i, 1] / (1 + sample.G[i, 1])

        rho_mean = sf * inv(U'U) * U' * y_tilde
        tau_store[i, :] =  rho_mean[2:(l+1)]
    end

    return mean(tau_store; dims = 1)[1, :]
end




