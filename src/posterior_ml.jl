

"""
    Helper functions for repeated small calculations
"""
function variances(Σ)
    Σ_xx = Σ[2:end, 2:end]
    Σ_yx = Σ[2:end, 1]
    σ_y_x = Σ[1,1] - Σ_yx' * inv(Σ_xx) * Σ_yx

    return (σ_y_x, Σ_yx, Σ_xx)
end


calc_B_Σ(σ_y_x, Σ_yx, Σ_xx) = I + Σ_yx * Σ_yx' * inv(Σ_xx) / σ_y_x


"""
    Functions to sample from the conditional posteriors and compute marginal likelihoods.
"""
function post_sample_outcome(y_tilde, X, U, σ_y_x, g)
    n = size(X, 1); l = size(X, 2)
    ι = ones(n)
    sf = g / (g+1)

    ρ = rand(MvNormal(sf * inv(U'U) * U' * y_tilde, Symmetric(σ_y_x * sf * inv(U'U))))
    α, τ, β = (ρ[1], ρ[2:(l+1)], ρ[(l+2):end])
    
    return (α, τ, β)
end

function marginal_likelihood_outcome(y_tilde, U, σ_y_x, g)
    n, k_U = size(U)
    P_U = U * inv(U'U) * U'

    ml = -(k_U/2) * log(g+1) - (1/(2*σ_y_x)) * y_tilde' * (I - g/(g+1) * P_U) * y_tilde
    return ml 
end


function post_sample_treatment(X_tilde, B, V, Σ_xx, g)
    n = size(X_tilde, 1)
    V_t_V_inv = inv(V'V)

    Λ = rand(MatrixNormal( V_t_V_inv * V'X_tilde * inv(I + 1/g * inv(B))', Symmetric(V_t_V_inv), Symmetric(inv(B + 1/g * I) * Σ_xx)))
    Γ, Δ = (Λ[1, :], Λ[2:end, :])
    
    return (Γ, Δ)
end

function marginal_likelihood_treatment(X_tilde, B, V, Σ_xx, g)
    n, k_M = size(V)
    l = size(X_tilde, 2)
    P_V = V * inv(V'V) * V'

    C = inv(I + 1/g * inv(B))
    D = (B + 1/g * I)
    S = - inv(Σ_xx) * D * C * X_tilde' * P_V * X_tilde * C'

    ml = -(l*k_M/2) * log(g) - (n/2) * log(det(B + 1/g * I)) - (1/2) * tr(S)
    return ml
end

function post_sample_cov(ϵ, H, ν)
    n = size(ϵ, 1)
    Q = [ϵ H]' * [ϵ H]

    Σ = rand(InverseWishart(ν + n, I + Q))
    return Σ
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


