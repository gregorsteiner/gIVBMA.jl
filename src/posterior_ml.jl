

"""
    Helper functions for repeated small calculations
"""
function variances(Σ)
    Σ_xx = Σ[2:end, 2:end]
    Σ_yx = Σ[2:end, 1]
    σ_y_x = Σ[1,1] - Σ_yx' * inv(Σ_xx) * Σ_yx

    return (σ_y_x, Σ_yx, Σ_xx)
end

function calc_A(U, g)
    ι = ones(size(U, 1))
    P_ι = ι * inv(ι'ι) * ι'
    
    A = inv(U' * ((g+1)/g * I - P_ι) * U)
    return A
end

calc_B_Σ(σ_y_x, Σ_yx, Σ_xx) = I + Σ_yx * Σ_yx' * inv(Σ_xx) / σ_y_x


"""
    Functions to sample from the conditional posteriors and compute marginal likelihoods.
"""
function post_sample_outcome(y_tilde, X, A, U, σ_y_x)
    n = size(X, 1); l = size(X, 2)
    ι = ones(n)
    M_ι = I - ι * inv(ι'ι) * ι'

    β_tilde = rand(MvNormal(A * U' * M_ι * y_tilde, Symmetric(σ_y_x * A)))
    τ, β = (β_tilde[1:l], β_tilde[(l+1):end])
    α = rand(Normal(ι' * (y_tilde - X * τ) / n, σ_y_x / n))

    return (α, τ, β)
end

function marginal_likelihood_outcome(y_tilde, A, U, σ_y_x, g)
    n = length(y_tilde)
    ι = ones(n)
    M_ι = I - ι * inv(ι'ι) * ι'

    ml = (1/2) * (log(det(A)) - log(det(g * inv(U'U)))) - (1/(2*σ_y_x)) * y_tilde' * M_ι * (I - U * A * U') * M_ι * y_tilde
    return ml 
end


function post_sample_treatment(X_tilde, B, V, Σ_xx, g)
    n = size(X_tilde, 1)
    V_t_V_inv = inv(V'V)
    ι = ones(n)
    sf = g / (g + 1) # shrinkage factor

    Γ = rand(MvNormal((ι' * X_tilde / n)[1,:], Symmetric((1/n) * inv(B) * Σ_xx)))
    Δ = rand(MatrixNormal(sf * V_t_V_inv * V'X_tilde, Symmetric(sf * V_t_V_inv), Symmetric(inv(B) * Σ_xx)))
    
    return (Γ, Δ)
end

function marginal_likelihood_treatment(X_tilde, B, V, Σ_xx, g)
    n, k_M = size(V)
    P_V = V * inv(V'V) * V'

    ml = -(k_M/2) * log(g+1) - (n/2) * det(B) - (1/2) * tr(inv(Σ_xx) * B * (-X_tilde' * P_V * X_tilde))
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


