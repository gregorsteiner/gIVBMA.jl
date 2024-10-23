

"""
    This file includes all files to sample from conditonal posteriors and compute conditional marginal likelihoods.
"""

# an auxiliary function to compute ψ
function calc_psi(Σ)
    return sqrt(Σ[1,1] - Σ[1,2]^2 / Σ[2,2])
end

function post_sample_outcome(y, U, U_t_U, η, Σ, g)
    n = length(y)
    ψ = calc_psi(Σ)

    y_bar = Statistics.mean(y)
    η_bar = Statistics.mean(η)

    if (rank(U_t_U) < size(U_t_U, 1))
        error("Non-full rank model!")
    end

    B = g/(g+1) * inv(U_t_U)

    α = rand(Normal(y_bar - Σ[1,2]/Σ[2,2] * η_bar, ψ^2/n))

    Mean = B * U' * (y - Σ[1,2]/Σ[2,2] * η)
    β_tilde = rand(MvNormal(Mean, Symmetric(ψ^2 * B)))
    τ = β_tilde[1]
    β = β_tilde[2:end]    
    
    return (α = α, τ = τ, β = β)
end

function post_sample_treatment(x, V, V_t_V, ϵ, Σ, g)
    n = length(x)

    if (rank(V_t_V) < size(V_t_V, 1))
        error("Non-full rank model!")
    end

    ψ = calc_psi(Σ)
    ϵ_bar = Statistics.mean(ϵ)

    a = Σ[1,2]^2/(Σ[2,2] * ψ^2) + 1
    A = (g / (a*g + 1)) * inv(V_t_V)

    γ = rand(Normal(-Σ[1,2]/a * ϵ_bar, Σ[2,2]/(a*n))) 

    δ = rand(MvNormal(a * A * V' * (x - (Σ[1,2]/Σ[1,1]) * ϵ), Σ[2,2] * Symmetric(A)))
    
    return (γ = γ, δ = δ)
end

function post_sample_cov(ϵ, η, ν)
    n = length(ϵ)

    Q = [ϵ η]' * [ϵ η]
    if any(map(!isfinite, Q))
        error("Infinite sample covariance: Try increasing ν!")
    end
    Σ = rand(InverseWishart(ν + n, I + Q))
    return (Σ = Σ)
end

function marginal_likelihood_outcome(y, U, U_t_U, η, Σ, g)
    n = length(y)
    k = size(U, 2)

    if (rank(U_t_U) < size(U_t_U, 1))
        error("Non-full rank model!")
    end
    
    ψ = calc_psi(Σ)
    y_bar = Statistics.mean(y)
    η_bar = Statistics.mean(η)

    y_tilde = y - Σ[1,2]/Σ[2,2] * η 
    s = y_tilde' * (I - g/(g+1) * (U * inv(U_t_U) * U')) * y_tilde - n * (y_bar - Σ[1,2]/Σ[2,2] * η_bar)^2
    
    log_ml =  (-(k)/2)*log(g+1) - s/(2*ψ^2)
    return log_ml
end

function marginal_likelihood_treatment(x, V, V_t_V, ϵ, Σ, g)
    n = length(x)
    k = size(V, 2)

    if (rank(V_t_V) < size(V_t_V, 1))
        error("Non-full rank model!")
    end

    ψ = calc_psi(Σ)
    ϵ_bar = Statistics.mean(ϵ)

    a = Σ[1,2]^2/(Σ[2,2] * ψ^2) + 1

    x_tilde = (x - (Σ[1,2]/Σ[1,1]) * ϵ)
    t = (Σ[2,2]/Σ[1,1]) * ϵ'ϵ + x'x - 2 * (Σ[1,2]/Σ[1,1]) * ϵ'x - n * (Σ[1,2]^2/a^2) * ϵ_bar^2 - (a*g / (a*g+1)) * (x_tilde' * (V * inv(V_t_V) * V') * x_tilde)
    
    log_ml = (-k/2)*log(1 + g*a) - a*t/(2*Σ[2,2])
    return log_ml
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


