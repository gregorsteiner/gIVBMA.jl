

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

function least_squares_and_inverse(X, y)
    L = cholesky(X'X)    
    beta = L \ X'y
    XtX_inv = L \ I
    return beta, XtX_inv
end

function projection(X)
    Q = Matrix(qr(X).Q)
    return Q * Q'
end

"""
    Functions to sample from the conditional posteriors and compute marginal likelihoods.
"""
function post_sample_outcome(y_tilde, X, U, σ_y_x, g)
    l = size(X, 2)
    sf = g / (g+1)

    LS, U_t_U_inv = least_squares_and_inverse(U, y_tilde)
    ρ = rand(MvNormal(sf * LS, Symmetric(σ_y_x * sf * U_t_U_inv)))
    α, τ, β = (ρ[1], ρ[2:(l+1)], ρ[(l+2):end])
    
    return (α, τ, β)
end

function marginal_likelihood_outcome(y_tilde, U, σ_y_x, g)
    k_U = size(U, 2)
    P_U = projection(U)

    ml = -(k_U/2) * log(g+1) - (1/(2*σ_y_x)) * y_tilde' * (I - g/(g+1) * P_U) * y_tilde
    return ml 
end


function post_sample_treatment(X_tilde, B, V, Σ_xx, g::Number)
    LS, V_t_V_inv = least_squares_and_inverse(V, X_tilde)

    Λ = rand(MatrixNormal( LS * inv(I + 1/g * inv(B))', Symmetric(V_t_V_inv), Symmetric(inv(B + 1/g * I) * Σ_xx)))
    Γ, Δ = (Λ[1, :], Λ[2:end, :])
    
    return (Γ, Δ)
end

function marginal_likelihood_treatment(X_tilde, B, V, Σ_xx, g::Number)
    k_M = size(V, 2)
    l = size(X_tilde, 2)
    P_V = projection(V)

    C = inv(I + 1/g * inv(B))
    D = (B + 1/g * I)
    S = - inv(Σ_xx) * D * C * X_tilde' * P_V * X_tilde * C'

    ml = -(l*k_M/2) * log(g) - (k_M/2) * log(det(B + 1/g * I)) - (1/2) * tr(S)
    return ml
end

function post_sample_cov(ϵ, H, ν)
    n = size(ϵ, 1)
    Q = [ϵ H]' * [ϵ H]

    Σ = rand(InverseWishart(ν + n, I + Q))
    return Σ
end



"""
    Modified methods for the treatment posterior and marginal likelihood based on the two-component prior.
"""
function post_sample_treatment(X_tilde, B, V, Σ_xx, G::AbstractMatrix)
    b = B[1, 1] # must be 1x1 (two-comp prior is only supported in the scalar case)
    A_G = inv(b * V'V + inv(G) * V'V * inv(G))
    Λ = rand(MatrixNormal(b * A_G * V' * X_tilde,  Symmetric(A_G), Symmetric(Σ_xx)))
    Γ, Δ = (Λ[1, :], Λ[2:end, :])
    
    return (Γ, Δ)
end

function marginal_likelihood_treatment(X_tilde, B, V, Σ_xx, G::AbstractMatrix)
    b = B[1, 1] # must be 1x1 (two-comp prior is only supported in the scalar case)
    A_G = inv(b * V'V + inv(G) * V'V * inv(G))

    log_ml = (1/2)*(log(det(A_G)) - log(det(G * inv(V'V) * G))) + b^2 / 2 * tr(inv(Σ_xx) * X_tilde' * V * A_G * V' * X_tilde)
    return log_ml
end

"""
    Helper function to construct the G matrix.
"""
function G_constr(g, ind, p, k)
    return Diagonal([sqrt(g[1]); [repeat([sqrt(g[2])], p); repeat([sqrt(g[1])], k)][ind]])
end


