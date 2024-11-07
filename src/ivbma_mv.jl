
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
    return inv(U' * (g/(g+1) * I - P_ι) * U)
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


function ivbma_mv_mcmc(
    y::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    Z::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real},
    iter::Integer = 2000,
    burn::Integer = 1000,
    ν::Number = size(X, 2) + 2,
    g_L::Union{Function, Number} = length(y),
    g_M::Union{Function, Number} = length(y)
)
    # dimensions
    n, l = size(X)
    p = size(Z, 2); k = size(W, 2)

    # centre instruments and covariates
    Z = Z .- mean(Z; dims = 1)
    W = W .- mean(W; dims = 1)

    # starting values
    α, τ, β = (0, zeros(l), zeros(k))
    Γ, Δ = (zeros(l), zeros(k+p, l))
    Σ = Diagonal(ones(l+1))
    L = sample([true, false], k, replace = true)
    M = sample([true, false], k+p, replace = true)

    # storage objects
    nsave = iter - burn
    α_samples = zeros(nsave)
    τ_samples = zeros(nsave, l)
    β_samples = zeros(nsave, k)
    Γ_samples = zeros(nsave, l)
    Δ_samples = zeros(nsave, k + p, l)
    Σ_samples = Array{Matrix{Float64}}(undef, nsave)
    L_samples = zeros(Bool, nsave, k)
    M_samples = zeros(Bool, nsave, k + p)

    # Some precomputations
    ι = ones(n)
    U = [X W[:, L]]
    A = calc_A(U, g_L)

    V = [Z W][:, M]
    H = X - (ι * Γ' + V * Δ[M, :])

    # Gibbs sampler
    for i in 1:iter

        # Some precomputations
        (σ_y_x, Σ_yx, Σ_xx) = variances(Σ)
        B = calc_B_Σ(σ_y_x, Σ_yx, Σ_xx)
        
        # Update y_tilde
        y_tilde = y - H * inv(Σ_xx) * Σ_yx

        # Step 1: Outcome model
        # Update model
        prop = mc3_proposal(L)
        U_prop = [X W[:, prop]]
        A_prop = calc_A(U_prop, g_L)
        
        acc = min(1, exp(
            marginal_likelihood_outcome(y_tilde, A_prop, U_prop, σ_y_x, g_L) - marginal_likelihood_outcome(y_tilde, A, U, σ_y_x, g_L)
        ))
        if rand() < acc
            L = prop
            U = U_prop
            A = A_prop
        end

        # Update parameters
        α, τ, β = post_sample_outcome(y_tilde, X, A, U, σ_y_x)

        # Update residuals
        ϵ = y - (α * ι + U * [τ; β])
        X_tilde = X - (1/σ_y_x) * ϵ * Σ_yx' * inv(B)'

        # Step 2: Treatment model

        # Update model
        prop = mc3_proposal(M)
        V_prop = [Z W][:, prop]

        acc = min(1, exp(
            marginal_likelihood_treatment(X_tilde, B, V_prop, σ_y_x, g_M) - marginal_likelihood_treatment(X_tilde, B, V, σ_y_x, g_M)
        ))
        if rand() < acc
            M = prop
            V = V_prop
        end

        # Update parameters
        Γ, Δ = post_sample_treatment(X_tilde, B, V, Σ_xx, g_M)

        # Update residuals
        H = X - (ι * Γ' + V * Δ)

        # Step 3: Update covariance
        Σ = post_sample_cov(ϵ, H, ν)

        # Step 4: Store sampled values after burn in
        if i > burn
            α_samples[i - burn] = α
            τ_samples[i - burn, :] = τ
            β_samples[i - burn, L] = β
            Γ_samples[i - burn, :] = Γ
            Δ_samples[i - burn, M, :] = Δ
            Σ_samples[i - burn] = Σ
            L_samples[i - burn, :] = L
            M_samples[i - burn, :] = M
        end

    end

    
    return (
        α = α_samples,
        τ = τ_samples,
        β = β_samples,
        Γ = Γ_samples,
        Δ = Δ_samples,
        Σ = Σ_samples,
        L = L_samples,
        M = M_samples
    )

end
