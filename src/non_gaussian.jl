

"""
    (Inverse) Logistic function 
"""
logit(x) = exp(x) / (1+exp(x))

"""
    Barker proposal, see Zens & Steel (2024) and Livingstone & Zanella (2022)
"""
function gradient(y, X, Q, Mean_y, Mean_Q, σ_y_x, Σ_yx, Σ_xx, dist, idx_d, r)
    H = Q - Mean_Q
    cov_ratio = Σ_yx[idx_d] / Σ_xx[idx_d, idx_d]

    grad = cov_ratio / σ_y_x * (y - (Mean_y + H * inv(Σ_xx) * Σ_yx)) - H[:, idx_d]  * tr(inv(Σ_xx))
    if dist == "PLN"
        grad .+= X[:, idx_d] - exp.(Q[:, idx_d])
    elseif dist == "BL"
        grad .+= r * exp.(Q[:, idx_d]) ./ (1 .+ exp.(Q[:, idx_d])).^2 .* log.(X[:, idx_d] ./ (1 .- X[:, idx_d]))
    end
    return grad
end

function barker_proposal(q, gradient, proposal_variance)
    n = length(q)

    Qi = [rand(Normal(0, sqrt(proposal_variance[j]))) for j in 1:n]
    bi = 2 * (rand(n) .< (1 ./ (1 .+ exp.(-gradient .* Qi)))) .- 1

    q_prop = q .+ Qi .* bi
    return q_prop
end

function posterior_q(y, X, Q, Mean_y, Mean_Q, σ_y_x, Σ_yx, Σ_xx, dist, idx_d, r)
    post = [(
            logpdf(Normal(Mean_y[j] + ((Q - Mean_Q) * inv(Σ_xx) * Σ_yx)[j], sqrt(σ_y_x)), y[j])
            + logpdf(MvNormal(Mean_Q[j,:], Σ_xx), Q[j,:])
        ) for j in eachindex(y)]

    if dist == "PLN"
        post .+= [logpdf(Poisson(exp(Q[j, idx_d])), X[j, idx_d]) for j in eachindex(y)]
    elseif dist == "BL"
        μ = logit.(Q[:, idx_d])
        B_α = μ * r
        B_β = r * (1 .- μ)
        post .+= [logpdf(Beta(B_α[j], B_β[j]), X[j, idx_d]) for j in eachindex(y)]
    end
    
    return post
end

function barker_correction_term(curr, prop, GradCurr, GradProp)
    beta1 = -GradProp .* (curr .- prop)
    beta2 = -GradCurr .* (prop .- curr)

    result = -(max.(beta1, zeros(length(curr))) .+ log1p.(exp.(-abs.(beta1)))) .+
             (max.(beta2, zeros(length(curr))) .+ log1p.(exp.(-abs.(beta2))))

    return result
end

# posterior density for r in logs (only relevant for Beta-Logistic)
function post_r(r, x, q, r_prior::Distribution)
    post = sum([logpdf(Beta(logit(q[j]), r), x[j]) for j in eachindex(x)]) + logpdf(r_prior, r)
    return post
end


# helper function to generate the set values (only relevant for BL if we have 0s or 1s)
function set_values_0_1(x, q, r)
    μ = logit.(q)
    B_α = μ * r
    B_β = r * (1 .- μ)
    for i in eachindex(x)
        if x[i] <= 0.001
            x[i] = rand(truncated(Beta(B_α[i], B_β[i]), 1e-12, 0.001))
        elseif x[i] >= 0.999
            x[i] = rand(truncated(Beta(B_α[i], B_β[i]), 0.999, 1 - 1e-12))
        end
    end
    return x
end