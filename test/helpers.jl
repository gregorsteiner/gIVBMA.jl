
using Random
using LinearAlgebra
using Distributions


function gen_instr_coeff(p::Integer, c_M::Number)
    res = zeros(p)
    for i in 1:p
        if i <= p/2
            res[i] = c_M * (1 - i/(p/2 + 1))^4
        end
    end
    return res
end

function gen_data_KO2010(n::Integer = 100, c_M::Number = 3/8, τ::Number = 0.1, p::Integer = 20, k::Integer = 10, c::Number = 1/2)
    V = rand(MvNormal(zeros(p+k), I), n)'
    Z = V[:,1:p]
    W = V[:,(p+1):(p+k)]

    α = 1; γ = 1
    δ_Z = gen_instr_coeff(p, c_M)
    δ_W = [ones(Int(k/2)); zeros(Int(k/2))] .* 0.1
    β = [ones(Int(k/2)); zeros(Int(k/2))]

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    x = γ .+ Z * δ_Z + W * δ_W + u[:,2]
    y = α .+ τ * x .+ W * β + u[:,1]

    # centre all regressors
    y = y .- mean(y)
    x = x .- mean(x)
    Z = Z .- mean(Z; dims = 1)
    W = W .- mean(W; dims = 1)

    return (y=y, x=x, Z=Z, W=W)
end

function gen_data_pln(n::Integer = 100, c_M::Number = 3/8, τ::Number = 0.1, p::Integer = 20, k::Integer = 10, c::Number = 1/2)
    V = rand(MvNormal(zeros(p+k), I), n)'
    Z = V[:,1:p]
    W = V[:,(p+1):(p+k)]

    α = 1; γ = -1
    δ_Z = gen_instr_coeff(p, c_M)
    δ_W = [ones(Int(k/2)); zeros(Int(k/2))] .* 0.1
    β = [ones(Int(k/2)); zeros(Int(k/2))]

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    q = γ .+ Z * δ_Z + W * δ_W + u[:,2]
    x = [rand(Poisson(exp(q[j]))) for j in eachindex(q)]
    y = α .+ τ * x .+ W * β + u[:,1]

    # centre all regressors
    Z = Z .- mean(Z; dims = 1)
    W = W .- mean(W; dims = 1)

    return (y=y, x=x, q=q, Z=Z, W=W)
end

function gen_data_bl(n::Integer = 100, c_M::Number = 3/8, τ::Number = 0.1, p::Integer = 20, k::Integer = 10, c::Number = 1/2)
    V = rand(MvNormal(zeros(p+k), I), n)'
    Z = V[:,1:p]
    W = V[:,(p+1):(p+k)]

    α = 1; γ = -1
    δ_Z = gen_instr_coeff(p, c_M)
    δ_W = [ones(Int(k/2)); zeros(Int(k/2))] .* 0.1
    β = [ones(Int(k/2)); zeros(Int(k/2))]
    r = 1

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    q = γ .+ Z * δ_Z + W * δ_W + u[:,2]

    μ = exp.(q) ./ (1 .+ exp.(q))
    B_α = μ * r
    B_β = r * (1 .- μ)
    x = [rand(Beta(B_α[j], B_β[j])) for j in eachindex(q)]
    y = α .+ τ * x .+ W * β + u[:,1]

    # centre all regressors
    Z = Z .- mean(Z; dims = 1)
    W = W .- mean(W; dims = 1)

    return (y=y, x=x, q=q, Z=Z, W=W)
end

function gen_data_Kang2016(n::Integer = 200, τ::Number = 0.1, p::Integer = 10, s::Integer = 2, c::Number = 0.5)
    Z = rand(MvNormal(zeros(p), I), n)'

    α = γ = 1
    δ = ones(p) .* 1/2 # chosen s.t. the first-staeg R^2 is approximately 0.2
    β = [ones(s); zeros(p-s)]

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    x = γ .+ Z * δ + u[:,2]
    y = α .+ τ * x .+ Z * β + u[:,1]

    # centre all regressors
    y = y .- mean(y)
    x = x .- mean(x)
    Z = Z .- mean(Z; dims = 1)

    return (y=y, x=x, Z=Z)
end


