

using Revise, Infiltrator, Distributions, LinearAlgebra, StatsPlots
using Pkg; Pkg.activate(".")
using IVBMA


function gen_data(n::Integer = 100, p::Integer = 10, k::Integer = 10)
    V = rand(MvNormal(zeros(p+k), I), n)'
    Z = V[:,1:p]
    W = V[:,(p+1):(p+k)]

    α = 0; γ = -1; τ = 1/2
    δ_Z = [ones(Int(p/2)); zeros(Int(p/2))] .* (1/5)
    δ_W = [ones(Int(k/2)); zeros(Int(k/2))] .* (2/5)
    β = [ones(Int(k/2)); zeros(Int(k/2))] .* 2

    u = rand(MvNormal([0, 0], [1 1/2; 1/2 1]), n)'
    q = γ .+ Z * δ_Z + W * δ_W + u[:,2]
    x = [rand(Poisson(exp(q[j]))) for j in eachindex(q)]
    y = α .+ τ * x .+ W * β + u[:,1]

    return (y=y, x=x, Z=Z, W=W, q=q)
end

d = gen_data(50)
res = ivbma(d.y, d.x, d.Z, d.W; iter = 100, burn = 0, pln = true)

plot(res)
plot([map(x -> x[1,1], res.Σ) map(x -> x[1,2], res.Σ) map(x -> x[2,2], res.Σ)])

[d.q d.x]