using IVBMA
using Test

include("helpers.jl")

# Define test parameters
true_tau = 0.1  # The true value of tau used to generate the data

@testset "IVBMA" begin
    Random.seed!(42)
    data = gen_data_KO2010(200, 1, true_tau)

    res = ivbma(data.y, data.x, data.Z, data.W)
    res_2c = ivbma(data.y, data.x, data.Z, data.W; two_comp = true)

    @test isapprox(mean(res.τ), true_tau; atol=0.1)
    @test isapprox(mean(res_2c.τ), true_tau; atol=0.1)
end

@testset "IVBMA-Inv" begin
    Random.seed!(42)
    data = gen_data_Kang2016(200, true_tau)

    res = ivbma(data.y, data.x, data.Z)
   
    @test isapprox(mean(res.τ), true_tau; atol=0.1)
end

@testset "IVBMA-PLN" begin
    Random.seed!(42)
    data = gen_data_pln(200, 1, true_tau)

    res = ivbma(data.y, data.x, data.Z, data.W; dist = "PLN")
    res_2c = ivbma(data.y, data.x, data.Z, data.W; dist = "PLN", two_comp = true)

    @test isapprox(mean(res.τ), true_tau; atol=0.1)
    @test isapprox(mean(res_2c.τ), true_tau; atol=0.1)
end

@testset "IVBMA-BL" begin
    Random.seed!(42)
    data = gen_data_bl(200, 1, true_tau)

    res = ivbma(data.y, data.x, data.Z, data.W; dist = "BL")
    res_2c = ivbma(data.y, data.x, data.Z, data.W; dist = "BL", two_comp = true)

    @test isapprox(mean(res.τ), true_tau; atol=0.2)
    @test isapprox(mean(res_2c.τ), true_tau; atol=0.2)
end

