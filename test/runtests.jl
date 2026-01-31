

using gIVBMA
using Test


using DataFrames, CSV, InvertedIndices, Random, LinearAlgebra, Distributions

data_path = joinpath(@__DIR__, "data", "Carstensen_Gundlach.csv")
df = CSV.read(data_path, DataFrame, missingstring="-999.999")

# change column names to match paper
rename!(df, :kaufman => "rule", :mfalrisk => "malfal", :exprop2 => "exprop", :lngdpc95 => "lngdpc",
        :frarom => "trade", :lat => "latitude", :landsea => "coast")

# only keep required columns  
needed_columns = ["lngdpc", "rule", "malfal", "maleco", "lnmort", "frost", "humid",
                  "latitude", "eurfrac", "engfrac", "coast", "trade"]
df = df[:, needed_columns]

# drop all observations with missing values in the variables
dropmissing!(df)

# fit models
y = df.lngdpc
X = [df.rule df.malfal]
Z = Matrix(df[:, needed_columns[Not(1:3)]])

# test if posterior means are close to expected values
@testset "CG" begin
    
    # fit different model specifications
    res = givbma(y, X, Z)
    res_hyperg = givbma(y, X, Z; g_prior = "hyper-g/n")
    res_BL = givbma(y, X, Z; dist = ["Gaussian", "Gaussian", "BL"])
    res_2c = givbma(y, X[:, 1], Z[:, 1:2], Z[:, 3:8]; two_comp = true, g_prior = "hyper-g/n")
    res_cholesky = givbma(y, X, Z; dist = ["Gaussian", "Gaussian", "BL"], g_prior = "hyper-g/n", cov_prior = "Cholesky")
    res_BL_full = givbma(y, X, Z; dist = ["Gaussian", "Gaussian", "BL"], model_start = "Full")

    # check if the estimated parameters match the expected values
    expected_taus = [0.8, -1.0]
    @test isapprox(map(mean, rbw(res)), expected_taus; atol = 0.2)
    @test isapprox(map(mean, rbw(res_hyperg)), expected_taus; atol = 0.2)
    @test isapprox(map(mean, rbw(res_BL)), expected_taus; atol = 0.2)
    @test isapprox(map(mean, rbw(res_2c))[1], 1.0; atol = 0.3)
    @test isapprox(map(mean, rbw(res_cholesky)), expected_taus; atol = 0.2)
    @test isapprox(map(mean, rbw(res_BL_full)), expected_taus; atol = 0.2)

    # Also check if the LPS computation and the Rao-Blackwellisation work
    res_lps = lps(res_BL, y, X, Z)
    res_rbw = rbw(res_BL)
    @test isapprox(res_lps, 0.545; atol = 0.1)
end

