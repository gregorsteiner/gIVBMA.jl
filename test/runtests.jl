

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
    expected_taus = [0.8, -1.0]

    res = givbma(y, X, Z)
    res_hyperg = givbma(y, X, Z; g_prior = "hyper-g/n")
    res_BL = givbma(y, X, Z; dist = ["Gaussian", "Gaussian", "BL"])

    post_pred = posterior_predictive(res_BL, X[1, :], Z[1, :])
    res_lps = lps(res_BL, y, X, Z)
    res_rbw = rbw(res_BL)

    @test isapprox(map(mean, rbw(res)), expected_taus; atol = 0.2)
    @test isapprox(map(mean, rbw(res_hyperg)), expected_taus; atol = 0.2)
    @test isapprox(map(mean, rbw(res_BL)), expected_taus; atol = 0.2)
    @test isapprox(res_lps, 0.545; atol = 0.1)
end

