

using IVBMA
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

    res = ivbma(y, X, Z)
    res_hyperg = ivbma(y, X, Z; g_prior = "hyper-g/n")
    res_BL = ivbma(y, X, Z; dist = ["Gaussian", "BL"])
    res_2c = ivbma(y, X[:, 1], Z[:, 3:end], [X[:, 2] Z[:, 1:2]]; two_comp = true)

    @test isapprox(mean(res.τ, dims = 1)[1,:], expected_taus; atol = 0.1)
    @test isapprox(mean(res_hyperg.τ, dims = 1)[1,:], expected_taus; atol = 0.2)
    @test isapprox(mean(res_BL.τ, dims = 1)[1,:], expected_taus; atol = 0.1)
    @test isapprox(mean(res_2c.τ), expected_taus[1]; atol = 0.1)
end

