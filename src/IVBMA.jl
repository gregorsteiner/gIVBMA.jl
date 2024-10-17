module IVBMA

using LinearAlgebra, Distributions, Statistics
using InvertedIndices, SpecialFunctions
using StatsPlots

export ivbma, ivbma_2c, iv_fit, lpd, posterior_predictive, plot


include("ivbma_priors.jl")
include("ivbma_fit.jl")
include("ivbma_2c.jl")
include("ivbma_tools.jl")
include("iv_fit.jl")


end
