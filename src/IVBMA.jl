module IVBMA

using LinearAlgebra, Distributions, Statistics
using InvertedIndices, SpecialFunctions
using StatsPlots

export ivbma, ivbma_2c, lpd, posterior_predictive, plot


include("ivbma_priors.jl")
include("ivbma_fit.jl")
include("ivbma_2c.jl")
include("ivbma_tools.jl")


end
