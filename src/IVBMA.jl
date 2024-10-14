module IVBMA

using LinearAlgebra, Distributions, Statistics
using NamedArrays, InvertedIndices, SpecialFunctions
using StatsPlots

export ivbma, ivbma_2c, lpd, posterior_predictive, plot


include("ivbma_tools.jl")
include("ivbma_priors.jl")
include("ivbma_fit.jl")


end
