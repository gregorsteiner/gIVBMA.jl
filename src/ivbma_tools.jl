

"""
    A helper function to calculate ψ.
"""
function calc_psi(Σ)
    return sqrt(Σ[1,1] - Σ[1,2]^2 / Σ[2,2])
end

"""
    This is a helper function to adaptively tune the proposal variances of the MH steps.
"""
function adjust_variance(propVar, n_acc, iter)
    acc_rate = n_acc / iter
    if acc_rate < 0.2
        return propVar * 0.99
    elseif acc_rate > 0.4
        return propVar * 1.01
    end
    return propVar
end


"""
    Auxiliary function to check if a model has an instrument.
"""
function has_instrument(L::Vector{Bool}, M::Vector{Bool})::Bool
    # Ensure L and M have the same length
    if length(L) != length(M)
        throw(ArgumentError("Vectors L and M must have the same length"))
    end

    # Check if there's any true element in M that is false in L
    for i in 1:length(L)
        if M[i] && !L[i]
            return true
        end
    end

    return false
end



"""
    Obtain a sample from the posterior predictive of y|x.
"""
function posterior_predictive(PostSample::Union{PostSampleIV, PostSampleIVBMA, PostSampleIVBMA2C}, x::AbstractVector, Z::AbstractMatrix, W::AbstractMatrix)
    n = length(x)
    y = Matrix{Float64}(undef, length(PostSample.α), n)
    for i in eachindex(PostSample.α)
        η = x - (ones(n) .* PostSample.γ[i] + [Z W] * PostSample.δ[i,:])
        ψ = calc_psi(PostSample.Σ[i])
        Mean = PostSample.α[i] .* ones(n) + PostSample.τ[i] .* x + W * PostSample.β[i,:] + (PostSample.Σ[i][1,2]/PostSample.Σ[i][2,2]) .* η
        y[i,:] = rand(MvNormal(Mean, ψ^2 * I))
    end
    return y
end

"""
    Compute the log predictive score on a holdout dataset.
"""
function lpd(PostSample::Union{PostSampleIV, PostSampleIVBMA, PostSampleIVBMA2C}, y::AbstractVector, x::AbstractVector, Z::AbstractMatrix, W::AbstractMatrix)
    n = length(y)
    pd = Vector{Float64}(undef, length(PostSample.α))
    for i in eachindex(PostSample.α)
        η = x - (ones(n) .* PostSample.γ[i] + [Z W] * PostSample.δ[i,:])
        ψ = calc_psi(PostSample.Σ[i])
        Mean = PostSample.α[i] .* ones(n) + PostSample.τ[i] .* x + W * PostSample.β[i,:] + (PostSample.Σ[i][1,2]/PostSample.Σ[i][2,2]) .* η
        pd[i] = pdf(MvNormal(Mean, ψ^2 * I), y)
    end
    
    return -log(mean(pd))
end

function lpd(PostSample::Union{PostSampleIV, PostSampleIVBMA, PostSampleIVBMA2C}, y::AbstractVector, x::AbstractVector, Z::AbstractMatrix)
    n = length(y)
    pd = Vector{Float64}(undef, length(PostSample.α))
    for i in eachindex(PostSample.α)
        η = x - (ones(n) .* PostSample.γ[i] + Z * PostSample.δ[i,:])
        ψ = calc_psi(PostSample.Σ[i])
        Mean = PostSample.α[i] .* ones(n) + PostSample.τ[i] .* x + Z * PostSample.β[i,:] + (PostSample.Σ[i][1,2]/PostSample.Σ[i][2,2]) .* η
        pd[i] = pdf(MvNormal(Mean, ψ^2 * I), y)
    end
    
    return -log(mean(pd))
end

"""
    Plot method for IV, IVBMA or IVBMA_2c objects.
    This function plots traceplots and posterior densities of τ and σ₁₂.
"""
function StatsPlots.plot(ivbma::PostSampleIV)
    tp_τ = plot(ivbma.τ, label = "", ylabel = "τ")
    dp_τ = density(ivbma.τ, fill = true, label = "p(τ | D)")

    σ12 = map(x -> x[1,2], ivbma.Σ)
    tp_σ12 = plot(σ12, ylabel = "σ₁₂", label = "")
    dp_σ12 = density(σ12, fill = true, label = "p(σ₁₂ | D)")

    p = plot(
        tp_τ, dp_τ,
        tp_σ12, dp_σ12,
        layout = (2, 2)
        )
    return p
end

function StatsPlots.plot(ivbma::PostSampleIVBMA)
    tp_τ = plot(ivbma.τ, label = "", ylabel = "τ")
    dp_τ = density(ivbma.τ, fill = true, label = "p(τ | D)")

    σ12 = map(x -> x[1,2], ivbma.Σ)
    tp_σ12 = plot(σ12, ylabel = "σ₁₂", label = "")
    dp_σ12 = density(σ12, fill = true, label = "p(σ₁₂ | D)")

    g = [ivbma.g_L ivbma.g_M]
    tp_g = plot(g, yaxis = :log, ylabel = "g", label = ["g_L" "g_M"])
    model_size = plot([sum(ivbma.L, dims = 2) sum(ivbma.M, dims = 2)], ylabel = "Model Size", label = ["L" "M"])

    p = plot(
        tp_τ, dp_τ,
        tp_σ12, dp_σ12,
        tp_g, model_size,
        layout = (3, 2)
        )
    return p
end

function StatsPlots.plot(ivbma::PostSampleIVBMA2C)
    tp_τ = plot(ivbma.τ, label = "", ylabel = "τ")
    dp_τ = density(ivbma.τ, fill = true, label = "p(τ | D)")

    σ12 = map(x -> x[1,2], ivbma.Σ)
    tp_σ12 = plot(σ12, ylabel = "σ₁₂", label = "")
    dp_σ12 = density(σ12, fill = true, label = "p(σ₁₂ | D)")

    g = [ivbma.g_L ivbma.g_s ivbma.g_l]
    tp_g = plot(g, yaxis = :log, ylabel = "g", label = ["g_L" "g_s" "g_l"])
    model_size = plot([sum(ivbma.L, dims = 2) sum(ivbma.M, dims = 2)], ylabel = "Model Size", label = ["L" "M"])

    p = plot(
        tp_τ, dp_τ,
        tp_σ12, dp_σ12,
        tp_g, model_size,
        layout = (3, 2)
        )
    return p
end


