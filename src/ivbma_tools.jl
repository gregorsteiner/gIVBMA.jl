


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
function posterior_predictive(PostSample::PostSample, x::AbstractVector, Z::AbstractMatrix, W::AbstractMatrix)
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
function lps(PostSample::PostSample, y::AbstractVector, x::AbstractVector, Z::AbstractMatrix, W::AbstractMatrix)
    n = length(y)
    pln = !isempty(PostSample.q) # Check if q matrix is empty to detect PLN => if true use q instead of x in η

    pd = Vector{Float64}(undef, length(PostSample.α))
    for i in eachindex(PostSample.α)
        if !pln
            η = x - (ones(n) .* PostSample.γ[i] + [Z W] * PostSample.δ[i,:])
            ψ = calc_psi(PostSample.Σ[i])
            Mean = PostSample.α[i] .* ones(n) + PostSample.τ[i] .* x + W * PostSample.β[i,:] + (PostSample.Σ[i][1,2]/PostSample.Σ[i][2,2]) .* η
            pd[i] = pdf(MvNormal(Mean, ψ^2 * I), y)
        elseif pln
            Mean = PostSample.α[i] .* ones(n) + PostSample.τ[i] .* x + W * PostSample.β[i,:]
            pd[i] = pdf(MvNormal(Mean, PostSample.Σ[i][1,1] * I), y)
        end
    end
    
    return -log(mean(pd))
end

function lps(PostSample::PostSample, y::AbstractVector, x::AbstractVector, Z::AbstractMatrix)
    n = length(y)
    pln = !isempty(PostSample.q) # Check if q matrix is empty to detect PLN => if true use q instead of x in η
    
    pd = Vector{Float64}(undef, length(PostSample.α))
    for i in eachindex(PostSample.α)
        if !pln
            η = x - (ones(n) .* PostSample.γ[i] + Z * PostSample.δ[i,:])
            ψ = calc_psi(PostSample.Σ[i])
            Mean = PostSample.α[i] .* ones(n) + PostSample.τ[i] .* x + Z * PostSample.β[i,:] + (PostSample.Σ[i][1,2]/PostSample.Σ[i][2,2]) .* η
            pd[i] = pdf(MvNormal(Mean, ψ^2 * I), y)
        elseif pln
            Mean = PostSample.α[i] .* ones(n) + PostSample.τ[i] .* x + Z * PostSample.β[i,:]
            pd[i] = pdf(MvNormal(Mean, PostSample.Σ[i][1,1] * I), y)
        end
        
    end
    
    return -log(mean(pd))
end

"""
    Plot method for IVBMA objects.
    This function plots traceplots and posterior densities of τ and σ₁₂.
"""

function StatsPlots.plot(ivbma::PostSample)
    tp_τ = plot(ivbma.τ, label = "", ylabel = "τ")
    dp_τ = density(ivbma.τ, fill = true, label = "p(τ | D)")

    σ12 = map(x -> x[1,2], ivbma.Σ)
    tp_σ12 = plot(σ12, ylabel = "σ₁₂", label = "")
    dp_σ12 = density(σ12, fill = true, label = "p(σ₁₂ | D)")

    if size(ivbma.g, 2) == 2
        lab = ["g_L" "g_M"]
    else
        lab = ["g_L" "g_l" "g_s"]
    end
    tp_g = plot(ivbma.g, yaxis = :log, ylabel = "g", label = lab)
    model_size = plot([sum(ivbma.L, dims = 2) sum(ivbma.M, dims = 2)], ylabel = "Model Size", label = ["L" "M"])

    p = plot(
        tp_τ, dp_τ,
        tp_σ12, dp_σ12,
        tp_g, model_size,
        layout = (3, 2)
        )
    return p
end

"""
    Create a summary table describing the MCMC output.
"""
function describe(post::PostSample; ci = 0.95)
    # Determine the CI bounds
    lower_quantile = (1 - ci) / 2
    upper_quantile = 1 - lower_quantile
    ci_percentage = Int(ci * 100)  # Convert to percentage for the header

    # Prepare storage for numerical data and parameter names
    numerical_data = []
    row_labels = []

    # Calculate summaries for τ parameters (inclusion probability is always 1)
    mean_val = mean(post.τ)
    std_dev = std(post.τ)
    lower_ci, upper_ci = quantile(post.τ, [lower_quantile, upper_quantile])
    push!(numerical_data, [mean_val, std_dev, lower_ci, upper_ci, 1.0])
    push!(row_labels, "τ")

    # Calculate summaries for β parameters
    for j in 1:size(post.β, 2)
        mean_val = mean(post.β[:, j])
        std_dev = std(post.β[:, j])
        lower_ci, upper_ci = quantile(post.β[:, j], [lower_quantile, upper_quantile])
        inclusion_prob = mean(post.L[:, j]) # posterior inclusion probability for β
        
        push!(numerical_data, [mean_val, std_dev, lower_ci, upper_ci, inclusion_prob])
        push!(row_labels, "β[$j]")
    end

    # Calculate summaries for δ parameters
    for j in 1:size(post.δ, 2)
        mean_val = mean(post.δ[:, j])
        std_dev = std(post.δ[:, j])
        lower_ci, upper_ci = quantile(post.δ[:, j], [lower_quantile, upper_quantile])
        inclusion_prob = mean(post.M[:, j]) # posterior inclusion probability for δ

        push!(numerical_data, [mean_val, std_dev, lower_ci, upper_ci, inclusion_prob])
        push!(row_labels, "δ[$j]")
    end

    # Convert numerical_data to a matrix
    numerical_matrix = hcat(numerical_data...)'

    # Create a header with the dynamic credible interval percentage
    header = ["Posterior Mean", "Posterior SD", "Lower $ci_percentage% CI", "Upper $ci_percentage% CI", "PIP"]
    
    # Display table with row labels
    pretty_table(numerical_matrix; header = header, row_labels = row_labels, alignment = [:r, :r, :r, :r, :r])
end


