
"""
    Compute prior model probabilities based on the prior
        ```math
        P(Mⱼ) = w^{k_j} (1-w)^{k - k_j},
        ```
    with a Beta(a, b) hyperprior on `w`. Note that the result is on a log-scale.
"""
function model_prior(x, k, a = 1, m = floor(k/2))
    b = (k - m) / m 
    kj = sum(x)
    
    lg(x) = SpecialFunctions.loggamma(x)
    res = lg(a+b) - (lg(a) + lg(b)) + lg(a+kj) + lg(b+k-kj) - lg(a+b+k)
    return res
end


"""
    Hyper-g/n prior.
"""
hyper_g_n(g; a = 3, n = 100) = (a-2)/(2*n) * (1 + g/n)^(-a/2)


