
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


"""
    Jeffreys prior for the t df parameter.
"""
jp_ν(ν, p) = ((ν+1)/(ν+3))^(p/2) * (ν/(ν+3))^(1/2) * (SpecialFunctions.trigamma(ν/2) -  SpecialFunctions.trigamma((ν+1)/2) - 2*(ν+3)/(ν*(ν+1)^2))^(1/2)

