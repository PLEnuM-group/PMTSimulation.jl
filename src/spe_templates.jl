module SPETemplates

import Base: @kwdef
using Random
using Distributions
using Optim
using SpecialFunctions
export SPEDistribution, ExponTruncNormalSPE
export make_spe_dist, spe_peak, peak_to_valley, get_expon_weight_for_pv
export ExGaussian, ExpGaussianSPEDist


"""
Abstract type for SPE distributions.

`Distribution` types can by created using [`make_spe_dist`]@ref
"""
abstract type SPEDistribution{T<:Real} end

"""
    spe_peak(::SPEDistribution)
Return the location of the spe peak (distribution mode)
"""
spe_peak(::SPEDistribution) = error("Not implemented")

"""
    peak_to_valley(spe::SPEDistribution)
Return the peak to valley ratio.

This is typically the ratio between the distribution value at the spe peak and the
minimum before the exponential increase
"""
function peak_to_valley(spe::SPEDistribution)
    spe_d = make_spe_dist(spe)
    res = optimize(x -> pdf(spe_d, x[1]), 0.0, 1.0)
    return spe_peak(spe) / minimum(res)
end

"""
    make_spe_dist(d::SPEDistribution)

Return a `Distribution` struct.
"""
make_spe_dist(d::SPEDistribution{T}) where {T} = error("not implemented")


"""
Mixture model of an exponential and a truncated normal distribution
"""
struct ExponTruncNormalSPE{T<:Real} <: SPEDistribution{T}
    expon_rate::T
    norm_sigma::T
    norm_mu::T
    trunc_low::T
    expon_weight::T
end

spe_peak(d::ExponTruncNormalSPE) = d.norm_mu

function make_spe_dist(d::ExponTruncNormalSPE{T}) where {T<:Real}

    norm = Normal(d.norm_mu, d.norm_sigma)
    tnorm = truncated(norm, d.trunc_low, Inf)

    expon = Exponential(d.expon_rate)
    dist = MixtureModel([expon, tnorm], [d.expon_weight, 1 - d.expon_weight])

    return dist
end

"""
    get_expon_weight_for_pv(ptov::Real, expon_rate::Real, norm_sigma::Real, norm_mu::Real, trunc_low::Real)

Calculate the weight of the exponential component for a given peak to valley ratio in an exp-normal mixture
"""
function get_expon_weight_for_pv(ptov::Real, expon_rate::Real, norm_sigma::Real, norm_mu::Real, trunc_low::Real)
    function _optim(w)
        pv = peak_to_valley(ExponTruncNormalSPE(
            expon_rate=expon_rate,
            norm_sigma=norm_sigma,
            norm_mu=norm_mu,
            trunc_low=trunc_low,
            expon_weight=w))
        return (pv - ptov)^2
    end
    res = optimize(_optim, 0.01, 0.99)

    return Optim.minimizer(res)

end

function ExponTruncNormalSPE(;
    expon_rate::Real,
    norm_sigma::Real,
    norm_mu::Real,
    trunc_low::Real,
    expon_weight::Union{<:Real,Nothing}=nothing,
    peak_to_valley::Union{<:Real,Nothing}=nothing)

    if isnothing(expon_weight) && isnothing(peak_to_valley)
        error("Set one of `expon_weight` and `peak_to_valley")
    end

    T = promote()

    if isnothing(expon_weight)
        expon_weight = get_expon_weight_for_pv(peak_to_valley, expon_rate, norm_sigma, norm_mu, trunc_low)
    end

    expon_rate, norm_sigma, norm_mu, trunc_low, expon_weight = promote(
        expon_rate, norm_sigma, norm_mu, trunc_low, expon_weight
    )

    ExponTruncNormalSPE(expon_rate, norm_sigma, norm_mu, trunc_low, expon_weight)
end


"""
Exponentially modified Gaussian.

Distribution of Z = X + Y, where:
```
    X ~ Normal(μ, σ)
    Y ~ Exponential(λ)
```
"""
struct ExGaussian{T} <: ContinuousUnivariateDistribution
    expon_rate::T
    norm_sigma::T
    norm_mu::T
end

function Base.rand(r::AbstractRNG, d::ExGaussian)
    dnorm = Normal(d.norm_mu, d.norm_sigma)
    dexp = Exponential(1/d.expon_rate)
    x = rand(r, dnorm)
    y = rand(r, dexp)
    return x + y
end

function Distributions.logpdf(d::ExGaussian, x::Real)
    λ = d.expon_rate
    μ = d.norm_mu
    σ = d.norm_sigma

    return log(λ / 2) + (λ / 2 * (2 * μ + λ*σ^2 - 2 * x)) + log(erfc((μ + λ * σ^2 - x) / (sqrt(2) * σ)))
end

function Distributions.cdf(d::ExGaussian, x::Real)
    λ = d.expon_rate
    μ = d.norm_mu
    σ = d.norm_sigma
    dnorm = Normal(μ, σ)

    return cdf(dnorm, x) - 0.5 * exp(λ / 2 * (2 * μ + λ*σ^2 - 2 * x)) * erfc((μ + λ * σ^2 - x) / (sqrt(2) * σ))
end

function Distributions.mean(d::ExGaussian)
    return d.norm_mu + 1 / d.expon_rate
end

function Distributions.var(d::ExGaussian)
    return d.norm_sigma^2 + 1 / d.expon_rate^2
end

function Distributions.mode(d::ExGaussian)
    function _optim(x)
        return -logpdf(d, x)
    end
    res = optimize(_optim, 0.01, 3 * mean(d))

    return Optim.minimizer(res)
end


"""
Exponentially modified Normal SPE Distribution
"""
struct ExpGaussianSPEDist{T<:Real} <: SPEDistribution{T}
    expon_rate::T
    norm_sigma::T
    norm_mu::T
end

function spe_peak(d::ExpGaussianSPEDist)
    return mode(d)
end

peak_to_valley(::ExpGaussianSPEDist) = error("Not implemented")

make_spe_dist(d::ExpGaussianSPEDist) = ExGaussian(d.expon_rate, d.norm_sigma, d.norm_mu)



end
