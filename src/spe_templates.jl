module SPETemplates

import Base: @kwdef
using Distributions
using Optim

export SPEDistribution, ExponTruncNormalSPE
export make_spe_dist, spe_peak, peak_to_valley, get_expon_weight_for_pv


"""
Abstract type for SPE distributions.

`Distribution` types can by created using [`make_spe_dist`]@ref
"""
abstract type SPEDistribution{T<:Real} end
spe_peak(::SPEDistribution) = error("Not implemented")
peak_to_valley(::SPEDistribution) = error("Not implemented")

"""
    make_spe_dist(d::SPEDistribution)

Return a `Distribution`
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

function peak_to_valley(spe::SPEDistribution)
    spe_d = make_spe_dist(spe)
    res = optimize(x -> pdf(spe_d, x[1]), 0.0, 1.0)
    return spe_peak(spe) / minimum(res)
end


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



#=
"""
Exponentially modified Normal SPE Distribution
"""
struct ExpModNormSPEDist{T<:Real} <: SPEDistribution{T}
    expon_rate::T
    norm_sigma::T
    norm_mu::T
end

function spe_peak(d::ExpModNormSPEDist)

    τ = 1 / d.expon_rate
    return d.norm_mu - τ * sqrt(2) * d.norm_sigma

end

peak_to_valley(::SPEDistribution) = error("Not implemented")

"""
    make_spe_dist(d::SPEDistribution)

Return a `Distribution`
"""
make_spe_dist(d::SPEDistribution{T}) where {T} = error("not implemented")
=#





end
