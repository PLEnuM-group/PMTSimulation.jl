module PMTPipeline

using DataFrames
using DSP
using PoissonRandom
using Distributions
import Base: @kwdef
import Pipe: @pipe
using PhysicalConstants.CODATA2018
using Unitful
using StatsBase
using Random
using Interpolations
using Roots
using PhysicsTools
using Optim

using ..SPETemplates
using ..PulseTemplates
using ..Waveforms
import ..PMTConfig


export make_reco_pulses
export apply_tt, apply_tt!, subtract_mean_tt, subtract_mean_tt!
export plot_hits, plot_pmt_map, map_f_over_pmts
export find_noise_scale


function find_noise_scale(target_adc_noise, adc_range, adc_bits)
    x = randn(10000)
    function _optim(sigma)

        adc_noise = std(digitize.(x .* sigma, Ref(adc_bins(adc_range, adc_bits))))

        return mean((adc_noise - target_adc_noise) .^ 2)
    end

    res = optimize(_optim, [1.0])
    return first(Optim.minimizer(res))
end




function apply_tt(hit_times::AbstractArray{<:Real}, tt_dist::UnivariateDistribution)

    tt = rand(tt_dist, size(hit_times))
    return hit_times .+ tt
end


function apply_tt!(df::AbstractDataFrame, tt_dist::UnivariateDistribution)
    tt = rand(tt_dist, nrow(df))

    df[!, :time] .+= tt
    return df
end


function subtract_mean_tt(hits::AbstractVector{<:Real}, tt_dist::UnivariateDistribution)
    hits .- mean(tt_dist)
end

function subtract_mean_tt!(df::AbstractDataFrame, tt_dist::UnivariateDistribution)
    df[!, :time] .-= mean(tt_dist)
    return df
end

function make_reco_pulses(results::AbstractDataFrame, pmt_config::PMTConfig, time_range)
    @pipe results |>
          apply_tt!(_, pmt_config.tt_dist) |>
          subtract_mean_tt!(_, pmt_config.tt_dist) |>
          PulseSeries(_, pmt_config) |>
          digitize_waveform(
              _,
              pmt_config,
              time_range=time_range
          ) |>
          unfold_waveform(_, pmt_config, alg=:nnls)
end

function plot_hits(target, groups...; ylabel="", title="")
    l = grid(4, 4)
    plots = []

    coords = rad2deg.(target.pmt_coordinates)

    for (i, (theta, phi)) in enumerate(eachcol(coords))
        p = plot(title=format("θ={:.2f}, ϕ={:.2f}", theta, phi), titlefontsize=8,)

        for grp in groups
            this_hits = get(grp, (pmt_id=i,), nothing)

            if !isnothing(this_hits)
                histogram!(p, this_hits[:, :time], bins=70:1:150, xlabel="Time (ns)", #yscale=:log10,
                    #ylim=(0.1, 5000),
                    label="",
                    yscale=:log10, ylim=(0.5, 1000),
                    alpha=0.7,
                    ylabel=ylabel,
                    margin=3.2Plots.mm, xlabelfontsize=8, ylabelfontsize=8,
                    legend_position=:outertopright,
                    #legend_columns=2,
                    legendfontsize=6,
                )
            end
        end

        push!(plots, p)

    end
    return plot(plots..., layout=l, size=(1200, 800), plot_title=title)
end

function plot_pmt_map(target, xmaps...; labels, ylabel="", title="")
    l = grid(4, 4)
    plots = []

    coords = rad2deg.(target.pmt_coordinates)
    first = true
    for (i, (theta, phi)) in enumerate(eachcol(coords))
        if first
            p = plot(title=format("θ={:.2f}, ϕ={:.2f}", theta, phi), titlefontsize=8,
                legend_column=2,
                legendfontsize=6,
                legend_position=:best)
        else
            p = plot(title=format("θ={:.2f}, ϕ={:.2f}", theta, phi), titlefontsize=8,
                legend_position=false)
        end
        first = false


        for (xmap, label) in zip(xmaps, labels)

            x = get(xmap, i, nothing)

            if isnothing(x)
                continue
            end
            plot!(p, x, label=label, ylabel=ylabel, xlabel="Time (ns)",
                margin=3.2Plots.mm, xlabelfontsize=8, ylabelfontsize=8,
                #legend_position=:outertopright,
            )
        end
        push!(plots, p)

    end
    return plot(plots..., layout=l, size=(1200, 800), plot_title=title)
end


function map_f_over_pmts(target, f, input)
    out_d = []
    for pmt_id in 1:get_pmt_count(target)
        if typeof(input) <: GroupedDataFrame
            in = get(input, (pmt_id=pmt_id,), nothing)
        else
            in = get(input, pmt_id, nothing)
        end
        if !isnothing(in)
            out = f(in)
            push!(out_d, (pmt_id, out))
        else
            push!(out_d, (pmt_id, nothing))
        end

    end

    return Dict(out_d)

end




end
