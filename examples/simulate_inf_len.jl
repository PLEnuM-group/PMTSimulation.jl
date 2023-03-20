using PMTSimulation
using CairoMakie
using Distributions
using Random
using StaticArrays
using DSP
using Profile
using DataFrames
import Pipe: @pipe
using PoissonRandom
using Format
using Unitful
using PhysicalConstants.CODATA2018
using Roots
using Base.Iterators
using Random
using Interpolations
using StatsBase
using Optim

ElementaryCharge * 5E6 / 5u"ns" * 50u"Ω" |> u"mV"

#snr_db = 10 * log10(7^2 / 0.5^2)

snr_db = 10 * log10(7^2 / 0.1^2)

fwhm = 6.0
#fwhm = 30
gumbel_scale = gumbel_width_from_fwhm(fwhm)
gumbel_loc = 15

adc_range = (0.0, 1000.0)



find_noise_scale(0.6, adc_range, 12)

function _func(sigma)




pmt_config = PMTConfig(
    st=ExponTruncNormalSPE(expon_rate=1.0, norm_sigma=0.3, norm_mu=1.0, trunc_low=0.0, peak_to_valley=3.1),
    pm=PDFPulseTemplate(
        dist=Truncated(Gumbel(0, gumbel_scale) + gumbel_loc, 0, 100),
        amplitude=7.0 # mV
    ),
    snr_db=snr_db,
    sampling_freq=2.0,
    unf_pulse_res=0.1,
    adc_freq=0.208,
    adc_bits=12,
    adc_dyn_range=(0.0, 1000.0), #mV
    lp_cutoff=0.025,
    tt_mean=25, # TT mean
    tt_fwhm=1.5 # TT FWHM
)
spe_d = make_spe_dist(pmt_config.spe_template)

pulse_charges = 10 .^ (-1:0.3:4)
dyn_ranges_end = (100.0, 1000.0, 3000.0, 10000.0) # mV
data_unf_res = []

for (dr_end, c) in product(dyn_ranges_end, pulse_charges)
    pulse_times = rand(Uniform(0, 10), 300)
    for t in pulse_times
        ps = PulseSeries([t], [c], pmt_config.pulse_model)
        digi_wf = digitize_waveform(ps, pmt_config.sampling_freq, pmt_config.adc_freq, pmt_config.noise_amp, pmt_config.lp_filter, time_range=[-100, 200], yrange=(0.0, dr_end),)
        unfolded = unfold_waveform(digi_wf, pmt_config.pulse_model_filt, pmt_config.unf_pulse_res, 0.1, :nnls)
        if length(unfolded) > 0
            amax = sortperm(unfolded.charges)[end]

            push!(data_unf_res, (dr_end=dr_end, charge=c, time=t, reco_time=unfolded.times[amax], reco_charge=sum(unfolded.charges)))
        end
    end
end


data_unf_res = DataFrame(data_unf_res)
data_unf_res[:, :dt] = data_unf_res[:, :reco_time] - data_unf_res[:, :time]

time_res = combine(groupby(data_unf_res, [:charge, :dr_end]), :dt => mean, :dt => std, :dt => iqr)

colors = Makie.wong_colors()

fig = Figure()
ax = Axis(fig[1, 1], xscale=log10, yscale=log10,
    xlabel="Pulse Charge (PE)", ylabel="Time Resolution [IQR] (ns)")


for (i, (grpkey, grp)) in enumerate(pairs(groupby(time_res, :dr_end)))
    lines!(ax, grp[:, :charge], grp[:, :dt_iqr], label=string(grpkey[1]), color=colors[i])
end

group_color = [PolyElement(color=color, strokecolor=:transparent)
               for color in colors[1:4]]

group_linestyles = [LineElement(color=:black, linestyle=:solid),
    LineElement(color=:black, linestyle=:dash)]

#ylims!(ax, 0, 5)

dyn_range_labels = getproperty.(keys(groupby(time_res, :dr_end)), :dr_end)


Legend(
    fig[1, 2],
    group_color,
    string.(dyn_range_labels),
    "Dynamic range (mV)")

#hlines!(ax, 1/(pmt_config.adc_freq) / sqrt(12*9))

fig


begin
    pulse_series = PulseSeries([0, 100], [1, 100], pmt_config.pulse_model)
    waveform = Waveform(pulse_series, pmt_config.sampling_freq, pmt_config.noise_amp,
        time_range=(-50.0, 300.0))
    digi_wf = digitize_waveform(
        waveform,
        pmt_config.sampling_freq,
        pmt_config.adc_freq,
        pmt_config.lp_filter,
        yrange=pmt_config.adc_dyn_range,
        yres_bits=pmt_config.adc_bits)

    fig, ax = lines(waveform.timestamps, waveform.values, axis=(; xlabel="Time (ns)", ylabel="Amplitude (mV)"), label="Raw Waveform")

    unfolded_sig = unfold_waveform(digi_wf, pmt_config.pulse_model_filt, pmt_config.unf_pulse_res, 0.3, :nnls)

    reco = PulseSeries(unfolded_sig.times, unfolded_sig.charges, pmt_config.pulse_model)
    xs = -50:0.1:400
    reco_eval = evaluate_pulse_series(xs, reco)
    #=
    ax2 = Axis(
        fig[1, 1],
        yaxisposition=:right,
        ylabel="ADC Counts"
    )
    hidexdecorations!(ax2)
    hidespines!(ax2)
    linkxaxes!(ax, ax2)
    =#

    lines!(ax, digi_wf.timestamps, digi_wf.values, label="Digitized Waveform",
        color=:black)
    lines!(ax, xs, reco_eval, label="Unfolded Waveform")
    bins = adc_bins(pmt_config.adc_dyn_range, pmt_config.adc_bits)

    xlims!(ax, 100, 200)
    #hlines!(ax, bins[1:10], alpha=0.1)
    fig
end
