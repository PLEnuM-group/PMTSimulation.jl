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

ElementaryCharge * 5E6 / 5u"ns" * 50u"Ω" |> u"mV"


pmt_config = PMTConfig(
    st=ExponTruncNormalSPE(expon_rate=1.0, norm_sigma=0.3, norm_mu=1.0, trunc_low=0.0, peak_to_valley=3.1),
    pm=PDFPulseTemplate(
        dist=truncated(Gumbel(0, gumbel_width_from_fwhm(5.0)) + 4, 0, 20),
        amplitude=10.0 # mV
    ),
    snr_db=30,
    sampling_freq=2.0,
    unf_pulse_res=0.1,
    adc_freq=0.25,
    adc_bits=12,
    adc_dyn_range=(0.0, 100.0), #mV
    lp_cutoff=0.125,
    tt_mean=25, # TT mean
    tt_fwhm=1.5 # TT FWHM
)

spe_d = make_spe_dist(pmt_config.spe_template)
lines(0:0.01:5, x -> pdf(spe_d, x),
    axis=(; title="SPE Template", xlabel="Charge (PE)", ylabel="PDF"))


fig, ax, l = lines(-10:0.1:50, x -> evaluate_pulse_template(pmt_config.pulse_model, 0.0, x),
    axis=(; ylabel="Amplitude (a.u.)", xlabel="Time (ns)", title="Pulse template"), label="Unfiltered")
lines!(ax, -10:0.1:50, x -> evaluate_pulse_template(pmt_config.pulse_model_filt, 0.0, x), label="Filtered (125Mhz LPF)")
axislegend(ax)
fig
save(joinpath(@__DIR__, "../figures/pulse_shape.png"), fig)



pulse_times = sort(rand(Uniform(0, 5), 6))
pulse_charges = [0.1, 0.3, 1, 5, 50, 100]
fig = Figure()
for (i, (t, c)) in enumerate(zip(pulse_times, pulse_charges))
    row, col = divrem(i - 1, 3)
    @show row, col
    ax = Axis(fig[row+1, col+1], ylabel="Amplitude (a.u.)", xlabel="Time (ns)", title=format("t={:.2f} ns, c= {:.2f} (PE)", t, c))
    pulses = PulseSeries([t], [c], pmt_config.pulse_model)
    waveform = make_waveform(pulses, pmt_config.sampling_freq, pmt_config.noise_amp)
    digi_wv = digitize_waveform(waveform, pmt_config.sampling_freq, pmt_config.adc_freq, pmt_config.lp_filter, yrange=(0, 100))

    lines!(waveform.timestamps, waveform.values, label="Unfiltered")
    lines!(ax, digi_wv.timestamps, digi_wv.values, label="Digitized")
    #axislegend(ax)
    CairoMakie.xlims!(ax, -10, 20)
end
fig
save(joinpath(@__DIR__, "../figures/pulses_with_digi.png"), fig)

pulses = PulseSeries([rand()], [5.0], pmt_config.pulse_model)
digi_wv = digitize_waveform(pulses, pmt_config.sampling_freq, pmt_config.adc_freq, 0.01, pmt_config.lp_filter; yrange=pmt_config.adc_dyn_range, yres_bits=pmt_config.adc_bits)
unfolded_pulses = unfold_waveform(digi_wv, pmt_config.pulse_model_filt, pmt_config.unf_pulse_res, 0.2, :nnls)



ts = -20:0.1:50
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Time (ns)", ylabel="Amplitude (a.u.)")

reco = PulseSeries(unfolded_pulses.times, unfolded_pulses.charges, pmt_config.pulse_model)

lines!(ax, ts, evaluate_pulse_series(ts, pulses), label="Original Pulse")
lines!(ax, digi_wv.timestamps, digi_wv.values, label="Digitized Pulse")
lines!(ax, ts, evaluate_pulse_series(ts, reco), label="Reconstructed Pulse")
axislegend(ax)
CairoMakie.xlims!(ax, -20, 50)
fig

save(joinpath(@__DIR__, "../figures/pulses_unfolding.png"), fig)


pulse_charges = 10 .^ (-1:0.1:2)
data_unf_res = []
for c in pulse_charges
    pulse_times = rand(Uniform(0, 10), 5000)
    for t in pulse_times
        ps = PulseSeries([t], [c], pmt_config.pulse_model)
        digi_wf = digitize_waveform(ps, pmt_config.sampling_freq, pmt_config.adc_freq, 0.01, pmt_config.lp_filter, time_range=[-10, 20])
        unfolded = unfold_waveform(digi_wf, pmt_config.pulse_model_filt, pmt_config.unf_pulse_res, 0.3, :nnls)
        if length(unfolded) > 0
            amax = sortperm(unfolded.charges)[end]
            push!(data_unf_res, (charge=c, time=t, reco_time=unfolded.times[amax], reco_charge=sum(unfolded.charges)))
        end
    end
end
data_unf_res = DataFrame(data_unf_res)
data_unf_res[:, :dt] = data_unf_res[:, :reco_time] - data_unf_res[:, :time]

data_unf_res

time_res = combine(groupby(data_unf_res, :charge), :dt => mean, :dt => std)
lines(time_res[:, :charge], time_res[:, :dt_std])

grpd = groupby(data_unf_res, :charge)
hist(grpd[end-5][:, :dt])


fig = hist(reco_dt, bins=-3:0.05:3, axis=(; xlabel="Reco time - pulse time (ns)"))
save(joinpath(@__DIR__, "../figures/spe_time_res.png"), fig)
fig = hist(dcharge ./ c, bins=-1:0.01:1, axis=(; xlabel="(Reco Charge - True Charge)/True Charge"))
save(joinpath(@__DIR__, "../figures/spe_charge_res.png"), fig)



time_sep_per_GeV = (50 / 0.3) / 1E6
tau_log_e = 3:0.05:5
time_sep = time_sep_per_GeV .* 10 .^ tau_log_e
pulse_times = rand(Uniform(0, 10), 200)

c = 1
rdt = []
rdc = []
sucess = []
results = []
for tle in tau_log_e
    time_sep = 10^tle * time_sep_per_GeV
    for t in pulse_times
        ps = PulseSeries([t, t + time_sep], [c, c], pmt_config.pulse_model)
        digi_wf = digitize_waveform(ps, pmt_config.sampling_freq, pmt_config.adc_freq, 0.01, pmt_config.lp_filter, time_range=[-10, 50])
        unfolded_sig = unfold_waveform(digi_wf, pmt_config.pulse_model_filt, pmt_config.unf_pulse_res, 0.3, :nnls)

        ps = PulseSeries([t, t], [c, c], pmt_config.pulse_model)
        digi_wf = digitize_waveform(ps, pmt_config.sampling_freq, pmt_config.adc_freq, 0.01, pmt_config.lp_filter, time_range=[-10, 50])
        unfolded_bg = unfold_waveform(digi_wf, pmt_config.pulse_model_filt, pmt_config.unf_pulse_res, 0.3, :nnls)

        push!(results, (np_sig=length(unfolded_sig), np_bg=length(unfolded_bg), tle=tle))
    end
end

results = DataFrame(results)
results_mean = combine(groupby(results, :tle), [:np_sig, :np_bg] .=> mean)

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Log10(Energy)", ylabel="Mean number of reco pulses")

lines!(ax, results_mean[:, :tle], results_mean[:, :np_sig_mean], label="Signal")
lines!(ax, results_mean[:, :tle], results_mean[:, :np_bg_mean], label="BG (2PE single)")
axislegend(ax, position=:lt)
fig
save(joinpath(@__DIR__, "../figures/double_pulse.png"), fig)



fig = lines(10 .^ tau_log_e, Float64.(sucess), axis=(; xscale=log10, limits=(1E3, 1E5, 0, 1.1)))
fig = lines(10 .^ tau_log_e, Float64.(rdt), axis=(; xscale=log10))



ps = PulseSeries([1.0], [10.0], pmt_config.pulse_model)
digi_wf = digitize_waveform(ps, pmt_config.sampling_freq, pmt_config.adc_freq, 0.01, pmt_config.lp_filter)
unfolded = unfold_waveform(digi_wf, pmt_config.pulse_model_filt, pmt_config.unf_pulse_res, 0.5, :nnls)




distance = 50.0f0
pmt_area = Float32((75e-3 / 2)^2 * π)
target_radius = 0.21f0
target = MultiPMTDetector(@SVector[0.0f0, 0.0f0, distance], target_radius, pmt_area,
    make_pom_pmt_coordinates(Float32))
medium = make_cascadia_medium_properties(0.99f0)
targets = [target]


zenith_angle = 20.0f0
azimuth_angle = 0.0f0

particle = Particle(
    @SVector[0.0f0, 0.0f0, 0.0f0],
    sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle)),
    0.0f0,
    Float32(1E5),
    PEMinus
)


prop_source_ext = ExtendedCherenkovEmitter(particle, medium, (300.0f0, 800.0f0))
#prop_source_che = PointlikeCherenkovEmitter(particle, medium, (300f0, 800f0))

spectrum = CherenkovSpectrum((300.0f0, 800.0f0), 30, medium)

res = propagate_photons(prop_source_ext, target, medium, spectrum)

coszen_orient = 0
phi_orient = 0

orientation = sph_to_cart(acos(coszen_orient), phi_orient,)



res = make_hits_from_photons(res, prop_source_ext, target, medium, orientation)
res_grp_pmt = groupby(res, :pmt_id)

results = res_grp_pmt[5]
pmt_config = STD_PMT_CONFIG
waveform = @pipe results |>
                 resample_simulation |>
                 apply_tt!(_, pmt_config.tt_dist) |>
                 subtract_mean_tt!(_, pmt_config.tt_dist) |>
                 PulseSeries(_, pmt_config.spe_template, pmt_config.pulse_model) |>
                 digitize_waveform(
                     _,
                     pmt_config.sampling_freq,
                     pmt_config.adc_freq,
                     pmt_config.noise_amp,
                     pmt_config.lp_filter
                 )



p = histogram(results[:, :time], bins=200:250, xlabel="Time (ns)",
    ylabel="Counts", label="")
savefig(p, joinpath(@__DIR__, "../figures/example_photons.png"))
reco_pulses = make_reco_pulses(results)
p = plot(waveform, xlabel="Time (ns)", xlim=(200, 250),
    ylabel="Amplitude (a.u.)", label="")#xlim=(-50, 200))
savefig(p, joinpath(@__DIR__, "../figures/example_waveform.png"))


function plot_chain(results::AbstractDataFrame, pmt_config::PMTConfig)
    layout = @layout [a; b]

    p1 = histogram(results[:, :tres], weights=results[:, :total_weight],
        bins=-10:1:50, label="Photons", xlabel="Time residual (ns)", ylabel="Counts")

    hit_times = resample_simulation(results)
    p1 = histogram!(p1, hit_times, bins=-10:1:50, label="Hits")

    ps = PulseSeries(hit_times, pmt_config.spe_template, pmt_config.pulse_model)
    p2 = plot(ps, -10:0.01:50, label="True waveform", xlabel="Time residual (ns)", ylabel="Amplitude (a.u.)")

    wf = digitize_waveform(
        ps,
        pmt_config.sampling_freq,
        pmt_config.adc_freq,
        pmt_config.noise_amp,
        pmt_config.lp_filter
    )

    p2 = plot!(p2, wf, label="Digitized waveform")

    reco_pulses = unfold_waveform(wf, pmt_config.pulse_model_filt, pmt_config.unf_pulse_res, 0.2, :fnnls)
    pulses_orig_temp = PulseSeries(reco_pulses.times, reco_pulses.charges, pmt_config.pulse_model)


    p2 = plot!(p2, reco_pulses, -10:0.01:50, label="Reconstructed waveform")
    p2 = plot!(p2, pulses_orig_temp, -10:0.01:50, label="Unfolded waveform")

    plot(p1, p2, layout=layout, xlim=(-10, 50))
end


plot_chain(results, STD_PMT_CONFIG)

reco_pulses_che.times

xs = -10:1.0:50
ys = reco_pulses_che(xs)


mZero = MeanZero()                   #Zero mean function
kern = Mat32Iso(1.0, 1.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

logObsNoise = -1.0                        # log standard deviation of observation noise (this is optional)
gp = GP(xs, ys, mZero, kern, logObsNoise)

plot(reco_pulses_che)

plot!(gp, obsv=false, linestyle=:dash)

optimize!(gp)

plot!(gp, obsv=false,)


gp








hit_times = resample_simulation(results_che)
histogram(hit_times, bins=-20:5:100, alpha=0.7)





hit_times = resample_simulation(results_ext)
histogram!(hit_times, bins=-20:5:100, alpha=0.7)

#wf_model = make_gumbel_waveform_model(hit_times)
ps = PulseSeries(hit_times, STD_PMT_CONFIG.spe_template, STD_PMT_CONFIG.pulse_model)

plot(ps, -10, 200)


wf = make_waveform(ps, STD_PMT_CONFIG.sampling_freq, STD_PMT_CONFIG.noise_amp)



l = @layout [a; b]
p1 = plot(wf, label="Waveform + noise")
p2 = histogram(ps.times, weights=ps.charges, bins=-50:1:250, label="PE")
plot(p1, p2, layout=l)

designmethod = Butterworth(1)
lp_filter = digitalfilter(Lowpass(STD_PMT_CONFIG.lp_cutoff, fs=STD_PMT_CONFIG.sampling_freq), designmethod)

pulse_model_filt = make_filtered_pulse(STD_PMT_CONFIG.pulse_model, STD_PMT_CONFIG.sampling_freq, (-1000.0, 1000.0), lp_filter)

digi_wf = digitize_waveform(
    ps,
    STD_PMT_CONFIG.sampling_freq,
    STD_PMT_CONFIG.adc_freq,
    STD_PMT_CONFIG.noise_amp,
    lp_filter,
)

plot(wf, label="Waveform + Noise", xlabel="Time (ns)", ylabel="Amplitude (a.u.)", right_margin=40Plots.px,
    xlim=(-20, 50))
plot!(digi_wf, label="Digitized Waveform")
#plot!(ps, -20, 50)
sticks!(twinx(), ps.times, ps.charges, legend=false, left_margin=30Plots.px, ylabel="Charge (PE)", ylim=(1, 20), color=:green, xticks=:none, xlim=(-20, 50))


reco_pulses = unfold_waveform(digi_wf, pulse_model_filt, STD_PMT_CONFIG.unf_pulse_res, 0.2, :fnnls)

plot_waveform(wf, digi_wf, reco_pulses, STD_PMT_CONFIG.pulse_model, pulse_model_filt, (0.0, maximum(wf.values) * 1.1))
