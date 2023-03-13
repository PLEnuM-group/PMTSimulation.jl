module Waveforms

using ..PulseTemplates
using NonNegLeastSquares
using CairoMakie
using DSP
export Waveform
export add_gaussian_white_noise, digitize_waveform, unfold_waveform, plot_waveform
export adc_bins

"""
Waveform struct. Stores time stamps and corresponding values.
"""
struct Waveform{T<:Real,U<:AbstractVector{T},V<:AbstractVector{T}}
    timestamps::U
    values::V

    function Waveform(timestamps::U, values::V) where {T<:Real,U<:AbstractVector{T},V<:AbstractVector{T}}
        if length(timestamps) != length(values)
            error("Timestamps and values have to have same length.")
        end
        new{T,U,V}(timestamps, values)
    end
end

Base.length(wf::Waveform) = length(wf.timestamps)

#=
@recipe function f(wf::T) where {T<:Waveform}
    wf.timestamps, wf.values
end
=#

"""
    add_gaussian_white_noise(values, scale)
Add gaussian white noise with scale `scale` to `values`.
"""
function add_gaussian_white_noise(values, scale)
    values .+ randn(size(values)) * scale
end

"""
    Waveform(
        ps::PulseSeries,
        sampling_freq::Real,
        noise_amp::Real;
        time_range=(-50.0, 150.0))

Create a waveform from a PulseSeries.

# Arguments
- `ps``: Input PulseSeries
- `sampling_freq`: Sampling frequency (GHz) for evaluating the PulseSeries
- `noise_amp`: Amplitude (in mV) of gaussian noise to add to waveform
- `time_range`: Time range in which pulse series is evaluates
"""
function Waveform(
    ps::PulseSeries,
    sampling_freq::Real,
    noise_amp::Real;
    time_range=(-50.0, 150.0))
    if length(ps) == 0
        return Waveform(empty(ps.times), empty(ps.times))
    end

    dt = 1 / sampling_freq # ns
    timestamps = range(time_range[1], time_range[2], step=dt)

    waveform_values = evaluate_pulse_series(timestamps, ps)
    if noise_amp > 0
        waveform_values_noise = add_gaussian_white_noise(waveform_values, noise_amp)
    else
        waveform_values_noise = waveform_values
    end

    Waveform(timestamps, waveform_values_noise)
end

"""
    adc_bins(yrange, bits)
Calculate ADC bins when using `bits` bits in `yrange`.
"""
function adc_bins(yrange, bits)
    n_bins = 2^bits
    return adc_bins = LinRange(yrange[1], yrange[2], n_bins)
end


"""
    digitize_waveform(
        waveform::Waveform,
        sampling_frequency::Real,
        digitizer_frequency::Real,
        filter;
        yrange=(0, 100),
        yres_bits=12
        )

Digitize `waveform` which has been sampled with `sampling_frequency` by applying `filter`
and resampling with `digitizer_frequency`.

# Arguments
- `waveform`: Raw waveform
- `sampling_freq`: Sampling frequency (GHz) with which `waveform` has been sampled
- `digitizer_frequency`: Frequency (GHz) to which waveform should be downsampled
- `filter`: Filter applied before resampling
- `yrange`: Min and max levels of the digitized waveform
- `yres_bits`: Number of levels (2^yres_bits) `yrange` that are placed in yrange
"""
function digitize_waveform(
    waveform::Waveform,
    sampling_frequency::Real,
    digitizer_frequency::Real,
    filter;
    yrange=(0, 100),
    yres_bits=12
)

    if length(waveform) == 0
        return Waveform(empty(waveform.timestamps), empty(waveform.values))
    end

    min_time, max_time = extrema(waveform.timestamps)
    waveform_filtered = filt(filter, waveform.values)

    resampling_rate = digitizer_frequency / sampling_frequency
    new_interval = range(min_time, max_time, step=1 / digitizer_frequency)
    waveform_resampled = resample(waveform_filtered, resampling_rate)

    # Discretize
    # TODO check if there are 2^yres_bits or 2^yres_bits -1 bins
    n_bins = 2^yres_bits
    adc_bins = LinRange(yrange[1], yrange[2], n_bins)
    bin_ix = searchsortedfirst.(Ref(adc_bins), waveform_resampled)
    bin_ix[bin_ix.>n_bins] .= n_bins
    waveform_discretized = adc_bins[bin_ix]

    return Waveform(collect(new_interval), waveform_discretized)
end

function digitize_waveform(
    ps::PulseSeries,
    sampling_frequency::Real,
    digitizer_frequency::Real,
    noise_amp::Real,
    filter; time_range=(-50.0, 150.0),
    yrange=(0, 100),
    yres_bits=12
)
    wf = make_waveform(ps, sampling_frequency, noise_amp; time_range=time_range)
    digitize_waveform(wf, sampling_frequency, digitizer_frequency, filter; yrange=yrange, yres_bits=yres_bits)
end


function make_nnls_matrix(
    pulse_times::V,
    pulse_shape::PulseTemplate,
    timestamps::V) where {T<:Real,V<:AbstractVector{T}}

    nnls_matrix = zeros(T, size(timestamps, 1), size(pulse_times, 1))

    # dt = 1/sampling_frequency # ns
    # timestamps_hires = range(eval_range[1], eval_range[2], step=dt)


    for i in eachindex(pulse_times)
        nnls_matrix[:, i] = evaluate_pulse_template(
            pulse_shape, pulse_times[i], timestamps)
    end

    nnls_matrix

end


function apply_nnls(
    pulse_times::V,
    pulse_shape::PulseTemplate,
    digi_wf::Waveform{T,V};
    alg::Symbol=:nnls) where {T<:Real,V<:AbstractVector{T}}


    # append zeros at waveform edges
    wf_delta = first(diff(digi_wf.timestamps))
    wf_tmin, wf_tmax = extrema(digi_wf.timestamps)

    wf_timestamps = collect((wf_tmin-5*wf_delta):wf_delta:(wf_tmax+5*wf_delta))
    wf_values = zeros(length(wf_timestamps))
    wf_values[6:6+length(digi_wf.timestamps)-1] = digi_wf.values

    #wf_values =
    matrix = make_nnls_matrix(pulse_times, pulse_shape, wf_timestamps)
    #charges = nonneg_lsq(matrix, digi_wf.values; alg=:nnls)[:, 1]


    if alg == :nnls_NNLS
        charges = nnls(matrix, digi_wf.values)
    else
        charges = nonneg_lsq(matrix, wf_values, alg=alg)[:, 1]
    end
    charges
end

"""
    unfold_waveform(
        digi_wf::Waveform,
        pulse_model::PulseTemplate,
        pulse_resolution::Real,
        min_charge::Real,
        alg::Symbol=:nnls;
        min_boundary_dist=3
    )

Unfold waveform into pulses using NNLS.

# Arguments
- `digi_wf`: Digitized waveform
- `pulse_model`: Pulse model to fit to waveform
- `pulse_resolution`: Difference of time steps at where pulse hypotheses are placed
- `min_charge`: Minimum charge after unfolding considered for true pulses
- `alg`: NNLS algorithm (default :nnls)
- `min_boundary_dist`: Minimum distance from waveform edge (in ns) to count as pulse

# Returns
PulseSeries with unfolded pulses
"""
function unfold_waveform(
    digi_wf::Waveform,
    pulse_model::PulseTemplate,
    pulse_resolution::Real,
    min_charge::Real,
    alg::Symbol=:nnls;
    min_boundary_dist=3
)
    #offset = get_template_mode(pulse_model)

    if length(digi_wf) == 0
        return PulseSeries(empty(digi_wf.timestamps), empty(digi_wf.values), pulse_model)
    end
    min_time, max_time = extrema(digi_wf.timestamps)
    min_time -= 20
    max_time += 20
    pulse_times = collect(range(min_time, max_time, step=pulse_resolution))
    pulse_charges = apply_nnls(pulse_times, pulse_model, digi_wf, alg=alg)

    nonzero = pulse_charges .> min_charge


    non_edge = (
        pulse_times .> (min_time + min_boundary_dist) .&&
        pulse_times .< (max_time - min_boundary_dist)
    )



    mask = nonzero .&& non_edge

    return PulseSeries(pulse_times[mask], pulse_charges[mask], pulse_model)

end

function plot_waveform(
    orig_waveform::Waveform,
    digitized_waveform::Waveform,
    reco_pulses::PulseSeries,
    pulse_template::PulseTemplate,
    xlim::Tuple{<:Real,<:Real},
    ylim::Tuple{<:Real,<:Real}
)
    pulses_orig_temp = PulseSeries(reco_pulses.times, reco_pulses.charges, pulse_template)

    p = plot(
        orig_waveform,
        label="Waveform + Noise",
        xlabel="Time (ns)",
        ylabel="Amplitude (a.u.)",
        right_margin=45Plots.px,
        legend=:topleft,
        lw=2,
        palette=:seaborn_colorblind,
        dpi=150,
        ylim=ylim,
        xlim=xlim,
        #xlim=(-100, 100)
    )
    p = plot!(digitized_waveform, label="Digitized Waveform", lw=2, xlim=xlim)
    p = plot!(reco_pulses, label="Reconstructed Waveform", lw=2, xlim=xlim)

    p = plot!(pulses_orig_temp, label="Unfolded Waveform", lw=2, xlim=xlim)

    p = sticks!(twinx(), reco_pulses.times, reco_pulses.charges, legend=false, left_margin=30Plots.px, ylabel="Charge (PE)", xlim=xlim, ylim=(0, 50), color=:red, xticks=:none)

    p
end

end
