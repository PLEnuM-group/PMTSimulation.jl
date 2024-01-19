module PMTSimulation

using Reexport
using PhysicsTools
using DSP
using Distributions

include("pmt_config.jl")
include("spe_templates.jl")
include("pulse_templates.jl")
include("waveforms.jl")
include("pipeline.jl")


export PMTConfig
export STD_PMT_CONFIG

@reexport using .PulseTemplates
@reexport using .Waveforms
@reexport using .SPETemplates
@reexport using .PMTPipeline


function PMTConfig(;
    st::SPEDistribution,
    pm::PulseTemplate,
    snr_db=nothing,
    noise_sigma=nothing,
    sampling_freq::Real,
    unf_pulse_res::Real,
    adc_freq::Real,
    adc_bits::Int,
    adc_dyn_range::Tuple,
    lp_cutoff::Real,
    tt_mean::Real,
    tt_fwhm::Real)

    if isnothing(snr_db) && isnothing(noise_sigma)
        error("Have to set at least `snr_db` or `noise_sigma`")
    end

    if isnothing(noise_sigma)
        mode = get_template_mode(pm)
        eval_at_mode = evaluate_pulse_template(pm, 0, mode)
        noise_sigma = eval_at_mode / 10^(snr_db / 20)
    end


    designmethod = Butterworth(3)
    lp_filter = digitalfilter(Lowpass(lp_cutoff, fs=sampling_freq), designmethod)
    filtered_pulse = make_filtered_pulse(pm, sampling_freq, adc_freq, (-100.0, 100.0), lp_filter)

    tt_theta = calc_gamma_shape_mean_fwhm(tt_mean, tt_fwhm)
    tt_alpha = tt_mean / tt_theta
    tt_dist = Gamma(tt_alpha, tt_theta)

    PMTConfig(st, pm, filtered_pulse, noise_sigma, sampling_freq, unf_pulse_res, adc_freq, adc_bits, adc_dyn_range, tt_dist, lp_filter)
end

adc_range = (0.0, 1000.0)
adc_bits = 12
STD_PMT_CONFIG = PMTConfig(
    st=ExponTruncNormalSPE(expon_rate=1.0, norm_sigma=0.3, norm_mu=1.0, trunc_low=0.0, peak_to_valley=3.1),
    pm=PDFPulseTemplate(
        dist=Truncated(Gumbel(0, gumbel_width_from_fwhm(6)) + 10., 0, 50),
        amplitude=7.0 # mV
    ),
    #snr_db=22.92,
    noise_sigma=find_noise_scale(0.6, adc_range, adc_bits),
    sampling_freq=2.0,
    unf_pulse_res=0.1,
    adc_freq=0.208,
    adc_bits=adc_bits,
    adc_dyn_range=adc_range, #mV
    lp_cutoff=0.125,
    tt_mean=50., # TT mean
    tt_fwhm=1.5 # TT FWHM
)


end
