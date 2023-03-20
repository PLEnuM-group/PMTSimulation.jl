using DSP
using PhysicsTools
using Distributions

export STD_PMT_CONFIG, PMTConfig


struct PMTConfig{T<:Real,S ,P ,U , V}
    spe_template::S
    pulse_model::P
    pulse_model_filt::U
    noise_amp::T
    sampling_freq::T # Ghz
    unf_pulse_res::T # ns
    adc_freq::T # Ghz
    adc_bits::Int64
    adc_dyn_range::Tuple{T,T}
    tt_dist::V
    lp_filter::ZeroPoleGain{:z,ComplexF64,ComplexF64,Float64}

end
