# PMTSimulation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://github.com/PLEnuM-group/PMTSimulation.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://github.com/PLEnuM-group/PMTSimulation.jl/dev/)
[![Build Status](https://github.com/PLEnuM-group/PMTSimulation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PLEnuM-group/PMTSimulation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/PLEnuM-group/PMTSimulation.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PLEnuM-group/PMTSimulation.jl)

Simulation suite for photo-multiplier tubes (PMT). Includes functions for simulating individual PMT pulses, filtering and unfolding of complex PMT waveforms.

## Installation
```{julia}
using Pkg
Pkg.add("https://github.com/PLEnuM-group/PMTSimulation.jl")
```