# PMTSimulation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://github.com/PLEnuM-group/PMTSimulation.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://github.com/PLEnuM-group/PMTSimulation.jl/dev/)
[![Build Status](https://github.com/PLEnuM-group/PMTSimulation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PLEnuM-group/PMTSimulation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/PLEnuM-group/PMTSimulation.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/PLEnuM-group/PMTSimulation.jl)

Simulation suite for photo-multiplier tubes (PMT). Includes functions for simulating individual PMT pulses, filtering and unfolding of complex PMT waveforms.

## Installation
This package is registered in the PLEnuM julia package [registry](https://github.com/PLEnuM-group/julia-registry). In order to use this registry, first install the (LocalRegistry.jl)(https://github.com/GunnarFarneback/LocalRegistry.jl) package and then add the PLEnuM registry:
```{julia}
using Pkg
pkg"add LocalRegistry"
pkg"registry add https://github.com/PLEnuM-group/julia-registry"
```

Finally, install the package:
```{julia}
using Pkg
pkg"add PMTSimulation"
```