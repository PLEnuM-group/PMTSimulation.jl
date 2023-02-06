using PMTSimulation
using Documenter

DocMeta.setdocmeta!(PMTSimulation, :DocTestSetup, :(using PMTSimulation); recursive=true)

makedocs(;
    modules=[PMTSimulation],
    authors="Christian Haack",
    repo="https://github.com/chrhck/PMTSimulation.jl/blob/{commit}{path}#{line}",
    sitename="PMTSimulation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chrhck.github.io/PMTSimulation.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chrhck/PMTSimulation.jl",
    devbranch="main",
)
