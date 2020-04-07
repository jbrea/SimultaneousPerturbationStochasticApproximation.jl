using Documenter, SimultaneousPerturbationStochasticApproximation

makedocs(
    modules = [SimultaneousPerturbationStochasticApproximation],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Johanni Brea",
    sitename = "SimultaneousPerturbationStochasticApproximation.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/jbrea/SimultaneousPerturbationStochasticApproximation.jl.git",
    push_preview = true
)
