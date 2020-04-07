var documenterSearchIndex = {"docs":
[{"location":"#SimultaneousPerturbationStochasticApproximation-1","page":"SimultaneousPerturbationStochasticApproximation","title":"SimultaneousPerturbationStochasticApproximation","text":"","category":"section"},{"location":"#","page":"SimultaneousPerturbationStochasticApproximation","title":"SimultaneousPerturbationStochasticApproximation","text":"using SimultaneousPerturbationStochasticApproximation\nf(x) = sum(abs2, x)\nresult = minimize(f, lower = fill(-10, 10), upper = fill(10, 10), maxfevals = 10^5)","category":"page"},{"location":"#","page":"SimultaneousPerturbationStochasticApproximation","title":"SimultaneousPerturbationStochasticApproximation","text":"Modules = [SimultaneousPerturbationStochasticApproximation]\nPages = [\"SimultaneousPerturbationStochasticApproximation.jl\"]","category":"page"},{"location":"#SimultaneousPerturbationStochasticApproximation.ConvergenceWelchTest-Tuple{}","page":"SimultaneousPerturbationStochasticApproximation","title":"SimultaneousPerturbationStochasticApproximation.ConvergenceWelchTest","text":"ConvergenceWelchTest(; N = 1000, K = 25, T = Float64, N_positivetests = 3)\n\n\n\n\n\n","category":"method"},{"location":"#SimultaneousPerturbationStochasticApproximation.LearningRateUpdate-Tuple{Any,Any}","page":"SimultaneousPerturbationStochasticApproximation","title":"SimultaneousPerturbationStochasticApproximation.LearningRateUpdate","text":"LearningRateUpdate(lower, upper; pi_a = .7, N = 1000, factor = 1.5)\n\n\n\n\n\n","category":"method"},{"location":"#SimultaneousPerturbationStochasticApproximation.NoInitializer","page":"SimultaneousPerturbationStochasticApproximation","title":"SimultaneousPerturbationStochasticApproximation.NoInitializer","text":"NoInitializer()\n\n\n\n\n\n","category":"type"},{"location":"#SimultaneousPerturbationStochasticApproximation.RandomInitializer","page":"SimultaneousPerturbationStochasticApproximation","title":"SimultaneousPerturbationStochasticApproximation.RandomInitializer","text":"RandomInitializer()\n\n\n\n\n\n","category":"type"},{"location":"#SimultaneousPerturbationStochasticApproximation.SPSA-Tuple{}","page":"SimultaneousPerturbationStochasticApproximation","title":"SimultaneousPerturbationStochasticApproximation.SPSA","text":"SPSA(; lower, upper,\n       heuristic = SimpleHeuristic(elementwise = true,\n                                   update = LearningRateUpdate(lower, upper)),\n       initializer = SobolInitializer(restarts = 1, N = 100),\n       convergencetest = ConvergenceWelchTest(),\n       pi_max = .1,\n       init = lower)\n\n\n\n\n\n","category":"method"},{"location":"#SimultaneousPerturbationStochasticApproximation.SimpleHeuristic-Tuple{}","page":"SimultaneousPerturbationStochasticApproximation","title":"SimultaneousPerturbationStochasticApproximation.SimpleHeuristic","text":"SimpleHeuristic(; elementwise = false, A = 500, c = 0., c_min = 1e-3,\n                  a_max = 1e4, α = .602, γ = .101, ninit = 100,\n                  update = nothing)\n\nupdate could also be LearningRateUpdate.\n\n\n\n\n\n","category":"method"},{"location":"#SimultaneousPerturbationStochasticApproximation.minimize!-Tuple{SPSA,Any}","page":"SimultaneousPerturbationStochasticApproximation","title":"SimultaneousPerturbationStochasticApproximation.minimize!","text":"minimize!(spsa::SPSA, f; kwargs...) = minimize!(Random.GLOBAL_RNG, spsa, f; kwargs...)\n\n\n\n\n\n","category":"method"},{"location":"#SimultaneousPerturbationStochasticApproximation.minimize!-Union{Tuple{T}, Tuple{Random.AbstractRNG,SPSA{#s104,#s105,#s106,T,Tl,Tu} where Tu where Tl where #s106 where #s105 where #s104,Any}} where T","page":"SimultaneousPerturbationStochasticApproximation","title":"SimultaneousPerturbationStochasticApproximation.minimize!","text":"minimize!(rng::Random.AbstractRNG,\n               spsa::SPSA{<:Any,<:Any,<:Any,T},\n               f;\n               callback = () -> nothing,\n               maxfevals,\n               gamma_fhat = .9, # just for tracking the learning progress\n               verbose = true) where T\n\n\n\n\n\n","category":"method"},{"location":"#SimultaneousPerturbationStochasticApproximation.minimize-Tuple{Any}","page":"SimultaneousPerturbationStochasticApproximation","title":"SimultaneousPerturbationStochasticApproximation.minimize","text":"minimize(f; callback = () -> nothing, maxfevals, verbose = true, kwargs...)\n\nkwargs are passed to SPSA.\n\n\n\n\n\n","category":"method"}]
}
