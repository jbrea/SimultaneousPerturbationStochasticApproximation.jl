# SimultaneousPerturbationStochasticApproximation

```@example
using SimultaneousPerturbationStochasticApproximation
f(x) = sum(abs2, x)
result = minimize(f, lower = fill(-10, 10), upper = fill(10, 10), maxfevals = 10^5)
```

```@autodocs
Modules = [SimultaneousPerturbationStochasticApproximation]
Pages = ["SimultaneousPerturbationStochasticApproximation.jl"]
```

