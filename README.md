# SimultaneousPerturbationStochasticApproximation.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/jbrea/SimultaneousPerturbationStochasticApproximation.jl.svg?branch=master)](https://travis-ci.com/jbrea/SimultaneousPerturbationStochasticApproximation.jl)
[![codecov.io](http://codecov.io/github/jbrea/SimultaneousPerturbationStochasticApproximation.jl/coverage.svg?branch=master)](http://codecov.io/github/jbrea/SimultaneousPerturbationStochasticApproximation.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://jbrea.github.io/SimultaneousPerturbationStochasticApproximation.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://jbrea.github.io/SimultaneousPerturbationStochasticApproximation.jl/dev)

A Julia implementation of [SPSA (Simultaneous Perturbation Stochastic Approximation)](https://www.jhuapl.edu/SPSA/).

Install it in a julia repl with
```
]add https://github.com/jbrea/SimultaneousPerturbationStochasticApproximation.jl
```

```@example
using SimultaneousPerturbationStochasticApproximation
f(x) = sum(abs2, x)
result = minimize(f, lower = fill(-10, 10), upper = fill(10, 10), maxfevals = 10^5)
```

This package is not (well) tested and has minimal documentation. I found SPSA generally
to be inferior to the standard methods of [BlackBoxOptim](https://github.com/robertfeldt/BlackBoxOptim.jl)
or [PyCMA](https://github.com/jbrea/PyCMA.jl).

