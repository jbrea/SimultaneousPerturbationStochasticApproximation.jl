using SimultaneousPerturbationStochasticApproximation
using Test, Random, Statistics, LinearAlgebra

using SimultaneousPerturbationStochasticApproximation: estimate_grad, init!, LearningRateUpdate, update!, ConvergenceWelchTest, converged!

@testset "estimate gradient" begin
f(x) = sum(abs2, x) + .2*randn()
@test mean(estimate_grad(Random.GLOBAL_RNG, [2, 1], .01, f)[1] for _ in 1:10^5) ≈ [4, 2] atol = .5
end # estimate gradient tests

@testset "heuristic" begin
h = SimpleHeuristic()
init!(Random.GLOBAL_RNG, h, f, [2, 1], 10^4, [.2, .2])
@test h.c ≈ .2 atol = .15

h = SimpleHeuristic()
init!(Random.GLOBAL_RNG, h, x -> sum(abs2, x), [2, 1], 10^4, [.2, .2])
@test h.c == h.c_min

h = SimpleHeuristic(c = .1)
init!(Random.GLOBAL_RNG, h, f, [2, 1], 10^4, [.2, .2])
@test h.c == .1

h = SimpleHeuristic(elementwise = true)
init!(Random.GLOBAL_RNG, h, f, [2, 1], 10^4, [.2, .2])
@test length(h.a) == 2

# Update heuristics
f(x) = dot([100, 1], x) + .1 * randn()
lb = fill(-10, 2)
ub = fill(10, 2)
h = SimpleHeuristic(elementwise = true,
                    update = LearningRateUpdate(lb, ub, N = 10))
theta = [2., 1]
init!(Random.GLOBAL_RNG, h, f, theta, 10^4, [.2, .2])
olda = copy(h.a)
for _ in 1:10
    theta .*= .8 # trend
    update!(h, theta)
end
@test (&)((h.a .> olda)...)

Random.seed!(2);
olda = copy(h.a)
for _ in 1:10
    theta  = 100*randn(2) # span
    update!(h, theta)
end
@test (&)((h.a .< olda)...)
end # heuristic tests

@testset "convergence" begin
f(x) = sum(abs2, x) + .1*randn()
Random.seed!(2);
c = ConvergenceWelchTest(N = 1, K = 500)
global ret = false
for _ in 1:10
    global ret = converged!(c, f, [0, 0]) # converged
end
@test ret == true

c = ConvergenceWelchTest(N = 10)
global ret = false
theta = [5., 1]
for _ in 1:41
    theta .*= .98 # not converged
    global ret = converged!(c, f, theta)
end
@test ret == false

spsa = SPSA(lower = fill(-10, 2), upper = fill(10, 2),
            convergencetest = ConvergenceWelchTest())
res = minimize!(spsa, x -> sum(abs2, x), maxfevals = 10^8, verbose = false)
@test res[1].evals < 10^8

spsa = SPSA(lower = fill(-10, 2), upper = fill(10, 2),
            convergencetest = ConvergenceWelchTest())
res = minimize!(spsa, f, maxfevals = 10^8, verbose = false)

end # convergence

@testset "minimize" begin
spsa = SPSA(lower = fill(-10, 2), upper = fill(10, 2))
res = minimize!(spsa, x -> sum(abs2, x), maxfevals = 10^3)
@test res[1].x ≈ zeros(2) atol = 1e-12

res = minimize(x -> sum(abs2, x), lower = fill(-10, 2), upper = fill(10, 2),
               maxfevals = 10^3)
@test res[1].x ≈ zeros(2) atol = 1e-12
end
