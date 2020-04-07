module SimultaneousPerturbationStochasticApproximation
using Sobol, Random, Statistics, Dates, Printf, HypothesisTests

export SimpleHeuristic, SPSA, LearningRateUpdate, NoInitializer, RandomInitializer,
SobolInitializer, minimize!, minimize, ConvergenceWelchTest

mutable struct SimpleHeuristic{Ta,U,T}
    a::Ta
    A::T
    c::T
    c_min::T
    a_max::T
    α::T
    γ::T
    update::U
    ninit::Int
end
"""
    SimpleHeuristic(; elementwise = false, A = 500, c = 0., c_min = 1e-3,
                      a_max = 1e4, α = .602, γ = .101, ninit = 100,
                      update = nothing)

`update` could also be [`LearningRateUpdate`](@ref).
"""
function SimpleHeuristic(; elementwise = false, A = 500, c = 0., c_min = 1e-3,
                           a_max = 1e4, α = .602, γ = .101, ninit = 100,
                           update = nothing)
    a = elementwise ? zeros(1) : 0.
    SimpleHeuristic(a, float(A), c, c_min, a_max, α, γ, update, ninit)
end
function init!(rng, s::SimpleHeuristic{Ta}, f, θ, N, maxstepsize) where Ta
    K = floor(Int, s.ninit/3)
    nfevals = 0
    if s.c == 0
        y = [f(θ) for _ in 1:K]
        s.c = max(s.c_min, std(y))
        nfevals += K
    end
    if sum(s.a) == 0
        grad = [estimate_grad(rng, θ, s.c, f)[1] for _ in 1:K]
        nfevals += 2K
        a = min.(s.a_max, abs.(maxstepsize * (s.A + 1)^s.α ./ mean(grad)))
        s.a = Ta <: Number ? mean(a) : a
    end
    s, nfevals
end
update!(h::SimpleHeuristic, θ) = h
mutable struct LearningRateUpdate
    i::Int
    N::Int
    factor::Float64
    range::Vector{Float64}
    θs::Vector{Vector{Float64}}
end
"""
    LearningRateUpdate(lower, upper; pi_a = .7, N = 1000, factor = 1.5)
"""
function LearningRateUpdate(lower, upper; pi_a = .7, N = 1000, factor = 1.5)
    LearningRateUpdate(0, N, factor, pi_a * (upper .- lower), [])
end
function update!(h::SimpleHeuristic{<:AbstractVector,LearningRateUpdate}, θ)
    u = h.update
    if u.i % u.N == u.N - 1
        @views differences = hcat((u.θs[2:end] .- u.θs[1:end-1])...)
        for i in eachindex(h.a)
            θs = [x[i] for x in u.θs]
            if maximum(θs) - minimum(θs) > u.range[i]
                h.a[i] /= u.factor
            elseif pvalue(OneSampleTTest(@view(differences[i, :]))) < .05
                h.a[i] *= u.factor
            end
        end
        empty!(u.θs)
    end
    u.i += 1
    push!(u.θs, copy(θ))
    h
end
struct SPSA{H,I,C,T,Tl,Tu}
    heuristic::H
    initializer::I
    convergencetest::C
    lower::Tl
    upper::Tu
    maxstepsize::Vector{T}
    theta::Vector{T}
end
converged!(::Any, f, θ) = false
mutable struct ConvergenceWelchTest{T}
    i::Int
    N::Int
    K::Int
    m_old::T
    s_old::T
    N_positivetests::Int
    previoustests::Vector{Bool}
end
"""
    ConvergenceWelchTest(; N = 1000, K = 25, T = Float64, N_positivetests = 3)
"""
function ConvergenceWelchTest(; N = 1000, K = 25, T = Float64, N_positivetests = 3)
    ConvergenceWelchTest(0, N, K, zero(T), one(T), N_positivetests, Bool[])
end
function converged!(c::ConvergenceWelchTest, f, θ)
    if c.i % c.N == 0
        f_new = [f(θ) for _ in 1:c.K]
        m_new = mean(f_new)
        s_new = var(f_new)/c.K + eps()
        if length(c.previoustests) != 0
            xbar = c.m_old - m_new
            stderr = sqrt(c.s_old + s_new)
            df = stderr^4/((c.s_old^2 + s_new^2)/(c.K-1))
            p = pvalue(UnequalVarianceTTest(c.K, c.K, xbar, df, stderr, xbar/stderr, 0), tail = :right)
            push!(c.previoustests, p > .05)
            if length(c.previoustests) >= c.N_positivetests && (&)(c.previoustests[end-c.N_positivetests+1:end]...)
                return true
            end
        else
            push!(c.previoustests, false)
        end
        c.m_old = m_new
        c.s_old = s_new
    end
    c.i += 1
    false
end
"""
    NoInitializer()
"""
struct NoInitializer end
init!(rng, spsa::SPSA{<:Any, NoInitializer}, f) = [spsa.theta]
Base.length(::NoInitializer) = 1
fevals(::NoInitializer) = 0
"""
    RandomInitializer()
"""
struct RandomInitializer end
function init!(rng, spsa::SPSA{<:Any, RandomInitializer, <:Any, T}, f) where T
    [rand(rng, T, length(spsa.lower)) .* (spsa.upper .- spsa.lower) .+ spsa.lower]
end
Base.length(::RandomInitializer) = 1
fevals(::RandomInitializer) = 0
"""
    SobolInitializer(; restarts, N)

During initialization, the `SobolInitializer` evaluates the objective function
at `N` points and returns the `restarts` best ones as initial conditions for SPSA.
"""
Base.@kwdef struct SobolInitializer
    restarts::Int
    N::Int
end
function init!(rng, spsa::SPSA{<:Any,SobolInitializer, <:Any, T}, f) where T
    s = SobolSeq(spsa.lower, spsa.upper)
    res = []
    θs = Vector{T}[]
    for θ in Iterators.take(s, spsa.initializer.N)
        push!(res, f(θ))
        push!(θs, copy(θ))
    end
    sp = sortperm(res, rev = true)
    θs[sp[1:spsa.initializer.restarts]]
end
fevals(i::SobolInitializer) = i.N
Base.length(i::SobolInitializer) = i.restarts
"""
    SPSA(; lower, upper,
           heuristic = SimpleHeuristic(elementwise = true,
                                       update = LearningRateUpdate(lower, upper)),
           initializer = SobolInitializer(restarts = 1, N = 100),
           convergencetest = ConvergenceWelchTest(),
           pi_max = .1,
           init = lower)
"""
function SPSA(; lower, upper,
                heuristic = SimpleHeuristic(elementwise = true,
                                            update = LearningRateUpdate(lower, upper)),
                initializer = SobolInitializer(restarts = 1, N = 100),
                convergencetest = ConvergenceWelchTest(),
                pi_max = .1,
                init = lower)
    SPSA(heuristic, initializer, convergencetest, lower, upper,
         pi_max * (upper .- lower), float.(init))
end
struct SPSA_Iterator{A}
    spsa::A
    N::Int
end
Base.length(a::SPSA_Iterator) = a.N
function max_iterations(maxfevals, initializer)
    repetitions = length(initializer)
    evals = initevals = fevals(initializer)
    n = 0
    while evals < maxfevals
        n += 1
        evals += 2 * repetitions
    end
    n, initevals
end
function Base.iterate(a::SPSA_Iterator{<:SPSA{<:SimpleHeuristic}}, n = 1)
    n > a.N && return nothing
    x = a.spsa.heuristic
    A = x.A
    (n, x.a/(n + A)^x.α, x.c/n^x.γ), n + 1
end
_clamp!(x, lo::Nothing, hi::Nothing) = x
_clamp!(x, lo::Number, hi::Number) = clamp!(x, lo, hi)
function _clamp!(x, lo::AbstractVector, hi::AbstractVector)
    @inbounds for i in eachindex(x)
        x[i] = clamp(x[i], lo[i], hi[i])
    end
    x
end

function estimate_grad(rng, θ, c, f)
    d = length(θ)
    δ = 2 * (rand(rng, d) .> .5) .- 1
    θ⁺ = θ .+ c * δ
    θ⁻ = θ .- c * δ
    S⁺ = f(θ⁺)
    S⁻ = f(θ⁻)
    (S⁺ - S⁻)/(2*c) * δ, (S⁺ + S⁻)/2
end
"""
    minimize(f; callback = () -> nothing, maxfevals, verbose = true, kwargs...)

`kwargs` are passed to [`SPSA`](@ref).
"""
function minimize(f; callback = () -> nothing, maxfevals, verbose = true, kwargs...)
    spsa = SPSA(; kwargs...)
    minimize!(spsa, f, callback = callback, maxfevals = maxfevals, verbose = verbose)
end
"""
    minimize!(spsa::SPSA, f; kwargs...) = minimize!(Random.GLOBAL_RNG, spsa, f; kwargs...)
"""
minimize!(spsa::SPSA, f; kwargs...) = minimize!(Random.GLOBAL_RNG, spsa, f; kwargs...)
"""
    minimize!(rng::Random.AbstractRNG,
                   spsa::SPSA{<:Any,<:Any,<:Any,T},
                   f;
                   callback = () -> nothing,
                   maxfevals,
                   gamma_fhat = .9, # just for tracking the learning progress
                   verbose = true) where T
"""
function minimize!(rng::Random.AbstractRNG,
                   spsa::SPSA{<:Any,<:Any,<:Any,T},
                   f;
                   callback = () -> nothing,
                   maxfevals,
                   gamma_fhat = .9, # this is still a hack; should be estimated
                   verbose = true) where T
    start_time = last_time = now()
    verbose && @printf "%6.s %6.s %13.s %10.s %10.s\n" "round" "iter" "elapsed" "fevals" "f"
    results = []
    N, evals = max_iterations(maxfevals, spsa.initializer)
    initial_points = init!(rng, spsa, f)
    _, tune_evals = init!(rng, spsa.heuristic, f, initial_points[1], N, spsa.maxstepsize)
    evals += tune_evals
    restarts = length(initial_points)
    N -= floor(Int, tune_evals/(2*restarts))
    fhat_average = Inf
    for (r, θ) in enumerate(initial_points)
        spsa.theta .= θ # this is a hack to have spsa.theta always point
        θ = spsa.theta  # to the currently updated vector
        for (n, aₙ, cₙ) in SPSA_Iterator(spsa, N)
            grad, fhat = estimate_grad(rng, θ, cₙ, f)
            evals += 2
            fhat_average = isinf(fhat_average) ? fhat : gamma_fhat * fhat_average + (1 - gamma_fhat) * fhat
            @. θ -= clamp(aₙ * grad, -spsa.maxstepsize, spsa.maxstepsize)
            _clamp!(θ, spsa.lower, spsa.upper)
            callback()
            if verbose && (n % div(N, 50) == 0 || now() - last_time > Second(5))
                last_time = now()
                @printf "%4.g/%g %6.g %13.s %10.g %10.f\n" r restarts n round(last_time - start_time, Second) evals fhat_average
            end
            update!(spsa.heuristic, θ)
            converged!(spsa.convergencetest, f, θ) && break # these evals are not counted currently
        end
        push!(results, (x = copy(θ), f = fhat_average, evals = evals))
    end
    sort(results, by = x -> x.f)
end

end # module
