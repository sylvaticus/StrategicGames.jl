# BetaGameTheory.jl
A set of functions in pure Julia for Game Theory

In particular, `nash_lcp(payoff_tensor;init,verbosity)` implements the LCP algorithm to find a Nash Equilibrium for n-players simultaneous games when mixed strategies are allowed.

The library is easily usable in R or Python with [`JuliaCall`](https://github.com/Non-Contradiction/JuliaCall) and [`PyJulia`](https://github.com/JuliaPy/pyjulia) packages respectively.


[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://sylvaticus.github.io/BetaGameTheory.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://sylvaticus.github.io/BetaGameTheory.jl/dev)
[![Build status (Github Actions)](https://github.com/sylvaticus/BetaGameTheory.jl/workflows/CI/badge.svg)](https://github.com/sylvaticus/BetaGameTheory.jl/actions)
[![codecov.io](http://codecov.io/github/sylvaticus/BetaGameTheory.jl/coverage.svg?branch=main)](http://codecov.io/github/sylvaticus/BetaGameTheory.jl?branch=main)



```julia
# Example from https://www.youtube.com/watch?v=bKrwQKUT0v8
julia> using BetaGameTheory

julia> U = [(0,0,0) ; (3,3,3) ;; (3,3,3)   ; (2,2,4) ;;;
            (3,3,3)  ; (2,4,2)  ;; (4,2,2) ; (1,1,1)  ;;;]
2×2×2 Array{Tuple{Int64, Int64, Int64}, 3}:
[:, :, 1] =
 (0, 0, 0)  (3, 3, 3)
 (3, 3, 3)  (2, 2, 4)

[:, :, 2] =
 (3, 3, 3)  (4, 2, 2)
 (2, 4, 2)  (1, 1, 1)

julia> payoff_array = expand_dimensions(U) # from n-dimensional array of tuples to n+1 arrays of scalars
2×2×2×3 Array{Int64, 4}:
[:, :, 1, 1] =
 0  3
 3  2

[:, :, 2, 1] =
 3  4
 2  1

[:, :, 1, 2] =
 0  3
 3  2

[:, :, 2, 2] =
 3  2
 4  1

[:, :, 1, 3] =
 0  3
 3  4

[:, :, 2, 3] =
 3  2
 2  1

julia> eq  = nash_lcp(payoff_array)
(status = MathOptInterface.LOCALLY_SOLVED, equilibrium_strategies = [[0.5811388300841898, 0.4188611699158103], [0.5811388300841898, 0.4188611699158103], [0.5811388300841898, 0.41886116991581035]], expected_payoffs = [2.16227766016838, 2.16227766016838, 2.16227766016838])

julia> eq_strategies = eq.equilibrium_strategies
3-element Vector{Vector{Float64}}:
 [0.5811388300841898, 0.4188611699158103]
 [0.5811388300841898, 0.4188611699158103]
 [0.5811388300841898, 0.41886116991581035]

julia> p = -1 + sqrt(10)/2
0.5811388300841898

julia> @test eq_strategies ≈ [[p,1-p],[p,1-p],[p,1-p]]
Test Passed

julia> expected_payoff(payoff_array,eq_strategies)
3-element Vector{Float64}:
 2.1622776601683795
 2.1622776601683795
 2.1622776601683795
 ```