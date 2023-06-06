# StrategicGames.jl

A set of functions in pure Julia for analysing strategic generic N-players games using concepts and tools of Game Theory.

While written in Julia, the library is easily accessible directly in Python or R using the [`PyJulia`](https://github.com/JuliaPy/pyjulia) and [`JuliaCall`](https://github.com/Non-Contradiction/JuliaCall) packages respectively ([Python and R examples](https://sylvaticus.github.io/StrategicGames.jl/dev/using_other_languages.html#using_other_languages)).

Check out the companion repository [GameTheoryNotes](https://github.com/sylvaticus/GameTheoryNotes) for introductory notes on the Game Theory approach and the documentation at the links below for further information on the package and how to use it.

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://sylvaticus.github.io/StrategicGames.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://sylvaticus.github.io/StrategicGames.jl/dev)
[![Build status (Github Actions)](https://github.com/sylvaticus/StrategicGames.jl/workflows/CI/badge.svg)](https://github.com/sylvaticus/StrategicGames.jl/actions)
[![codecov.io](http://codecov.io/github/sylvaticus/StrategicGames.jl/coverage.svg?branch=main)](http://codecov.io/github/sylvaticus/StrategicGames.jl?branch=main)


## Basic example

Other examples are available in the [`Examples`](https://sylvaticus.github.io/StrategicGames.jl/dev/#examples) section of the documentation.

```julia
julia> # Prisoner's dilemma (N players are also supported in all functions)
       payoff = [(-1,-1) (-3,  0);
                 ( 0,-3) (-2, -2)];
julia> # From N-dimensional array of tuples to N+1 arrays of scalars    
       payoff_array = expand_dimensions(payoff);
julia> # Find all the dominated strategies for the two players
       dominated_strategies(payoff_array)
2-element Vector{Vector{Int64}}:
 [1]
 [1]
julia> # Compute one Nash Equilibrium of the Game using complementarity formulation
       eq = nash_cp(payoff_array).equilibrium_strategies
2-element Vector{Vector{Float64}}:
 [0.0, 0.9999999887780999]
 [0.0, 0.9999999887780999]
julia> # Compute all isolated Nash equilibria using support enumeration
       eqs = nash_se(payoff_array,max_samples=Inf)
1-element Vector{NamedTuple{(:equilibrium_strategies, :expected_payoffs, :supports), Tuple{Vector{Vector{Float64}}, Vector{Float64}, Vector{Vector{Int64}}}}}:
 (equilibrium_strategies = [[0.0, 0.9999999999999999], [0.0, 0.9999999999999999]], expected_payoffs = [-1.9999999999999678, -1.9999999999999678], supports = [[2], [2]])
julia> # Best response for player 2
       best_response(payoff_array,[[0.5,0.5],[0.5,0.5]],2).optimal_strategy
2-element Vector{Float64}:
 0.0
 1.0
julia> # Expected payoffs given a specific strategy profile
       expected_payoff(payoff_array,[[1,0],[1,0]])
2-element Vector{Int64}:
 -1
 -1
julia> # Is this strategy profile a Nash equilibrium ?
       is_nash(payoff_array,[[1,0],[1,0]]) 
false
```

## Other game-theory libraries & benchmarks

**Julia**
 - **[Nash.jl](https://github.com/KrainskiL/Nash.jl)** Has several functions to generate games in normal form, determine best response and if a strategy profile is a Nash Equilibrium (NE) but it doesn't provide a functionality to retrieve a NE, except for 2 players simmetric games
 - **[GameTheory.jl](https://github.com/QuantEcon/GameTheory.jl)** Inter alia, compute N-players pure strategy NE, 2-players mixed strategy games (`lrsnash`, using exact arithmetics) and N-players mixed strategies NE using a solver of the polynomial equation representation of the complementarity conditions (`hc_solve`). However this "generic" N-player solver is slow, as it doesn't seem to have a dominance check. Further, compilation times are huge.

**Non-Julia**
 - [Nashpy](https://github.com/drvinceknight/Nashpy): two players only
 - [Gambit](http://www.gambit-project.org/): many algorithms require installation of gambit other than pygambit. No decimal/rational payoffs in Python
 - [Sage Math](https://doc.sagemath.org/html/en/reference/game_theory/sage/game_theory/normal_form_game.html): two players only
 - [GtNash](https://forgemia.inra.fr/game-theory-tools-group/gtnash/-/tree/main/)

### Benchmarks
The following benchmarks have been run on a Intel Core 5 laptop on StrategicGames v0.0.4.
See [benchmarks/benchmarks_other_libraries.jl](/benchmarks/benchmarks_other_libraries.jl) for details.

| benchmark_name | library | method | time (ms) | memory (MB) | alloc | n eqs | notes |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| small_3x2 | StrategicGames | nash_se | 3.43 | 0.55 | 17694 | 3 |  |
| small_3x2 | GameTheory | hc_solve | 15.85 | 2.73 | 38255 | 3 |  |
| small_3x2 | nashpy | vertex_enumeration | 21.78 |  |  | 3 |  |
| small_3x2 | nashpy | lemke_howson_enumeration | 1.66 |  |  | 5 | repeated results |
| small_3x2 | nashpy | support_enumeration | 2.68 |  |  | 3 |  |
| small_3x2 | pygambit | lcp_solve | 0.58 |  |  | 3 |  |
| small_3x2 | pygambit | ExternalEnumPolySolver | 2.84 |  |  | 3 |  |
| rand_6x7 | StrategicGames | nash_se | 223.78 | 346.44 | 7113996 | 1 |  |
| rand_6x7 | GameTheory | hc_solve | 24319.04 | 219.29 | 6639449 | 1 |  |
| rand_6x7 | nashpy | vertex_enumeration | 483.39 |  |  | 1 |  |
| rand_6x7 | nashpy | lemke_howson_enumeration | 10.20 |  |  | 13 | repeated results |
| rand_6x7 | nashpy | support_enumeration | 1002.63 |  |  | 0 |  |
| rand_6x7 | pygambit | lcp_solve | 8.61 |  |  | 1 |  |
| rand_6x7 | pygambit | ExternalEnumPolySolver | 466356.13 |  |  | 1 |  |
| rand_dec_6x5 | StrategicGames | nash_se | 61.64 | 61.19 | 1383871 | 3 |  |
| rand_dec_6x5 | GameTheory | hc_solve | 2891.12 | 12.38 | 129350 | 3 |  |
| rand_dec_6x5 | nashpy | vertex_enumeration | 115.39 |  |  | 3 |  |
| rand_dec_6x5 | nashpy | lemke_howson_enumeration | 4.75 |  |  | 11 | repeated results |
| rand_dec_6x5 | nashpy | support_enumeration | 247.32 |  |  | 3 |  |
| rand_4x4x2 | StrategicGames | nash_se | 2990.28 | 68.61 | 1243570 | 7 | 1 eq repeated |
| rand_4x4x2 | GameTheory | hc_solve | 5085.48 | 14.03 | 163760 | 4 | 2 eq missing |
| rand_4x4x2 | pygambit | ExternalEnumPolySolver | 924.56 |  |  | 5 | 1 eq missed |
| rand_6x7_1st_eq | StrategicGames | nash_se | 7.90 | 3.70 | 81730 | 1 |  |
| rand_6x7_1st_eq | GameTheory | hc_solve | 20529.34 | 193.50 | 5212846 | 1 |  |
| rand_6x7_1st_eq | nashpy | vertex_enumeration | 221.68 |  |  | 1 |  |
| rand_6x7_1st_eq | nashpy | lemke_howson_enumeration | 0.83 |  |  | 1 |  |
| rand_6x7_1st_eq | nashpy | support_enumeration | (0.00) |  |  | 0 | no eq reported |

## Acknowledgements

The development of this package at the _Bureau d'Economie Théorique et Appliquée_ (BETA, Nancy) was supported by the French National Research Agency through the [Laboratory of Excellence ARBRE](http://mycor.nancy.inra.fr/ARBRE/), a part of the “Investissements d'Avenir” Program (ANR 11 – LABX-0002-01).

[![BLogos](assets/logos_betaumr.png)](hhttp://www.beta-umr7522.fr/)
