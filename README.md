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
       payoff_array = expand_dimensions(U);
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


## Acknowledgements

The development of this package at the _Bureau d'Economie Théorique et Appliquée_ (BETA, Nancy) was supported by the French National Research Agency through the [Laboratory of Excellence ARBRE](http://mycor.nancy.inra.fr/ARBRE/), a part of the “Investissements d'Avenir” Program (ANR 11 – LABX-0002-01).

[![BLogos](assets/logos_betaumr.png)](hhttp://www.beta-umr7522.fr/)
