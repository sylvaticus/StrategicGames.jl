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
       dominated_actions(payoff_array)
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
 - **[GameTheory.jl](https://github.com/QuantEcon/GameTheory.jl)** Inter alia, compute N-players pure strategy NE, 2-players mixed strategy games (`lrsnash`, using exact arithmetics) and N-players mixed strategies NE using a solver of the polynomial equation representation of the complementarity conditions (`hc_solve`). However this "generic" N-player solver is slow, as it doesn't seem to have a dominance check.

**Non-Julia**
 - [Nashpy](https://github.com/drvinceknight/Nashpy): two players only
 - [Gambit](http://www.gambit-project.org/): many algorithms require installation of gambit other than pygambit. No decimal/rational payoffs in Python
 - [Sage Math](https://doc.sagemath.org/html/en/reference/game_theory/sage/game_theory/normal_form_game.html): two players only
 - [GtNash](https://forgemia.inra.fr/game-theory-tools-group/gtnash/-/tree/main/)

### Benchmarks
The following benchmarks have been run on an Intel Core Ultra 7 165H laptop (22 threads) with Julia 1.13.1, on StrategicGames v0.0.8, GameTheory.jl v0.8.0, Nashpy v0.0.43 and pygambit v16.7.0.
See [benchmarks/benchmarks_other_libraries.jl](/benchmarks/benchmarks_other_libraries.jl) for details.

Times are medians in nanoseconds and memory is in bytes (both from BenchmarkTools, Julia libraries only for the memory), `neqs` is the number of equilibria returned.
`nash_se` is run with its default `domrem_mixedactions=false`, as none of the other algorithms removes the actions dominated by mixed strategies (setting it to `true` would be much slower on these games, with the same equilibria found).
In the `rand_6x7_1st_eq` benchmark only the first equilibrium is searched for, and `nash_se` is reported both using all the 22 threads and single-threaded.

```text
 Row │ benchmark_name   library         method                    time            memory           alloc    neqs   notes
     │ String           String          String                    Float64         Float64?         Int64?   Int64  String
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Test             rand            rand(2,2)                     20.256          112.0              2      1
   2 │ small_3x2        StrategicGames  nash_cp                        1.892e7          1.2118e6     37141      1
   3 │ small_3x2        StrategicGames  nash_se                        1.84491e6   372208.0          12095      3
   4 │ small_3x2        GameTheory      hc_solve                       9.74171e6        5.15837e6    77294      3
   5 │ small_3x2        GameTheory      support_enumeration        13415.0           5120.0            124      3
   6 │ small_3x2        GameTheory      lrsnash                    51035.5          40456.0           1269      3
   7 │ small_3x2        nashpy          vertex_enumeration             8.14761e6  missing          missing      3
   8 │ small_3x2        nashpy          lemke_howson_enumeration  739972.0        missing          missing      5  repeated results
   9 │ small_3x2        nashpy          support_enumeration            1.49346e6  missing          missing      3
  10 │ small_3x2        pygambit        lcp_solve                 432796.0        missing          missing      3
  11 │ small_3x2        pygambit        enumpoly_solve                 3.96886e5  missing          missing      3
  12 │ rand_6x7         StrategicGames  nash_cp                        1.14288e8        8.00475e6   244963      0
  13 │ rand_6x7         StrategicGames  nash_se                        3.27694e7        7.21902e7  1579071      1
  14 │ rand_6x7         GameTheory      hc_solve                       7.19525e9        2.14659e8  1373479      1
  15 │ rand_6x7         GameTheory      support_enumeration            4.95192e5   208864.0           4566      1
  16 │ rand_6x7         GameTheory      lrsnash                        1.01465e6   353528.0          12312      1
  17 │ rand_6x7         nashpy          vertex_enumeration             1.74133e8  missing          missing      1
  18 │ rand_6x7         nashpy          lemke_howson_enumeration       5.594e6    missing          missing     13  repeated results
  19 │ rand_6x7         nashpy          support_enumeration            6.43585e8  missing          missing      0
  20 │ rand_6x7         pygambit        lcp_solve                      7.65117e6  missing          missing      1
  21 │ rand_6x7         pygambit        enumpoly_solve                 6.72629e8  missing          missing      1  single run
  22 │ rand_dec_6x5     StrategicGames  nash_cp                        1.00546e8        4.11159e6   125727      1
  23 │ rand_dec_6x5     StrategicGames  nash_se                        2.54142e7        1.46028e7   337455      3
  24 │ rand_dec_6x5     GameTheory      hc_solve                       1.60984e9        1.90183e7   221355      3
  25 │ rand_dec_6x5     GameTheory      support_enumeration            1.7658e5     61360.0           1398      3
  26 │ rand_dec_6x5     GameTheory      lrsnash                        1.39889e6   277232.0          10087      3
  27 │ rand_dec_6x5     nashpy          vertex_enumeration             5.44506e7  missing          missing      3
  28 │ rand_dec_6x5     nashpy          lemke_howson_enumeration       2.02323e6  missing          missing     11  repeated results
  29 │ rand_dec_6x5     nashpy          support_enumeration            1.27051e8  missing          missing      3
  30 │ rand_4x4x2       StrategicGames  nash_cp                        1.01613e8        5.02784e6   145097      0
  31 │ rand_4x4x2       StrategicGames  nash_se                        1.09057e9        7.65537e7  1518860      7  1 eq repeated
  32 │ rand_4x4x2       GameTheory      hc_solve                       2.01035e9        6.58439e7   519644      4  2 eq missing
  33 │ rand_4x4x2       pygambit        enumpoly_solve                 3.63245e7  missing          missing      3  3 eq missed
  34 │ rand_6x7_1st_eq  StrategicGames  nash_se                        3.19987e6        6.94906e6   155236      1  22 threads
  35 │ rand_6x7_1st_eq  StrategicGames  nash_se                        4.05169e6        3.31806e6    74993      1  single thread
  36 │ rand_6x7_1st_eq  GameTheory      hc_solve                       2.29806e9        7.24779e7   497224      1
  37 │ rand_6x7_1st_eq  nashpy          vertex_enumeration             8.72077e7  missing          missing      1
  38 │ rand_6x7_1st_eq  nashpy          lemke_howson_enumeration  343807.0        missing          missing      1
  39 │ rand_6x7_1st_eq  nashpy          support_enumeration         5931.17       missing          missing      0  no eq reported
  40 │ degenerated_3x2  StrategicGames  nash_cp                        2.31078e7        1.21561e6    37211      1
  41 │ degenerated_3x2  StrategicGames  nash_se                        3.13353e6   461872.0          15360      3
  42 │ degenerated_3x2  GameTheory      hc_solve                       1.28393e7        5.16101e6    77299      1
  43 │ degenerated_3x2  GameTheory      support_enumeration        17366.0           4816.0            116      2
  44 │ degenerated_3x2  GameTheory      lrsnash                    74776.0          38344.0           1204      3
  45 │ degenerated_3x2  nashpy          vertex_enumeration             8.0939e6   missing          missing      3
  46 │ degenerated_3x2  nashpy          lemke_howson_enumeration  931296.0        missing          missing      5  2 repeated results
  47 │ degenerated_3x2  nashpy          support_enumeration            1.6891e6   missing          missing      2
  48 │ degenerated_3x2  pygambit        lcp_solve                 494266.0        missing          missing      3
  49 │ degenerated_3x2  pygambit        enumpoly_solve                 4.4146e5   missing          missing      2

```
## Acknowledgements

The development of this package at the _Bureau d'Economie Théorique et Appliquée_ (BETA, Nancy) was supported by the French National Research Agency through the [Laboratory of Excellence ARBRE](http://mycor.nancy.inra.fr/ARBRE/), a part of the “Investissements d'Avenir” Program (ANR 11 – LABX-0002-01).

[![BLogos](assets/logos_betaumr.png)](hhttp://www.beta-umr7522.fr/)
