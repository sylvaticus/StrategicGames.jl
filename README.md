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
 - **[GameTheory.jl](https://github.com/QuantEcon/GameTheory.jl)** Inter alia, compute N-players pure strategy NE, 2-players mixed strategy games (`lrsnash`, using exact arithmetics) and N-players mixed strategies NE using a solver of the polynomial equation representation of the complementarity conditions (`hc_solve`). However this "generic" N-player solver is slow, as it doesn't seem to have a dominance check.

**Non-Julia**
 - [Nashpy](https://github.com/drvinceknight/Nashpy): two players only
 - [Gambit](http://www.gambit-project.org/): many algorithms require installation of gambit other than pygambit. No decimal/rational payoffs in Python
 - [Sage Math](https://doc.sagemath.org/html/en/reference/game_theory/sage/game_theory/normal_form_game.html): two players only
 - [GtNash](https://forgemia.inra.fr/game-theory-tools-group/gtnash/-/tree/main/)

### Benchmarks
The following benchmarks have been run on a Intel Core 5 laptop on StrategicGames v0.0.5.
See [benchmarks/benchmarks_other_libraries.jl](/benchmarks/benchmarks_other_libraries.jl) for details.

```text
 Row │ benchmark_name   library         method                    time             memory           alloc    neqs   notes              
     │ String           String          String                    Float64          Float64?         Int64?   Int64  String             
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Test             rand            rand(2,2)                     59.4469           96.0              1      1
   2 │ small_3x2        StrategicGames  nash_cp                        1.58605e7    306112.0           6002      1
   3 │ small_3x2        StrategicGames  nash_se                        2.42753e6    348928.0           8022      3
   4 │ small_3x2        GameTheory      hc_solve                       1.54957e7         2.90228e6    38420      3
   5 │ small_3x2        GameTheory      support_enumeration            1.26744e5      6848.0            103      3
   6 │ small_3x2        GameTheory      lrsnash                    81660.0           25208.0           1064      3
   7 │ small_3x2        nashpy          vertex_enumeration             2.24663e7   missing          missing      3
   8 │ small_3x2        nashpy          lemke_howson_enumeration       1.70237e6   missing          missing      5  repeated results
   9 │ small_3x2        nashpy          support_enumeration            2.88622e6   missing          missing      3
  10 │ small_3x2        pygambit        lcp_solve                 626370.0         missing          missing      3
  11 │ small_3x2        pygambit        ExternalEnumPolySolver         3.04543e6   missing          missing      3
  12 │ rand_6x7         StrategicGames  nash_cp                        6.79384e7    832992.0          16774      0
  13 │ rand_6x7         StrategicGames  nash_se                        5.81445e7         8.39402e7  1489215      1
  14 │ rand_6x7         GameTheory      hc_solve                       2.29249e10        2.13365e8  5674406      1
  15 │ rand_6x7         GameTheory      support_enumeration            1.07393e6    255888.0           3223      1
  16 │ rand_6x7         GameTheory      lrsnash                        1.22114e6    189432.0          11058      1
  17 │ rand_6x7         nashpy          vertex_enumeration             4.69925e8   missing          missing      1
  18 │ rand_6x7         nashpy          lemke_howson_enumeration       1.12535e7   missing          missing     13  repeated results
  19 │ rand_6x7         nashpy          support_enumeration            1.14592e9   missing          missing      0
  20 │ rand_6x7         pygambit        lcp_solve                      9.02179e6   missing          missing      1
  21 │ rand_6x7         pygambit        ExternalEnumPolySolver         5.88408e11  missing          missing      1
  22 │ rand_dec_6x5     StrategicGames  nash_cp                        2.62495e7    666384.0          12585      1
  23 │ rand_dec_6x5     StrategicGames  nash_se                        1.91476e7         1.63948e7   301926      3
  24 │ rand_dec_6x5     StrategicGames  nash_se                        2.10698e7         1.63948e7   301928      3
  25 │ rand_dec_6x5     GameTheory      hc_solve                       2.81279e9         1.31916e7   129193      3
  26 │ rand_dec_6x5     GameTheory      support_enumeration            3.6849e5      82976.0           1112      3
  27 │ rand_dec_6x5     GameTheory      lrsnash                        1.18709e6    194672.0           9179      3
  28 │ rand_dec_6x5     nashpy          vertex_enumeration             1.22334e8   missing          missing      3
  29 │ rand_dec_6x5     nashpy          lemke_howson_enumeration       5.08469e6   missing          missing     11  repeated results
  30 │ rand_dec_6x5     nashpy          support_enumeration            2.52825e8   missing          missing      3
  31 │ rand_4x4x2       StrategicGames  nash_cp                        7.2921e8          2.99718e6    85361      0
  32 │ rand_4x4x2       StrategicGames  nash_se                        3.10279e9         7.36352e7  1353042      7  1 eq repeated
  33 │ rand_4x4x2       GameTheory      hc_solve                       6.81009e9         1.87611e7   168777      4  2 eq missing
  34 │ rand_4x4x2       pygambit        ExternalEnumPolySolver         6.67698e8   missing          missing      5  1 eq missed
  35 │ rand_6x7_1st_eq  StrategicGames  nash_se                        7.89602e6         3.82298e6    69988      1
  36 │ rand_6x7_1st_eq  GameTheory      hc_solve                       1.17059e10        1.06781e8  2643505      1
  37 │ rand_6x7_1st_eq  nashpy          vertex_enumeration             2.20641e8   missing          missing      1
  38 │ rand_6x7_1st_eq  nashpy          lemke_howson_enumeration       8.8455e5    missing          missing      1
  39 │ rand_6x7_1st_eq  nashpy          support_enumeration         3977.38        missing          missing      0  no eq reported
  40 │ degenerated_3x2  StrategicGames  nash_cp                        1.74314e7    315424.0           6320      1
  41 │ degenerated_3x2  StrategicGames  nash_se                        2.39501e6    430528.0           9999      3
  42 │ degenerated_3x2  GameTheory      hc_solve                       1.58658e7         2.91026e6    38445      1
  43 │ degenerated_3x2  GameTheory      support_enumeration       128884.0            6544.0             98      2
  44 │ degenerated_3x2  GameTheory      lrsnash                    79857.5           23688.0           1002      3
  45 │ degenerated_3x2  nashpy          vertex_enumeration             1.8437e7    missing          missing      3
  46 │ degenerated_3x2  nashpy          lemke_howson_enumeration       1.75092e6   missing          missing      5  2 repeated results
  47 │ degenerated_3x2  nashpy          support_enumeration            3.03576e6   missing          missing      2
  48 │ degenerated_3x2  pygambit        lcp_solve                 672832.0         missing          missing      3
  49 │ degenerated_3x2  pygambit        ExternalEnumPolySolver         3.39349e6   missing          missing      2

```
## Acknowledgements

The development of this package at the _Bureau d'Economie Théorique et Appliquée_ (BETA, Nancy) was supported by the French National Research Agency through the [Laboratory of Excellence ARBRE](http://mycor.nancy.inra.fr/ARBRE/), a part of the “Investissements d'Avenir” Program (ANR 11 – LABX-0002-01).

[![BLogos](assets/logos_betaumr.png)](hhttp://www.beta-umr7522.fr/)
