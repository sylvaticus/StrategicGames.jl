# StrategicGames package documentation

[GitHub repository](https://github.com/sylvaticus/StrategicGames.jl)


```@docs
StrategicGames
```


## Index

StrategicGames provides the following functions:

```@index
Modules = [StrategicGames]
Order   = [:constant, :type, :function, :macro]
```
 

 ## [Some examples](@id examples)

```julia
julia> using Pkg; Pkg.add("StrategicGames") # only once to install the library
julia> using StrategicGames
```

#### 3-players game with payoff a function of the number of players choosing the same option

The 3 players in this game can choose between `X` or `Y` with payoffs respectively `2nₓ - 2nₓ²+3` and `4-nᵧ` where `nₓ` and `nᵧ` represent the number of players that choose `X` or `Y`.
(_this example is taken from https://www.youtube.com/watch?v=bKrwQKUT0v8 where it is analytically solved_)
```julia
julia> U = [(0,0,0) ; (3,3,3) ;; (3,3,3) ; (2,2,4) ;;;
            (3,3,3) ; (2,4,2) ;; (4,2,2) ; (1,1,1) ;;;]
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
julia> eq  = nash_cp(payoff_array)
(status = MathOptInterface.LOCALLY_SOLVED, equilibrium_strategies = [[0.5811388300841898, 0.4188611699158103], [0.5811388300841898, 0.4188611699158103], [0.5811388300841898, 0.41886116991581035]], expected_payoffs = [2.16227766016838, 2.16227766016838, 2.16227766016838])
julia> eq_strategies = eq.equilibrium_strategies
3-element Vector{Vector{Float64}}:
 [0.5811388300841898, 0.4188611699158103]
 [0.5811388300841898, 0.4188611699158103]
 [0.5811388300841898, 0.41886116991581035]
julia> p = -1 + sqrt(10)/2
0.5811388300841898
julia> eq_strategies ≈ [[p,1-p],[p,1-p],[p,1-p]]
true
julia> expected_payoff(payoff_array,eq_strategies)
3-element Vector{Float64}:
 2.1622776601683795
 2.1622776601683795
 2.1622776601683795
```

#### Prisoner's dilemma
```julia
julia> # Pay-off matrix
       U = [(-1,-1) (-3,0); (0, -3) (-2, -2)]
2×2 Matrix{Tuple{Int64, Int64}}:
 (-1, -1)  (-3, 0)
 (0, -3)   (-2, -2)
julia> # From N-dimensional array of tuples to N+1 arrays of scalars    
       payoff_array = expand_dimensions(U);
julia> # Find all the dominated strategies for the two players
       dominated_strategies(payoff_array)
2-element Vector{Vector{Int64}}:
 [1]
 [1]
julia> # Compute one Nash Equilibrium of the Game using LCP (linear complementarity) formulation       
       eq = nash_cp(payoff_array).equilibrium_strategies
2-element Vector{Vector{Float64}}:
 [-4.049752569180346e-11, 1.0000000000404976]
 [-4.0497525691839856e-11, 1.0000000000404976]
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

#### Head or tail
```julia
julia> payoff_matrix = [(1,-1) (-1,1); (-1,1) (1, -1)]
2×2 Matrix{Tuple{Int64, Int64}}:
 (1, -1)  (-1, 1)
 (-1, 1)  (1, -1)
julia> eq            = nash_cp(expand_dimensions(payoff_matrix));
julia> eq_strategies = eq.equilibrium_strategies
2-element Vector{Vector{Float64}}:
 [0.5, 0.5]
 [0.5, 0.5]
```

#### Battle of the sex
```julia
julia> payoff_matrix = [(2,1) (0,0); (0,0) (1,2)]
2×2 Matrix{Tuple{Int64, Int64}}:
 (2, 1)  (0, 0)
 (0, 0)  (1, 2)
julia> eq            = nash_cp(expand_dimensions(payoff_matrix));
julia> eq_strategies = eq.equilibrium_strategies 
2-element Vector{Vector{Float64}}:
 [0.6666666663602984, 0.33333333363970163]
 [0.33333333363970163, 0.6666666663602984]
```

#### Rock, paper, scissor
```julia
julia> # Actions: Rock, Paper, Scissor (in the order)
       U = [(0,0) (-1,1) (1,-1); (1,-1) (0,0) (-1,1); (-1,1) (1,-1) (0,0) ]
3×3 Matrix{Tuple{Int64, Int64}}:
 (0, 0)   (-1, 1)  (1, -1)
 (1, -1)  (0, 0)   (-1, 1)
 (-1, 1)  (1, -1)  (0, 0)
julia> eq = nash_cp(expand_dimensions(U)).equilibrium_strategies
2-element Vector{Vector{Float64}}:
 [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
 [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
```

#### A biased penalty kick game
The row player is the kicker and the column player is the goalkeeper, first action is kick/jump on one direction, second one is kick/jump on the other direction.
The kicker is more efficient (or, alternatively, the goalkeeper is less efficient) on the second direction.

```julia
julia> payoff_matrix = [(-1,1) (1,-1); (1,-1) (0, 0)]
2×2 Matrix{Tuple{Int64, Int64}}:
 (-1, 1)  (1, -1)
 (1, -1)  (0, 0)
julia> eq            = nash_cp(expand_dimensions(payoff_matrix));
julia> eq_strategies = eq.equilibrium_strategies
2-element Vector{Vector{Float64}}:
 [0.3333333333332723, 0.6666666666667278]
 [0.33333333333337, 0.6666666666666301]
```

