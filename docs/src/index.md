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
 
## [Payoff representations](@id payoff_representations)

The payoffs of a game in normal form can be written in two equivalent forms. The first one is the most natural to type, while the functions of this package work with the second one, obtained from the first with [`expand_dimensions`](@ref).

#### First form: N-dimensional array of N-tuples

In a game with `N` players, the payoffs are an `N`-dimensional array `U` where:
- dimension `n` corresponds to player `n`, and the index along it is the action chosen by that player;
- each element is a tuple with the payoffs of all the `N` players for that action profile.

So `U[a₁,a₂,...,aₙ]` is the tuple of payoffs when player 1 plays action `a₁`, player 2 plays action `a₂`, and so on, and `U[a₁,a₂,...,aₙ][n]` is the payoff of player `n` alone.

With 2 players this is the usual bimatrix, with the actions of player 1 on the rows and those of player 2 on the columns. In the Prisoner's dilemma below, `U[1,2] = (-3,0)` means that when player 1 plays its first action and player 2 its second one, player 1 gets `-3` and player 2 gets `0`.

With 3 players the array has a third dimension, the action of player 3, and it is convenient to write it with Julia's `;` (concatenate along the first dimension, i.e. player 1's actions), `;;` (along the second dimension, player 2's actions) and `;;;` (along the third dimension, player 3's actions) syntax:

```julia
julia> U = [(0,0,0) ; (3,3,3) ;; (3,3,3) ; (2,2,4) ;;;   # player 3 plays action 1
            (3,3,3) ; (2,4,2) ;; (4,2,2) ; (1,1,1) ;;;]  # player 3 plays action 2
2×2×2 Array{Tuple{Int64, Int64, Int64}, 3}:
[:, :, 1] =
 (0, 0, 0)  (3, 3, 3)
 (3, 3, 3)  (2, 2, 4)
[:, :, 2] =
 (3, 3, 3)  (4, 2, 2)
 (2, 4, 2)  (1, 1, 1)
julia> U[1,2,2] # player 1 plays action 1, player 2 action 2, player 3 action 2
(4, 2, 2)
julia> U[1,2,2][1] # payoff of player 1 for this action profile
4
```

#### Second form: (N+1)-dimensional array of scalars

[`expand_dimensions`](@ref) moves the payoffs out of the tuples into an additional, last, dimension that indicates the player receiving the payoff:

```julia
julia> payoff_array = expand_dimensions(U);
```

The result is an `N+1` dimensional array of scalars, where:

```julia
payoff_array[a₁,a₂,...,aₙ,n] == U[a₁,a₂,...,aₙ][n]
```

For the 3-players game above, `payoff_array` has size `(2,2,2,3)`: 2 actions for each of the 3 players, plus a last dimension of size 3 for the 3 players' payoffs. The tuple `U[1,2,2] = (4,2,2)` is now spread across the last dimension:

| First form          | Second form               | Meaning                                                        |
|:------------------- |:------------------------- |:-------------------------------------------------------------- |
| `U[1,2,2]`          | `payoff_array[1,2,2,:]`   | payoffs of all players for actions 1, 2 and 2: `(4,2,2)` / `[4,2,2]` |
| `U[1,2,2][1]`       | `payoff_array[1,2,2,1]`   | payoff of player 1 for that action profile: `4`                 |
| `U[1,2,2][2]`       | `payoff_array[1,2,2,2]`   | payoff of player 2 for that action profile: `2`                 |
| `U[1,2,2][3]`       | `payoff_array[1,2,2,3]`   | payoff of player 3 for that action profile: `2`                 |

```julia
julia> size(payoff_array)
(2, 2, 2, 3)
julia> payoff_array[1,2,2,1]
4
julia> payoff_array[1,2,2,:]
3-element Vector{Int64}:
 4
 2
 2
```

When Julia prints `payoff_array`, each slice `[:, :, a₃, n]` is the matrix of the payoffs of player `n` when player 3 plays action `a₃`, with the actions of player 1 on the rows and those of player 2 on the columns. The `4` of player 1 above is therefore in the first row, second column of the slice `[:, :, 2, 1]`:

```julia
julia> payoff_array[:,:,2,1] # payoffs of player 1 when player 3 plays action 2
2×2 Matrix{Int64}:
 3  4
 2  1
```

Similarly, `payoff_array[:,:,:,n]` is the whole payoff array of player `n` alone.

#### Games from GameTheory.jl

When [GameTheory.jl](https://github.com/QuantEcon/GameTheory.jl) is loaded, [`nash_cp`](@ref) and [`nash_se`](@ref) also accept its `NormalFormGame` (or any other subtype of `AbstractNormalFormGame`) directly; see the [GameTheory.jl extension](ext.md) page.


## [Some examples](@id examples)

```julia
julia> using Pkg; Pkg.add("StrategicGames") # only once to install the library
julia> using StrategicGames
```

#### 3-players game with payoff a function of the number of players choosing the same option

The 3 players in this game can choose between `X` or `Y` with payoffs respectively `2nₓ - nₓ²+3` and `4-nᵧ` where `nₓ` and `nᵧ` represent the number of players that choose `X` or `Y`.
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
       dominated_actions(payoff_array)
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

"Battle of the sex" game has indeed 3 different equilibria. We can find them using the support enumeration method:

```julia
julia> eqs           = nash_se(expand_dimensions(payoff_matrix),max_samples=Inf);
julia> eq_strategies = [eq.equilibrium_strategies for eq in eqs] 
3-element Vector{Vector{Vector{Float64}}}:
 [[0.9999999999999999, 0.0], [0.9999999999999999, 0.0]]
 [[0.0, 0.9999999999999999], [0.0, 0.9999999999999999]]
 [[0.6666666666666666, 0.33333333333333337], [0.3333333333333333, 0.6666666666666667]]
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


## Acknowledgements

The development of this package at the _Bureau d'Economie Théorique et Appliquée_ (BETA, Nancy) was supported by the French National Research Agency through the [Laboratory of Excellence ARBRE](http://mycor.nancy.inra.fr/ARBRE/), a part of the “Investissements d'Avenir” Program (ANR 11 – LABX-0002-01).

[![BLogos](assets/logos_betaumr.png)](hhttp://www.beta-umr7522.fr/)
