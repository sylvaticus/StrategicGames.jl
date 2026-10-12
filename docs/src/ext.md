# GameTheory.jl extension

When [GameTheory.jl](https://github.com/QuantEcon/GameTheory.jl) is loaded,
`nash_cp` and `nash_se` accept a `NormalFormGame` or a custom subtype of
`AbstractNormalFormGame` from GameTheory.jl.

## Example usage

Consider the 3-player 2-action game with 9 Nash equilibria in McKelvey and
McLennan (1996) "Computation of Equilibria in Finite Games", constructed as a
`NormalFormGame`:

```julia
julia> using StrategicGames, GameTheory

julia> g = NormalFormGame((2, 2, 2));

julia> g[1, 1, 1] = [9, 8, 12];

julia> g[2, 2, 1] = [9, 8, 2];

julia> g[1, 2, 2] = [3, 4, 6];

julia> g[2, 1, 2] = [3, 4, 4];

julia> g
2×2×2 NormalFormGame{3, Float64}:
[:, :, 1] =
 (9.0, 8.0, 12.0)  (0.0, 0.0, 0.0)
 (0.0, 0.0, 0.0)   (9.0, 8.0, 2.0)

[:, :, 2] =
 (0.0, 0.0, 0.0)  (3.0, 4.0, 6.0)
 (3.0, 4.0, 4.0)  (0.0, 0.0, 0.0)

julia> Base.active_repl.options.iocontext[:compact] = true;  # Reduce digits to display

julia> eq = nash_cp(g);

julia> eq.equilibrium_strategies
3-element Vector{Vector{Float64}}:
 [0.5, 0.5]
 [0.5, 0.5]
 [1.0, -1.35301e-32]

julia> eqs = nash_se(g, max_samples=Inf);

julia> [eq.equilibrium_strategies for eq in eqs]
8-element Vector{Vector{Vector{Float64}}}:
 [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
 [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
 [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
 [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
 [[0.5, 0.5], [0.5, 0.5], [1.0, 0.0]]
 [[0.25, 0.75], [1.0, 0.0], [0.25, 0.75]]
 [[0.0, 1.0], [0.333333, 0.666667], [0.333333, 0.666667]]
 [[0.5, 0.5], [0.333333, 0.666667], [0.25, 0.75]]

julia> all([GameTheory.is_nash(g, Tuple(eq.equilibrium_strategies)) for eq in eqs])
true
```

This game in fact has two full-support equilibria: `nash_se` returns one
equilibrium per support and thus does not reach one of those equilibria.

A game can also be defined as a custom subtype of `AbstractNormalFormGame`,
which must define `GameTheory.nums_actions` and `GameTheory.payoff_profile`.
Consider the 3-player game from [Some examples](@ref examples), but for any
number of players:

```julia
julia> struct SameOptionGame{N} <: AbstractNormalFormGame{N,Int} end

julia> GameTheory.nums_actions(::SameOptionGame{N}) where {N} = ntuple(_ -> 2, Val(N));

julia> function GameTheory.payoff_profile(::SameOptionGame{N}, a) where {N}
           nx = count(==(1), a)  # X is action 1, Y action 2
           ny = N - nx
           return ntuple(i -> a[i] == 1 ? 2nx - nx^2 + 3 : 4 - ny, Val(N))
       end;

julia> g = SameOptionGame{3}();

julia> eq = nash_cp(g);

julia> eq.equilibrium_strategies
3-element Vector{Vector{Float64}}:
 [0.581139, 0.418861]
 [0.581139, 0.418861]
 [0.581139, 0.418861]

julia> eqs = nash_se(g, max_samples=Inf);

julia> [eq.equilibrium_strategies for eq in eqs]
7-element Vector{Vector{Vector{Float64}}}:
 [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
 [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
 [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
 [[0.25, 0.75], [0.25, 0.75], [1.0, 0.0]]
 [[0.25, 0.75], [1.0, 0.0], [0.25, 0.75]]
 [[1.0, 0.0], [0.25, 0.75], [0.25, 0.75]]
 [[0.581139, 0.418861], [0.581139, 0.418861], [0.581139, 0.418861]]

julia> all([GameTheory.is_nash(g, Tuple(eq.equilibrium_strategies)) for eq in eqs])
true
```

## API

Methods available when GameTheory.jl is loaded:

```@autodocs
Modules = [StrategicGamesGameTheoryExt]
```
