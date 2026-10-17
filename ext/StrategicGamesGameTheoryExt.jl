module StrategicGamesGameTheoryExt

using StrategicGames
using GameTheory: AbstractNormalFormGame, GAMPayoffVector, nums_actions

"""
    expand_dimensions(g)

Convenience function that returns the payoff array of an N-player normal form
game `g` of GameTheory.jl in the N+1 dimensional format used in this package:
`payoff_array[a_1, ..., a_N, i]` is the payoff to player `i` at the action
profile `(a_1, ..., a_N)`.

# Parameters
- `g::AbstractNormalFormGame`: `NormalFormGame` or any `AbstractNormalFormGame`
  from GameTheory.jl.

# Returns
- An N+1 dimensional array of payoff values.
"""
function StrategicGames.expand_dimensions(g::AbstractNormalFormGame{N}) where {N}
    p = GAMPayoffVector(g)
    return reshape(p.payoffs, (nums_actions(g)..., N))
end

"""
    nash_cp(g; kwargs...)

Find a Nash Equilibrium for an N-player normal form game `g` of GameTheory.jl,
by solving the complementarity problem of the whole game with Ipopt, after the
iterated removal of dominated actions.

# Parameters
- `g::AbstractNormalFormGame`: `NormalFormGame` or any `AbstractNormalFormGame`
  from GameTheory.jl.
- `kwargs`: Keyword arguments of [`nash_cp`](@ref).

# Returns
- A named tuple with the following information: `status`,
  `equilibrium_strategies`, `expected_payoffs`.
"""
StrategicGames.nash_cp(g::AbstractNormalFormGame; kwargs...) =
    nash_cp(expand_dimensions(g); kwargs...)

"""
    nash_se(g; kwargs...)

Compute `max_samples` (default one) Nash equilibria for an N-player normal form
game `g` of GameTheory.jl, by support enumeration: for each support profile
that passes a dominance check, the indifference conditions are solved as a
linear system with HiGHS for 2 players and as a nonlinear system with Ipopt for
N >= 3 players.

# Parameters
- `g::AbstractNormalFormGame`: `NormalFormGame` or any `AbstractNormalFormGame`
  from GameTheory.jl.
- `kwargs`: Keyword arguments of [`nash_se`](@ref).

# Returns
- A vector of named tuples with the following information:
  `equilibrium_strategies`, `expected_payoffs`, `supports`.
"""
StrategicGames.nash_se(g::AbstractNormalFormGame; kwargs...) =
    nash_se(expand_dimensions(g); kwargs...)

end
