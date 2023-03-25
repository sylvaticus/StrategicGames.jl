"""
    StrategicGames package

Strategic Games provides functionalities to work with strategic games, including finding Nash equilibrium in simultaneous games when mixed strategies are allowed. 

Unless otherwise stated, all functions work with generic n players (the examples generally show 2 players for simplicity) and assume the payoff "matrix" to be in the form of N-players + 1 dimensional arrays, where each dimension except the last one is given by the number of discrete actions available to each player and the last dimension is given by the number of players.
A convenient function allows to transform a N-dimensional array of tuples (the payoff for the various players) to the Nplayers+1 dimensional array


"""
module StrategicGames

using LinearAlgebra, JuMP, Ipopt, GLPK

export expand_dimensions
export expected_payoff, nash_cp
export dominated_strategies, best_response, is_best_response, is_nash

"""
    expand_dimensions(x::AbstractArray{T}

Convenience function to transform a _N_ dimensional array of tuples in a _N+1_ dimensional array of scalars (the format used for most functions in this package).

# Example:
```julia
julia> using StrategicGames
julia> payoff_tuple = [(1,-1) (-1,1) (1,0); (-1,1) (1, -1) (0,1)] # 2 players, with 2 and 3 actions respectively
2×3 Matrix{Tuple{Int64, Int64}}:
 (1, -1)  (-1, 1)  (1, 0)
 (-1, 1)  (1, -1)  (0, 1)

julia> payoff_array = expand_dimensions(payoff_tuple)
2×3×2 Array{Int64, 3}:
[:, :, 1] =
  1  -1  1
 -1   1  0

[:, :, 2] =
 -1   1  0
  1  -1  1
```
"""
function expand_dimensions(x::AbstractArray{T}) where {T <: Tuple{Vararg{T2}}} where {T2}
    xsize    = size(x)
    ndims_in = ndims(x)
    size_add = length(x[ones(Int64,ndims_in)...])
    out      = Array{T2,ndims_in+1}(undef,vcat([xsize...],size_add)...)
    for idx in CartesianIndices(xsize)
        for j in 1:size_add
            out[Tuple(idx)...,j] = x[idx][j]
        end
    end
    return out
end

"""
    outer_product(vs)

Return the outer product of several vectors (or a vector of vectors)

# Example:
```julia
julia> using StrategicGames
julia> vs = [[0.2,0.8], [0.1,0.9], [0.2,0.2,0.6]];

julia> outer_product(vs)
2×2×3 Array{Float64, 3}:
[:, :, 1] =
 0.004  0.036
 0.016  0.144

[:, :, 2] =
 0.004  0.036
 0.016  0.144

[:, :, 3] =
 0.012  0.108
 0.048  0.432
 ```
"""
#outer_product(vs...) =  .*([reshape(vs[d], (ntuple(Returns(1), d-1)..., :)) for d in 1:length(vs)]... )
#outer_product(vs) = Base.splat(*).(Iterators.product(vs...))
#outer_product(vs...) = Base.splat(*).(Iterators.product(vs...))
outer_product(vs)    = prod.(Iterators.product(vs...))
outer_product(vs...) = prod.(Iterators.product(vs...))



"""
    expected_value(v::Array{N,Real},p::Vector{Vector{Real}}) --> Real

Compute the expected value (scalar) of a N-dimensional value array given a vector of N probability vectors, one per each dimension of the value array. 
"""
function expected_value(v,p)
    sum((v .* outer_product(p)))
end
#=
function expected_value_old(v,p)
    (ndims(v) == length(p) && all(size(v) .== length.(p))) || error("Mismatch dimension or size between the value array and the probability vectors")
    (all([all(>=(0-1e-06 ), v) for v in p])  && all(sum.(p) .> 1 .- 1e-06 ) && all(sum.(p) .< 1 .+ 1e-06 )) || error("p is not a vector of probabilities")
    outsum = 0.0
    comp   = 0.0 #https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    nd = ndims(v)
    for idx in CartesianIndices(size(v)) 
        y = *(v[idx],[p[d][idx[d]] for d in 1:nd]...) - comp
        t = outsum + y
        comp = (t - outsum) - y
        outsum = t
    end
    return outsum
end
=#

function expected_payoff(payoff_array,s,n)
    (ndims(payoff_array) == length(s)+1 && all(size(payoff_array)[1:end-1] .== length.(s))) || error("Mismatch dimension or size between the value array and the probability vectors")
    (all([all(>=(0-1e-06 ), v) for v in s])  && all(sum.(s) .> 1 .- 1e-06 ) && all(sum.(s) .< 1 .+ 1e-06 )) || error("s is not a vector of probabilities")
    playerdim  = ndims(payoff_array)
    expected_value(selectdim(payoff_array,playerdim,n),s)
end

function expected_payoff(payoff_array,s)
    [expected_payoff(payoff_array,s,i) for i in 1:length(s)]
end

# use normalise_strategies only when the solver find an approximate solution and you want to return approximate probabilities
"""
    nash_cp(payoff_array;init,verbosity)

Find a Nash Equilibrium for N-players simultaneous games when mixed strategies are allowed using the Complementarity Problem formulation.

# Parameters
- `payoff_array`: the Nplayers+1 dimensional array of payoffs for the N players
- `init`: a vector of vector of mixed strategies (i.e. PMFs) for each players to start the algorithm with. Different init points may reach different equilibrium points [def: equal probabilities for each available action of the players]
- `verbosity``: an integer parameter to pass to the inner solver (currently Ipopt) [def: `0`]

# Notes
- This function uses a complementarity formulation. For N <= 2 the problem, except the complementarity equation, the problem is linear and known as LCP (Linear Complementarity Problem)
- This implementation uses the JuMP modelling language with the Ipopt solver engine (and hence it uses an interior point method instead of the pivotal approach used in the original Lemke-Howson [1964] algorithm)
- There is no guarantee on timing and even that the algorithm converge to an equilibrium. Different Nash equilibriums may be reached by setting different initial points 

# Returns
- A named tuple with the following elements: `status`,`equilibrium_strategies`,`expected_payoffs`

# Example
```julia
julia> using StrategicGames
julia> payoff = [(-1,-1) (-3,0); (0, -3) (-2, -2)] # prisoner's dilemma
2×2 Matrix{Tuple{Int64, Int64}}:
 (-1, -1)  (-3, 0)
 (0, -3)   (-2, -2)
julia> eq     = nash_cp(expand_dimensions(payoff));
julia> eq_strategies = eq.equilibrium_strategies
2-element Vector{Vector{Float64}}:
 [-4.049752569180346e-11, 1.0000000000404976]
 [-4.0497525691839856e-11, 1.0000000000404976]
```
"""
function nash_cp(payoff;allow_mixed=true,init=[fill(1/size(payoff,d),size(payoff,d)) for d in 1:ndims(payoff)-1],verbosity=0)  
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    (length(nActions) == nPlayers) || error("Mismatch dimension or size between the payoff array and the number of players")

    # Vector of 2-elements tuples where the first one is the player index and the second one is the action index
    # We specify it as a vector of tuples as the number of actions can be different for the different players
    idxSet = Vector{Tuple{Int64,Int64}}()
    setPosByPlayerIdx = Vector{Vector{Int64}}()
    pos = 1
    for np in 1:nPlayers
        push!(setPosByPlayerIdx,collect(pos:pos+nActions[np]-1))
        for na in 1:nActions[np]
            push!(idxSet,(np,na))
            pos += 1
        end
    end

    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level", verbosity)
    @variables m begin
        r[idx in idxSet ] >= 0
        u[n in 1:nPlayers]
    end
    @variable(m, 0 <= s[idx in idxSet ] <= 1,  start=init[idx[1]][idx[2]])
    @constraints m begin
        slack[idx in idxSet], # either rₙⱼ or sₙⱼ must be zero
            r[idx] * s[idx] == 0
        utility[idx in idxSet], # the expected utility for each action must be constant, for each nPlayers
            sum(   
                selectdim(selectdim(payoff,nPlayers+1,idx[1]),idx[1],idx[2])[cidx]
                * prod(s[sidx] for sidx in zip(collect(1:nPlayers)[[1:idx[1]-1;idx[1]+1:end]], Tuple(cidx)) )
                for cidx in CartesianIndices(([nActions...][[1:idx[1]-1;idx[1]+1:end]]...,))
            ) +   r[idx] == u[idx[1]]
        probabilities[n in 1:nPlayers],
            sum(s[j] for j in idxSet[setPosByPlayerIdx[n]]) == 1
    end

    if !allow_mixed
        @NLconstraint(m, pure_only[n in 1:nPlayers], # if we want only pure strategies we add this quadratic constraint to force boolean probabilities
            sum(s[j]^2 for j in idxSet[setPosByPlayerIdx[n]]) >= 1-0.00000001
        )
    end

    @objective m Min sum(u[n] for n in 1:nPlayers)

    optimize!(m)
    #print(m)
    status = termination_status(m)

    optStrategies = Vector{Vector{Float64}}()
    optU          = Float64[]
    if (status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL]) && has_values(m)
        optStrategies = Vector{Vector{Float64}}()
        for n in 1:nPlayers
            optStrategies_by_player = Float64[]
            for j in 1:nActions[n]
                optStr = value(s[(n,j)])
                push!(optStrategies_by_player,optStr)
            end
            #if normalise_strategies # sometimes numerically not 100% correct probabilities
            #    optStrategies_by_player = BetaML.softmax(optStrategies_by_player)
            #end
            push!(optStrategies,optStrategies_by_player)
        end
        optU = value.(u)
    end
    return (status=status,equilibrium_strategies=optStrategies,expected_payoffs=optU)
end

"""
    dominated_strategies(payoff,nplayer;strict=true)

Return a vector with the positions of the actions for player `nplayer` that are dominates by at least one of his other actions.

# Notes
- This function is available also as `dominated_strategies(payoff;strict=true)` returning a vector of vectors of dominated strategies for all players

# Example
```julia
julia> using StrategicGames
julia> payoff_array = expand_dimensions([(3,4) (1,5) (6,2); (2,6) (3,7) (1,7)])
2×3×2 Array{Int64, 3}:
[:, :, 1] =
 3  1  6
 2  3  1
[:, :, 2] =
 4  5  2
 6  7  7
 julia> dominated_strategies(payoff_array,2) 
1-element Vector{Int64}:
 1
julia> dominated_strategies(payoff_array,2,strict=false) 
2-element Vector{Int64}:
 1
 3
```
"""
function dominated_strategies(payoff,player;strict=true)
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    payoff_n = selectdim(payoff,nPlayers+1,player)
    dominated = Int64[]
    for i in 1:nActions[player]
        ai = selectdim(payoff_n,player,i) # action to test 
        for j in 1:nActions[player]
            i != j || continue
            aj = selectdim(payoff_n,player,j)           
            check_dominated = strict ? aj .> ai : aj .>= ai
            if all(check_dominated)
                push!(dominated,i)
                break
            end 
        end
    end
    return dominated 
end
function dominated_strategies(payoff;strict=true)
    return [dominated_strategies(payoff,n,strict=strict) for n in 1:ndims(payoff)-1]
end

"""
    best_response(payoff_array,strategy_profile,nplayer)

Return (possibly one of many) best strategy and corrsponding expected payoff for a given player.

# Parameters:
- `payoff_array`: the N_players+1 dimensional array of payoffs
- `strategy_profile`: the vector of vectors defining the strategies for the N players. The strategy for player n for which the best response is computed is used as initial values in the inner optimisation
- `nplayer`: counter of the player for which we want to compute the best_response (e.g. 1 or 3)

# Returns:
- A named tuple with: `expected_payoff`, `optimal_strategy`, `status` (of the underlying optimisation)

# Example:

```julia
julia> using StrategicGames
julia> payoff_array  = [(3,4) (1,5); (4,2) (2,3)] # prisoner's dilemma
2×2 Matrix{Tuple{Int64, Int64}}:
 (3, 4)  (1, 5)
 (4, 2)  (2, 3)
julia> best_response(expand_dimensions(payoff_array),[[0.5,0.5],[0.5,0.5]],2)
(expected_payoff = 4.0, optimal_strategy = [0.0, 1.0], status = MathOptInterface.OPTIMAL)
```
"""
function best_response(payoff,strategy_profile,player)
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    payoff_n = selectdim(payoff,nPlayers+1,player)
    init = strategy_profile[player]
    m = Model(GLPK.Optimizer)
    @variable(m, 0 <= s[j in 1:nActions[player] ] <= 1,  start=init[j])
    @constraints m begin
        probabilities,
            sum(s[j] for j in 1:nActions[player]) == 1
    end
    @objective m Max sum(
        [*(payoff_n[idx],[strategy_profile[n][idx[n]] for n in [1:player-1;player+1:nPlayers]]...,s[idx[player]]) for idx in CartesianIndices(nActions)]  
    )
    optimize!(m)
    #print(m)
    status = termination_status(m)
    
    return (expected_payoff=objective_value(m), optimal_strategy=value.(s), status=status)
end

"""
    is_best_response(payoff_array,strategy_profile,nplayer;atol=1e-07,rtol=1e-07)

Determine if a given strategy for player `nplayer` is a best response to a given payoff array and strategies of the other players

# Parameters:
- `payoff_array`: the Nplayers+1 dimensional array of payoffs
- `strategy_profile`: the vector of vectors defining the strategies for the N players
- `nplayer`: counter of the player for which we want to verify if its strategy is a best_response (e.g. 1 or 3)
- `atol`: absolute tollerance in comparing the expected payoff from the given strategy and those from the optimal one [def: `1e-07`]
- `rtol`: relative tollerance in comparing the expected payoff from the given strategy and those from the optimal one [def: `1e-07`]

# Example : 
```julia
julia> using StrategicGames
julia> payoff_array = [(3,4) (1,5); (4,2) (2,3)] # prisoner's dilemma
2×2 Matrix{Tuple{Int64, Int64}}:
 (3, 4)  (1, 5)
 (4, 2)  (2, 3)
julia> is_best_response(expand_dimensions(payoff_array),[[0,1],[0.5,0.5]],1)
true
julia> is_best_response(expand_dimensions(payoff_array),[[0,1],[0.5,0.5]],2)
false
```
"""
function is_best_response(payoff,strategy_profile,player;atol=1e-07,rtol=1e-07)
    best_u = best_response(payoff,strategy_profile,player).expected_payoff
    this_u = expected_payoff(payoff,strategy_profile,player)
    return isapprox(this_u,best_u,atol=atol,rtol=rtol)
end

"""
    is_nash(payoff_array,strategy_profile;atol=1e-07,rtol=1e-07)

Determine if a strategy profile is a Nash equilibrium for a given payoff array, i.e. if all strategies are (at least weak) best responses.

# Parameters:
- `payoff_array`: the Nplayers+1 array of payoffs
- `strategy_profile`: the vector of vectors defining the strategies for the N players
- `atol`: absolute tollerance in comparing the expected payoffs from the given strategies and those from the optimal ones [def: `1e-07`]
- `rtol`: relative tollerance in comparing the expected payoffs from the given strategies and those from the optimal ones [def: `1e-07`]

# Example : 
```julia
julia> using StrategicGames
julia> payoff_array  = [(3,4) (1,5); (4,2) (2,3)] # prisoner's dilemma
2×2 Matrix{Tuple{Int64, Int64}}:
 (3, 4)  (1, 5)
 (4, 2)  (2, 3)
julia> is_nash(expand_dimensions(payoff_array),[[0,1],[0,1]])
true
```
"""
function is_nash(payoff,strategy_profile;atol=1e-07,rtol=1e-07)
    all([isapprox( best_response(payoff,strategy_profile,i).expected_payoff,
                   expected_payoff(payoff,strategy_profile,i),
                   atol=atol,rtol=rtol) for i in 1:ndims(payoff)-1])
end


end # module StrategicGames
