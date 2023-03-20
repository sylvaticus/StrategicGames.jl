"""
    StrategicGames module

Provide utility functions to work with strategic games, including finding Hash equilibrium for n-player simultaneous games when mixed strategies are allowed. 
"""
module StrategicGames

using LinearAlgebra, JuMP, Ipopt

export expand_dimensions, outer_product
export expected_value, expected_payoff, nash_lcp

"""
    expand_dimensions(x::AbstractArray{T}

Convenence function to transform a _N_ dimensional array of tuples in a _N+1_ dimensional array of scalars (the format used for most functions in this package).

# Example:
```julia
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

Compute the expected value (scalar) of a N-dimensional value tensor given a vector of N probability vectors, one per each dimension of the value tensor. 
"""
function expected_value(v,p)
    sum((v .* outer_product(p)))
end
#=
function expected_value_old(v,p)
    (ndims(v) == length(p) && all(size(v) .== length.(p))) || error("Mismatch dimension or size between the value tensor and the probability vectors")
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
    (ndims(payoff_array) == length(s)+1 && all(size(payoff_array)[1:end-1] .== length.(s))) || error("Mismatch dimension or size between the value tensor and the probability vectors")
    (all([all(>=(0-1e-06 ), v) for v in s])  && all(sum.(s) .> 1 .- 1e-06 ) && all(sum.(s) .< 1 .+ 1e-06 )) || error("s is not a vector of probabilities")
    playerdim  = ndims(payoff_array)
    expected_value(selectdim(payoff_array,playerdim,n),s)
end

function expected_payoff(payoff_array,s)
    [expected_payoff(payoff_array,s,i) for i in 1:length(s)]
end

# use normalise_strategies only when the solver find an approximate solution and you want to return approximate probabilities
"""
    nash_lcp(payoff_tensor;init,verbosity)

Find a Nash Equilibrium for n-players simultaneous games when mixed strategies are allowed using the LCP algorithm.

# Parameters
- `payoff_tensor`: the nplayers+1 dimension payoff tensor of payoffs for the various players
- `init`: a vector of vector of mixed strategies (i.e. PMFs) for each players to start the algorithm with. Different init points may reach different equilibrium points [def: equal probabilities for each available action by the players]
- verbosity: an integer parameter to pass to the inner solver (currently Ipopt) [def: `0`]

# Notes
- This function uses the LCP (Linear Complementarity) formulation from  Lemke-Howson [1964] (for nplayers >3 the algorithm actually is not linear)
- The implementation uses the JuMP modelling language with the Ipopt solver engine
- There is no guarantee on timing and even that the algorithm converge to an equilibrium. Different equilibriums may be reached by setting different initial points 

# Returns
- A named tuple with the following elements: `status`,`equilibrium_strategies`,`expected_payoffs`

# Example
```julia
julia> payoff = [(-1,-1) (-3,0); (0, -3) (-2, -2)] # prisoner's dilemma
2×2 Matrix{Tuple{Int64, Int64}}:
 (-1, -1)  (-3, 0)
 (0, -3)   (-2, -2)
julia> eq     = nash_lcp(expand_dimensions(payoff));
julia> eq_strategies = eq.equilibrium_strategies
2-element Vector{Vector{Float64}}:
 [-4.049752569180346e-11, 1.0000000000404976]
 [-4.0497525691839856e-11, 1.0000000000404976]
```
"""
function nash_lcp(payoff;allow_mixed=true,init=[fill(1/size(payoff,d),size(payoff,d)) for d in 1:ndims(payoff)-1],verbosity=0)  
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    (length(nActions) == nPlayers) || error("Mismatch dimension or size between the payoff tensor and the number of players")

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

end # module StrategicGames
