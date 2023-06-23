"""
    StrategicGames package

Strategic Games provides functionalities to work with strategic games, including finding mixed or pure Nash equilibria in simultaneous games (currently [`nash_se`](@ref), using support enumeration, and [`nash_cp`](@ref) solving the complementarity problem).

Unless otherwise stated, all functions work with generic n players (the examples generally show 2 players for simplicity) and assume the payoff "matrix" to be in the form of N-players + 1 dimensional arrays, where each dimension except the last one is given by the number of discrete actions available to each player, while the last dimension is given by the number of players.
Convenient functions ([`expand_dimensions`](@ref) and [`unstack_payoff`](@ref)) allows to transform payoff encoded in other form to the  N-players+1 dimensional array format used in this package.
"""
module StrategicGames

using LinearAlgebra, Combinatorics, JuMP, Ipopt, GLPK, HiGHS,  PrecompileTools

export Verbosity, NONE, LOW, STD, HIGH, FULL
export expand_dimensions, unstack_payoff
export expected_payoff
export dominated_actions, best_response, is_best_response, is_nash
export nash_cp, nash_se


"""
    Verbosity

Many functions accept a `verbosity` parameter.

Choose between: `NONE`, `LOW`, `STD` [default], `HIGH` and `FULL`.

Under default verbosity (`STD`) no output is printed, unless something unexpected in most conditions (but not necessarily an error) is detected 
"""
@enum Verbosity NONE=0 LOW=10 STD=20 HIGH=30 FULL=40

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
#function expand_dimensions(x::AbstractArray{T}) where {T <: Tuple{Vararg{T2}}} where {T2}
function expand_dimensions(x::AbstractArray{T}) where {T <: Tuple}    
    xsize    = size(x)
    ndims_in = ndims(x)
    size_add = length(x[ones(Int64,ndims_in)...])
    if eltype(eltype(x)) <: Integer
        out      = Array{Int,ndims_in+1}(undef,vcat([xsize...],size_add)...)
    else
        out      = Array{Float64,ndims_in+1}(undef,vcat([xsize...],size_add)...)
    end
    for idx in CartesianIndices(xsize)
        for j in 1:size_add
            out[Tuple(idx)...,j] = x[idx][j]
        end
    end
    return out
end

"""
    unstack_payoff(x::AbstractMatrix)

Unstack a payoff encoded in _long_ format, where the first half of the columns are the action positions for each player and the second half of the columns are the payoff for the various players, to the Nplayers+1 dimensional array used in the library.

# Example
```julia
julia> # 2 players with 2 and 3 actions respectively
       long_payoff = [
            1 1 0.1 0.3;
            1 2 4 6;
            1 3 4 2;
            2 2 4 5;
            2 1 0.1 0.3;
            2 3 1.3 2;];
julia> unstack_payoff(long_payoff)
2×3×2 Array{Float64, 3}:
[:, :, 1] =
 0.1  4.0  4.0
 0.1  4.0  1.3
[:, :, 2] =
 0.3  6.0  2.0
 0.3  5.0  2.0
```
"""
function unstack_payoff(x::AbstractMatrix)
    nCols = size(x,2) 
    nCols % 2 == 0 || error("The matrix should have an even number of column, when the first half are actions ids and the second half are the relative payoffs for the various players")
    nActions = (Int.([maximum(x[:,c]) for c in 1:Int(nCols/2)])...,)
    nPlayers = Int(nCols/2)
    out = Array{Float64,nPlayers+1}(undef,nActions...,nPlayers)
    for r in eachrow(x)
        pos  = Int.(r[1:nPlayers])
        data = r[nPlayers+1:end]
        for n in 1:nPlayers
            out[pos...,n] = data[n]
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
  batch(n,bsize;sequential=false,rng)

Return a vector of `bsize` vectors of indeces from `1` to `n`.
Randomly unless the optional parameter `sequential` is used.

# Example:
```julia
julia> Utils.batch(6,2,sequential=true)
3-element Array{Array{Int64,1},1}:
 [1, 2]
 [3, 4]
 [5, 6]
 ```
"""
function batch(n::Integer,bsize::Integer)
    ridx = collect(1:n)
    if bsize > n
        return [ridx]
    end
    n_batches = Int64(floor(n/bsize))
    batches = Array{Int64,1}[]
    for b in 1:n_batches
        push!(batches,ridx[b*bsize-bsize+1:b*bsize])
    end
    return batches
end


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

Find a Nash Equilibrium for N-players simultaneous games when mixed strategies are allowed using the Complementarity Problem formulation and implementing iterated removal of dominated strategies

# Parameters
- `payoff_array`: the Nplayers+1 dimensional array of payoffs for the N players
- `init`: a vector of vector of mixed strategies (i.e. PMFs) for each players to start the algorithm with. Different init points may reach different equilibrium points [def: equal probabilities for each available action of the players]
- `strict_domination_removal`: wether to remove only strictly dominated strategies in the pre-optimisation or also weakly dominated ones [def: `true`]
- `verbosity`: either `NONE`, `LOW`, `STD` [default], `HIGH` or `FULL`

# Notes
- This function uses a complementarity formulation. For N <= 2 the problem, except the complementarity equation, is linear and known as LCP (Linear Complementarity Problem)
- This implementation uses the JuMP modelling language with the Ipopt solver engine (and hence it uses an interior point method instead of the pivotal approach used in the original Lemke-Howson [1964] algorithm)
- There is no guarantee on timing and even that the algorithm converge to an equilibrium. Different Nash equilibriums may be reached by setting different initial points
- By default the iterative removal of dominated strategies concerns only _strictly_ dominated ones. For some games where the algorithm doesn't find a Nash equilibrium, you can often get success setting the algorithm to remove also weakly dominated strategies. 

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
function nash_cp(payoff;allow_mixed=true,init=[fill(1/size(payoff,d),size(payoff,d)) for d in 1:ndims(payoff)-1],verbosity=STD,ϵ=0.0,strict_domination_removal=true)  
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    (length(nActions) == nPlayers) || error("Mismatch dimension or size between the payoff array and the number of players")

    dominated = dominated_actions(payoff,strict=strict_domination_removal) # TODO: this is not used !!!!

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
    if verbosity <= STD
        set_optimizer_attribute(m, "print_level", 0)
    end

    @variables m begin
        r[idx in idxSet ] >= 0-ϵ
        u[n in 1:nPlayers]
    end
    @variable(m, 0-ϵ <= s[idx in idxSet ] <= 1+ϵ,  start=init[idx[1]][idx[2]])

    for n in 1:nPlayers
        for ds in dominated[n] 
            fix(s[(n,ds)], 0.0; force=true);
        end
    end

    # explicit complementarity constraint not supported by the IPOPT solver:
    # @constraints m begin
    #     slack[idx in idxSet], # either rₙⱼ or sₙⱼ must be zero
    #     complements(r[idx],s[idx])
    # end

    @NLconstraints m begin
        slack[idx in idxSet], # either rₙⱼ or sₙⱼ must be zero
            r[idx] * s[idx] == 0
        utility[idx in idxSet], # the expected utility for each action must be constant, for each nPlayers
            sum(   
                selectdim(selectdim(payoff,nPlayers+1,idx[1]),idx[1],idx[2])[cidx]
                * prod(s[sidx] for sidx in zip(collect(1:nPlayers)[[1:idx[1]-1;idx[1]+1:end]], Tuple(cidx)) )
                for cidx in CartesianIndices(([nActions...][[1:idx[1]-1;idx[1]+1:end]]...,))
            ) +   r[idx] == u[idx[1]]
        probabilities_lb[n in 1:nPlayers],
            sum(s[j] for j in idxSet[setPosByPlayerIdx[n]]) >= 1-ϵ
        probabilities_ub[n in 1:nPlayers],
            sum(s[j] for j in idxSet[setPosByPlayerIdx[n]]) <= 1+ϵ            
    end

    if !allow_mixed
        @NLconstraint(m, pure_only[n in 1:nPlayers], # if we want only pure strategies we add this quadratic constraint to force boolean probabilities
            sum(s[j]^2 for j in idxSet[setPosByPlayerIdx[n]]) >= 1-ϵ
        )
    end

    @objective m Max 1;sum(u[n] for n in 1:nPlayers)

    optimize!(m)
    if verbosity == FULL
        print(m)
    end
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
    restrict_payoff(payoff_n, player;dominated,support)

Compose a sub-payoff considering only non dominated actions and, for the other players, only the actions in the specified support (while for the given player we ignore the support)
"""
function restrict_payoff(payoff_n, player;dominated=[Int64[] for n in 1:ndims(payoff_n)], support=[1:nA for nA in size(payoff_n)])
    nActions = size(payoff_n)
    nPlayers = length(nActions)

    outidx = [Int64[] for i in 1:nPlayers]

    for i in 1:nPlayers
        if i != player
            outidx[i] = intersect(setdiff(1:nActions[i],dominated[i]),support[i])
        else
            outidx[i] = setdiff(1:nActions[i],dominated[i])
        end
    end

    return @view payoff_n[outidx...]
end

"""
    dominated_actions(payoff;strict=true,iterated=true,dominated,support,verbosity)

Implements the "Iterated [by default] Removal of Strictly [by default] Dominated Strategies" (IRSDS) algorithm, returning a vector (for each player) of vectors (action positions) of actions that for a given player are dominates by at least one of his other actions. This function is iterative (recursive) by default.

# Parameters
- `payoff_array`: the Nplayers+1 dimensional array of payoffs for the N players
- `strict`: wheter to look for strictly dominated actions [def: `true`]
- `iterated`: wheter to look for dominated actions iteractively [def: `true`]

- `verbosity`: either `NONE`, `LOW`, `STD` [default], `HIGH` or `FULL`


# Notes
- This function is available also as `dominated_actions(payoff,player;strict)` returning a vector of dominated actions for a given players (computed not iteractively)
- The function has also two arguments that are internally used, `dominated`, a vector of vectors of actions that can be skip-checked as already deemed as dominated (default to empty vectors) and `support`, a vector of indices reppresenting the support profile to consider, defaulting to all actions by all players.
- The former is used by this same function to recursivelly look-up for dominated actions, the later is of importance when the function is called by algorithms that work over a specific support. While for looking at the actions of other players these two function arguments work similarly (I can skip considering actions that are either dominated or not in their supports) the difference is for the own player actions, where I can skip checking dominated/not in the support actions, but I can consider in the comparation also actions not in the specific support I am looking at. 
- To get the list of retrieved dominated actions at each iteration, use a verbosity level higher than `STD` 

# Example
```julia
julia> using StrategicGames
julia> payoff = [(13,3) (1,4) (7,3); (4,1) (3,3) (6,2); (-1,9) (2,8) (8,-1)]
3×3 Matrix{Tuple{Int64, Int64}}:
 (13, 3)  (1, 4)  (7, 3)
 (4, 1)   (3, 3)  (6, 2)
 (-1, 9)  (2, 8)  (8, -1)
julia> payoff_array = expand_dimensions(payoff);
julia> dominated_player2 = dominated_actions(payoff_array,2)
1-element Vector{Int64}:
 3
julia> dominated = dominated_actions(payoff_array,verbosity=HIGH)
Dominated strategies at step 1: [Int64[], [3]]
Dominated strategies at step 2: [[3], [3]]
Dominated strategies at step 3: [[3], [3, 1]]
Dominated strategies at step 4: [[3, 1], [3, 1]]
Dominated strategies at step 5: [[3, 1], [3, 1]]
2-element Vector{Vector{Int64}}:
 [3, 1]
 [3, 1]
```
"""
function dominated_actions(payoff;strict=true,iterated=true,dominated=[Int64[] for n in 1:size(payoff)[end]],support=[1:nA for nA in size(payoff)[1:end-1]],iteration=1,allow_mixed=true,tol=1e-06,solver="HiGHS",verbosity=STD)

    #println(support)
    nActions      = size(payoff)[1:end-1]
    nPlayers      = size(payoff)[end]
    new_dominated = false
    mask          = fill(true,size(payoff))
    mask_support  = fill(false,size(payoff))
    for n in 1:nPlayers
        selectdim(mask,n,dominated[n]) .= false
    end
    mask_support[support...,:] .= true
    #println(mask)
    
    for n in 1:nPlayers
        payoff_n        = selectdim(payoff,nPlayers+1,n)
        mask_n          = selectdim(mask,nPlayers+1,n)
        mask_support_n  = selectdim(mask_support,nPlayers+1,n)
        dominated_n     = dominated[n]
        #println(mask_support_n)
        #println(payoff_n)
        #println(mask_n)
        #println(dominated_n)
        for i in support[n] # we check only actions in the support
            #println("i: $i")
            i in dominated_n && continue # we don't check this action, it is already deemed as dominated
            if allow_mixed == true
                if is_mixdominated(payoff_n,n,i,dominated=dominated, support=support, solver=solver, tol=tol, strict=strict, verbosity=verbosity)
                    push!(dominated_n,i)
                    new_dominated = true
                end 
            else
                ai = selectdim(payoff_n,n,i) # action to test 
                maski         = .! selectdim(mask_n,n,i)
                maski_support = .! selectdim(mask_support_n,n,i)
                for j in 1:nActions[n] # we check for any possible dominating action, even outside support
                    #println("j: $j")
                    i != j || continue
                    maskj         = .! selectdim(mask_n,n,j)
                    #maskj_support =  selectdim(mask_support_n,n,j) # as this action to check may be outside support, maskj_support could be all false
                    j in dominated_n && continue # we don't check this action, it is already deemed as dominated
                    maski == maskj || error("Maski should always be equal to maskj!\n$maski \n$maskj")
                    aj = selectdim(payoff_n,n,j)           
                    check_dominated = strict ? ( (aj .> ai) .|| maski .|| maski_support ) : (aj .>= ai .|| maski .|| maski) # we don't care if action j is not higher than action i for other player actions that are dominated or not in the support
                    if all(check_dominated)
                        push!(dominated_n,i)
                        new_dominated = true
                        break
                    end 
                end
            end
        end
        #dominated[n] = dominated_n # maybe not even needed 
        sort!(dominated_n) # not really needed but nice to have them in order
    end
    if verbosity > STD
        println("Dominated strategies at step $iteration: $dominated")
    end
    if (new_dominated && iterated)
        return dominated_actions(payoff,strict=strict,iterated=true,dominated=dominated,support=support,allow_mixed=allow_mixed,tol=tol,solver=solver,verbosity=verbosity,iteration=iteration+1)
    else
        return dominated
    end
end
function dominated_actions(payoff,player;support=[1:nA for nA in size(payoff)[1:end-1]],strict=true, allow_mixed=true,tol=1e-06,solver="HiGHS",verbosity=STD)
    return dominated_actions(payoff;strict=strict,support=support,iterated=false, allow_mixed=allow_mixed,tol=tol,solver=solver,verbosity=verbosity)[player]
end

"""
Solver: "HiGHS", "Ipopt", "GLPK"
"""
function is_mixdominated(payoff_n, player, action ; dominated=[Int64[] for n in 1:ndims(payoff_n)], support=[1:nA for nA in size(payoff_n)], solver="HiGHS", tol =1e-06, strict=true, verbosity=STD)
 

    if action in dominated[player]
        return true
    end

    # we restrict the payoff to domain/undominated action and we remap the action id
    payoff_n_res = restrict_payoff(payoff_n,player,dominated=dominated,support=support)

    action = action - sum(dominated[player] .< action)
    nActions = size(payoff_n_res)
   
    #nPlayers = ndims(payoff_n_res)
    nActions_player = nActions[player]
    if(nActions_player == 1)
        return false # only action can't be dominated
    end

    oth_actions   = setdiff(1:nActions_player,action)
    payoff_n_a    = selectdim(payoff_n_res,player,action)
    payoff_n_otha = [selectdim(payoff_n_res,player,i) for i in oth_actions]

    nOthActions = length(oth_actions)

    m = Model(getfield(eval(Symbol(solver)),:Optimizer))
    if verbosity <= STD
        solver == "HiGHS" && set_optimizer_attribute(m, "output_flag", false)
        solver == "Ipopt" && set_optimizer_attribute(m, "print_level", 0)
        solver == "GLPK"  && set_optimizer_attribute(m, "msg_lev", GLPK.GLP_MSG_OFF)
    else
        solver == "HiGHS" && set_optimizer_attribute(m, "output_flag", true)
        solver == "Ipopt" && set_optimizer_attribute(m, "print_level", 5)
        solver == "GLPK"  && set_optimizer_attribute(m, "msg_lev", GLPK.GLP_MSG_ALL)
    end

    @variable(m, 0 <= p[j in 1:nOthActions ] <= 1, start=1/nOthActions)
    @variable(m, dist >= -1)
    @constraint(m, probability, sum(p) == 1)

    for idx in CartesianIndices(size(payoff_n_a))
        @constraint(m, sum(p[j] .* payoff_n_otha[j][idx] for j in 1:nOthActions) + dist >= payoff_n_a[idx] )
    end

    @objective m Min dist 
    if verbosity >= FULL
        print(m)
    end
    optimize!(m)

    status = termination_status(m)
    if !(status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL]) || ! has_values(m)
        @error("Error in computing dominance check at player $player")
        return false
    else
        optDist = value.(dist)
        ps = value.(p)
        if verbosity >= FULL
            println("optDist: ", optDist)
            println("p: ", ps)
        end
        if strict
            return (optDist < -tol )
        else
            return optDist <= 0.0+eps()
        end
    end
end

function check_domination_2p(payoff,player)
    nPlayers = size(payoff)[end]
    nPlayers == 2 || error("This function supports only 2 players")
    payoff_n = selectdim(payoff,nPlayers+1,player)
    if player == 1
        for r in eachrow(payoff_n)
            for r2 in eachrow(payoff_n)
                if all(r2 .> r) # r2 strictly dominates r1 
                    return true
                end
            end
        end
    else
        for c in eachcol(payoff_n)
            for c2 in eachcol(payoff_n)
                if all(c2 .> c) # c2 strictly dominates c1 
                    return true
                end
            end
        end
    end
    return false
end

"""
    best_response(payoff_array,strategy_profile,nplayer;solver)

Return (possibly one of many) best strategy and corrsponding expected payoff for a given player.

# Parameters:
- `payoff_array`: the N_players+1 dimensional array of payoffs
- `strategy_profile`: the vector of vectors defining the strategies for the N players (i.e. vectors of the probabilities for each action for each player). The strategy for player n for which the best response is computed is used as initial values in the inner optimisation
- `nplayer`: counter of the player for which we want to compute the best_response (e.g. 1 or 3)
- `solver`: currently either "GLPK" or "HiGHS" [default]
- `verbosity`: either `NONE`, `LOW`, `STD` [default], `HIGH` or `FULL`

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
function best_response(payoff,strategy_profile,player;solver="HiGHS",verbosity=STD)
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    payoff_n = selectdim(payoff,nPlayers+1,player)
    init = strategy_profile[player]
    m = Model(getfield(eval(Symbol(solver)),:Optimizer))
    if solver == "HiGHS" && verbosity <= STD
        set_optimizer_attribute(m, "output_flag", false)
    end

    @variable(m, 0 <= s[j in 1:nActions[player] ] <= 1,  start=init[j])
    @constraints m begin
        probabilities,
            sum(s[j] for j in 1:nActions[player]) == 1
    end
    @objective m Max sum(
        [*(payoff_n[idx],[strategy_profile[n][idx[n]] for n in [1:player-1;player+1:nPlayers]]...,s[idx[player]]) for idx in CartesianIndices(nActions)]  
    )
    optimize!(m)
    if verbosity == FULL
        print(m)
    end
    status = termination_status(m)
    
    return (expected_payoff=objective_value(m), optimal_strategy=value.(s), status=status)
end

"""
    is_best_response(payoff_array,strategy_profile,nplayer;atol=1e-07,rtol=1e-07,solver,verbosity=STD)

Determine if a given strategy for player `nplayer` is a best response to a given payoff array and strategies of the other players

# Parameters:
- `payoff_array`: the Nplayers+1 dimensional array of payoffs
- `strategy_profile`: the vector of vectors defining the strategies for the N players
- `nplayer`: counter of the player for which we want to verify if its strategy is a best_response (e.g. 1 or 3)
- `atol`: absolute tollerance in comparing the expected payoff from the given strategy and those from the optimal one [def: `1e-07`]
- `rtol`: relative tollerance in comparing the expected payoff from the given strategy and those from the optimal one [def: `1e-07`]
- `solver`: currently either "GLPK" or "HiGHS"
- `verbosity`: either `NONE`, `LOW`, `STD` [default], `HIGH` or `FULL`

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
function is_best_response(payoff,strategy_profile,player;atol=1e-07,rtol=1e-07,solver="GLPK",verbosity=STD)
    best_u = best_response(payoff,strategy_profile,player,solver=solver,verbosity=verbosity).expected_payoff
    this_u = expected_payoff(payoff,strategy_profile,player)
    return isapprox(this_u,best_u,atol=atol,rtol=rtol)
end

"""
    is_nash(payoff_array,strategy_profile;atol=1e-07,rtol=1e-07,solver,verbosity)

Determine if a strategy profile is a Nash equilibrium for a given payoff array, i.e. if all strategies are (at least weak) best responses.

# Parameters:
- `payoff_array`: the Nplayers+1 array of payoffs
- `strategy_profile`: the vector of vectors defining the strategies for the N players
- `atol`: absolute tollerance in comparing the expected payoffs from the given strategies and those from the optimal ones [def: `1e-07`]
- `rtol`: relative tollerance in comparing the expected payoffs from the given strategies and those from the optimal ones [def: `1e-07`]
- `solver`: currently either "GLPK" or "HiGHS"
- `verbosity`: either `NONE`, `LOW`, `STD` [default], `HIGH` or `FULL`

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
function is_nash(payoff,strategy_profile;atol=1e-07,rtol=1e-07,solver="GLPK",verbosity=STD)
    all([isapprox( best_response(payoff,strategy_profile,i,solver=solver,verbosity=verbosity).expected_payoff,
                   expected_payoff(payoff,strategy_profile,i),
                   atol=atol,rtol=rtol) for i in 1:ndims(payoff)-1])
end

"""
    nash_on_support(payoff_array,support;init,verbosity)

Find (if it exists) a Nash equilibrium for N-players simultaneous games when mixed strategies are allowed on a specific support.

# Parameters
- `payoff_array`: the Nplayers+1 dimensional array of payoffs for the N players
- `support`: vector of vector of action counts that are in the tested support for each player [def: full support]
- `init`: a vector of vector of mixed strategies (i.e. PMFs) for each players to start the algorithm with. Different init points may reach different equilibrium points [def: equal probabilities for each available action of the players]
- `verbosity`: either `NONE`, `LOW`, `STD` [default], `HIGH` or `FULL`

# Notes
- This implementation uses the JuMP modelling language with the Ipopt solver engine

# Returns
- A named tuple with the following elements: `status`,`equilibrium_strategies`,`expected_payoffs`, `solved`

# Example
```julia
julia> using StrategicGames
julia> payoff = [(-1,-1) (-3,0); (0, -3) (-2, -2)] # prisoner's dilemma. Only Nash eq is [[0,1],[0,1]]
2×2 Matrix{Tuple{Int64, Int64}}:
 (-1, -1)  (-3, 0)
 (0, -3)   (-2, -2)
julia> payoff_array = expand_dimensions(payoff);
julia> nash_on_support(payoff_array,[[1,2],[1,2]]).solved # false
false
julia> nash_on_support(payoff_array,[[2],[2]]).solved     # true
true
```
"""
function nash_on_support(payoff,support= collect.(range.(1,size(payoff)[1:end-1]));verbosity=STD)  
    verbosity == FULL && println("Looking for NEq on support: $support")
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    (length(nActions) == nPlayers) || error("Mismatch dimension or size between the payoff array and the number of players")
    if isempty(support)
        support = [collect(1:nActions[d]) for d in 1:length(nActions)]
    end
    init = [fill(0.0,size(payoff,d)) for d in 1:ndims(payoff)-1]
    [[init[d][j] = 1/length(support[d]) for j in support[d]] for d in 1:length(nActions)]
    support_flag = [fill(false,size(payoff,d)) for d in 1:ndims(payoff)-1]
    [[support_flag[d][j] = true for j in support[d]] for d in 1:length(nActions)]

    # Vector of 2-elements tuples where the first one is the player index and the second one is the action index
    # We specify it as a vector of tuples as the number of actions can be different for the different players
    # Here we create 3 versions: full (check if needed), in support, and not in support
    idxSet  = Vector{Tuple{Int64,Int64}}()
    idxSet1 = Vector{Tuple{Int64,Int64}}() # index of combinations in support
    idxSet0 = Vector{Tuple{Int64,Int64}}() # index of combinations out of support
    setPosByPlayerIdx = Vector{Vector{Int64}}()
    setPosByPlayerIdx1 = Vector{Vector{Int64}}()
    setPosByPlayerIdx0 = Vector{Vector{Int64}}()
    pos  = 1
    pos1 = 1
    pos0 = 1
    for np in 1:nPlayers
        push!(setPosByPlayerIdx,collect(pos:pos+nActions[np]-1))
        setPosByPlayerIdx1_n = Int64[]
        setPosByPlayerIdx0_n = Int64[]
        for na in 1:nActions[np]
            push!(idxSet,(np,na))
            pos += 1
            if support_flag[np][na]
                push!(idxSet1,(np,na))
                push!(setPosByPlayerIdx1_n,pos1)
                pos1 += 1
            else
                push!(idxSet0,(np,na))
                push!(setPosByPlayerIdx0_n,pos0)
                pos0 += 1
            end 
        end
        push!(setPosByPlayerIdx1,setPosByPlayerIdx1_n)          
        push!(setPosByPlayerIdx0,setPosByPlayerIdx0_n)
    end

    
    #println("idxSet: \n$idxSet")
    #println("idxSet0: \n$idxSet0")
    #println("idxSet1: \n$idxSet1")

    m = Model(Ipopt.Optimizer)
    if verbosity < FULL
        set_optimizer_attribute(m, "print_level", 0)
    end
 
    @variables m begin
        u[n in 1:nPlayers]
    end
    @variable(m, 0 <= s[idx in idxSet ] <= 1,  start=init[idx[1]][idx[2]])
    for idx0 in idxSet0 
        fix(s[idx0], 0.0; force=true);
    end

    @NLconstraints m begin
        
        utility1[idx in idxSet1], # the expected utility for each action in the support must be constant, for each nPlayers
            sum(   
                selectdim(selectdim(payoff,nPlayers+1,idx[1]),idx[1],idx[2])[cidx]
                 * prod(s[sidx] for sidx in zip(collect(1:nPlayers)[[1:idx[1]-1;idx[1]+1:end]], Tuple(cidx))   )
                for cidx in CartesianIndices(([nActions...][[1:idx[1]-1;idx[1]+1:end]]...,)) 
            )  == u[idx[1]]
        utility0[idx in idxSet0], # the expected utility for each action not in the support must be constant, for each nPlayers
            sum(   
                selectdim(selectdim(payoff,nPlayers+1,idx[1]),idx[1],idx[2])[cidx]
                * prod(s[sidx] for sidx in zip(collect(1:nPlayers)[[1:idx[1]-1;idx[1]+1:end]], Tuple(cidx)) )
                for cidx in CartesianIndices(([nActions...][[1:idx[1]-1;idx[1]+1:end]]...,)) #if Tuple(cidx) in idxSet1
            ) <= u[idx[1]]      
      
        probabilities[n in 1:nPlayers],
            sum(s[j] for j in idxSet1[setPosByPlayerIdx1[n]]) == 1
    end

    @objective m Max sum(u[n] for n in 1:nPlayers)
    if verbosity == FULL
        println("Optimisation model to be solved:")
        println(m)
    end
    optimize!(m)
    status = termination_status(m)

    optStrategies = Vector{Vector{Float64}}()
    optU          = Float64[]
    solved = false
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
        solved = true
    elseif (status in [MOI.LOCALLY_INFEASIBLE, MOI.INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE,MOI.INFEASIBLE_OR_UNBOUNDED, MOI.DUAL_INFEASIBLE])
        solved = false
    else
        if verbosity >= STD
            @warn "The feasibility check for support $support returned neither solved neither unsolved ($status). Returning no Nash equilibrium for this support."
        end
        solved = false
    end
    return (status=status,equilibrium_strategies=optStrategies,expected_payoffs=optU,solved)
end


"""
    nash_on_support_2p(payoff_array,support;init,verbosity)

Find (if it exists) a Nash equilibrium for 2-players simultaneous games when mixed strategies are allowed on a specific support. This is a specialised 2-players version of the [`nash_on_support`](@ref) function.

# Parameters
- `payoff_array`: the 3 dimensional array of payoffs for the 2 players
- `support`: vector of vector of action counts that are in the tested support for each player [def: full support]
- `init`: a vector of vector of mixed strategies (i.e. PMFs) for each players to start the algorithm with. Different init points may reach different equilibrium points [def: equal probabilities for each available action of the players]
- `solver`: the linear solver to use. Currently only "HiGHS" [default] is supported, as "GLPK" seems to have problems as this function is multithreaded
- `verbosity`: either `NONE`, `LOW`, `STD` [default], `HIGH` or `FULL`

# Notes
- This implementation uses the JuMP modelling language with either the HiGHS solver engine

# Returns
- A named tuple with the following elements: `status`,`equilibrium_strategies`,`expected_payoffs`, `solved`

# Example
```julia
julia> using StrategicGames
julia> payoff = [(-1,-1) (-3,0); (0, -3) (-2, -2)] # prisoner's dilemma. Only Nash eq is [[0,1],[0,1]]
2×2 Matrix{Tuple{Int64, Int64}}:
 (-1, -1)  (-3, 0)
 (0, -3)   (-2, -2)
julia> payoff_array = expand_dimensions(payoff);
julia> nash_on_support2p(payoff_array,[[1,2],[1,2]]).solved # false
false
julia> nash_on_support2p(payoff_array,[[2],[2]]).solved     # true
true
```
"""
function nash_on_support_2p(payoff,support= collect.(range.(1,size(payoff)[1:end-1]));solver="HiGHS",verbosity=STD)  
    #println("boo")
    verbosity == FULL && println("Looking for NEq on support: $support")
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    nPlayers == 2 || error("This function works only for 2 players games")
    (length(nActions) == nPlayers) || error("Mismatch dimension or size between the payoff array and the number of players")
    if isempty(support)
        support = [collect(1:nActions[d]) for d in 1:length(nActions)]
    end
    init=[fill(1/size(payoff,1), size(payoff,1)),fill(1/size(payoff,2), size(payoff,2))]

    m = Model(getfield(eval(Symbol(solver)),:Optimizer))
    if solver == "HiGHS" && verbosity <= STD
        set_optimizer_attribute(m, "output_flag", false)
    end
    @variables m begin
        u[n in 1:2]
    end
    @variable(m, 0-eps() <= s1[j in 1:nActions[1] ] <= 1,  start=init[1][j])
    @variable(m, 0-eps() <= s2[j in 1:nActions[2] ] <= 1,  start=init[2][j])
    for Σ1 in setdiff(1:nActions[1],support[1])
        fix(s1[Σ1], 0.0; force=true);
    end
    for Σ2 in setdiff(1:nActions[2],support[2])
        fix(s2[Σ2], 0.0; force=true);
    end

    @constraints m begin
        utility1_insupport[σ1 in support[1]], # the expected utility for each action in the support of player 1 must be constant
            sum( payoff[σ1,j2,1] * s2[j2] for j2 in 1:nActions[2] )  == u[1]
        utility1_outsupport[Σ1 in setdiff(1:nActions[1],support[1]) ], # the expected utility for each action not in the support of player 1 must be lower than the costant utility above
            sum( payoff[Σ1,j2,1] * s2[j2] for j2 in 1:nActions[2] )  <= u[1]
        utility2_insupport[σ2 in support[2]], # the expected utility for each action in the support of player 2 must be constant
            sum( payoff[j1,σ2,2] * s1[j1] for j1 in 1:nActions[1] )  == u[2]
        utility2_outsupport[Σ2 in setdiff(1:nActions[2],support[2])], # the expected utility for each action not in the support of player 2 must be lower than the costant utility above
            sum( payoff[j1,Σ2,2] * s1[j1] for j1 in 1:nActions[1] )  <= u[2]
        probabilities1,
            sum(s1[j] for j in 1:nActions[1]) == 1
        probabilities2,
            sum(s2[j] for j in 1:nActions[2]) == 1
    end

    @objective m Max u[1] + u[2]

    if verbosity == FULL
        println("Optimisation model to be solved:")
        println(m)
    end
    optimize!(m)
    status = termination_status(m)

    optStrategies = Vector{Vector{Float64}}()
    optU          = Float64[]
    solved = false
    if (status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL]) && has_values(m)
        optStrategies1 = value.(s1)
        optStrategies2 = value.(s2)
        optStrategies = [optStrategies1,optStrategies2]
        optU = value.(u)
        solved = true
    elseif (status in [MOI.LOCALLY_INFEASIBLE, MOI.INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE,MOI.INFEASIBLE_OR_UNBOUNDED, MOI.DUAL_INFEASIBLE])
        solved = false
    else
        if verbosity >= STD
            @warn "The feasibility check for support $support returned neither solved neither unsolved ($status). Returning no Nash equilibrium for this support."
        end
        solved = false
    end
    return (status=status,equilibrium_strategies=optStrategies,expected_payoffs=optU,solved)
end

"""
    nash_se2(payoff; allow_mixed=true, max_samples=1, verbosity=STD)

ONLY FOR BENCHMARKS, UNEXPORTED
Solves Nash eqs using support enumeration for 2 players game using strictly the approach of [Porter-Nudelman-Shoham (2008)](https://doi.org/10.1016/j.geb.2006.03.015)
"""
function nash_se2(payoff; allow_mixed=true, max_samples=1, verbosity=STD)
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    nPlayers == 2 || error("This function works only for 2 players games")
    nSupportSizes = allow_mixed ? prod(nActions) : 1
    eqs = NamedTuple{(:equilibrium_strategies, :expected_payoffs, :supports), Tuple{Vector{Vector{Float64}}, Vector{Float64},Vector{Vector{Int64}}}}[]

    support_sizes = Matrix{Union{Int64,NTuple{nPlayers,Int64}}}(undef,nSupportSizes,3) # sum, diff, support sizes

    if allow_mixed
        i = 1
        for idx in CartesianIndices(nActions)
            support_sizes[i,:] = [sum(Tuple(idx)),maximum(Tuple(idx))-minimum(Tuple(idx)),Tuple(idx)] 
            i += 1
        end
    else
        support_sizes[1,:] = [nPlayers,0,(ones(Int64,nPlayers)...,)]
    end
    if nPlayers <= 2
        support_sizes = sortslices(support_sizes,dims=1,by=x->(x[2],x[1]))
    else
        support_sizes = sortslices(support_sizes,dims=1,by=x->(x[1],x[2]))
    end   

    for support_size in eachrow(support_sizes)
        for S1 in combinations(1:nActions[1],support_size[3][1])
            A2 = setdiff(1:nActions[2],dominated_actions(payoff[S1,:,:],2))
            #if !isempty(dominated_actions(payoff[S1,A2,:],1))
            #    continue
            #end
            if check_domination_2p(payoff[S1,A2,:],1)
                continue
            end

            for S2 in combinations(A2,support_size[3][2])
                #if !isempty(dominated_actions(payoff[S1,S2,:],1))
                #        continue
                #end
                if check_domination_2p(payoff[S1,S2,:],1)
                    continue
                end
                eq_test =  nash_on_support_2p(payoff,[S1,S2],verbosity=verbosity)
                if eq_test.solved
                        push!(eqs,(equilibrium_strategies=eq_test.equilibrium_strategies, expected_payoffs=eq_test.expected_payoffs,supports=[S1,S2]))
                        if length(eqs) == max_samples
                            return eqs
                        end
                end
            end
        end 
    end
    return eqs
end

function eq_on_support(payoff,S,verbosity)
    if all(isempty.(dominated_actions(payoff,iterated=false,support=S))) # if there is even a single dominated strategy, this can not be a valid support for a Nash eq
        nPlayers = size(payoff)[end]
        eq_test = (nPlayers == 2) ? nash_on_support_2p(payoff,S,verbosity=verbosity) : nash_on_support(payoff,S,verbosity=verbosity)
        if eq_test.solved
            eq = (equilibrium_strategies=eq_test.equilibrium_strategies, expected_payoffs=eq_test.expected_payoffs,supports=S)
            return eq
        else
            return nothing
        end
    end
end    

"""
    nash_se(payoff_array; allow_mixed=true, max_samples=1, isolated_eq_only=true, mt=true, verbosity=STD)

Compute `max_samples` (default one) Nash equilibria for a N-players generic game in normal form using support enumeration method.

# Parameters
- `payoff_array`: the Nplayers+1 dimensional array of payoffs for the N players
- `allow_mixed`: wether to look and report also mixed strategies (default) or look only for pure strategies (if any)
- `max_samples`: number of found sample Nash equilibria needed to stop the algorithm [def: `1`]. Set it to `Inf` to look for all the possible isolated equilibria of the game
- `mt`: wheter to use multithreads (def: `true`). Note that currently multithreading is always disable for a single eq search due to performance issues
- `isolated_eq_only`: wheter to look only for isolated equilibria (def: true)
- `verbosity`: either `NONE`, `LOW`, `STD` [default], `HIGH` or `FULL`

# Notes
- This function uses a support enumeration method to avoid the complementarity conditions and solve simpler problems conditional to a specific support.  More specifically we use the heuristic of [Porter-Nudelman-Shoham (2008)](https://doi.org/10.1016/j.geb.2006.03.015) and a dominance check, altought not recursively as in  [Turocy (2007)](https://web.archive.org/web/20230401080619/https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=36ba7977838df2bfdeab22157b0ed6ce940fb2be)
- To reduce computational costs, in 2 players game by default only (and all) existing isolated Nash equilibria that are unique for a given support are returned by this function. To retrieve a "sample" of equilbria that are not isolated but represent instead a continuum in degenerated games in two players games you can set `isolated_eq_only` option to false. This will allow the search to eq within supports of different sizes for the two players. This option is always false for more than 2 players.

# Returns
- A vector of named tuples (even for the default single `max_samples`) with the following information: `equilibrium_strategies`, `expected_payoffs`, `supports` 

# Example
```julia
julia> using StrategicGames
julia> payoff = expand_dimensions([(3,3) (3,2);
                                   (2,2) (5,6);
                                   (0,3) (6,1)]);
julia> eqs = nash_se(payoff,max_samples=Inf)
3-element Vector{NamedTuple{(:equilibrium_strategies, :expected_payoffs, :supports), Tuple{Vector{Vector{Float64}}, Vector{Float64}, Vector{Vector{Int64}}}}}:
 (equilibrium_strategies = [[0.9999999999999999, 0.0, 0.0], [0.9999999999999999, 0.0]], expected_payoffs = [2.9999999999999516, 2.9999999999999516], supports = [[1], [1]])
 (equilibrium_strategies = [[0.8, 0.2, 0.0], [0.6666666666666666, 0.33333333333333337]], expected_payoffs = [3.0, 2.8000000000000003], supports = [[1, 2], [1, 2]])
 (equilibrium_strategies = [[0.0, 0.33333333333333337, 0.6666666666666666], [0.33333333333333315, 0.6666666666666669]], expected_payoffs = [4.000000000000001, 2.6666666666666665], supports = [[2, 3], [1, 2]])
```
"""
function nash_se(payoff; allow_mixed=true, max_samples=1, isolated_eq_only=true, mt=true, verbosity=STD)
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    nSupportSizes = allow_mixed ? ( (nPlayers >2 || !isolated_eq_only) ? prod(nActions) : minimum(nActions) ) : 1
    
    eqs = NamedTuple{(:equilibrium_strategies, :expected_payoffs, :supports), Tuple{Vector{Vector{Float64}}, Vector{Float64},Vector{Vector{Int64}}}}[]

    mt = ( max_samples == 1 ) ? false : mt # as  mutithreaded for a single eq is currently very slow

    support_sizes = Matrix{Union{Int64,NTuple{nPlayers,Int64}}}(undef,nSupportSizes,3) # sum, diff, support sizes
    if allow_mixed
        if nPlayers>2 || !isolated_eq_only
            i = 1
            for idx in CartesianIndices(nActions)
                support_sizes[i,:] = [sum(Tuple(idx)),maximum(Tuple(idx))-minimum(Tuple(idx)),Tuple(idx)] 
                i += 1
            end
        else
            for i in 1:minimum(nActions)
                support_sizes[i,:] = [i*nPlayers,0,(i,i)]
            end
        end
    else
        support_sizes[1,:] = [nPlayers,0,(ones(Int64,nPlayers)...,)]
    end
    if nPlayers <= 2
        support_sizes = sortslices(support_sizes,dims=1,by=x->(x[2],x[1]))
    else
        support_sizes = sortslices(support_sizes,dims=1,by=x->(x[1],x[2]))
    end   

    for support_size in eachrow(support_sizes)
        support_size = support_size[3]   # tuple of a specific set of support sizes
        D = [combinations(1:nActions[n],support_size[n]) for n in 1:nPlayers] # vector for each player of its possible way to get the specific support size of actions for him out of all his possible nActions[n]
        #println(D)

        nSupportsToTest  = *(length.(D)...)
        eq_still_to_find = Int(min(maxintfloat(Float64),max_samples)-length(eqs))
        #eq_still_to_find = 0 # force single thread algo
        
        # This could be reunited using Floops: https://juliafolds.github.io/FLoops.jl/dev/tutorials/parallel/
        # https://discourse.julialang.org/t/programming-pattern-for-when-you-want-thread-race/97457
        
        
        if max_samples == 1 && mt # multithreaded first find the treasure race....
            eqs_ssize = Array{NamedTuple{(:equilibrium_strategies, :expected_payoffs, :supports), Tuple{Vector{Vector{Float64}}, Vector{Float64},Vector{Vector{Int64}}}}}(undef,nSupportsToTest)
            eqs_ssize_mask = fill(false, nSupportsToTest)
            idxs_supports  = reshape([idx for idx in CartesianIndices((length.(D)...,))],nSupportsToTest)
            lk = ReentrantLock()
            found = true
            Threads.@threads for i in 1:length(idxs_supports)
                idx = idxs_supports[i]
                #println(idx)
                S = [collect(D[n])[idx[n]] for n in 1:nPlayers]
                eq = eq_on_support(payoff,S,verbosity)
                #println(eq)
                if !isnothing(eq)
                    eqs_ssize[i] = eq
                    eqs_ssize_mask[i] = true
                    lock(lk) do 
                        found = true
                    end
                end
                if !found
                    break
                end
            end
            if any(eqs_ssize_mask) # could well noone if found on this support size and we need to look for another one...
                return eqs_ssize[eqs_ssize_mask]
            end      
        elseif (!mt) || eq_still_to_find < nSupportsToTest # we go sequencing single trheading...
            for idx in CartesianIndices((length.(D)...,))
                #println(idx)
                S = [collect(D[n])[idx[n]] for n in 1:nPlayers]
                #println(S)
                eq = eq_on_support(payoff,S,verbosity)
                if !isnothing(eq)
                    push!(eqs,eq)
                    if verbosity > STD
                        println("Eq #$(length(eqs))/$max_samples found: $eq")
                    end
                    if length(eqs) == max_samples
                        return eqs
                    end
                end
            end
        else   # we run multithreaded code
            eqs_ssize = Array{NamedTuple{(:equilibrium_strategies, :expected_payoffs, :supports), Tuple{Vector{Vector{Float64}}, Vector{Float64},Vector{Vector{Int64}}}}}(undef,nSupportsToTest)
            eqs_ssize_mask = fill(false, nSupportsToTest)
            idxs_supports  = reshape([idx for idx in CartesianIndices((length.(D)...,))],nSupportsToTest)
            #GC.gc()
            Threads.@threads for i in 1:length(idxs_supports)
                idx = idxs_supports[i]
                # batch here min the number of eq remaining and the number of threads
                #println(idx)
                S = [collect(D[n])[idx[n]] for n in 1:nPlayers]
                eq = eq_on_support(payoff,S,verbosity)
                if !isnothing(eq)
                    eqs_ssize[i] = eq
                    eqs_ssize_mask[i] = true
                end
            end
            eqs_ssize_final = eqs_ssize[eqs_ssize_mask]
            #println(eqs_ssize_final)
            if verbosity > STD
                for (i,eq) in enumerate(eqs_ssize_final)
                    println("Eq #$(length(eqs)+i)/$max_samples found: $eq")
                end
            end
            append!(eqs,eqs_ssize_final)
            if length(eqs) >= max_samples
                return eqs
            end
        end # mutithread case
    end # end analysis of this support size
    return eqs
end

include("Precompilation.jl") 

end # module StrategicGames
