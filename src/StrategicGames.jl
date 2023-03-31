"""
    StrategicGames package

Strategic Games provides functionalities to work with strategic games, including finding Nash equilibrium in simultaneous games when mixed strategies are allowed. 

Unless otherwise stated, all functions work with generic n players (the examples generally show 2 players for simplicity) and assume the payoff "matrix" to be in the form of N-players + 1 dimensional arrays, where each dimension except the last one is given by the number of discrete actions available to each player and the last dimension is given by the number of players.
A convenient function allows to transform a N-dimensional array of tuples (the payoff for the various players) to the Nplayers+1 dimensional array


"""
module StrategicGames

using LinearAlgebra, Combinatorics, JuMP, Ipopt, GLPK, HiGHS

export Verbosity, NONE, LOW, STD, HIGH, FULL
export expand_dimensions
export expected_payoff
export dominated_strategies, best_response, is_best_response, is_nash
export nash_on_support
export nash_cp, nash_se


"""
    Verbosity

Many functions accept a `verbosity` parameter.

Choose between: `NONE`, `LOW`, `STD` [default], `HIGH` and `FULL`.

Under default verbosity (`STD`) no output is print unless something unexpected in most conditions (but not necessarily an error) is detected 
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

    dominated = dominated_strategies(payoff,strict=strict_domination_removal)

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
    dominated_strategies(payoff;strict=true,iterated=true,dominated,support,verbosity)

Implements the "Iterated [by default] Removal of Strictly [by default] Dominated Strategies" (IRSDS) algorithm, returning a vector (for each player) of vectors (action positions) of actions that for a given player are dominates by at least one of his other actions. This function is iterative (recursive) by default.

# Parameters
- `payoff_array`: the Nplayers+1 dimensional array of payoffs for the N players
- `strict`: wheter to look for strictly dominated actions [def: `true`]
- `iterated`: wheter to look for dominated actions iteractively [def: `true`]
- `dominated`: vector of vectors of actions to skip check as already deemed as dominated
- `support`: support profile to consider (vector of vectors, see notes). Default all actions
- `verbosity`: either `NONE`, `LOW`, `STD` [default], `HIGH` or `FULL`


# Notes
- This function is available also as `dominated_strategies(payoff,player;strict)` returning a vector of dominated strategies for a given players (computed not iteractively)
- Whae a particolar support is specified, the function looks for dominated strategies within the support for a given player, but considering also for possibly dominating actions outside its support. In both cases only the actions in the supports of the other players are considered.
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
julia> dominated_player2 = dominated_strategies(payoff_array,2)
1-element Vector{Int64}:
 3
julia> dominated = dominated_strategies(payoff_array,verbosity=HIGH)
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
function dominated_strategies(payoff;strict=true,iterated=true,dominated=[Int64[] for n in 1:size(payoff)[end]],support=[1:nA for nA in size(payoff)[1:end-1]],verbosity=STD,iteration=1)

    #println(support)
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    new_dominated = false
    mask  = fill(true,size(payoff))
    mask_support = fill(false,size(payoff))
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
        #dominated[n] = dominated_n # maybe not even needed 
        sort!(dominated_n) # not really needed but nice to have them in order
    end
    if verbosity > STD
        println("Dominated strategies at step $iteration: $dominated")
    end
    if (new_dominated && iterated)
        return dominated_strategies(payoff,strict=strict,iterated=true,dominated=dominated,support=support,verbosity=verbosity,iteration=iteration+1)
    else
        return dominated
    end
end
function dominated_strategies(payoff,player;support=[1:nA for nA in size(payoff)[1:end-1]],strict=true)
    return dominated_strategies(payoff;strict=strict,support=support,iterated=false)[player]
end

"""
    best_response(payoff_array,strategy_profile,nplayer;solver)

Return (possibly one of many) best strategy and corrsponding expected payoff for a given player.

# Parameters:
- `payoff_array`: the N_players+1 dimensional array of payoffs
- `strategy_profile`: the vector of vectors defining the strategies for the N players. The strategy for player n for which the best response is computed is used as initial values in the inner optimisation
- `nplayer`: counter of the player for which we want to compute the best_response (e.g. 1 or 3)
- `solver`: currently either "GLPK" or "HiGHS"
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
    verbosity == FULL && println(support)
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
    if verbosity <= STD
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
        print(m)
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
            warn("The feasibility check for support $support returned neither solved neither unsolved ($status). Returning no Nash equilibrium for this support.")
        end
        solved = false
    end
    return (status=status,equilibrium_strategies=optStrategies,expected_payoffs=optU,solved)
end

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
            A2 = setdiff(1:nActions[2],dominated_strategies(payoff[S1,:,:],2))
            if !isempty(dominated_strategies(payoff[S1,A2,:],1))
                continue
            end
            for S2 in combinations(A2,support_size[3][2])
                if !isempty(dominated_strategies(payoff[S1,S2,:],1))
                        continue
                end
                eq_test =  nash_on_support(payoff,[S1,S2],verbosity=verbosity)
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

"""
    nash_se(payoff; allow_mixed=true, max_samples=1, verbosity=STD)

Compute 1 [def] - `max_samples` Nash equilibria for a N-players generic game in normal form using support enumeration method.

# Parameters

# Notes
- the output is always an array of named tuple with the following information: `equilibrium_strategies`, `expected_payoffs`, `supports` 

"""
function nash_se(payoff; allow_mixed=true, max_samples=1, verbosity=STD)
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
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
        support_size = support_size[3]   
        D = [combinations(1:nActions[n],support_size[n]) for n in 1:nPlayers]
        for idx in CartesianIndices((length.(D)...,))
            #println(idx)
            S = [collect(D[n])[idx[n]] for n in 1:nPlayers]
            #println(S)
            if all(isempty.(dominated_strategies(payoff,iterated=false,support=S)))
                eq_test = nash_on_support(payoff,S,verbosity=verbosity)
                if eq_test.solved
                    eq = (equilibrium_strategies=eq_test.equilibrium_strategies, expected_payoffs=eq_test.expected_payoffs,supports=S)
                    push!(eqs,eq)
                    if length(eqs) == max_samples
                        return eqs
                    end
                end
            end
        end
    end
    return eqs
end

end # module StrategicGames
