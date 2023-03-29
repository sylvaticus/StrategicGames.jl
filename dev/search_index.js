var documenterSearchIndex = {"docs":
[{"location":"api.html#Detailed-API","page":"API","title":"Detailed API","text":"","category":"section"},{"location":"api.html","page":"API","title":"API","text":"Modules = [StrategicGames]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"api.html#StrategicGames.Verbosity","page":"API","title":"StrategicGames.Verbosity","text":"Verbosity\n\nMany functions accept a verbosity parameter.\n\nChoose between: NONE, LOW, STD [default], HIGH and FULL.\n\nUnder default verbosity (STD) no output is print unless something unexpected in most conditions (but not necessarily an error) is detected \n\n\n\n\n\n","category":"type"},{"location":"api.html#StrategicGames.best_response-Tuple{Any, Any, Any}","page":"API","title":"StrategicGames.best_response","text":"best_response(payoff_array,strategy_profile,nplayer;solver)\n\nReturn (possibly one of many) best strategy and corrsponding expected payoff for a given player.\n\nParameters:\n\npayoff_array: the N_players+1 dimensional array of payoffs\nstrategy_profile: the vector of vectors defining the strategies for the N players. The strategy for player n for which the best response is computed is used as initial values in the inner optimisation\nnplayer: counter of the player for which we want to compute the best_response (e.g. 1 or 3)\nsolver: currently either \"GLPK\" or \"HiGHS\"\nverbosity: either NONE, LOW, STD [default], HIGH or FULL\n\nReturns:\n\nA named tuple with: expected_payoff, optimal_strategy, status (of the underlying optimisation)\n\nExample:\n\njulia> using StrategicGames\njulia> payoff_array  = [(3,4) (1,5); (4,2) (2,3)] # prisoner's dilemma\n2×2 Matrix{Tuple{Int64, Int64}}:\n (3, 4)  (1, 5)\n (4, 2)  (2, 3)\njulia> best_response(expand_dimensions(payoff_array),[[0.5,0.5],[0.5,0.5]],2)\n(expected_payoff = 4.0, optimal_strategy = [0.0, 1.0], status = MathOptInterface.OPTIMAL)\n\n\n\n\n\n","category":"method"},{"location":"api.html#StrategicGames.dominated_strategies-Tuple{Any}","page":"API","title":"StrategicGames.dominated_strategies","text":"dominated_strategies(payoff;strict=true,iterated=true,verbosity,dominated)\n\nImplements the \"Iterated [by default] Removal of Strictly [by default] Dominated Strategies\" (IRSDS) algorithm, returning a vector (for each player) of vectors (action positions) of actions that for a given player are dominates by at least one of his other actions. This function is iterative (recursive) by default.\n\nParameters\n\npayoff_array: the Nplayers+1 dimensional array of payoffs for the N players\nstrict: wheter to look for strictly dominated actions [def: true]\niterated: wheter to look for strictly dominated actions iteractively [def: true]\nverbosity: either NONE, LOW, STD [default], HIGH or FULL\ndominated: vector of vectors of actions to skip check as already deemed as dominated\n\nNotes\n\nThis function is available also as dominated_strategies(payoff,player;strict) returning a vector of dominated strategies for a given players (computed not iteractively)\nTo get the list of retrieved dominated actions at each iteration, use a verbosity level higher than STD \n\nExample\n\njulia> using StrategicGames\njulia> payoff = [(13,3) (1,4) (7,3); (4,1) (3,3) (6,2); (-1,9) (2,8) (8,-1)]\n3×3 Matrix{Tuple{Int64, Int64}}:\n (13, 3)  (1, 4)  (7, 3)\n (4, 1)   (3, 3)  (6, 2)\n (-1, 9)  (2, 8)  (8, -1)\njulia> payoff_array = expand_dimensions(payoff);\njulia> dominated_player2 = dominated_strategies(payoff_array,2)\n1-element Vector{Int64}:\n 3\njulia> dominated = dominated_strategies(payoff_array,verbosity=HIGH)\nDominated strategies at step 1: [Int64[], [3]]\nDominated strategies at step 2: [[3], [3]]\nDominated strategies at step 3: [[3], [3, 1]]\nDominated strategies at step 4: [[3, 1], [3, 1]]\nDominated strategies at step 5: [[3, 1], [3, 1]]\n2-element Vector{Vector{Int64}}:\n [3, 1]\n [3, 1]\n\n\n\n\n\n","category":"method"},{"location":"api.html#StrategicGames.expand_dimensions-Union{Tuple{AbstractArray{T}}, Tuple{T}, Tuple{T2}} where {T2, T<:Tuple{Vararg{T2}}}","page":"API","title":"StrategicGames.expand_dimensions","text":"expand_dimensions(x::AbstractArray{T}\n\nConvenience function to transform a N dimensional array of tuples in a N+1 dimensional array of scalars (the format used for most functions in this package).\n\nExample:\n\njulia> using StrategicGames\njulia> payoff_tuple = [(1,-1) (-1,1) (1,0); (-1,1) (1, -1) (0,1)] # 2 players, with 2 and 3 actions respectively\n2×3 Matrix{Tuple{Int64, Int64}}:\n (1, -1)  (-1, 1)  (1, 0)\n (-1, 1)  (1, -1)  (0, 1)\n\njulia> payoff_array = expand_dimensions(payoff_tuple)\n2×3×2 Array{Int64, 3}:\n[:, :, 1] =\n  1  -1  1\n -1   1  0\n\n[:, :, 2] =\n -1   1  0\n  1  -1  1\n\n\n\n\n\n","category":"method"},{"location":"api.html#StrategicGames.expected_value-Tuple{Any, Any}","page":"API","title":"StrategicGames.expected_value","text":"expected_value(v::Array{N,Real},p::Vector{Vector{Real}}) --> Real\n\nCompute the expected value (scalar) of a N-dimensional value array given a vector of N probability vectors, one per each dimension of the value array. \n\n\n\n\n\n","category":"method"},{"location":"api.html#StrategicGames.is_best_response-Tuple{Any, Any, Any}","page":"API","title":"StrategicGames.is_best_response","text":"is_best_response(payoff_array,strategy_profile,nplayer;atol=1e-07,rtol=1e-07,solver,verbosity=STD)\n\nDetermine if a given strategy for player nplayer is a best response to a given payoff array and strategies of the other players\n\nParameters:\n\npayoff_array: the Nplayers+1 dimensional array of payoffs\nstrategy_profile: the vector of vectors defining the strategies for the N players\nnplayer: counter of the player for which we want to verify if its strategy is a best_response (e.g. 1 or 3)\natol: absolute tollerance in comparing the expected payoff from the given strategy and those from the optimal one [def: 1e-07]\nrtol: relative tollerance in comparing the expected payoff from the given strategy and those from the optimal one [def: 1e-07]\nsolver: currently either \"GLPK\" or \"HiGHS\"\nverbosity: either NONE, LOW, STD [default], HIGH or FULL\n\nExample :\n\njulia> using StrategicGames\njulia> payoff_array = [(3,4) (1,5); (4,2) (2,3)] # prisoner's dilemma\n2×2 Matrix{Tuple{Int64, Int64}}:\n (3, 4)  (1, 5)\n (4, 2)  (2, 3)\njulia> is_best_response(expand_dimensions(payoff_array),[[0,1],[0.5,0.5]],1)\ntrue\njulia> is_best_response(expand_dimensions(payoff_array),[[0,1],[0.5,0.5]],2)\nfalse\n\n\n\n\n\n","category":"method"},{"location":"api.html#StrategicGames.is_nash-Tuple{Any, Any}","page":"API","title":"StrategicGames.is_nash","text":"is_nash(payoff_array,strategy_profile;atol=1e-07,rtol=1e-07,solver,verbosity)\n\nDetermine if a strategy profile is a Nash equilibrium for a given payoff array, i.e. if all strategies are (at least weak) best responses.\n\nParameters:\n\npayoff_array: the Nplayers+1 array of payoffs\nstrategy_profile: the vector of vectors defining the strategies for the N players\natol: absolute tollerance in comparing the expected payoffs from the given strategies and those from the optimal ones [def: 1e-07]\nrtol: relative tollerance in comparing the expected payoffs from the given strategies and those from the optimal ones [def: 1e-07]\nsolver: currently either \"GLPK\" or \"HiGHS\"\nverbosity: either NONE, LOW, STD [default], HIGH or FULL\n\nExample :\n\njulia> using StrategicGames\njulia> payoff_array  = [(3,4) (1,5); (4,2) (2,3)] # prisoner's dilemma\n2×2 Matrix{Tuple{Int64, Int64}}:\n (3, 4)  (1, 5)\n (4, 2)  (2, 3)\njulia> is_nash(expand_dimensions(payoff_array),[[0,1],[0,1]])\ntrue\n\n\n\n\n\n","category":"method"},{"location":"api.html#StrategicGames.nash_cp-Tuple{Any}","page":"API","title":"StrategicGames.nash_cp","text":"nash_cp(payoff_array;init,verbosity)\n\nFind a Nash Equilibrium for N-players simultaneous games when mixed strategies are allowed using the Complementarity Problem formulation and implementing iterated removal of dominated strategies\n\nParameters\n\npayoff_array: the Nplayers+1 dimensional array of payoffs for the N players\ninit: a vector of vector of mixed strategies (i.e. PMFs) for each players to start the algorithm with. Different init points may reach different equilibrium points [def: equal probabilities for each available action of the players]\nstrict_domination_removal: wether to remove only strictly dominated strategies in the pre-optimisation or also weakly dominated ones [def: true]\nverbosity: either NONE, LOW, STD [default], HIGH or FULL\n\nNotes\n\nThis function uses a complementarity formulation. For N <= 2 the problem, except the complementarity equation, is linear and known as LCP (Linear Complementarity Problem)\nThis implementation uses the JuMP modelling language with the Ipopt solver engine (and hence it uses an interior point method instead of the pivotal approach used in the original Lemke-Howson [1964] algorithm)\nThere is no guarantee on timing and even that the algorithm converge to an equilibrium. Different Nash equilibriums may be reached by setting different initial points\nBy default the iterative removal of dominated strategies concerns only strictly dominated ones. For some games where the algorithm doesn't find a Nash equilibrium, you can often get success setting the algorithm to remove also weakly dominated strategies. \n\nReturns\n\nA named tuple with the following elements: status,equilibrium_strategies,expected_payoffs\n\nExample\n\njulia> using StrategicGames\njulia> payoff = [(-1,-1) (-3,0); (0, -3) (-2, -2)] # prisoner's dilemma\n2×2 Matrix{Tuple{Int64, Int64}}:\n (-1, -1)  (-3, 0)\n (0, -3)   (-2, -2)\njulia> eq     = nash_cp(expand_dimensions(payoff));\njulia> eq_strategies = eq.equilibrium_strategies\n2-element Vector{Vector{Float64}}:\n [-4.049752569180346e-11, 1.0000000000404976]\n [-4.0497525691839856e-11, 1.0000000000404976]\n\n\n\n\n\n","category":"method"},{"location":"api.html#StrategicGames.nash_on_support","page":"API","title":"StrategicGames.nash_on_support","text":"nash_on_support(payoff_array,support;init,verbosity)\n\nFind (if it exists) a Nash equilibrium for N-players simultaneous games when mixed strategies are allowed on a specific support.\n\nParameters\n\npayoff_array: the Nplayers+1 dimensional array of payoffs for the N players\nsupport: vector of vector of action counts that are in the tested support for each player [def: full support]\ninit: a vector of vector of mixed strategies (i.e. PMFs) for each players to start the algorithm with. Different init points may reach different equilibrium points [def: equal probabilities for each available action of the players]\nverbosity: either NONE, LOW, STD [default], HIGH or FULL\n\nNotes\n\nThis implementation uses the JuMP modelling language with the Ipopt solver engine\n\nReturns\n\nA named tuple with the following elements: status,equilibrium_strategies,expected_payoffs, solved\n\nExample\n\njulia> using StrategicGames\njulia> payoff = [(-1,-1) (-3,0); (0, -3) (-2, -2)] # prisoner's dilemma. Only Nash eq is [[0,1],[0,1]]\n2×2 Matrix{Tuple{Int64, Int64}}:\n (-1, -1)  (-3, 0)\n (0, -3)   (-2, -2)\njulia> payoff_array = expand_dimensions(payoff);\njulia> nash_on_support(payoff_array,[[1,2],[1,2]]).solved # false\nfalse\njulia> nash_on_support(payoff_array,[[2],[2]]).solved     # true\ntrue\n\n\n\n\n\n","category":"function"},{"location":"index.html#StrategicGames-package-documentation","page":"Index","title":"StrategicGames package documentation","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"GitHub repository","category":"page"},{"location":"index.html","page":"Index","title":"Index","text":"StrategicGames","category":"page"},{"location":"index.html#StrategicGames","page":"Index","title":"StrategicGames","text":"StrategicGames package\n\nStrategic Games provides functionalities to work with strategic games, including finding Nash equilibrium in simultaneous games when mixed strategies are allowed. \n\nUnless otherwise stated, all functions work with generic n players (the examples generally show 2 players for simplicity) and assume the payoff \"matrix\" to be in the form of N-players + 1 dimensional arrays, where each dimension except the last one is given by the number of discrete actions available to each player and the last dimension is given by the number of players. A convenient function allows to transform a N-dimensional array of tuples (the payoff for the various players) to the Nplayers+1 dimensional array\n\n\n\n\n\n","category":"module"},{"location":"index.html#Index","page":"Index","title":"Index","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"StrategicGames provides the following functions:","category":"page"},{"location":"index.html","page":"Index","title":"Index","text":"Modules = [StrategicGames]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"index.html#examples","page":"Index","title":"Some examples","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"julia> using Pkg; Pkg.add(\"StrategicGames\") # only once to install the library\njulia> using StrategicGames","category":"page"},{"location":"index.html#players-game-with-payoff-a-function-of-the-number-of-players-choosing-the-same-option","page":"Index","title":"3-players game with payoff a function of the number of players choosing the same option","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"The 3 players in this game can choose between X or Y with payoffs respectively 2nₓ - 2nₓ²+3 and 4-nᵧ where nₓ and nᵧ represent the number of players that choose X or Y. (this example is taken from https://www.youtube.com/watch?v=bKrwQKUT0v8 where it is analytically solved)","category":"page"},{"location":"index.html","page":"Index","title":"Index","text":"julia> U = [(0,0,0) ; (3,3,3) ;; (3,3,3) ; (2,2,4) ;;;\n            (3,3,3) ; (2,4,2) ;; (4,2,2) ; (1,1,1) ;;;]\n2×2×2 Array{Tuple{Int64, Int64, Int64}, 3}:\n[:, :, 1] =\n (0, 0, 0)  (3, 3, 3)\n (3, 3, 3)  (2, 2, 4)\n[:, :, 2] =\n (3, 3, 3)  (4, 2, 2)\n (2, 4, 2)  (1, 1, 1)\njulia> payoff_array = expand_dimensions(U) # from n-dimensional array of tuples to n+1 arrays of scalars\n2×2×2×3 Array{Int64, 4}:\n[:, :, 1, 1] =\n 0  3\n 3  2\n[:, :, 2, 1] =\n 3  4\n 2  1\n[:, :, 1, 2] =\n 0  3\n 3  2\n[:, :, 2, 2] =\n 3  2\n 4  1\n[:, :, 1, 3] =\n 0  3\n 3  4\n[:, :, 2, 3] =\n 3  2\n 2  1\njulia> eq  = nash_cp(payoff_array)\n(status = MathOptInterface.LOCALLY_SOLVED, equilibrium_strategies = [[0.5811388300841898, 0.4188611699158103], [0.5811388300841898, 0.4188611699158103], [0.5811388300841898, 0.41886116991581035]], expected_payoffs = [2.16227766016838, 2.16227766016838, 2.16227766016838])\njulia> eq_strategies = eq.equilibrium_strategies\n3-element Vector{Vector{Float64}}:\n [0.5811388300841898, 0.4188611699158103]\n [0.5811388300841898, 0.4188611699158103]\n [0.5811388300841898, 0.41886116991581035]\njulia> p = -1 + sqrt(10)/2\n0.5811388300841898\njulia> eq_strategies ≈ [[p,1-p],[p,1-p],[p,1-p]]\ntrue\njulia> expected_payoff(payoff_array,eq_strategies)\n3-element Vector{Float64}:\n 2.1622776601683795\n 2.1622776601683795\n 2.1622776601683795","category":"page"},{"location":"index.html#Prisoner's-dilemma","page":"Index","title":"Prisoner's dilemma","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"julia> # Pay-off matrix\n       U = [(-1,-1) (-3,0); (0, -3) (-2, -2)]\n2×2 Matrix{Tuple{Int64, Int64}}:\n (-1, -1)  (-3, 0)\n (0, -3)   (-2, -2)\njulia> # From N-dimensional array of tuples to N+1 arrays of scalars    \n       payoff_array = expand_dimensions(U);\njulia> # Find all the dominated strategies for the two players\n       dominated_strategies(payoff_array)\n2-element Vector{Vector{Int64}}:\n [1]\n [1]\njulia> # Compute one Nash Equilibrium of the Game using LCP (linear complementarity) formulation       \n       eq = nash_cp(payoff_array).equilibrium_strategies\n2-element Vector{Vector{Float64}}:\n [-4.049752569180346e-11, 1.0000000000404976]\n [-4.0497525691839856e-11, 1.0000000000404976]\njulia> # Best response for player 2\n       best_response(payoff_array,[[0.5,0.5],[0.5,0.5]],2).optimal_strategy\n2-element Vector{Float64}:\n 0.0\n 1.0\njulia> # Expected payoffs given a specific strategy profile\n       expected_payoff(payoff_array,[[1,0],[1,0]])\n2-element Vector{Int64}:\n -1\n -1\njulia> # Is this strategy profile a Nash equilibrium ?\n       is_nash(payoff_array,[[1,0],[1,0]]) \nfalse","category":"page"},{"location":"index.html#Head-or-tail","page":"Index","title":"Head or tail","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"julia> payoff_matrix = [(1,-1) (-1,1); (-1,1) (1, -1)]\n2×2 Matrix{Tuple{Int64, Int64}}:\n (1, -1)  (-1, 1)\n (-1, 1)  (1, -1)\njulia> eq            = nash_cp(expand_dimensions(payoff_matrix));\njulia> eq_strategies = eq.equilibrium_strategies\n2-element Vector{Vector{Float64}}:\n [0.5, 0.5]\n [0.5, 0.5]","category":"page"},{"location":"index.html#Battle-of-the-sex","page":"Index","title":"Battle of the sex","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"julia> payoff_matrix = [(2,1) (0,0); (0,0) (1,2)]\n2×2 Matrix{Tuple{Int64, Int64}}:\n (2, 1)  (0, 0)\n (0, 0)  (1, 2)\njulia> eq            = nash_cp(expand_dimensions(payoff_matrix));\njulia> eq_strategies = eq.equilibrium_strategies \n2-element Vector{Vector{Float64}}:\n [0.6666666663602984, 0.33333333363970163]\n [0.33333333363970163, 0.6666666663602984]","category":"page"},{"location":"index.html#Rock,-paper,-scissor","page":"Index","title":"Rock, paper, scissor","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"julia> # Actions: Rock, Paper, Scissor (in the order)\n       U = [(0,0) (-1,1) (1,-1); (1,-1) (0,0) (-1,1); (-1,1) (1,-1) (0,0) ]\n3×3 Matrix{Tuple{Int64, Int64}}:\n (0, 0)   (-1, 1)  (1, -1)\n (1, -1)  (0, 0)   (-1, 1)\n (-1, 1)  (1, -1)  (0, 0)\njulia> eq = nash_cp(expand_dimensions(U)).equilibrium_strategies\n2-element Vector{Vector{Float64}}:\n [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]\n [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]","category":"page"},{"location":"index.html#A-biased-penalty-kick-game","page":"Index","title":"A biased penalty kick game","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"The row player is the kicker and the column player is the goalkeeper, first action is kick/jump on one direction, second one is kick/jump on the other direction. The kicker is more efficient (or, alternatively, the goalkeeper is less efficient) on the second direction.","category":"page"},{"location":"index.html","page":"Index","title":"Index","text":"julia> payoff_matrix = [(-1,1) (1,-1); (1,-1) (0, 0)]\n2×2 Matrix{Tuple{Int64, Int64}}:\n (-1, 1)  (1, -1)\n (1, -1)  (0, 0)\njulia> eq            = nash_cp(expand_dimensions(payoff_matrix));\njulia> eq_strategies = eq.equilibrium_strategies\n2-element Vector{Vector{Float64}}:\n [0.3333333333332723, 0.6666666666667278]\n [0.33333333333337, 0.6666666666666301]","category":"page"},{"location":"index.html#Other-game-theory-libraries","page":"Index","title":"Other game-theory libraries","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"Julia","category":"page"},{"location":"index.html","page":"Index","title":"Index","text":"Nash.jl\nGameTheory.jl","category":"page"},{"location":"index.html","page":"Index","title":"Index","text":"Non-Julia","category":"page"},{"location":"index.html","page":"Index","title":"Index","text":"Nashpy: two players only\nGambit","category":"page"}]
}
