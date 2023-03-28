using Test, StrategicGames
import StrategicGames: outer_product, expected_value, nash_se2

payoff_tuple = [(4,4) (0,0); (0,0) (6,6)]
payoff_array = expand_dimensions(payoff_tuple)
@test payoff_array == [4; 0;; 0; 6;;; 4; 0;; 0; 6;;; ]


vs = [[0.2,0.8], [0.1,0.9], [0.2,0.2,0.6]];
product  = outer_product(vs)
product2 = outer_product(vs...)
@test product ≈ [0.004; 0.016;; 0.036; 0.144;;; 0.004; 0.016;; 0.036; 0.144;;; 0.012; 0.048;; 0.108; 0.432;;;]
@test product2 == product  

payoff_array = [4 0 0.1 0.8 0.99 1000 10000 0.0001; 4 0 0.1 0.8 0.99 1000 10000 0.0001; 4 0 0.1 0.8 0.99 1000 10000 0.0001; 4 0 0.1 0.8 0.99 1000 10000 0.0001;]
s = [[0.9, 0.01, 0.03, 0.06], [0.1, 0.01, 0.09, 0.7, 0.01, 0.04, 0.02, 0.03]]
payoff = expected_value(payoff_array,s)
payoff ≈ 240.978903

payoff_array = expand_dimensions([(4,4) (0,0); (0,0) (6,6)])
s = [[0.5,0.5],[0.5,0.5]]
@test expected_payoff(payoff_array,s) == [2.5,2.5]

# Example from https://www.youtube.com/watch?v=bKrwQKUT0v8
U = [(0,0,0) ; (3,3,3) ;; (3,3,3)   ; (2,2,4) ;;;
     (3,3,3)  ; (2,4,2)  ;; (4,2,2) ; (1,1,1)  ;;;]
payoff_array = expand_dimensions(U)
eq  = nash_cp(payoff_array)
eq_strategies = eq.equilibrium_strategies
p = -1 + sqrt(10)/2
@test eq_strategies ≈ [[p,1-p],[p,1-p],[p,1-p]]
is_nash(payoff_array,eq_strategies)


# Testing dominated_strategies
u = expand_dimensions([(3,4) (1,5) (6,2); (2,6) (3,7) (1,7)])
@test dominated_strategies(u,2,strict=true) == [1]
@test dominated_strategies(u,2,strict=false) == [1,3]

u = [(3,4,2) (1,5,3) (6,2,3); (2,6,1) (3,7,1) (1,7,2);;;
     (4,6,4) (2,7,6) (4,2,3); (3,7,2) (4,5,2) (0,4,2);;;]
payoff = expand_dimensions(u)
@test dominated_strategies(payoff,2,strict=false) == [3]


#=
# TODO: check why nash_cp() doesn't work with this one:
u = [(3,4,2) (1,5,3) (6,2,3); (2,6,1) (3,7,1) (1,7,2);;;
     (4,6,4) (2,7,6) (4,2,3); (3,7,2) (4,5,2) (0,4,2);;;]
=#
u = [(0,0,0) ; (3,3,3) ;; (3,3,3)   ; (2,2,4) ;;;
     (3,3,3)  ; (2,4,2)  ;; (4,2,2) ; (1,1,1)  ;;;]
payoff        = expand_dimensions(u)
eq            = nash_cp(payoff)
eq_strategies = eq.equilibrium_strategies
opt_u         = best_response(payoff,[eq_strategies[1],[0.33,0.33,0.34],eq_strategies[3]],2).expected_payoff
nash_u         = eq.expected_payoffs[2]
@test isapprox(opt_u,nash_u)

@test is_best_response(payoff,eq_strategies,2)

@test is_nash(payoff,eq_strategies) == true

U = [(0,0) (-1,1) (1,-1); (1,-1) (0,0) (-1,1); (-1,1) (1,-1) (0,0) ] # head, rock, scissor Only eq is [[0.33,0.33,0.33],[0.33,0.33,0.33]]
payoff = expand_dimensions(U)
@test nash_on_support(payoff,[[1,2,3],[1,2,3]]).solved == true
@test nash_on_support(payoff,[[1,2],[1,2,3]]).solved == false

U = [(-1,-1) (-3,0); (0, -3) (-2, -2)] # prisoner's dilemma. Only Nash eq is [[0,1],[0,1]]
payoff = expand_dimensions(U)
@test nash_on_support(payoff,[[1,2],[1,2]]).solved == false
@test nash_on_support(payoff,[[1],[1]]).solved  == false
@test nash_on_support(payoff,[[2],[2]]).solved  ==true


#=
# ----------------------------------------------------------------
u = [(0,0) (-1,1) (1,-1); (1,-1) (0,0) (-1,1); (-1,1) (1,-1) (0,0) ]
payoff   = expand_dimensions(u)
eq = nash_se2(payoff,max_samples=Inf)
u = [(-1,-1) (-3,0); (0, -3) (-2, -2)]
payoff=(expand_dimensions(u))
eq = nash_se2(payoff,max_samples=Inf,allow_mixed=false)
u = [(1,-1) (-1,1); (-1,1) (1, -1)]
payoff=(expand_dimensions(u))
eq = nash_se2(payoff,max_samples=Inf)
u = [(2,1) (0,0); (0,0) (1,2)]
payoff=(expand_dimensions(u))
eq = nash_se2(payoff,max_samples=Inf)

is_nash(payoff,eq[1].equilibrium_strategies)
=#