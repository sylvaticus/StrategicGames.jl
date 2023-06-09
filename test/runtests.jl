using Test, StrategicGames
import StrategicGames: outer_product, expected_value, nash_se2, nash_on_support, nash_on_support_2p, check_domination_2p, restrict_payoff, is_mixdominated

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
@test is_nash(payoff_array,eq_strategies) == true

u = [(3,4,2) (1,5,3) (6,2,3); (2,6,1) (3,7,1) (1,7,2);;;
     (4,6,4) (2,7,6) (4,2,4); (3,7,2) (4,5,2) (0,4,3);;;]
payoff = expand_dimensions(u)
domrem_strict=true; domrem_mixedactions=true; domrem_tol=1e-06; domrem_solver="HiGHS"; verbosity=STD 
eq  = nash_cp(payoff)

# Testing restrict_payoff
dominated = dominated_actions(payoff,domrem_strict=domrem_strict,domrem_mixedactions=domrem_mixedactions, domrem_tol=domrem_tol,domrem_solver=domrem_solver,verbosity=verbosity) #
respayoff = restrict_payoff(payoff,dominated=dominated)
@test respayoff == [4 2; 3 4;;;; 6 7; 7 5;;;; 4 6; 2 2;;;;]

# Testing dominated_actions
u = expand_dimensions([(3,4) (1,5) (6,2); (2,6) (3,7) (1,7)])
@test dominated_actions(u,2,domrem_strict=true) == [1]
@test dominated_actions(u,2,domrem_strict=false) == [1,3]
u = [(3,4,2) (1,5,3) (6,2,3); (2,6,1) (3,7,1) (1,7,2);;;
     (4,6,4) (2,7,6) (4,2,4); (3,7,2) (4,5,2) (0,4,3);;;]
payoff = expand_dimensions(u)
dominated_actions(payoff,2,domrem_strict=false,domrem_mixedactions=false)
@test dominated_actions(payoff,2,domrem_strict=false) == [3]
@test dominated_actions(payoff,domrem_strict=false,iterated=false) == [[],[3],[1]]

@test dominated_actions(payoff,domrem_strict=true,iterated=true) == [[],[3],[1]]

# from https://www.youtube.com/watch?v=Pp5cF4RWuU0 :
u = [(13.0,3.0) (1.0,4.0) (7.0,3.0); (4.0,1.0) (3.0,3.0) (6.0,2.0); (-1.0,9.0) (2.0,8.0) (8.0,-1.0)]
payoff = expand_dimensions(u)
@test dominated_actions(payoff,domrem_strict=true,iterated=true) == [[1,3],[1,3]] 

# from https://www.youtube.com/watch?v=ErJNYh8ejSA : 
u = [(1,1) (-1,2) (5,0) (1,1);
     (2,3) ( 1,2) (3,0) (5,1);
     (1,1) ( 0,5) (1,7) (0,1)]
payoff = expand_dimensions(u)

@test dominated_actions(payoff,iterated=false,domrem_mixedactions=false)                           == [[3],[4]]
@test dominated_actions(payoff)                                          == [[1,3],[2,3,4]]
@test dominated_actions(payoff,iterated=false,support=[[1,2,3],[1,3,4]], domrem_mixedactions=false) == [[3],[4]] # p2.a4 remains dominated by p2.a.2 even if it is not in the support 
@test dominated_actions(payoff,iterated=false,support=[[1,2,3],[1,2,3]], domrem_mixedactions=false) == [[3],[]] # p2.a4 is not checked for domination as not inthe support
@test dominated_actions(payoff,iterated=false,support=[[1,2,3],[1,2,4]], domrem_mixedactions=false) == [[1,3],[4]]# without p2.a3 also p1.a1 become dominated
@test dominated_actions(payoff,support=[[1,2,3],[1,2,4]])                == [[1,3],[2,4]]# p2.a3 not deemed as dominated in inner loops as not in the domain

payoff = [(-1,-1) (-3,0); (0, -3) (-2, -2)]
payoff_array = expand_dimensions(payoff)
@test check_domination_2p(payoff_array,1) == true
@test check_domination_2p(payoff_array,2) == true
payoff = [(-1,-1) (-3,0); (-4, -3) (-2, -2)]
payoff_array = expand_dimensions(payoff)
@test check_domination_2p(payoff_array,1) == false
@test check_domination_2p(payoff_array,2) == true
payoff = [(-1,-1) (-3,0); (-4, -3) (-2, -4)]
payoff_array = expand_dimensions(payoff)
@test check_domination_2p(payoff_array,1) == false
@test check_domination_2p(payoff_array,2) == false

#=
# TODO: check why nash_cp() doesn't work with this one:
u = [(3,4,2) (1,5,3) (6,2,3); (2,6,1) (3,7,1) (1,7,2);;;
     (4,6,4) (2,7,6) (4,2,3); (3,7,2) (4,5,2) (0,4,2);;;]
# also this one: 
u = [(13,3) (1,4) (7,3); (4,1) (3,3) (6,2); (-1,9) (2,8) (8,-1)]
# check with iterated removal of strictly dominated strategies
=#
u = [(0,0,0) ; (3,3,3) ;; (3,3,3)   ; (2,2,4) ;;;
     (3,3,3)  ; (2,4,2)  ;; (4,2,2) ; (1,1,1)  ;;;]
payoff        = expand_dimensions(u)
eq            = nash_cp(payoff)
eq_strategies = eq.equilibrium_strategies
opt_u         = StrategicGames.best_response(payoff,[eq_strategies[1],[0.33,0.33,0.34],eq_strategies[3]],2).expected_payoff
nash_u        = eq.expected_payoffs[2]
@test isapprox(opt_u,nash_u)
@test is_best_response(payoff,eq_strategies,2)
@test is_nash(payoff,eq_strategies) == true
u = [(3,4,2) (1,5,3) (6,2,3); (2,6,1) (3,7,1) (1,7,2);;;
     (4,6,4) (2,7,6) (4,2,3); (3,7,2) (4,5,2) (0,4,2);;;]
payoff        = expand_dimensions(u)
eq            = nash_cp(payoff,domrem_strict=false)
@test StrategicGames.is_nash(payoff,eq.equilibrium_strategies) == true
eq_se         = nash_se(payoff,max_samples=Inf)
@test length(eq_se) == 1
@test eq.equilibrium_strategies ≈ eq_se[1].equilibrium_strategies

# 2 players with 2 and 3 actions respectively
long_payoff = [
     1 1 0.1 0.3;
     1 2 4 6;
     1 3 4 2;
     2 2 4 5;
     2 1 0.1 0.3;
     2 3 1.3 2;
]
@test unstack_payoff(long_payoff) == [0.1  4.0  4.0; 0.1  4.0  1.3;;;
                                     0.3  6.0  2.0; 0.3  5.0  2.0;;;]
long_payoff = [
     1 1 1 0.3 0.1 0.3;
     1 1 2 0.3 0.5 1.2;
     1 2 1 3 4 5;
     1 2 2 4 2 1;
     1 3 1 1.1 4 2;
     1 3 2 1.2 1.3 2;
     2 2 1 3 4 5;
     2 1 1 0.3 0.1 0.3;
     2 1 2 0.3 0.5 1.2;
     2 2 2 4 2 1;
     2 3 1 1.1 4 2;
     2 3 2 1.2 10 2;
]
p = unstack_payoff(long_payoff)
@test size(p) == (2,3,2,3)
@test p[2,3,2,2] == 10

u = [(13,3) (1,4) (7,3); (4,1) (3,3) (6,2); (-1,9) (2,8) (8,-1)]
payoff        = expand_dimensions(u)
eq            = nash_cp(payoff)
@test is_nash(payoff,eq.equilibrium_strategies) == true

U = [(0,0) (-1,1) (1,-1); (1,-1) (0,0) (-1,1); (-1,1) (1,-1) (0,0) ] # head, rock, scissor Only eq is [[0.33,0.33,0.33],[0.33,0.33,0.33]]
payoff = expand_dimensions(U)
@test nash_on_support(payoff,[[1,2,3],[1,2,3]]).solved == true
@test nash_on_support(payoff,[[1,2],[1,2,3]]).solved == false
@test nash_on_support_2p(payoff,[[1,2,3],[1,2,3]]).solved == true
@test nash_on_support_2p(payoff,[[1,2],[1,2,3]]).solved == false

U = [(-1,-1) (-3,0); (0, -3) (-2, -2)] # prisoner's dilemma. Only Nash eq is [[0,1],[0,1]]
payoff = expand_dimensions(U)
@test nash_on_support(payoff,[[1,2],[1,2]]).solved == false
@test nash_on_support(payoff,[[1],[1]]).solved  == false
@test nash_on_support(payoff,[[2],[2]]).solved  ==true
@test nash_on_support_2p(payoff,[[1,2],[1,2]]).solved == false
@test nash_on_support_2p(payoff,[[1],[1]]).solved  == false
@test nash_on_support_2p(payoff,[[2],[2]]).solved  ==true

payoff = expand_dimensions([(3,3) (3,2);
                            (2,2) (5,6);
                            (0,3) (6,1)]);
eqs = nash_se(payoff,max_samples=Inf)
@test length(eqs) == 3
@test eqs[1].equilibrium_strategies ≈  [[1.0, -0.0, 0.0], [1.0, 0.0]]
@test eqs[2].equilibrium_strategies ≈ [[0.8, 0.19999999999999998, 0.0], [0.6666666666666667, 0.3333333333333333]]
@test eqs[3].equilibrium_strategies ≈  [[0.0, 0.33333333333333337, 0.6666666666666666],[0.33333333333333315, 0.6666666666666669]]

eqs = nash_se2(payoff,max_samples=Inf)
@test length(eqs) == 3
@test eqs[1].equilibrium_strategies ≈  [[1.0, -0.0, 0.0], [1.0, 0.0]]
@test eqs[2].equilibrium_strategies ≈ [[0.8, 0.19999999999999998, 0.0], [0.6666666666666667, 0.3333333333333333]]
@test eqs[3].equilibrium_strategies ≈  [[0.0, 0.33333333333333337, 0.6666666666666666],[0.33333333333333315, 0.6666666666666669]]

# Degenerated game
u = [(3,3) (3,3);
     (2,2) (5,6);
     (0,3) (6,1)]
payoff = expand_dimensions(u)

eqs = nash_se(payoff,max_samples=Inf, mt=true, isolated_eq_only=true)
@test length(eqs) == 3
eqs = nash_se(payoff,max_samples=Inf, mt=true, isolated_eq_only=false)
@test length(eqs) == 4


# Testing restrict_payoff...
nActions  = (5,4,5)
player    = 2
dominated = [[2,3],[1,2],[1]]
support   = [[1,3,4,5],[2,3],[1,2,3]]
payoff_n  = reshape(1:prod(nActions), nActions)

restricted = restrict_payoff(payoff_n,2,dominated=dominated,support=support)
@test restricted[:,:,1] == [31 36; 34 39; 35 40]

# Testing is_mixdominated
u = [(3,1) (0,1);
     (9/4,1) (1,1);
     (0,1) (4,1)]
payoff= expand_dimensions(u)
player=1
payoff_n = selectdim(payoff,ndims(payoff),player)
@test is_mixdominated(payoff_n,1,2,domrem_strict=true) == false
@test is_mixdominated(payoff_n,1,2,domrem_strict=false) == true
payoff_n[2,1] = 9/4 + 0.01
@test is_mixdominated(payoff_n,1,2) == false
payoff_n[2,1] = 9/4 - 0.1
@test is_mixdominated(payoff_n,1,2) == true

u = [(1,3,2) (2,2.24,2) (2,0,3);  (2,0,3) (2,1,3) (2,4,1);;; (2,0,1) (3,1,3) (2,4,2); (1,3,3) (2,1,3) (1,4,0)]
payoff= expand_dimensions(u)
player=2
payoff_n = selectdim(payoff,ndims(payoff),player)
@test is_mixdominated(payoff_n,2,2,domrem_strict=true) == true

dominated = [[2],[3],[2]]
support = [[1,2],[3],[1,2]]
@test is_mixdominated(payoff_n,2,2,domrem_strict=true,support=support) == true
@test is_mixdominated(payoff_n,2,2,domrem_strict=true,support=support, dominated=dominated) == true

dominated = [[2],[3],[]]
@test is_mixdominated(payoff_n,2,2,domrem_strict=true,support=support, dominated=dominated) == false


payoff = [(13,3) (1,4) (7, 3);
          ( 4,1) (3,3) (6, 2);
          (-1,9) (2,8) (8,-1)]

payoff_array = expand_dimensions(payoff);
dominated_player2 = dominated_actions(payoff_array,2)
dominated = dominated_actions(payoff_array)
dominated = dominated_actions(payoff_array,domrem_mixedactions=false)

u = [(3,1) (0,1);
     (9/4-0.01,1) (1,1);
     (0,1) (4,1)]
payoff_array = expand_dimensions(u);
@test dominated_actions(payoff_array,domrem_mixedactions=true) == [[2],[]]
@test dominated_actions(payoff_array,domrem_mixedactions=false) == [[],[]]
@test dominated_actions(payoff_array,1,domrem_mixedactions=true) == [2]


