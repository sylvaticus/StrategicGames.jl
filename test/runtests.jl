using Test, BetaGameTheory


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
eq  = nash_lcp(payoff_array)
eq_strategies = eq.equilibrium_strategies
p = -1 + sqrt(10)/2
@test eq_strategies ≈ [[p,1-p],[p,1-p],[p,1-p]]
