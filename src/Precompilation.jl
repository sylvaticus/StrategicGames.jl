@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    u2 = [(4,4) (0,0); (0,0) (6,6)]
    u3 = [(0,0,0) ; (3,3,3) ;; (3,3,3)   ; (2,2,4) ;;;
    (3,3,3)  ; (2,4,2)  ;; (4,2,2) ; (1,1,1)  ;;;]
    @compile_workload begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)
        payoff2 = expand_dimensions(u2)
        payoff3 = expand_dimensions(u3)
        nash_cp(payoff2)
        nash_se(payoff2)
        nash_se(payoff3)
        is_nash(payoff2,[[1,0],[0,1]])
        dominated_strategies(payoff3,iterated=true)
    end
end