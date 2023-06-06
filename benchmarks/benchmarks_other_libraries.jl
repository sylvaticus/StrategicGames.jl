# Benchmark with other Game Theory libraries: GameTheory.jl, Nashpy, pygambit

using Pkg
cd(@__DIR__)
Pkg.activate(".")
using BenchmarkTools
using StrategicGames
using GameTheory
using CSV, DataFrames
using PyCall
#ENV["JULIA_PYTHONCALL_EXE"] = "/home/lobianco/.pyenv/shims/python3"
#import PythonCall
#using Nash.jl # https://github.com/KrainskiL/Nash.jl
#import StrategicGames: nash_se2

const nash     = PyCall.pyimport("nashpy")  # Equiv. of Python `import nashpy as nash` 
const pygambit = PyCall.pyimport("pygambit")

# ------------------------------------------------------------------------------
# Preliminary set up....
bms = DataFrame(benchmark_name = String[], library = String[], method = String[], time = Float64[], memory = Union{Float64,Missing}[], alloc = Int64[], neqs = Int64[], notes=String[])

macro binfo(bexpr)
    runexpr = quote
      b = @benchmark $bexpr
      return (median(b).time, median(b).memory, median(b).allocs)
    end
    return eval(runexpr)
end

py"""
import numpy as np
import pygambit
def setGambitGame2pInt(P1,P2):
    p1 = np.array(P1, dtype=pygambit.Rational)
    p2 = np.array(P2, dtype=pygambit.Rational)
    return pygambit.Game.from_arrays(p1,p2)
def setGambitGame3pInt(P1,P2,P3):
    p1 = np.array(P1, dtype=pygambit.Rational)
    p2 = np.array(P2, dtype=pygambit.Rational)
    p3 = np.array(P3, dtype=pygambit.Rational)
    return pygambit.Game.from_arrays(p1,p2,p3)
def setGambitGame2pDec(P1,P2):
    p1 = np.array(P1, dtype=pygambit.Decimal)
    p2 = np.array(P2, dtype=pygambit.Decimal)
    print(p1)
    return pygambit.Game.from_arrays(p1,p2)
"""

# For test
res = @binfo rand(2,2)
push!(bms,["Test","rand","rand(2,2)",res...,1,""])
CSV.write("bms.csv",bms)

# ------------------------------------------------------------------------------
# Test 1 - small 3x2 game

game_name = "small_3x2"
u = [(3,3) (3,2);
     (2,2) (5,6);
     (0,3) (6,1)]
payoff = expand_dimensions(u)
small_3x2_StrategicGames = payoff
small_3x2_GameTheory     = NormalFormGame(payoff)
small_3x2_nash           = nash.Game(payoff[:,:,1], payoff[:,:,2])
small_3x2_gambit         = py"""setGambitGame2pInt"""(payoff[:,:,1],payoff[:,:,2])

eqs  = nash_cp(small_3x2_StrategicGames,verbosity=NONE)
neqs = 1
res  = @binfo nash_cp($small_3x2_StrategicGames,verbosity=NONE)
push!(bms,[game_name,"StrategicGames","nash_cp",res...,neqs,""])

eqs = nash_se(small_3x2_StrategicGames,max_samples=Inf, mt=true)
neqs = length(eqs)
res = @binfo nash_se($small_3x2_StrategicGames,max_samples=Inf)
push!(bms,[game_name,"StrategicGames","nash_se",res...,neqs,""])

eqs  = hc_solve(small_3x2_GameTheory)
neqs = length(eqs)
res  = @binfo hc_solve($small_3x2_GameTheory)
push!(bms,[game_name,"GameTheory","hc_solve",res..., neqs,""])

eqs  = support_enumeration(small_3x2_GameTheory)
neqs = length(eqs)
res  = @binfo support_enumeration($small_3x2_GameTheory)
push!(bms,[game_name,"GameTheory","support_enumeration",res..., neqs,""])

eqs  = lrsnash(small_3x2_GameTheory)
neqs = length(eqs)
res  = @binfo lrsnash($small_3x2_GameTheory)
push!(bms,[game_name,"GameTheory","lrsnash",res..., neqs,""])

eqs_gen = small_3x2_nash.vertex_enumeration(); eqs = [eq for eq in eqs_gen]
neqs= length(eqs)
res = @binfo begin eqs = $small_3x2_nash.vertex_enumeration(); [eq for eq in eqs] end
res[2] = missing; res[3] = missing
push!(bms,[game_name,"nashpy","vertex_enumeration",res...,neqs,""])

eqs_gen = small_3x2_nash.lemke_howson_enumeration(); eqs = [eq for eq in eqs_gen]
neqs= length(eqs)
res = @binfo begin eqs = $small_3x2_nash.lemke_howson_enumeration(); [eq for eq in eqs] end
res[2] = missing; res[3] = missing
push!(bms,[game_name,"nashpy","lemke_howson_enumeration",res...,neqs,"repeated results"])

eqs_gen = small_3x2_nash.support_enumeration(); eqs = [eq for eq in eqs_gen]
neqs= length(eqs)
res = @binfo begin eqs = $small_3x2_nash.support_enumeration(); [eq for eq in eqs] end
res[2] = missing; res[3] = missing
push!(bms,[game_name,"nashpy","support_enumeration",res...,neqs,""])

eqs = pygambit.nash.lcp_solve(small_3x2_gambit)
neqs= length(eqs)
res = @binfo pygambit.nash.lcp_solve(small_3x2_gambit)
res[2] = missing; res[3] = missing
push!(bms,[game_name,"pygambit","lcp_solve",res...,neqs,""])

solver = pygambit.nash.ExternalEnumPolySolver()
eqs = solver.solve(small_3x2_gambit)
neqs= length(eqs)
res = @binfo solver.solve(small_3x2_gambit)
res[2] = missing; res[3] = missing
push!(bms,[game_name,"pygambit","ExternalEnumPolySolver",res...,neqs,""])

CSV.write("bms.csv",bms)

# ------------------------------------------------------------------------------
# Test 2 - mid rand 6x7 game
game_name = "rand_6x7"
#payoff = rand(1:1000,6,7,2)
p1 = [
    253  646  641  395  258  153  375
    713   17  145  582  338  258  145
    174  265  282  588   80  996  478
    346  517  963  829  976  735  334
    492  106  199  986  278  658  407
    506  288  439  345  549  869  986]
p2 = [
    296   23  932  537  515  130  679
 491   49  315  432  977  799  777
 284  761  914  268  313  864   25
  75  944  444  209  804  452  502
 869  837  950  707  755  452    9
 579  794  747  272  929  288  780
]
payoff = cat(p1,p2,dims=3)
rand_6x7_StrategicGames = payoff
rand_6x7_GameTheory     = NormalFormGame(payoff)
rand_6x7_nash           = nash.Game(payoff[:,:,1], payoff[:,:,2])
rand_6x7_gambit         = py"""setGambitGame2pInt"""(payoff[:,:,1],payoff[:,:,2])

eqs  = nash_cp(rand_6x7_StrategicGames)
neqs = 0
res  = @binfo nash_cp($rand_6x7_StrategicGames,verbosity=NONE)
push!(bms,[game_name,"StrategicGames","nash_cp",res...,neqs,""])

eqs = nash_se(rand_6x7_StrategicGames,max_samples=Inf)
neqs = length(eqs)
res = @binfo nash_se($rand_6x7_StrategicGames,max_samples=Inf)
push!(bms,[game_name,"StrategicGames","nash_se",res...,neqs,""])

eqs  = hc_solve(rand_6x7_GameTheory)
neqs = length(eqs)
res  = @binfo hc_solve($rand_6x7_GameTheory)
push!(bms,[game_name,"GameTheory","hc_solve",res..., neqs,""])

eqs  = support_enumeration(rand_6x7_GameTheory)
neqs = length(eqs)
res  = @binfo support_enumeration($rand_6x7_GameTheory)
push!(bms,[game_name,"GameTheory","support_enumeration",res..., neqs,""])

eqs  = lrsnash(rand_6x7_GameTheory)
neqs = length(eqs)
res  = @binfo lrsnash($rand_6x7_GameTheory)
push!(bms,[game_name,"GameTheory","lrsnash",res..., neqs,""])

eqs_gen = rand_6x7_nash.vertex_enumeration(); eqs = [eq for eq in eqs_gen]
neqs= length(eqs)
res = @binfo begin eqs = $rand_6x7_nash.vertex_enumeration(); [eq for eq in eqs] end
res[2] = missing; res[3] = missing
push!(bms,[game_name,"nashpy","vertex_enumeration",res...,neqs,""])

eqs_gen = rand_6x7_nash.lemke_howson_enumeration(); eqs = [eq for eq in eqs_gen]
neqs= length(eqs)
res = @binfo begin eqs = $rand_6x7_nash.lemke_howson_enumeration(); [eq for eq in eqs] end
res[2] = missing; res[3] = missing
push!(bms,[game_name,"nashpy","lemke_howson_enumeration",res...,neqs,"repeated results"])

eqs_gen = rand_6x7_nash.support_enumeration(); eqs = [eq for eq in eqs_gen]
neqs= length(eqs)
res = @binfo begin eqs = $rand_6x7_nash.support_enumeration(); [eq for eq in eqs] end
res[2] = missing; res[3] = missing
push!(bms,[game_name,"nashpy","support_enumeration",res...,neqs,""])

CSV.write("bms.csv",bms)

eqs = pygambit.nash.lcp_solve(rand_6x7_gambit)
neqs= length(eqs)
res = @binfo pygambit.nash.lcp_solve(rand_6x7_gambit)
res[2] = missing; res[3] = missing
push!(bms,[game_name,"pygambit","lcp_solve",res...,neqs,""])

solver = pygambit.nash.ExternalEnumPolySolver()
eqs = solver.solve(rand_6x7_gambit) # takes a while !!!!
neqs= length(eqs)
res = @binfo solver.solve(rand_6x7_gambit) # takes even more !
res[2] = missing; res[3] = missing
push!(bms,[game_name,"pygambit","ExternalEnumPolySolver",res...,neqs,""])

CSV.write("bms.csv",bms)

# ------------------------------------------------------------------------------
# Test 3 - rand decimal 6x5 game
game_name = "rand_dec_6x5"

# Note that I can't get Gambit working with decimal data, not even expressed as rational:
# This would both result in "TypeError: payoff argument should be a numeric type instance"
# g = pygambit.Game.from_arrays(np.array([[11/10,21/10,31/10],[41/10,51/10,61/10]], dtype = pygambit.Rational),np.array([[101/100,201/100,301/100],[401/100,501/100,601/600]], dtype = pygambit.Rational))
# g = pygambit.Game.from_arrays(np.array([[1.1,2.1,3.1],[4.1,5.1,6.1]], dtype = pygambit.Decimal),np.array([[10.1,20.1,30.1],[40.1,50.1,60.1]], dtype = pygambit.Decimal))


#payoff = rand(6,5,2)
p1 = [
    0.843789   0.0777291  0.0265584  0.664203  0.716576
    0.399443   0.356958   0.54122    0.817734  0.22748
    0.346581   0.632591   0.712136   0.250977  0.797303
    0.315566   0.861972   0.089729   0.801083  0.386999
    0.895148   0.279619   0.83699    0.844866  0.871585
    0.0835494  0.647902   0.0942865  0.405726  0.542989]
p2 = [
    0.443203   0.810953  0.904374  0.270971   0.293723
    0.527435   0.72811   0.4115    0.496543   0.453296
    0.689995   0.612925  0.201368  0.578151   0.572826
    0.0396185  0.722874  0.486394  0.100332   0.527595
    0.0671089  0.318471  0.968878  0.0495921  0.803702
    0.207152   0.848768  0.445008  0.293703   0.239207
]
payoff = cat(p1,p2,dims=3)
rand_dec_6x5_StrategicGames = payoff
rand_dec_6x5_GameTheory     = NormalFormGame(payoff)
rand_dec_6x5_nash           = nash.Game(payoff[:,:,1], payoff[:,:,2])

eqs  = nash_cp(rand_dec_6x5_StrategicGames)
neqs = 1
res  = @binfo nash_cp($rand_dec_6x5_StrategicGames,verbosity=NONE)
push!(bms,[game_name,"StrategicGames","nash_cp",res...,neqs,""])

eqs = nash_se(rand_dec_6x5_StrategicGames,max_samples=Inf)
neqs = length(eqs)
res = @binfo nash_se($rand_dec_6x5_StrategicGames,max_samples=Inf)
push!(bms,[game_name,"StrategicGames","nash_se",res...,neqs,""])
res[2] = missing; res[3] = missing
eqs  = hc_solve(rand_dec_6x5_GameTheory)
neqs = length(eqs)
res  = @binfo hc_solve($rand_dec_6x5_GameTheory)
push!(bms,[game_name,"GameTheory","hc_solve",res..., neqs,""])

eqs  = support_enumeration(rand_dec_6x5_GameTheory)
neqs = length(eqs)
res  = @binfo support_enumeration($rand_dec_6x5_GameTheory)
push!(bms,[game_name,"GameTheory","support_enumeration",res..., neqs,""])

eqs  = lrsnash(rand_dec_6x5_GameTheory)
neqs = length(eqs)
res  = @binfo lrsnash($rand_dec_6x5_GameTheory)
push!(bms,[game_name,"GameTheory","lrsnash",res..., neqs,""])

eqs_gen = rand_dec_6x5_nash.vertex_enumeration(); eqs = [eq for eq in eqs_gen]
neqs= length(eqs)
res = @binfo begin eqs = $rand_dec_6x5_nash.vertex_enumeration(); [eq for eq in eqs] end
res[2] = missing; res[3] = missing
push!(bms,[game_name,"nashpy","vertex_enumeration",res...,neqs,""])

eqs_gen = rand_dec_6x5_nash.lemke_howson_enumeration(); eqs = [eq for eq in eqs_gen]
neqs= length(eqs)
res = @binfo begin eqs = $rand_dec_6x5_nash.lemke_howson_enumeration(); [eq for eq in eqs] end
res[2] = missing; res[3] = missing
push!(bms,[game_name,"nashpy","lemke_howson_enumeration",res...,neqs,"repeated results"])

eqs_gen = rand_dec_6x5_nash.support_enumeration(); eqs = [eq for eq in eqs_gen]
neqs= length(eqs)
res = @binfo begin eqs = $rand_dec_6x5_nash.support_enumeration(); [eq for eq in eqs] end
res[2] = missing; res[3] = missing
push!(bms,[game_name,"nashpy","support_enumeration",res...,neqs,""])

CSV.write("bms.csv",bms)

# ------------------------------------------------------------------------------
# Test 4 - rand 4x4x2 game
game_name = "rand_4x4x2"
#payoff = rand(1:1000,4,4,2,3)
p1 = [656 207 580 548; 208 337 496 805; 640 422 152 691; 89 183 56 167;;; 44 789 965 925; 917 475 148 990; 78 563 732 25; 424 110 965 431]
p2 = [955 92 480 205; 436 135 802 301; 206 316 346 699; 912 470 317 787;;; 207 536 967 212; 990 335 573 61; 865 455 698 356; 200 141 978 848]
p3 = [364 963 199 81; 332 994 362 972; 421 780 852 150; 72 308 723 828;;; 737 785 653 960; 389 791 125 709; 421 483 10 404; 552 622 161 810]
payoff = cat(p1,p2,p3,dims=4)
rand_4x4x2_StrategicGames = payoff
rand_4x4x2_GameTheory     = NormalFormGame(payoff)
rand_4x4x2_gambit         = py"""setGambitGame3pInt"""(payoff[:,:,:,1],payoff[:,:,:,2],payoff[:,:,:,3])

eqs = nash_cp(rand_4x4x2_StrategicGames)
neqs = 0
res = @binfo nash_cp($rand_4x4x2_StrategicGames,verbosity=NONE)
push!(bms,[game_name,"StrategicGames","nash_cp",res...,neqs,""])

eqs = nash_se(rand_4x4x2_StrategicGames,max_samples=Inf)
neqs = length(eqs)
res = @binfo nash_se($rand_4x4x2_StrategicGames,max_samples=Inf)
push!(bms,[game_name,"StrategicGames","nash_se",res...,neqs,"1 eq repeated"])

eqs  = hc_solve(rand_4x4x2_GameTheory)
neqs = length(eqs)
res  = @binfo hc_solve($rand_4x4x2_GameTheory)
push!(bms,[game_name,"GameTheory","hc_solve",res..., neqs,"2 eq missing"])

CSV.write("bms.csv",bms)

solver = pygambit.nash.ExternalEnumPolySolver()
eqs = solver.solve(rand_4x4x2_gambit)
neqs= length(eqs)
res = @binfo solver.solve(rand_4x4x2_gambit)
res[2] = missing; res[3] = missing
push!(bms,[game_name,"pygambit","ExternalEnumPolySolver",res...,neqs,"1 eq missed"])

CSV.write("bms.csv",bms)

# s = [[0.5531496062992137, 0.0, 0.0, 0.4468503937007864], [0.0, 0.0, 1.0, 0.0], [-9.755303440999721e-15, 1.0000000000000098]]
# StrategicGames.is_nash(payoff,s)

# ------------------------------------------------------------------------------
# Test 5 - mid rand 6x7 game first solution only
game_name = "rand_6x7_1st_eq"
#payoff = rand(1:1000,6,7,2)
p1 = [
    253  646  641  395  258  153  375
    713   17  145  582  338  258  145
    174  265  282  588   80  996  478
    346  517  963  829  976  735  334
    492  106  199  986  278  658  407
    506  288  439  345  549  869  986]
p2 = [
    296   23  932  537  515  130  679
 491   49  315  432  977  799  777
 284  761  914  268  313  864   25
  75  944  444  209  804  452  502
 869  837  950  707  755  452    9
 579  794  747  272  929  288  780
]
payoff = cat(p1,p2,dims=3)
rand_6x7_StrategicGames = payoff
rand_6x7_GameTheory     = NormalFormGame(payoff)
rand_6x7_nash           = nash.Game(payoff[:,:,1], payoff[:,:,2])
rand_6x7_gambit         = py"""setGambitGame2pInt"""(payoff[:,:,1],payoff[:,:,2])

eqs = nash_se(rand_6x7_StrategicGames,max_samples=1,mt=true)
neqs = length(eqs)
res = @binfo nash_se($rand_6x7_StrategicGames,max_samples=1,mt=true)
push!(bms,[game_name,"StrategicGames","nash_se",res...,neqs,""])

eqs  = hc_solve(rand_6x7_GameTheory,ntofind=1)
neqs = length(eqs)
res  = @binfo hc_solve($rand_6x7_GameTheory,ntofind=1)
push!(bms,[game_name,"GameTheory","hc_solve",res..., neqs,""])

eqs_gen = rand_6x7_nash.vertex_enumeration(); eqs = py"next($eqs_gen)"
neqs= 1
res = @binfo begin eqs = $rand_6x7_nash.vertex_enumeration(); py"next($eqs)" end
res[2] = missing; res[3] = missing
push!(bms,[game_name,"nashpy","vertex_enumeration",res...,neqs,""])

eqs_gen = rand_6x7_nash.lemke_howson_enumeration(); eqs = py"next($eqs_gen)"
neqs= 1
res = @binfo begin eqs = $rand_6x7_nash.lemke_howson_enumeration(); py"next($eqs)" end
res[2] = missing; res[3] = missing
push!(bms,[game_name,"nashpy","lemke_howson_enumeration",res...,neqs,""])

eqs_gen = rand_6x7_nash.support_enumeration();
# eqs = py"next($eqs_gen)"
neqs= 0
res = @binfo begin eqs = $rand_6x7_nash.support_enumeration(); end
res[2] = missing; res[3] = missing
push!(bms,[game_name,"nashpy","support_enumeration",res...,neqs,"no eq reported"])

CSV.write("bms.csv",bms)

# ------------------------------------------------------------------------------
# Test 6 - degenerated 3x2 game

game_name = "degenerated_3x2"
u = [(3,3) (3,3);
     (2,2) (5,6);
     (0,3) (6,1)]
payoff = expand_dimensions(u)
degenerated_3x2_StrategicGames = payoff
degenerated_3x2_GameTheory     = NormalFormGame(payoff)
degenerated_3x2_nash           = nash.Game(payoff[:,:,1], payoff[:,:,2])
degenerated_3x2_gambit         = py"""setGambitGame2pInt"""(payoff[:,:,1],payoff[:,:,2])

eqs  = nash_cp(degenerated_3x2_StrategicGames,verbosity=NONE)
neqs = 1
res  = @binfo nash_cp($degenerated_3x2_StrategicGames,verbosity=NONE)
push!(bms,[game_name,"StrategicGames","nash_cp",res...,neqs,""])

eqs = nash_se(degenerated_3x2_StrategicGames,max_samples=Inf, mt=true)
neqs = length(eqs)
res = @binfo nash_se($degenerated_3x2_StrategicGames,max_samples=Inf)
push!(bms,[game_name,"StrategicGames","nash_se",res...,neqs,""])

eqs  = hc_solve(degenerated_3x2_GameTheory)
neqs = length(eqs)
res  = @binfo hc_solve($degenerated_3x2_GameTheory)
push!(bms,[game_name,"GameTheory","hc_solve",res..., neqs,""])

eqs  = support_enumeration(degenerated_3x2_GameTheory)
neqs = length(eqs)
res  = @binfo support_enumeration($degenerated_3x2_GameTheory)
push!(bms,[game_name,"GameTheory","support_enumeration",res..., neqs,""])

eqs  = lrsnash(degenerated_3x2_GameTheory)
neqs = length(eqs)
res  = @binfo lrsnash($degenerated_3x2_GameTheory)
push!(bms,[game_name,"GameTheory","lrsnash",res..., neqs,""])

eqs_gen = degenerated_3x2_nash.vertex_enumeration(); eqs = [eq for eq in eqs_gen]
neqs= length(eqs)
res = @binfo begin eqs = $degenerated_3x2_nash.vertex_enumeration(); [eq for eq in eqs] end
res = (res[1], missing, missing)
push!(bms,[game_name,"nashpy","vertex_enumeration",res...,neqs,""])

eqs_gen = degenerated_3x2_nash.lemke_howson_enumeration(); eqs = [eq for eq in eqs_gen]
neqs= length(eqs)
res = @binfo begin eqs = $degenerated_3x2_nash.lemke_howson_enumeration(); [eq for eq in eqs] end
res = (res[1], missing, missing)
push!(bms,[game_name,"nashpy","lemke_howson_enumeration",res...,neqs,"2 repeated results"])

eqs_gen = degenerated_3x2_nash.support_enumeration(); eqs = [eq for eq in eqs_gen]
neqs= length(eqs)
res = @binfo begin eqs = $degenerated_3x2_nash.support_enumeration(); [eq for eq in eqs] end
res = (res[1], missing, missing)
push!(bms,[game_name,"nashpy","support_enumeration",res...,neqs,""])

eqs = pygambit.nash.lcp_solve(degenerated_3x2_gambit)
neqs= length(eqs)
res = @binfo pygambit.nash.lcp_solve(degenerated_3x2_gambit)
res = (res[1], missing, missing)
push!(bms,[game_name,"pygambit","lcp_solve",res...,neqs,""])

solver = pygambit.nash.ExternalEnumPolySolver()
eqs = solver.solve(degenerated_3x2_gambit)
neqs= length(eqs)
res = @binfo solver.solve(degenerated_3x2_gambit)
res = (res[1], missing, missing)
push!(bms,[game_name,"pygambit","ExternalEnumPolySolver",res...,neqs,""])

CSV.write("bms.csv",bms)