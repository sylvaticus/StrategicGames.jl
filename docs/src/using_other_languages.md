# [Using StrategicGames.jl from other programming languages](@id using_other_languages)


In this section we provide two examples of using `StrategicGames` directly in Python or R (with automatic object conversion). Click `Details` for a more extended explanation of these examples.
While I have no experience, the same approach can be used to access `StrategicGames` from any language with a binding to Julia, like Matlab or Javascript. 

## Use StrategicGames in Python

```
$ python3 -m pip install --user juliacall
```
```python
>>> from juliacall import Main as jl
>>> import numpy as np
>>> jl.seval('using Pkg; Pkg.add("StrategicGames")') # Only once 
>>> jl.seval("using StrategicGames")
>>> sg     = jl.StrategicGames
>>> payoff = np.array([[[-1,-1],[-3,0]],[[0,-3],[-2,-2]]]) # prisoner's dilemma
>>> eq     = sg.nash_cp(payoff)
>>> eq._jl_display() # force a "Julian" way of displaying of Julia objects
(status = MathOptInterface.LOCALLY_SOLVED, equilibrium_strategies = [[0.0, 0.9999999887780999], [0.0, 0.9999999887780999]], expected_payoffs = [-1.9999999807790678, -1.9999999807790678])
>>> sg.is_nash(payoff,eq.equilibrium_strategies)
True
>>> sg.dominated_actions(payoff)
<jl [[1], [1]]>
```

```@raw html
<details><summary>Details</summary>
```

We show for Python two separate "Julia from Python" interfaces, [PyJulia](https://github.com/JuliaPy/pyjulia) and [JuliaCall](https://github.com/cjdoris/PythonCall.jl) with the second one being the most recent one.

#### With the classical `pyjulia` package

[PyJulia](https://github.com/JuliaPy/pyjulia) is a relativelly old method to use Julia code and libraries in Python. It works great but it requires that you already have a Julia working installation on your PC, so we need first to download and install the Julia binaries for our operating system from [JuliaLang.org](https://julialang.org/). Be sure that Julia is working by opening the Julia terminal and e.g. typing `println("hello world")`

Install `PyJulia` with: 

```
$ python3 -m pip install --user julia   # the name of the package in `pip` is `julia`, not `PyJulia`
```

We can now open a Python terminal and, to obtain an interface to Julia, just run:

```python
>>> import julia
>>> julia.install() # Only once to set-up in julia the julia packages required by PyJulia
>>> jl = julia.Julia(compiled_modules=False)
```
If we have multiple Julia versions, we can specify the one to use in Python passing `julia="/path/to/julia/binary/executable"` (e.g. `julia = "/home/myUser/lib/julia-1.8.0/bin/julia"`) to the `install()` function.

The `compiled_module=False` in the Julia constructor is a workaround to the common situation when the Python interpreter is statically linked to `libpython`, but it will slow down the interactive experience, as it will disable Julia packages pre-compilation, and every time we will use a module for the first time, this will need to be compiled first.
Other, more efficient but also more complicate, workarounds are given in the package documentation, under the https://pyjulia.readthedocs.io/en/stable/troubleshooting.html[Troubleshooting section].

Let's now add to Julia the StrategicGames package. We can surely do it from within Julia, but we can also do it while remaining in Python:

```python
>>> jl.eval('using Pkg; Pkg.add("StrategicGames")') # Only once to install StrategicGames
```

While `jl.eval('some Julia code')` evaluates any arbitrary Julia code (see below), most of the time we can use Julia in a more direct way. Let's start by importing the StrategicGames Julia package as a submodule of the Python Julia module:

```python
>>> from julia import StrategicGames
>>> jl.seval("using StrategicGames")
```

As you can see, it is no different than importing any other Python module.

For the data, let's load the payoff matrix "Python side" using Numpy:

```python
>>> import numpy as np
>>> payoff = np.array([[[-1,-1],[-3,0]],[[0,-3],[-2,-2]]]) # prisoner's dilemma
```

We can now call StrategicGames functions as we would do for any other Python library functions. In particular, we can pass to the functions (and retrieve) complex data types without worrying too much about the conversion between Python and Julia types, as these are converted automatically:

```python
>>> eq = StrategicGames.nash_cp(payoff)
>>> # Note that array indexing in Julia start at 1
>>> eq
<PyCall.jlwrap (status = MathOptInterface.LOCALLY_SOLVED, equilibrium_strategies = [[0.0, 0.9999999887780999], [0.0, 0.9999999887780999]], expected_payoffs = [-1.9999999807790678, -1.9999999807790678])
>>> StrategicGames.is_nash(payoff,eq.equilibrium_strategies)
True
>>> StrategicGames.dominated_actions(payoff)
[array([1], dtype=int64), array([1], dtype=int64)]
```

Note: If we are using the `jl.eval()` interface, the objects we use must be already known to julia. To pass objects from Python to Julia, import the julia `Main` module (the root module in julia) and assign the needed variables, e.g.

```python
>>> X_python = [1,2,3,2,4]
>>> from julia import Main
>>> Main.X_julia = X_python
>>> jl.eval('sum(X_julia)')
12
```

Another alternative is to "eval" only the function name and pass the (python) objects in the function call:

```python
>>> jl.eval('sum')(X_python)
12
```

#### With the newer `JuliaCall` python package

[JuliaCall](https://github.com/cjdoris/PythonCall.jl) is a newer way to use Julia in Python that doesn't require separate installation of Julia.

Istall it in Python using `pip` as well:

```
$ python3 -m pip install --user juliacall
```

We can now open a Python terminal and, to obtain an interface to Julia, just run:

```python
>>> from juliacall import Main as jl
```
If you have `julia` on PATH, it will use that version, otherwise it will automatically download and install a private version for `JuliaCall`

If we have multiple Julia versions, we can specify the one to use in Python passing `julia="/path/to/julia/binary/executable"` (e.g. `julia = "/home/myUser/lib/julia-1.8.0/bin/julia"`) to the `install()` function.

To add `StrategicGames` to the JuliaCall private version we evaluate the julia package manager `add` function:

```python
>>> jl.seval('using Pkg; Pkg.add("StrategicGames")') # Only once to install StrategicGames
```

As with `PyJulia` we can evaluate arbitrary Julia code either using `jl.seval('some Julia code')` and by direct call, but let's first import `StrategicGames`:

```python
>>> jl.seval("using StrategicGames")
>>> sg = jl.StrategicGames
```

For the data, we reuse the `payoff` Numpy arrays we created earlier.

We can now call StrategicGames functions as we would do for any other Python library functions. In particular, we can pass to the functions (and retrieve) complex data types without worrying too much about the conversion between Python and Julia types, as these are converted automatically:


```python
>>> eq = sg.nash_cp(payoff)
>>> eq._jl_display() # force a "Julian" way of displaying of Julia objects
(status = MathOptInterface.LOCALLY_SOLVED, equilibrium_strategies = [[0.0, 0.9999999887780999], [0.0, 0.9999999887780999]], expected_payoffs = [-1.9999999807790678, -1.9999999807790678])
>>> sg.is_nash(payoff,eq.equilibrium_strategies)
True
>>> sg.dominated_actions(payoff)
<jl [[1], [1]]>

```

Note: If we are using the `jl.eval()` interface, the objects we use must be already known to julia. To pass objects from Python to Julia, we can write a small Julia _macro_:

```python
>>> X_python = [1,2,3,2,4]
>>> jlstore = jl.seval("(k, v) -> (@eval $(Symbol(k)) = $v; return)")
>>> jlstore("X_julia",X_python)
>>> jl.seval("sum(X_julia)")
12
```

Another alternative is to "eval" only the function name and pass the (python) objects in the function call:

```python
>>> X_python = [1,2,3,2,4]
>>> jl.seval('sum')(X_python)
12
```

#### Conclusions about using StrategicGames in Python

Using either the direct call or the `eval` function, wheter in `Pyjulia` or `JuliaCall`, we should be able to use all the StrategicGames functionalities directly from Python. If you run into problems using StrategicGames from Python, [open an issue](https://github.com/sylvaticus/StrategicGames.jl/issues/new) specifying your set-up.

```@raw html
</details>
```

## Use StrategicGames in R


```{r}
> install.packages("JuliaCall") # only once
> library(JuliaCall)
> julia_setup(installJulia = TRUE) # use installJulia = TRUE to let R download and install a private copy of julia, FALSE to use an existing Julia local installation
> julia_eval('using Pkg; Pkg.add("StrategicGames")') # only once
> julia_eval("using StrategicGames")
> payoff = array(c(-1,0,-3,-2,-1,-3,0,-2), dim=c(2,2,2)) # rows: players 1, cols: players 2, 3rd dim: players
> eq     = julia_call("nash_cp",payoff)
> eq
Julia Object of type NamedTuple{(:status, :equilibrium_strategies, :expected_payoffs), Tuple{MathOptInterface.TerminationStatusCode, Vector{Vector{Float64}}, Vector{Float64}}}.
(status = MathOptInterface.LOCALLY_SOLVED, equilibrium_strategies = [[0.0, 0.9999999887780999], [0.0, 0.9999999887780999]], expected_payoffs = [-1.9999999807790678, -1.9999999807790678])
> equilibrium_strategies = field(eq,"equilibrium_strategies")
> strategy_player1       = julia_call("getindex",equilibrium_strategies,as.integer(1))
> strategy_player1
[1] 0 1
> julia_call("is_nash",payoff,equilibrium_strategies)
[1] TRUE
> julia_call("dominated_actions",payoff)
Julia Object of type Vector{Vector{Int64}}.
[[1], [1]]
```


```@raw html
<details><summary>Details</summary>
```
For R we show how to access `StrategicGames` functionalities using the [JuliaCall](https://github.com/Non-Contradiction/JuliaCall) R package (no relations with the homonymous Python package).

Let's start by installing [`JuliaCall`](https://cran.r-project.org/web/packages/JuliaCall/index.html) in R:

```{r}
> install.packages("JuliaCall")
> library(JuliaCall)
> julia_setup(installJulia = TRUE) # use installJulia = TRUE to let R download and install a private copy of julia, FALSE to use an existing Julia local installation
```

Note that, differently than `PyJulia`, the "setup" function needs to be called every time we start a new R section, not just when we install the `JuliaCall` package.
If we don't have `julia` in the path of our system, or if we have multiple versions and we want to specify the one to work with, we can pass the `JULIA_HOME = "/path/to/julia/binary/executable/directory"` (e.g. `JULIA_HOME = "/home/myUser/lib/julia-1.1.0/bin"`) parameter to the `julia_setup` call. Or just let `JuliaCall` automatically download and install a private copy of julia.

`JuliaCall` depends for some things (like object conversion between Julia and R) from the Julia `RCall` package. If we don't already have it installed in Julia, it will try to install it automatically.

As in Python, let's start from creating the payoff array in R and do some work with it in Julia:

```{r}
> payoff = array(c(-1,0,-3,-2,-1,-3,0,-2), dim=c(2,2,2)) # rows: players 1, cols: players 2, 3rd dim: players
```

Let's install StrategicGames. As we did in Python, we can install a Julia package from Julia itself or from within R:

```{r}
> julia_eval('using Pkg; Pkg.add("StrategicGames")')
```

We can now "import" the StrategicGames julia package (in julia a "Package" is basically a module plus some metadata that facilitate its discovery and integration with other packages) and call its functions with the `julia_call("juliaFunction",args)` R function:

```{r}
> julia_eval("using StrategicGames")
> eq = julia_call("nash_cp",payoff)
> eq
Julia Object of type NamedTuple{(:status, :equilibrium_strategies, :expected_payoffs), Tuple{MathOptInterface.TerminationStatusCode, Vector{Vector{Float64}}, Vector{Float64}}}.
(status = MathOptInterface.LOCALLY_SOLVED, equilibrium_strategies = [[0.0, 0.9999999887780999], [0.0, 0.9999999887780999]], expected_payoffs = [-1.9999999807790678, -1.9999999807790678])
> equilibrium_strategies = field(eq,"equilibrium_strategies")
> strategy_player1       = julia_call("getindex",equilibrium_strategies,as.integer(1))
> strategy_player1
[1] 0 1
> julia_call("is_nash",payoff,equilibrium_strategies)
[1] TRUE
> julia_call("dominated_actions",payoff)
Julia Object of type Vector{Vector{Int64}}.
[[1], [1]]
```

As alternative, we can embed Julia code directly in R using the `julia_eval()` function:

```{r}
get_eq_strategies <- julia_eval('
  function get_strategy(payoff,n)
    eq = nash_cp(payoff)
    return eq.equilibrium_strategies[Int(n)]
  end
')
```

We can then call the above function in R in one of the following three ways:
1. `get_eq_strategies(payoff,2)`
2. `julia_assign("payoff_julia_obj",payoff); julia_eval("get_strategy(payoff_julia_obj,2)")`
3. `julia_call("get_strategy",payoff,2)`

While other "convenience" functions are provided by the package, using  `julia_call`, or `julia_assign` followed by `julia_eval`, should suffix to use `StrategicGames` from R. If you run into problems using StrategicGames from R, [open an issue](https://github.com/sylvaticus/StrategicGames.jl/issues/new) specifying your set-up.

```@raw html
</details>
```