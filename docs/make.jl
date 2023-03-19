using Documenter
using BetaGameTheory

push!(LOAD_PATH,"../src/")
makedocs(
    sitename="BetaGameTheory Documentation",
    pages = [
        "Index" => "index.md",
        "API"   => "api.md"
    ],
    format = Documenter.HTML(prettyurls = false)
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/sylvaticus/BetaGameTheory.jl.git",
    devbranch = "main"
)
