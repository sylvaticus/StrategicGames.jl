using Documenter
using StrategicGames

push!(LOAD_PATH,"../src/")
makedocs(
    sitename="StrategicGames Documentation",
    pages = [
        "Index" => "index.md",
        "API"   => "api.md"
    ],
    format = Documenter.HTML(prettyurls = false,
        analytics = "G-DC4KL97F1C",
    )
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/sylvaticus/StrategicGames.jl.git",
    devbranch = "main"
)
