using Documenter, BenchmarkFreeFermions

makedocs(;
     modules = [BenchmarkFreeFermions],
     sitename="BenchmarkFreeFermions.jl",
     authors = "Qiaoyi Li"
)

deploydocs(
    repo = "github.com/Qiaoyi-Li/BenchmarkFreeFermions.jl",
    devbranch="main",
)