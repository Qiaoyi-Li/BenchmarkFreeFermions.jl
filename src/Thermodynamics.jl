# Fermi-Dirac distribution
"""
     n_fermion(x::Real, β::Real) -> ::Float64
     n_fermion(x::Real, lsβ::AbstractVector{<:Real}) -> ::Vector{Float64}
      n_fermion(lsx::AbstractVector{<:Real}, β::Real) -> ::Vector{Float64}

The Fermi-Dirac distribution `x -> 1 / (e^{βx} + 1)`.
"""
function n_fermion(x::Real, β::Real)::Float64
     if isinf(β)
          iszero(x) && return NaN
          return x < 0 ? 1.0 : 0.0
     else
          return 1.0 / (exp(β * x) + 1.0)
     end
end
function n_fermion(x::Real, lsβ::AbstractVector{<:Real})::Vector{Float64}
     return [n_fermion(x, β) for β in lsβ]
end
function n_fermion(lsx::AbstractVector{<:Real}, β::Real)::Vector{Float64}
     return [n_fermion(x, β) for x in lsx]
end

"""
     ParticleNumber(ϵ::Vector{Float64}, β::Real, μ::Real) -> Ntot::Float64

Return the total particle number `Ntot = \\sum_k nk` with given chemical potential `μ`.
"""
function ParticleNumber(ϵ::Vector{Float64}, β::Real, μ::Real)::Float64
     return sum(ϵ) do x
          n_fermion(x - μ, β)
     end
end

"""
     LogPartition(ϵ::Vector{Float64}, β::Real, μ::Real) -> lnZ::Float64

Return the logarithm of the partition function.
"""
function LogPartition(ϵ::Vector{Float64}, β::Real, μ::Real)::Float64
     @assert !isinf(β)
     return sum(ϵ) do x
          log1p(exp(-β * (x - μ)))
     end
end

"""
     Energy(ϵ::Vector{Float64}, β::Real, μ::Real) -> E::Float64

Return the total inner energy `E = \\sum_k ϵk nk`.
"""
function Energy(ϵ::Vector{Float64}, β::Real, μ::Real)::Float64
     return sum(ϵ) do x
          n_fermion(x - μ, β) * x
     end
end

"""
     FreeEnergy(ϵ::Vector{Float64}, β::Real, μ::Real) -> F::Float64

Return the free energy `F = - lnZ/β + μN`.
"""
function FreeEnergy(ϵ::Vector{Float64}, β::Real, μ::Real)::Float64
     isinf(β) && return Energy(ϵ, β, μ)
     return -LogPartition(ϵ, β, μ) / β + μ * ParticleNumber(ϵ, β, μ)
end

"""
     Entropy(ϵ::Vector{Float64}, β::Real, μ::Real)) -> S::Float64

Return the thermal entropy `S = β(E - F)`.
"""
function Entropy(ϵ::Vector{Float64}, β::Real, μ::Real)::Float64
     isinf(β) && return 0.0
     # βE + lnZ - βμN
     return β * Energy(ϵ, β, μ) + LogPartition(ϵ, β, μ) - β * μ * ParticleNumber(ϵ, β, μ)
end

"""
     SpecificHeat_μ(ϵ::Vector{Float64}, β::Real, μ::Real) -> C_μ::Float64

Return the fixed-chemical-potential specific heat `C_μ = (∂(E - μN) / ∂T)_μ`.
"""
function SpecificHeat_μ(ϵ::Vector{Float64}, β::Real, μ::Real)::Float64
     isinf(β) && return 0.0
     return sum(ϵ) do x
          β^2 * (x - μ)^2 / (2 + 2 * cosh(β * (x - μ)))
     end
end

"""
     SpecificHeat_N(ϵ::Vector{Float64}, β::Real, μ::Real) -> C_N::Float64

Return the fixed-particle-number specific heat `C_N = (∂E / ∂T)_N`.
"""
function SpecificHeat_N(ϵ::Vector{Float64}, β::Real, μ::Real)::Float64
     isinf(β) && return 0.0
     coefs = sech.(β * (ϵ .- μ) / 2).^2 / 4
     dμ_T = - β * dot(coefs, ϵ .- μ) / sum(coefs)
     return dot(coefs, ϵ .* (β^2 .* (ϵ .- μ) .+ β * dμ_T))
end

"""
     SolveChemicalPotential(ϵ::Vector{Float64},   
          β::Real,
          n::Real) -> μ::Float64

Return the chemical potential `μ` that leads to the given average particle number `n ∈ (0, 1)`.  
"""
function SolveChemicalPotential(ϵ::Vector{Float64}, β::Real, n::Real; tol::Float64=1e-12)::Float64
     @assert 0 < n < 1
     @assert tol > 0
     @assert !isinf(β)

     # relative error (per site)
     f_err(x) = ParticleNumber(ϵ, β, x) / length(ϵ) - n

     # solve manually via bisection method
     bound = map([extrema(ϵ)...]) do x
          # 1 / (e ^ {β(x - μ)} + 1) = n
          x - log(1 / n - 1) / β
     end
     μ = mean(bound)

     err_n = f_err(μ)
     while abs(err_n) > tol
          if err_n > 0
               bound[2] = μ
               μ = (bound[1] + μ) / 2
          else
               bound[1] = μ
               μ = (bound[2] + μ) / 2
          end
          err_n = f_err(μ)
          μ, err_n
     end
     return μ
end