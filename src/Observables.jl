"""
     GreenFunction(ξ::Vector{Float64}, V::Matrix{F}, β::Real; τ::Number=0.0) -> G::Matrix{F}
     GreenFunction(ξ::Vector{Float64}, V::Matrix{F}, β::Real,
     i::Int64, j::Int64;
     τ::Number=0.0,
     reverse::Bool = false) -> Gij::F

Compute the Green's funtion `Gij(τ) = ⟨c_i(τ) c_j^dag⟩` (up to a coefficient) with the shifted single particle spectrum `ξ = ϵ - μ` and the eigenvectors `V` obtained from `Tij = - V diagm(ϵ) V'`.

# Kwargs
     reverse::Bool = false
Return `⟨c_i^dag(τ) c_j⟩` instead if `reverse = true`.
"""
function GreenFunction(ξ::Vector{Float64}, V::Matrix{F}, β::Real;
     τ::Number=0.0,
     reverse::Bool=false)::Matrix{F} where {F<:Union{Float64,ComplexF64}}
     # Gij(τ) = ⟨c_i(τ) c_j^dag⟩ = \sum_k Vik Vjk^* e^{-τ ξ_k}(1 - n_k)
     # ⟨c_i^dag(τ) c_j⟩ = \sum_k Vik^* Vjk e^{τ ξ_k}n_k
     if reverse
          if isinf(β)
               D = map(ξ) do x
                    exp(τ * x) * n_fermion(x, β)
               end
          else
               D = map(ξ) do x
                    exp(τ * x) / (exp(β * x) + 1.0)
               end
          end

          return transpose(V * diagm(D) * V')
     end

     if isinf(β)
          D = map(ξ) do x
               exp(-τ * x) * (1.0 - n_fermion(x, β))
          end
     else
          D = map(ξ) do x
               exp((β - τ) * x) / (exp(β * x) + 1.0)
          end
     end
     return V * diagm(D) * V'
end
function GreenFunction(ξ::Vector{Float64}, V::Matrix{F}, β::Real,
     i::Int64, j::Int64;
     τ::Number=0.0,
     reverse::Bool = false)::F where {F<:Union{Float64,ComplexF64}}

     if reverse
          if isinf(β)
               return mapreduce(+, ξ, view(V, i, :), view(V, j, :)) do x, vi, vj
                    conj(vi) * vj * exp(τ * x) * n_fermion(x, β)
               end
          else
               return mapreduce(+, ξ, view(V, i, :), view(V, j, :)) do x, vi, vj
                    conj(vi) * vj * exp(τ * x) / (exp(β * x) + 1.0)
               end
          end

     end

     if isinf(β)
          return mapreduce(+, ξ, view(V, i, :), view(V, j, :)) do x, vi, vj
               vi * conj(vj) * exp(- τ * x) * (1.0 - n_fermion(x, β))
          end
     else
          return mapreduce(+, ξ, view(V, i, :), view(V, j, :)) do x, vi, vj
               vi * conj(vj) * exp((β - τ) * x) / (exp(β * x) + 1.0)
          end
     end
end

"""
     ExpectationValue(G::Matrix{F},
          si::Vector{Int64},
          dagidx::Vector{Int64}) -> ::F

Return the expectation value of severial fermionic operators. `si` denotes the sites and `dagidx` tells whether the operator is daggered or not. For example, `si = [i, j]` and `dagidx = [2]` means the single particle correlation `c_i c_j^dag` and `si = [i, j, k, l]` and `dagidx = [3, 4]` means the pairing correlation `c_i c_j c_k^dag c_l^dag`. 
"""
function ExpectationValue(G::Matrix{F},
     si::Vector{Int64},
     dagidx::Vector{Int64}
)::F where {F<:Union{Float64,ComplexF64}}

     A = zeros(F, length(si), length(si))
     for i in 1:length(si)
          for j in i+1:length(si)
               if !in(i, dagidx)
                    if in(j, dagidx) # c c^dag
                         A[i, j] = G[si[i], si[j]]
                         A[j, i] = -A[i, j]
                    end
               elseif !in(j, dagidx) # c^dag c
                    if si[i] == si[j]
                         A[i, j] = 1.0 - G[si[i], si[i]]
                    else
                         A[i, j] = -G[si[j], si[i]]
                    end
                    A[j, i] = -A[i, j]
               end
          end
     end
     return pfaffian!(A)
end

"""
     TimeCorrelation(ξ::Vector{Float64},
          V::Matrix,
          β::Real,
          si::Vector{Int64},
          dagidx::Vector{Int64},
          lsτ::AbstractVector{<:Number}) -> ::Float64(::ComplexF64)

Return the time correlation function with the shifted single particle spectrum `ξ = ϵ - μ` and the eigenvectors `V` obtained from `Tij = - V diagm(ϵ) V'`. `si` denotes the sites and `dagidx` tells whether the operator is daggered or not. `lsτ` has the same length as `si` and tells the time of each operator. For example, `si = [i, j, k, l]`, `dagidx = [1, 3]` and `lsτ = [τ1, τ2, 0, 0]` means `c_i^dag(τ1) c_j(τ2) c_k^dag c_l`.   
"""
function TimeCorrelation(ξ::Vector{Float64},
     V::Matrix{F},
     β::Real,
     si::Vector{Int64},
     dagidx::Vector{Int64},
     lsτ::AbstractVector{<:Number}) where {F<:Union{Float64,ComplexF64}}
     @assert length(lsτ) == length(si)

     T = promote_type(F, typeof.(lsτ)...)
     A = zeros(T, length(si), length(si))
     for i in 1:length(si)
          for j in i+1:length(si)
               if !in(i, dagidx)
                    if in(j, dagidx) # c(τi) c^dag(τj)
                         A[i, j] = GreenFunction(ξ, V, β, si[i], si[j]; τ=lsτ[i] - lsτ[j])
                         A[j, i] = -A[i, j]
                    end
               elseif !in(j, dagidx) # c^dag(τi) c(τj)

                    A[i, j] =  GreenFunction(ξ, V, β, si[i], si[j]; τ=lsτ[i] - lsτ[j], reverse = true)
                    A[j, i] = -A[i, j]
               end
          end
     end
     return pfaffian!(A)
end


"""
     Density(G::Matrix, si::Int64) -> ::Float64
     Density(G::Matrix, lssi::AbstractVector{Int64}) -> ::Vector{Float64}

Return the local density `n_i` with the green function `G`.
"""
function Density(G::Matrix, si::Int64)::Float64
     return 1.0 - real(G[si, si])
end
function Density(G::Matrix,
     lssi::AbstractVector{Int64}=1:size(G, 1))::Vector{Float64}
     return [1.0 - real(G[i, i]) for i in lssi]
end

