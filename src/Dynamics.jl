"""
     function TwoParticleGreenFunction(ξ::Vector{Float64},
          V::Matrix,
          β::Real,
          O₁::Matrix,
          O₂::Matrix;
          ωtol::Float64 = 1e-8,
          wtol::Float64 = 1e-8) -> lsω::Vector, lsw::Vector 

Return the two-particle Green's function `G^R(ω) = ⟨⟨O₁; O₂⟩⟩_{ω+iη}`, represented by the pole positions `lsω` and the corresponding weights `lsw`. Here `O₁` is represented by a matrix in the real-space basis, i.e., `O₁ = ∑ᵢⱼ O₁[i, j] cᵢ^dag cⱼ`, and similarly for `O₂`.

# Kwargs
     ωtol::Float64 = 1e-8
Tolerance to merge nearly degenerate poles.

     wtol::Float64 = 1e-8
Weight tolerance to drop negligible poles.
"""
function TwoParticleGreenFunction(ξ::Vector{Float64},
     V::Matrix,
     β::Real,
     O₁::Matrix,
     O₂::Matrix;
     ωtol::Float64 = 1e-8,
     wtol::Float64 = 1e-8)

     # matrix elements of O in eigenbasis
     M₁ = V' * O₁ * V
     M₂ = V' * O₂ * V

     # G^R(ω) = sum_{k1,k2} M₁(k1, k2) M₂(k2, k1) * (n_F(ξ_k1) - n_F(ξ_k2)) / (ω + ξ_k1 - ξ_k2 + iη)

     lsω = Float64[]
     lsw = ComplexF64[] 

     for k1 in 1:length(ξ)
          for k2 in 1:length(ξ)
               (k1 == k2) && continue
               ω = ξ[k2] - ξ[k1]
               abs(ω) < ωtol && continue

               w = M₁[k1, k2] * M₂[k2, k1] * (n_fermion(ξ[k1], β) - n_fermion(ξ[k2], β))

               push!(lsω, ω)
               push!(lsw, w)
          end
     end

     # merge nearly degenerate poles
     perms = sortperm(lsω)
     lsω = lsω[perms]
     lsw = lsw[perms]

     lsgroups = [[1]]
     for i in 2:length(lsω)
          if abs(lsω[i] - lsω[i - 1]) < ωtol
               push!(lsgroups[end], i)
          else
               push!(lsgroups, [i])
          end
     end

     lsω = map(lsgroups) do ids 
          sum(lsω[ids] .* abs.(lsw[ids])) / sum(abs.(lsw[ids]))
     end
     lsw = map(lsgroups) do ids
          sum(lsw[ids])
     end

     # drop negligible weights
     ids = findall(x -> abs(x) ≥ wtol, lsw)
     lsω = lsω[ids]
     lsw = lsw[ids]

     if (O₁ == O₂') || (isreal(O₁) && isreal(O₂))
          lsw = real(lsw)
     end

     return lsω, lsw

end