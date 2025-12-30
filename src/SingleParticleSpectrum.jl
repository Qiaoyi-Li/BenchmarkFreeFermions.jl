"""
     SingleParticleSpectrum(Tij::AbstractMatrix) -> ϵ::Vector{Float64}

Numerically diagonalize the hopping matrix `Tij` to get the single particle spectrum `ϵ`. Note the convention is `H = - \\sum_ij Tij c_i^dag c_j`.
"""
function SingleParticleSpectrum(Tij::AbstractMatrix)::Vector{Float64}
     @assert ishermitian(Tij)
     return eigvals(-Tij)
end

"""
     EigenModes(Tij::AbstractMatrix{F};
          tol::Float64 = 1e-12) -> ϵ::Vector{Float64}, V::Matrix{F}

Numerically diagonalize the hopping matrix `Tij = - V diagm(ϵ) V'` to get the single particle spectrum `ϵ` and the eigenvectors `V`. Note the convention is `H = - \\sum_ij Tij c_i^dag c_j` and the decoupled Hamiltonian reads `H = \\sum_k ϵ_k f_k^dag f_k` where `c_i = \\sum_k V[i,k] f_k`.

# Kwargs
     tol::Float64 = 1e-12 : tolerance to identify degenerate eigenvalues.
"""
function EigenModes(Tij::AbstractMatrix{F}; tol::Float64 = 1e-12) where F <: Union{Float64, ComplexF64}
     @assert ishermitian(Tij)
     ϵ::Vector{Float64}, V::Matrix{F} = eigen(-Tij)

     # carefully handle degenerate eigenvalues
     lsgroups = [[1]]
     lslabel = [1]
     for i in 2:length(ϵ)
          if abs(ϵ[i] - ϵ[i - 1]) < tol
               push!(lsgroups[end], i)
          else
               push!(lsgroups, [i])
          end
          push!(lslabel, length(lsgroups))
     end
     lsϵ_mean = map(lsgroups) do ids
          mean(ϵ[ids])
     end
     ϵ = map(lslabel) do l
          lsϵ_mean[l]
     end

     return ϵ, V
end




