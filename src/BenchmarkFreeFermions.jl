module BenchmarkFreeFermions

using LinearAlgebra
using Statistics:mean
using SkewLinearAlgebra

export SingleParticleSpectrum, EigenModes
include("SingleParticleSpectrum.jl")

export n_fermion, ParticleNumber, LogPartition, Energy, FreeEnergy, Entropy, SpecificHeat_μ, SpecificHeat_N, SolveChemicalPotential
include("Thermodynamics.jl")

export GreenFunction, ExpectationValue, TimeCorrelation, Density
include("Observables.jl")

end
