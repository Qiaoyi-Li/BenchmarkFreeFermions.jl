---
name: benchmark-free-fermions
description: Compute exact free-fermion reference results (spectra, Green's functions, correlators, thermodynamics, two-particle response) for benchmarking tensor network algorithms (DMRG/MPS/PEPS/TRG/etc.) against O(N^3) exact solutions. Use when validating tensor network code on free-fermion test cases, comparing ground-state energies/observables/correlations, or generating thermal reference data.
license: MIT
compatibility: opencode
metadata:
  package: "BenchmarkFreeFermions"
  uuid: "5b68a00d-ca4d-4b58-ab0c-dc082ade624e"
---

# BenchmarkFreeFermions — Exact Free-Fermion Reference for Tensor Network Benchmarks

This skill teaches you how to use the `BenchmarkFreeFermions.jl` Julia package to compute
exact results for free fermion systems and use them as ground-truth references when
testing or validating finite-size tensor network algorithms.

Free fermions are exactly solvable in O(N^3), so they provide a cheap alternative to full ED
for benchmarking newly implemented tensor network codes in real space.

## When to Use

- Validating a tensor network algorithm (DMRG/MPS/PEPS/tanTRG/AutomataMPO/...) on a free-fermion test case
- Computing exact ground-state energy, Green's functions, or correlators to compare against variational results
- Generating thermal/thermodynamic reference data (`ln Z`, `E`, `F`, `S`, `C`) for finite-temperature tensor network benchmarks
- Computing two-particle response functions (structure factors, optical conductivity) as exact references

**Do NOT use when:**
- The system has interactions (the package only handles free/quadratic fermions, `H = -∑ Tᵢⱼ cᵢ† cⱼ`)
- You need real-time dynamics beyond two-particle retarded response
- The Hamiltonian is not expressible as a quadratic form in fermion operators

## Core Convention

The Hamiltonian convention is:

    H = -∑ᵢⱼ Tᵢⱼ cᵢ† cⱼ

where `Tij` is a Hermitian matrix of hopping coefficients. The minus sign is built into
`EigenModes` — do **not** add it yourself. `BenchmarkFreeFermions` diagonalizes `Tij` and
provides thermodynamic, observable, and dynamic quantities built from the single-particle
spectrum.

## Workflow

### Step 1: Build the hopping matrix `Tij`

Construct a Hermitian `L×L` matrix. For random tests:

```julia
using BenchmarkFreeFermions
L = 8
Tij = rand(ComplexF64, L, L)
Tij += Tij'
```

For a lattice model, enumerate nearest-neighbor pairs and fill `Tij`. If you have the
companion package [`FiniteLattices.jl`](https://github.com/Qiaoyi-Li/FiniteLattices.jl)
(optional, same author), you can use it to enumerate neighbors:

```julia
using FiniteLattices, BenchmarkFreeFermions
Latt = YCSqua(L, W)
matTij = zeros(size(Latt), size(Latt))
for (i, j) in neighbor(Latt; ordered=true)
    matTij[i, j] = t
end
```

Without `FiniteLattices.jl`, build the neighbor list by hand, e.g. for a 1D chain with
open boundaries (standard tight-binding `H = -t ∑ cᵢ†cᵢ₊₁` with `t > 0` corresponds to
`Tij = t`, since the minus sign is built into `EigenModes`):

```julia
L = 8
t = 1.0
Tij = zeros(ComplexF64, L, L)
for i in 1:L-1
    Tij[i, i+1] = Tij[i+1, i] = t   # Hermitian nearest-neighbor hopping
end
```

For spinful fermions, build a `2L×2L` block-diagonal matrix (spin-↑ top-left, spin-↓
bottom-right), or stack two identical blocks:

```julia
Tij = rand(ComplexF64, L, L)
Tij += Tij'
Tij = hcat(vcat(Tij, zeros(ComplexF64, L, L)),
           vcat(zeros(ComplexF64, L, L), Tij))
```

### Step 2: Diagonalize

```julia
# Eigenvalues only (cheaper, sufficient for thermodynamic quantities)
ϵ = SingleParticleSpectrum(Tij)

# Eigenvalues AND eigenvectors (needed for Green's functions and correlators)
# Convention: Tij = -V diagm(ϵ) V';  cᵢ = ∑ₖ V[i,k] fₖ
ϵ, V = EigenModes(Tij)
```

**Avoid degeneracy at the Fermi level** before computing zero-temperature observables —
perturb `Tij` slightly until the gap is non-zero:

```julia
while abs(ϵ[div(length(ϵ), 2)+1] - ϵ[div(length(ϵ), 2)]) < 1e-2
    for i in 1:size(Tij,1), j in i:size(Tij,2)
        if !iszero(Tij[i, j])
            Tij[i, j] += 0.01 * randn()
            Tij[j, i] = Tij[i, j]'
        end
    end
    ϵ, V = EigenModes(Tij)
end
```

### Step 3: Set the chemical potential

For half-filling at zero temperature, place `μ` at the gap midpoint:

```julia
μ = (ϵ[div(L, 2)] + ϵ[div(L, 2)+1]) / 2
ξ = ϵ .- μ          # shifted energies — GreenFunction takes ξ, not raw ϵ
```

For a target filling fraction `n ∈ (0,1)` at finite temperature, solve for `μ`:

```julia
μ = SolveChemicalPotential(ϵ, β, n)
```

### Step 4: Compute the reference Green's function

```julia
# Zero temperature, equal-time: Gᵢⱼ = ⟨cᵢ cⱼ†⟩
G = GreenFunction(ξ, V, Inf)

# Finite temperature
G = GreenFunction(ξ, V, β)

# Imaginary-time dependent
G = GreenFunction(ξ, V, β; τ=τ)

# Reversed convention: ⟨cᵢ† cⱼ⟩
G = GreenFunction(ξ, V, β; reverse=true)

# Single element (more efficient for sparse queries)
Gij = GreenFunction(ξ, V, β, i, j)
```

### Step 5: Compute exact observables and compare

**Ground-state energy** (spinless, half-filling):

```julia
Eg_ex = sum(ϵ[1:div(L, 2)])
```

Spinful SU(2) (2 particles per mode):

```julia
Eg_ex = 2 * sum(ϵ[1:div(L, 2)])
# Or for a 2L×2L block-diagonal Tij:
Eg_ex = sum(ϵ[1:L])
```

**Local density:**

```julia
n_i = Density(G, i)              # = 1 - G[i,i]
n  = Density(G)                  # vector over all sites
```

**Multi-fermion correlators** via `ExpectationValue(G, si, dagidx)`:

- `si` = list of site indices (e.g. `[i, i, j, j]`)
- `dagidx` = which positions in `si` carry a dagger (†)

| Observable | Operator | `si` | `dagidx` |
|---|---|---|---|
| `⟨nᵢ nⱼ⟩` | `cᵢ† cᵢ cⱼ† cⱼ` | `[i,i,j,j]` | `[1,3]` |
| `⟨cᵢ† cⱼ cₖ† cₗ⟩` | as written | `[i,j,k,l]` | `[1,3]` |
| `⟨cᵢ† cⱼ† cₖ cₗ⟩` (pairing) | as written | `[i,j,k,l]` | `[1,2]` |
| `⟨cᵢ cⱼ cₖ† cₗ†⟩` | as written | `[i,j,k,l]` | `[3,4]` |

```julia
# Density-density
O_ex = ExpectationValue(G, [i, i, j, j], [1, 3])

# 4-fermion hopping correlator
O_ex = ExpectationValue(G, [i, j, k, l], [1, 3])

# Pairing correlator
O_ex = ExpectationValue(G, [i, j, k, l], [1, 2])
```

**Two-point functions** (no `ExpectationValue` needed):

```julia
@test abs(Obs.FFdag[(i, j)] - G[i, j]) < tol                                          # ⟨cᵢ cⱼ†⟩
@test abs(Obs.FdagF[(i, j)] - (i == j ? 1 - G[i,i] : -G[j, i])) < tol                 # ⟨cᵢ† cⱼ⟩
```

**Spinful convention**: spin-↓ site indices are offset by `+L`. Examples (singlet/triplet
pairing, charge/spin bonds):

```julia
# Singlet pairing: (c↑† c↓† - c↓† c↑†)(c↓ c↑ - c↑ c↓)/2
O_ex = ExpectationValue(G, [i, j+L, k+L, l], [1, 2]) -
       ExpectationValue(G, [i, j+L, k, l+L], [1, 2])

# Triplet pairing: 3 c↑† c↑† c↑ c↑
O_ex = 3 * ExpectationValue(G, [i, j, l, k], [1, 2])

# Charge bond: (c↑† c↑ + c↓† c↓)(c↑† c↑ + c↓† c↓)
O_ex = 2 * (ExpectationValue(G, [i, j, k, l], [1, 3]) +
            ExpectationValue(G, [i, j, k+L, l+L], [1, 3]))

# Spin bond: 3/2 c↑† c↓ c↓† c↑
O_ex = 3/2 * ExpectationValue(G, [i, j+L, k+L, l], [1, 3])
```

For spinful SU(2) systems where you build the Slater determinant manually, construct
`ψ_ex` from `V` and compute `G` via a Slater-determinant Green's function routine
(`G′ = ψ_ex * ψ_ex'` for an orthonormal occupied subspace, or use your own Slater
Green's function routine):

```julia
ψ_ex = zeros(ComplexF64, 2*L, L)
ψ_ex[1:L, 1:div(L,2)]       = V[:, 1:div(L,2)]   # ψ↑
ψ_ex[L+1:2*L, div(L,2)+1:L] = V[:, 1:div(L,2)]   # ψ↓
G′ = ψ_ex * ψ_ex'                                   # Slater Green's function
```

### Step 6 (optional): Thermodynamic references

```julia
lnZ = LogPartition(ϵ, β, μ)
E   = Energy(ϵ, β, μ)
F   = FreeEnergy(ϵ, β, μ)
S   = Entropy(ϵ, β, μ)
Cμ  = SpecificHeat_μ(ϵ, β, μ)
CN  = SpecificHeat_N(ϵ, β, μ)
```

For a sweep over `β`:

```julia
lslnZ_ex = [LogPartition(ϵ, β, μ) for β in lsβ]
lsE_ex   = [Energy(ϵ, β, μ)       for β in lsβ]
```

### Step 7 (optional): Two-particle response

For quadratic operators `O₁`, `O₂` (each an `L×L` matrix in the real-space basis):

```julia
lsω, lsw = TwoParticleGreenFunction(ξ, V, β, O₁, O₂)
```

The retarded Green's function at frequency `ω + iη`:

```julia
GR_ω = sum(w ./ (ω .- ω₀ .+ im*η) for (ω₀, w) in zip(lsω, lsw))
```

Density structure factor (positive-frequency poles only):

```julia
sel = findall(x -> x > 0, lsω)
Dq  = -2 * imag(sum(lsw[sel] ./ (ω .- lsω[sel] .+ im*η)))
```

Optical conductivity: build current operators `O1`, `O2` and call
`TwoParticleGreenFunction(ξ, V, Inf, O1, O2)`.

### Step 8: Compare against the tensor network result

Typical tolerances for well-converged tensor network results: `1e-8` to `1e-12`.

```julia
@test abs(Eg_tn - Eg_ex) < 1e-8
@test abs(Obs.nn[(i, j)] - ExpectationValue(G, [i, i, j, j], [1, 3])) < 1e-8
```

## API Reference

### Single-Particle Spectrum (`src/SingleParticleSpectrum.jl`)

| Function | Signature | Returns |
|---|---|---|
| `SingleParticleSpectrum` | `(Tij::AbstractMatrix)` | `ϵ::Vector{Float64}` |
| `EigenModes` | `(Tij::AbstractMatrix{F}; tol=1e-12)` | `(ϵ, V)` — `Tij = -V diagm(ϵ) V'` |

### Thermodynamics (`src/Thermodynamics.jl`)

| Function | Signature | Returns |
|---|---|---|
| `n_fermion` | `(x::Real, β::Real)` / `(x, lsβ)` / `(lsx, β)` | Fermi-Dirac value(s) |
| `ParticleNumber` | `(ϵ, β, μ)` | `N = ∑ₖ n_F(ϵₖ - μ)` |
| `LogPartition` | `(ϵ, β, μ)` | `ln Z = ∑ₖ ln(1 + e^{-β(ϵₖ-μ)})` |
| `Energy` | `(ϵ, β, μ)` | `E = ∑ₖ ϵₖ n_F(ϵₖ - μ)` |
| `FreeEnergy` | `(ϵ, β, μ)` | `F = -ln Z / β + μ N` |
| `Entropy` | `(ϵ, β, μ)` | `S = β(E - F)` |
| `SpecificHeat_μ` | `(ϵ, β, μ)` | `C_μ` (fixed μ) |
| `SpecificHeat_N` | `(ϵ, β, μ)` | `C_N` (fixed N) |
| `SolveChemicalPotential` | `(ϵ, β, n; tol=1e-12)` | `μ` for desired average filling `n ∈ (0,1)` |

### Observables (`src/Observables.jl`)

| Function | Signature | Returns |
|---|---|---|
| `GreenFunction` | `(ξ, V, β; τ=0.0, tol=1e-12, reverse=false)` | `Gᵢⱼ = ⟨cᵢ(τ) cⱼ†⟩` |
| `GreenFunction` | `(ξ, V, β, i, j; τ=0.0, reverse=false)` | single element |
| `ExpectationValue` | `(G, si::Vector{Int}, dagidx::Vector{Int})` | multi-fermion correlator via Pfaffian |
| `TimeCorrelation` | `(ξ, V, β, si, dagidx, lsτ)` | time-ordered correlator on imaginary-time grid |
| `Density` | `(G, i::Int)` or `(G, lssi=1:size(G,1))` | `n_i = 1 - G[i,i]` or vector |

### Dynamics (`src/Dynamics.jl`)

| Function | Signature | Returns |
|---|---|---|
| `TwoParticleGreenFunction` | `(ξ, V, β, O₁, O₂; ωtol=1e-8, wtol=1e-8)` | `(lsω, lsw)` — pole positions and weights for `⟨⟨O₁; O₂⟩⟩_{ω+iη}` |

## Core Contract

- **Hamiltonian convention**: `H = -∑ᵢⱼ Tᵢⱼ cᵢ† cⱼ`. The minus sign is built into `EigenModes` — do not add it yourself.
- **`Tij` must be Hermitian**: enforce `Tij += Tij'` or symmetrize explicitly.
- **Zero-temperature = `β = Inf`**: pass `Inf` to `GreenFunction`, `TwoParticleGreenFunction`, etc.
- **Shift before Green's function**: `GreenFunction` takes `ξ = ϵ .- μ`, not raw `ϵ`.
- **Avoid Fermi-level degeneracy** at `T=0` — perturb `Tij` until the gap is non-zero.
- **Spinful site indexing**: spin-↓ indices are offset by `+L` in a `2L`-site layout.
- **`ExpectationValue` needs the full `G` matrix** (uses a Pfaffian via `SkewLinearAlgebra.pfaffian!`).
- **Tolerances**: use `1e-8` to `1e-12` for comparisons against well-converged tensor network results.

## Installation

The package is not on the General registry — install it from GitHub:

```julia
] add https://github.com/Qiaoyi-Li/BenchmarkFreeFermions.jl.git
```

Or add it manually to your `Project.toml`:

```toml
[deps]
BenchmarkFreeFermions = "5b68a00d-ca4d-4b58-ab0c-dc082ade624e"
```

and run `] instantiate` (Pkg will resolve it from the GitHub URL recorded in your
`Manifest.toml`, or you can `] dev /path/to/your/local/clone` for development).

Dependencies: `LinearAlgebra`, `SkewLinearAlgebra ~ 1`, `Statistics`. Julia ≥ 1.6.7.

## Verification Checklist

- [ ] `Tij` is Hermitian (`Tij == Tij'`)
- [ ] No degeneracy at the Fermi level for `T=0` calculations
- [ ] `ξ = ϵ .- μ` passed to `GreenFunction`, not raw `ϵ`
- [ ] `β = Inf` for ground-state (zero-temperature) references
- [ ] Site indices for spin-↓ offset by `+L` in spinful systems
- [ ] `dagidx` positions match the operator string you intend
- [ ] Comparison tolerance chosen appropriately (`1e-8` … `1e-12`)
- [ ] Ground-state energy formula matches the filling:
      `sum(ϵ[1:L÷2])` (spinless half-filling) ·
      `2*sum(ϵ[1:L÷2])` (spinful SU(2)) ·
      `sum(ϵ[1:L])` (block-diagonal `2L`)

## Reference Test Files

Real-world usage examples to imitate, in tensor-network packages that consume this package
as a benchmark oracle (all public on GitHub):

- [`MPSKit.jl`](https://github.com/QuantumKitHub/MPSKit.jl) —
  `examples/quantum1d/7.xy-finiteT/main.jl`: finite-temperature 1D XY model, compares
  MPSKit numerical methods (TaylorCluster MPO, MPO multiplication, TDVP imaginary-time
  evolution) against exact `LogPartition`, `FreeEnergy`, `Energy`, and ground-state
  energy from `BenchmarkFreeFermions`.
- [`FiniteMPS.jl`](https://github.com/Qiaoyi-Li/FiniteMPS.jl) (same author) —
  `test/mulsiteIntr.jl` (spinless, 1/2/3/4-site observables),
  `test/mulsiteIntr2.jl` (spinful SU(2), Slater-determinant Green's function),
  `test/mulsiteIntr3.jl` (spinful, pairing correlations),
  `test/FreeFermion.jl` (AutomataMPO correctness via free-fermion GS energy),
  `example/FreeFermion/tanTRG.jl` (thermal benchmarking of tanTRG).

These repos are **not** required to use `BenchmarkFreeFermions.jl` itself; they are
referenced only as concrete examples of how the free-fermion reference data is consumed
by tensor-network benchmarks.
