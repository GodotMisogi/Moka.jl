module EquationOfState

export AbstractEOS, LinearEOS, compute_density!, LinearEOSParams

using KernelAbstractions
using MOKA: Mesh, DEFAULT_NTHREADS
const KA = KernelAbstractions

"""
    AbstractEOS

Supertype for equation-of-state formulations. Dispatch [`compute_density!`](@ref)
on a concrete subtype to select the EOS. See [`LinearEOS`](@ref).
"""
abstract type AbstractEOS end

"""
    LinearEOS <: AbstractEOS

Linear equation of state

```math
\\rho = \\rho_0 \\left(1 - \\alpha (T - T_0) + \\beta (S - S_0)\\right)
```

with thermal-expansion coefficient ``\\alpha``, haline-contraction coefficient
``\\beta``, and reference state ``(\\rho_0, T_0, S_0)``. The coefficients are
carried in a [`LinearEOSParams`](@ref) passed to [`compute_density!`](@ref).
"""
abstract type LinearEOS <: AbstractEOS end

"""
    LinearEOSParams(; ρ0=1000.0, T0=0.0, S0=0.0, α=2.0e-4, β=8.0e-4)

Coefficients of the [`LinearEOS`](@ref). Defaults follow common ocean values
(``\\alpha \\approx 2\\times10^{-4}\\,\\mathrm{K^{-1}}``,
``\\beta \\approx 8\\times10^{-4}\\,\\mathrm{psu^{-1}}``). A plain immutable of
scalars — passed by value to the kernel launcher (not into the kernel), so it does
not interfere with Enzyme AD of the array arguments.
"""
Base.@kwdef struct LinearEOSParams
    ρ0::Float64 = 1000.0
    T0::Float64 = 0.0
    S0::Float64 = 0.0
    α::Float64  = 2.0e-4
    β::Float64  = 8.0e-4
end

"""
    compute_density!(density, temperature, salinity, mesh, LinearEOS, params;
                     nthreads=DEFAULT_NTHREADS)

Fill `density` `(nVertLevels, nCells)` from the `temperature` and `salinity`
tracer fields using the [`LinearEOS`](@ref). The scalar EOS coefficients are
carried as compile-time `Val` kernel arguments (so no active `Float64` kernel
argument is introduced — the same rule the integrator kernels follow for AD).
"""
function compute_density!(density, temperature, salinity, Mesh::Mesh,
                          ::Type{LinearEOS}, params::LinearEOSParams;
                          nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(density)
    nCells      = Mesh.HorzMesh.PrimaryCells.nCells
    nVertLevels = Mesh.VertMesh.nVertLevels

    kernel! = linear_eos_kernel!(backend, nthreads)
    kernel!(density, temperature, salinity, nCells,
            Val(params.ρ0), Val(params.T0), Val(params.S0),
            Val(params.α), Val(params.β),
            ndrange=(nCells, nVertLevels))
end

# ρ = ρ0 (1 - α(T-T0) + β(S-S0)). The five EOS scalars are compile-time Val type
# parameters (not runtime args) so Enzyme sees no active Float64 kernel argument.
@kernel function linear_eos_kernel!(density,
                                    @Const(temperature),
                                    @Const(salinity),
                                    arrayLength,
                                    ::Val{ρ0}, ::Val{T0}, ::Val{S0},
                                    ::Val{α}, ::Val{β}) where {ρ0, T0, S0, α, β}
    iCell, k = @index(Global, NTuple)
    if iCell < arrayLength + 1
        @inbounds density[k, iCell] =
            ρ0 * (1.0 - α * (temperature[k, iCell] - T0) +
                        β * (salinity[k, iCell] - S0))
    end
end

end # module EquationOfState
