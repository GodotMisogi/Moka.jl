# Moka.jl — Domain Glossary

Moka.jl is a Julia implementation of the MPAS-Ocean shallow-water solver using the TRiSK discrete operator framework on unstructured hexagonal meshes.

## Core concepts

**TRiSK** — Discrete operator framework for MPAS hexagonal meshes. Defines divergence, curl, gradient, and Coriolis operators that are energy- and enstrophy-conserving by construction.

**Normal velocity** — Edge-normal velocity component; the prognostic momentum variable in TRiSK. Stored as `normalVelocity[nEdges, nVertLevels, nTimeLevels]`.

**Layer thickness** — Prognostic scalar field `h = H + η` where `H` is resting depth and `η` is SSH perturbation. Stored on cells.

**SSH (sea surface height)** — Diagnostic field derived from layer thickness: `ssh = layerThickness - restingThickness`.

**Forcing** — Externally applied surface wind stress projected to edge-normal direction. Loaded from `forcing.nc` into `ForcingVars`. Fields: `windStressZonal(nCells, nVertLevels)`, `windStressMeridional(nCells, nVertLevels)`.

## Benchmark: barotropic gyre

**Barotropic gyre** — Wind-driven depth-independent ocean circulation. The linearised shallow-water equations driven by a sinusoidal zonal wind stress `τ_x = −τ₀ cos(π y / L_y)` produce a double-gyre circulation.

**Munk layer** — Western boundary layer that balances the beta-effect with Laplacian viscosity. Width `δ_m = (ν₂/β)^(1/3)`; must be resolved by ≥ 3 grid cells.

**Munk layer width (no-slip)** — `δ_m = (2π/√3)(ν₂/β)^(1/3)`. At `ν₂ = 400 m²/s`, `β = 1e-11 s⁻¹m⁻¹`: δ_m ≈ 34 km, resolved at 40 km (0.85 cells) — marginal but the full formula above gives ~34 km → ≥ 3 cells requires ≤ 11 km, so 10 km is the finest benchmark resolution.

**No-slip BC** — Tangential velocity vanishes at domain walls. Produces a wider Munk layer than free-slip, easier to resolve.

**Beta-plane** — Approximation `f = f₀ + βy` linearising the variation of the Coriolis parameter with latitude.

## Physics parameters (barotropic gyre benchmark)

| Symbol | Value | Description |
|--------|-------|-------------|
| ν₂ | 400 m²/s | Laplacian (del2) horizontal viscosity |
| f₀ | 1×10⁻⁴ s⁻¹ | Reference Coriolis parameter |
| β | 1×10⁻¹¹ s⁻¹m⁻¹ | Beta-plane gradient |
| τ₀ | 0.1 N/m² | Wind stress amplitude |
| ρ | 1000 kg/m³ | Reference density |
| H | 5000 m | Resting ocean depth |
| g | 9.81 m/s² | Gravitational acceleration |
| Lx, Ly | 1200 km | Domain extent |

## Time integration

**ForwardEuler** — First-order explicit; stable but requires many steps. Used for short tests.

**RungeKutta4 (RK4)** — Classical 4-stage explicit method. ~3× fewer steps than ForwardEuler for equivalent accuracy over a 2-year gyre spin-up.

CFL condition for gravity waves: `dt ≤ dx / (2 c)` where `c = √(gH) ≈ 221 m/s`. At 40 km: dt ≤ ~90 s; 4 min timestep gives CFL ≈ 1.33 (stable with RK4).
