# ADR 0001: Extend RK4 time-stepper to accept ForcingVars

**Date**: 2026-06-09
**Status**: Accepted

## Context

The barotropic gyre benchmark requires wind-stress forcing applied at every tendency evaluation. The existing `ocn_run_loop` dispatcher always passes a `ForcingVars` argument to `ocn_timestep`, but the RK4 method signature did not accept it. This left the RK4 code path permanently broken (it also called a non-existent `computeTendency!` function).

A 2-year spin-up to gyre steady state requires approximately:
- ~350,000 steps at 40 km with ForwardEuler (1-minute dt)
- ~123,000 steps at 40 km with RK4 (4-minute dt at CFL ≈ 1.33)

## Decision

Replace the broken `ocn_timestep(..., ::Type{RungeKutta4})` method with one that matches the full signature used by `ocn_run_loop`:

```julia
ocn_timestep(_timestep, Prog, Diag, Tend, Forcing, S, ::Type{RungeKutta4}; backend)
```

Inside the method:
- `dt` is derived from `S.timeManager.timeStep` (same source as the passed `_timestep`, but avoids GPU scalar indexing in the tendency kernels).
- `Forcing` is threaded through to `computeNormalVelocityTendency!`, which projects wind stress onto edges.
- `diagnostic_compute!` gains the `backend` keyword that was previously missing.

## Consequences

- RK4 is now the recommended integrator for all production runs (gyre and future benchmarks).
- ForwardEuler remains available for short tests and as a simpler reference.
- The old IGW driver (`_research/inertialGravityWave/mpas_ocean.jl`) uses the pre-fix 4-return `ocn_init` signature and must be updated separately before it can be run.
