# Phase 5 (tracers + EOS + baroclinic PGF) verification. Standalone; uses the BG
# mesh. Checks:
#   1. LinearEOS matches the closed-form ρ = ρ0(1 - α(T-T0) + β(S-S0)), per level.
#   2. Tracer horizontal advection preserves a spatially-CONSTANT tracer (zero
#      tendency) and, with a structured field, produces a nonzero tendency.
#   3. Tracer del2 mixing leaves a constant tracer untouched and diffuses a
#      structured one (interior tendency nonzero).
#   4. Baroclinic PGF with UNIFORM density (ρ ≡ ρ0) reduces EXACTLY to the
#      barotropic sshGradient; with a density anomaly it differs.
#   5. A full ForwardEuler step advances tracers (end-to-end wiring works).

using MOKA
using NCDatasets
using Printf
using Statistics
import KernelAbstractions as KA
using Dates
using UnPack

const NV  = MOKA.NormalVelocity
const EOS = MOKA.EquationOfState
const BG20 = joinpath(@__DIR__, "..", "examples", "barotropic_gyre", "20km")
const INIT = joinpath(BG20, "initial_state.nc")

function build_mesh(; nVertLevels=3, backend=KA.CPU())
    h  = read_horz_mesh(INIT; backend=backend)
    vm = VerticalMesh(h; nVertLevels=nVertLevels, backend=backend)
    return Mesh(h, vm)
end

function check_eos()
    println(">> LinearEOS vs closed form")
    backend = KA.CPU(); nVL = 3
    mesh = build_mesh(; nVertLevels=nVL, backend=backend)
    nC = mesh.HorzMesh.PrimaryCells.nCells
    T = KA.zeros(backend, Float64, nVL, nC); fill!(T, 5.0)
    S = KA.zeros(backend, Float64, nVL, nC); fill!(S, 35.0)
    ρ = KA.zeros(backend, Float64, nVL, nC)
    p = EOS.LinearEOSParams(ρ0=1000.0, T0=0.0, S0=0.0, α=2e-4, β=8e-4)
    EOS.compute_density!(ρ, T, S, mesh, EOS.LinearEOS, p)
    expected = 1000.0*(1 - 2e-4*5.0 + 8e-4*35.0)
    err = maximum(abs, Array(ρ) .- expected) / expected
    @printf("   rel err vs closed form = %.3e (ρ=%.4f, expected=%.4f)\n",
            err, Array(ρ)[1,1], expected)
    @assert err < 1e-12
    println("   PASS: EOS matches closed form")
end

# minimal Prog/Diag/Tend with nTracers tracers, seeded thickness flux
function make_state(mesh; nTracers=1, backend=KA.CPU())
    nC  = mesh.HorzMesh.PrimaryCells.nCells
    nE  = mesh.HorzMesh.Edges.nEdges
    nVL = mesh.VertMesh.nVertLevels
    ssh = KA.zeros(backend, Float64, nC)
    vel = KA.zeros(backend, Float64, nVL, nE)
    lth = KA.zeros(backend, Float64, nVL, nC); fill!(lth, 100.0)
    tr  = KA.zeros(backend, Float64, nTracers, nVL, nC)
    Prog = MOKA.PrognosticVars(ssh, vel, lth, 2; tracers=tr)
    Diag = MOKA.DiagnosticVars(mesh; backend=backend)
    Tend = MOKA.TendencyVars(mesh; backend=backend, nTracers=nTracers)
    return Prog, Diag, Tend
end

function check_tracer_advection()
    println(">> Tracer advection: constant preserved, structured nonzero")
    backend = KA.CPU()
    mesh = build_mesh(; backend=backend)
    nVL = mesh.VertMesh.nVertLevels
    nC  = mesh.HorzMesh.PrimaryCells.nCells

    # constant tracer with a NONZERO thickness flux => advection tendency must be 0
    Prog, Diag, Tend = make_state(mesh; nTracers=1, backend=backend)
    Prog.tracers[end] .= 7.0
    fill!(Diag.thicknessFlux, 3.0)          # arbitrary nonzero flux
    MOKA.Tracer.compute_tracer_tendency!(Tend, Prog, Diag, mesh)
    m1 = maximum(abs, Array(Tend.tendTracers))
    @printf("   constant tracer, nonzero flux: max|tend| = %.3e\n", m1)
    @assert m1 < 1e-10 "advection of a constant tracer must give zero tendency"

    # structured tracer => nonzero tendency somewhere
    xc = Array(mesh.HorzMesh.PrimaryCells.xᶜ); Lx = maximum(xc)-minimum(xc)
    for k in 1:nVL, c in 1:nC
        Prog.tracers[end][1, k, c] = sin(2π*xc[c]/Lx)
    end
    MOKA.Tracer.compute_tracer_tendency!(Tend, Prog, Diag, mesh)
    m2 = maximum(abs, Array(Tend.tendTracers))
    @printf("   structured tracer: max|tend| = %.3e\n", m2)
    @assert m2 > 1e-8 "structured tracer should advect (nonzero tendency)"
    println("   PASS: tracer advection correct")
end

function check_tracer_mixing()
    println(">> Tracer del2 mixing: constant preserved, structured diffuses")
    backend = KA.CPU()
    mesh = build_mesh(; backend=backend)
    nVL = mesh.VertMesh.nVertLevels
    nC  = mesh.HorzMesh.PrimaryCells.nCells

    Prog, Diag, Tend = make_state(mesh; nTracers=1, backend=backend)
    Prog.tracers[end] .= 7.0
    MOKA.Tracer.compute_tracer_tendency!(Tend, Prog, Diag, mesh; tracerDel2=100.0)
    m1 = maximum(abs, Array(Tend.tendTracers))
    @printf("   constant tracer + mixing: max|tend| = %.3e\n", m1)
    @assert m1 < 1e-10 "mixing of a constant tracer must give zero tendency"

    xc = Array(mesh.HorzMesh.PrimaryCells.xᶜ); Lx = maximum(xc)-minimum(xc)
    for k in 1:nVL, c in 1:nC
        Prog.tracers[end][1, k, c] = sin(2π*xc[c]/Lx)
    end
    fill!(Diag.thicknessFlux, 0.0)   # isolate mixing (no advection)
    MOKA.Tracer.compute_tracer_tendency!(Tend, Prog, Diag, mesh; tracerDel2=100.0)
    m2 = maximum(abs, Array(Tend.tendTracers))
    @printf("   structured tracer + mixing: max|tend| = %.3e\n", m2)
    @assert m2 > 1e-8 "structured tracer should diffuse"
    println("   PASS: tracer mixing correct")
end

function check_baroclinic_pgf()
    println(">> Baroclinic PGF reduces to barotropic for uniform density")
    backend = KA.CPU()
    mesh = build_mesh(; backend=backend)
    nVL = mesh.VertMesh.nVertLevels
    nC  = mesh.HorzMesh.PrimaryCells.nCells
    nE  = mesh.HorzMesh.Edges.nEdges

    # structured ssh so the barotropic gradient is nonzero
    xc = Array(mesh.HorzMesh.PrimaryCells.xᶜ); Lx = maximum(xc)-minimum(xc)
    ssh = KA.zeros(backend, Float64, nC)
    for c in 1:nC; ssh[c] = 0.1*sin(2π*xc[c]/Lx); end
    vel = KA.zeros(backend, Float64, nVL, nE)
    lth = KA.zeros(backend, Float64, nVL, nC); fill!(lth, 100.0)
    Prog = MOKA.PrognosticVars(ssh, vel, lth, 2)
    Diag = MOKA.DiagnosticVars(mesh; backend=backend)

    ρ0 = Array(mesh.Constants.density)[1]

    # barotropic reference
    Tend_b = MOKA.TendencyVars(mesh; backend=backend)
    NV.pressure_gradient_tendency!(Tend_b, Prog, Diag, mesh, NV.sshGradient)
    tb = copy(Array(Tend_b.tendNormalVelocity))

    # baroclinic with UNIFORM density ρ ≡ ρ0 -> must match barotropic
    Tend_u = MOKA.TendencyVars(mesh; backend=backend)
    fill!(Diag.density, ρ0)
    NV.pressure_gradient_tendency!(Tend_u, Prog, Diag, mesh, NV.baroclinicGradient)
    tu = Array(Tend_u.tendNormalVelocity)
    err = maximum(abs, tu .- tb) / max(maximum(abs, tb), eps())
    @printf("   uniform-ρ baroclinic vs barotropic: rel err = %.3e\n", err)
    @assert err < 1e-10 "uniform-density baroclinic PGF must equal the barotropic term"

    # baroclinic with a density anomaly -> must differ
    Tend_a = MOKA.TendencyVars(mesh; backend=backend)
    for k in 1:nVL, c in 1:nC
        Diag.density[k, c] = ρ0 + 2.0*sin(2π*xc[c]/Lx)
    end
    NV.pressure_gradient_tendency!(Tend_a, Prog, Diag, mesh, NV.baroclinicGradient)
    ta = Array(Tend_a.tendNormalVelocity)
    d = sum(abs2, ta .- tb) / max(sum(abs2, tb), eps())
    @printf("   density-anomaly baroclinic vs barotropic: rel diff = %.3e\n", d)
    @assert d > 1e-6 "density anomaly should change the PGF"
    println("   PASS: baroclinic PGF correct")
end

function check_end_to_end_tracer()
    println(">> ForwardEuler step advances tracers end-to-end")
    backend = KA.CPU()
    mesh = build_mesh(; backend=backend)
    nVL = mesh.VertMesh.nVertLevels
    nC  = mesh.HorzMesh.PrimaryCells.nCells
    nE  = mesh.HorzMesh.Edges.nEdges

    xc = Array(mesh.HorzMesh.PrimaryCells.xᶜ); Lx = maximum(xc)-minimum(xc)
    xe = Array(mesh.HorzMesh.Edges.xᵉ)
    ssh = KA.zeros(backend, Float64, nC)
    vel = KA.zeros(backend, Float64, nVL, nE)
    for k in 1:nVL, e in 1:nE; vel[k,e] = 0.5*sin(2π*xe[e]/Lx); end
    lth = KA.zeros(backend, Float64, nVL, nC); fill!(lth, 100.0)
    tr  = KA.zeros(backend, Float64, 1, nVL, nC)
    for k in 1:nVL, c in 1:nC; tr[1,k,c] = sin(2π*xc[c]/Lx); end
    Prog = MOKA.PrognosticVars(ssh, vel, lth, 2; tracers=tr)
    Diag = MOKA.DiagnosticVars(mesh; backend=backend)
    Tend = MOKA.TendencyVars(mesh; backend=backend, nTracers=1)

    before = copy(Array(Prog.tracers[end]))
    dt = KA.zeros(backend, Float64, 1); dt[1] = 1.0
    # Two steps: diagnostic_compute! fills layerThicknessEdge last, so the
    # thicknessFlux (and hence tracer advection) is only nonzero from step 2 on —
    # matching the real run loop where diagnostics are recomputed each step.
    MOKA.ocn_timestep(dt, Prog, Diag, Tend, mesh, ForwardEuler)
    MOKA.ocn_timestep(dt, Prog, Diag, Tend, mesh, ForwardEuler)
    after = Array(Prog.tracers[end])
    d = maximum(abs, after .- before)
    @printf("   max tracer change after 2 steps = %.3e\n", d)
    @assert d > 1e-8 "tracer should change over steps with nonzero velocity"
    @assert all(isfinite, after) "tracer became non-finite"
    println("   PASS: end-to-end tracer stepping works")
end

function main()
    check_eos()
    check_tracer_advection()
    check_tracer_mixing()
    check_baroclinic_pgf()
    check_end_to_end_tracer()
    println("\nAll Phase 5 verification checks PASSED.")
end

main()
