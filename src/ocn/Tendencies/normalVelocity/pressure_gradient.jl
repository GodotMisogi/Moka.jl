# define our parent abstract type
abstract type PressureGradient end

using KernelAbstractions
const KA=KernelAbstractions

# define the supported PressureGradient types to dispatch on.
abstract type sshGradient <: PressureGradient end

# Baroclinic pressure gradient: barotropic SSH term + hydrostatic pressure from
# the density field (diagnosed from tracers via the EOS). Reduces exactly to
# `sshGradient` when the density is horizontally uniform.
abstract type baroclinicGradient <: PressureGradient end

function pressure_gradient_tendency!(Tend::TendencyVars,
                                     Prog::PrognosticVars,
                                     Diag::DiagnosticVars,
                                     Mesh::Mesh,
                                     ::Type{sshGradient};
                                     nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendNormalVelocity)

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh
    @unpack nEdges, dcEdge, cellsOnEdge, boundaryEdge = Edges

    # gravity as a 1-element (inactive) device array, like momentumDel2 — a bare
    # Float64 kernel arg is classified Active by Enzyme reverse mode.
    gravity = Mesh.Constants.gravity

    ssh = Prog.ssh[end]
    @unpack tendNormalVelocity = Tend

    kernel! = SSHGradOnEdge!(backend, nthreads)
    kernel!(tendNormalVelocity,
            ssh,
            cellsOnEdge,
            dcEdge,
            boundaryEdge,
            maxLevelEdge.Top,
            gravity,
            ndrange=nEdges)

    @pack! Tend = tendNormalVelocity
end

@kernel function SSHGradOnEdge!(tendency,
                                ssh,
                                cellsOnEdge,
                                dcEdge,
                                boundaryEdge,
                                maxLevelEdgeTop,
                                gravity)

    iEdge = @index(Global, Linear)

    if boundaryEdge[iEdge] != 1
        @inbounds jCell1 = cellsOnEdge[1,iEdge]
        @inbounds jCell2 = cellsOnEdge[2,iEdge]

        @inbounds InvDcEdge = 1.0 / dcEdge[iEdge]

        @inbounds g = gravity[1]
        for k in 1:maxLevelEdgeTop[iEdge]
            tendency[k, iEdge] -= g * InvDcEdge * (ssh[jCell2] - ssh[jCell1])
        end
    end
end

# ---------------------------------------------------------------------------
# Baroclinic pressure gradient (Phase 5)
# ---------------------------------------------------------------------------
# Hydrostatic pressure on cells, split into a barotropic (surface) part and a
# baroclinic part built from the density ANOMALY (ρ - ρ0), where ρ is diagnosed
# from tracers via the EOS into Diag.density:
#   p[k,i] = g ρ0 ssh[i] + g Σ_{k'=1}^{k} (ρ[k',i] - ρ0) h[k',i]
# and the horizontal PGF is  du/dt -= (1/ρ0) ∇_e p. Using the anomaly (not the full
# density) is what makes this reduce EXACTLY to the barotropic -g ∇ssh when the
# density is uniform (ρ ≡ ρ0 ⇒ anomaly = 0 ⇒ p = g ρ0 ssh).
function pressure_gradient_tendency!(Tend::TendencyVars,
                                     Prog::PrognosticVars,
                                     Diag::DiagnosticVars,
                                     Mesh::Mesh,
                                     ::Type{baroclinicGradient};
                                     nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendNormalVelocity)

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, Edges = HorzMesh
    @unpack nVertLevels, maxLevelEdge = VertMesh
    @unpack nEdges, dcEdge, cellsOnEdge, boundaryEdge = Edges
    nCells = PrimaryCells.nCells

    gravity = Mesh.Constants.gravity
    ρ0      = Mesh.Constants.density   # reference density ρ0 (1-element array)

    ssh            = Prog.ssh[end]
    layerThickness = Prog.layerThickness[end]
    @unpack pressure, density = Diag   # density: in-situ, pre-filled by the EOS
    @unpack tendNormalVelocity = Tend

    # 1. hydrostatic pressure column integral (per cell)
    p! = hydrostatic_pressure_kernel!(backend, nthreads)
    p!(pressure, density, layerThickness, ssh, gravity, ρ0, nVertLevels, nCells,
       ndrange=nCells)

    # 2. edge gradient of pressure -> tendency
    g! = pressure_grad_on_edge_kernel!(backend, nthreads)
    g!(tendNormalVelocity, pressure, cellsOnEdge, dcEdge, boundaryEdge,
       maxLevelEdge.Top, ρ0, ndrange=nEdges)

    @pack! Tend = tendNormalVelocity
end

@kernel function hydrostatic_pressure_kernel!(pressure,
                                              @Const(density),
                                              @Const(layerThickness),
                                              @Const(ssh),
                                              @Const(gravity),
                                              @Const(rho0),
                                              nVertLevels,
                                              arrayLength)
    iCell = @index(Global, Linear)
    if iCell < arrayLength + 1
        @inbounds g  = gravity[1]
        @inbounds ρ0 = rho0[1]
        # surface (barotropic) contribution
        acc = g * ρ0 * ssh[iCell]
        @inbounds for k in 1:nVertLevels
            # accumulate the density-anomaly weight, then store pressure AT layer k.
            # Using (ρ - ρ0) makes p = g ρ0 ssh when density is uniform, so the
            # gradient reduces to the barotropic term exactly.
            acc += g * (density[k, iCell] - ρ0) * layerThickness[k, iCell]
            pressure[k, iCell] = acc
        end
    end
end

@kernel function pressure_grad_on_edge_kernel!(tendency,
                                               @Const(pressure),
                                               @Const(cellsOnEdge),
                                               @Const(dcEdge),
                                               @Const(boundaryEdge),
                                               @Const(maxLevelEdgeTop),
                                               @Const(rho0))
    iEdge = @index(Global, Linear)
    if boundaryEdge[iEdge] != 1
        @inbounds jCell1 = cellsOnEdge[1, iEdge]
        @inbounds jCell2 = cellsOnEdge[2, iEdge]
        @inbounds invDc = 1.0 / dcEdge[iEdge]
        @inbounds invρ0 = 1.0 / rho0[1]
        @inbounds for k in 1:maxLevelEdgeTop[iEdge]
            tendency[k, iEdge] -= invρ0 * invDc *
                (pressure[k, jCell2] - pressure[k, jCell1])
        end
    end
end
