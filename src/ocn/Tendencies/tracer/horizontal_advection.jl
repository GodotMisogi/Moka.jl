"""
Horizontal advection of tracers by the thickness flux, in **advective form** so a
spatially-constant tracer is preserved exactly and boundary (wall) edges need no
special data. For each tracer ``\\phi`` and cell ``i``:

```math
\\frac{\\partial \\phi_i}{\\partial t} \\mathrel{-}=
    \\frac{1}{h_i A_i} \\sum_{e \\in \\mathrm{EC}(i)}
        n_{e,i}\\, F_e\\, (\\bar\\phi_e - \\phi_i)\\, l_e
```

where ``F_e`` is the thickness flux, ``\\bar\\phi_e = \\tfrac12(\\phi_i + \\phi_j)``
the edge average (``j`` the neighbour across edge ``e``), and ``h_i`` the layer
thickness. Subtracting ``\\phi_i`` turns the flux form into the advective form:
``\\bar\\phi_e - \\phi_i = \\tfrac12(\\phi_j - \\phi_i)`` vanishes for a uniform
tracer, so ``\\partial\\phi/\\partial t = 0`` exactly. Solid-wall edges
(`boundaryEdge == 1`) are skipped (no-normal-flux), which also avoids index-0 loads
from the missing neighbour cell.
"""
function tracer_horizontal_advection_tendency!(Tend::TendencyVars,
                                               Prog::PrognosticVars,
                                               Diag::DiagnosticVars,
                                               Mesh::Mesh;
                                               nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendTracers)

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, Edges = HorzMesh
    @unpack dvEdge, cellsOnEdge, boundaryEdge = Edges
    @unpack maxLevelEdge = VertMesh
    @unpack nCells, nEdgesOnCell = PrimaryCells
    @unpack edgesOnCell, edgeSignOnCell, areaCell = PrimaryCells

    @unpack thicknessFlux = Diag
    tracers        = Prog.tracers[end]
    layerThickness = Prog.layerThickness[end]
    @unpack tendTracers = Tend

    kernel! = tracer_flux_div_kernel!(backend, nthreads)
    kernel!(tendTracers,
            tracers,
            thicknessFlux,
            layerThickness,
            nEdgesOnCell,
            edgesOnCell,
            cellsOnEdge,
            boundaryEdge,
            maxLevelEdge.Top,
            edgeSignOnCell,
            dvEdge,
            areaCell,
            ndrange=nCells)

    @pack! Tend = tendTracers
end

@kernel function tracer_flux_div_kernel!(tendency,
                                         @Const(tracers),
                                         @Const(thicknessFlux),
                                         @Const(layerThickness),
                                         @Const(nEdgesOnCell),
                                         @Const(edgesOnCell),
                                         @Const(cellsOnEdge),
                                         @Const(boundaryEdge),
                                         @Const(maxLevelEdgeTop),
                                         @Const(edgeSignOnCell),
                                         @Const(dvEdge),
                                         @Const(areaCell))

    iCell = @index(Global, Linear)

    @inbounds invArea = 1.0 / areaCell[iCell]

    @inbounds for i in 1:nEdgesOnCell[iCell]
        iEdge = edgesOnCell[i, iCell]
        # skip solid-wall edges: no-normal-flux, and avoids index-0 neighbour loads.
        if boundaryEdge[iEdge] != 1
            c1 = cellsOnEdge[1, iEdge]
            c2 = cellsOnEdge[2, iEdge]
            # the neighbour across this edge (the cell that is not iCell)
            jCell = (c1 == iCell) ? c2 : c1
            coef = dvEdge[iEdge] * edgeSignOnCell[i, iCell] * invArea
            @inbounds for k in 1:maxLevelEdgeTop[iEdge]
                # advective form: (φ̄_e − φ_i) = ½(φ_j − φ_i) vanishes for a uniform
                # tracer, so a constant is preserved exactly. Divide the h*φ flux
                # divergence by the layer thickness (guard h==0).
                hk = layerThickness[k, iCell]
                invh = hk > 0.0 ? 1.0 / hk : 0.0
                fluxk = thicknessFlux[k, iEdge] * coef * invh
                @inbounds for t in 1:size(tracers, 1)
                    dφe = 0.5 * (tracers[t, k, jCell] - tracers[t, k, iCell])
                    tendency[t, k, iCell] += fluxk * dφe
                end
            end
        end
    end
end
