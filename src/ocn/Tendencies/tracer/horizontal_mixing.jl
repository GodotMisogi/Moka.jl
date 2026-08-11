"""
Laplacian (del2) horizontal mixing of tracers:

```math
\\frac{\\partial \\phi_i}{\\partial t} \\mathrel{+}=
    \\frac{\\kappa}{A_i} \\sum_{e \\in \\mathrm{EC}(i)}
        n_{e,i}\\, \\frac{\\phi_{c_2(e)} - \\phi_{c_1(e)}}{d_e}\\, l_e
```

i.e. the divergence of the tracer gradient scaled by the diffusivity ``\\kappa``
(`tracerDel2`). Boundary edges (index-0 neighbour) are skipped, giving a
no-normal-flux wall. Uniform tracers have zero gradient, so a constant field is
untouched — verified in the tests.
"""
function tracer_horizontal_mixing_tendency!(Tend::TendencyVars,
                                            Prog::PrognosticVars,
                                            Diag::DiagnosticVars,
                                            Mesh::Mesh;
                                            tracerDel2::Float64=0.0,
                                            nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendTracers)

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, Edges = HorzMesh
    @unpack dvEdge, dcEdge, cellsOnEdge, boundaryEdge = Edges
    @unpack maxLevelEdge = VertMesh
    @unpack nCells, nEdgesOnCell = PrimaryCells
    @unpack edgesOnCell, edgeSignOnCell, areaCell = PrimaryCells

    tracers = Prog.tracers[end]
    @unpack tendTracers = Tend

    kernel! = tracer_del2_kernel!(backend, nthreads)
    kernel!(tendTracers,
            tracers,
            nEdgesOnCell,
            edgesOnCell,
            cellsOnEdge,
            maxLevelEdge.Top,
            edgeSignOnCell,
            dvEdge,
            dcEdge,
            areaCell,
            boundaryEdge,
            tracerDel2,
            ndrange=nCells)

    @pack! Tend = tendTracers
end

@kernel function tracer_del2_kernel!(tendency,
                                     @Const(tracers),
                                     @Const(nEdgesOnCell),
                                     @Const(edgesOnCell),
                                     @Const(cellsOnEdge),
                                     @Const(maxLevelEdgeTop),
                                     @Const(edgeSignOnCell),
                                     @Const(dvEdge),
                                     @Const(dcEdge),
                                     @Const(areaCell),
                                     @Const(boundaryEdge),
                                     κ)

    iCell = @index(Global, Linear)

    @inbounds invArea = 1.0 / areaCell[iCell]

    @inbounds for i in 1:nEdgesOnCell[iCell]
        iEdge = edgesOnCell[i, iCell]
        # skip solid-wall edges (no-normal-flux); structured branch for AD safety.
        if boundaryEdge[iEdge] != 1
            c1 = cellsOnEdge[1, iEdge]
            c2 = cellsOnEdge[2, iEdge]
            # -edgeSignOnCell so the divergence of the outward gradient diffuses
            # toward neighbours (sign matches the momentum del2 convention).
            coef = -κ * edgeSignOnCell[i, iCell] * dvEdge[iEdge] * invArea / dcEdge[iEdge]
            @inbounds for k in 1:maxLevelEdgeTop[iEdge]
                @inbounds for t in 1:size(tracers, 1)
                    grad = tracers[t, k, c2] - tracers[t, k, c1]
                    tendency[t, k, iCell] += coef * grad
                end
            end
        end
    end
end
