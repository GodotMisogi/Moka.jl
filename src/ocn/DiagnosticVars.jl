import Adapt

using CUDA
using KernelAbstractions

mutable struct DiagnosticVars{F <: AbstractFloat, FV2 <: AbstractArray{F,2}}

    # var: layer thickness averaged from cell centers to edges [m]
    # dim: (nVertLevels, nEdges)
    layerThicknessEdge::FV2

    # var: layer thickness averaged from cell centers to vertices [m]
    # dim: (nVertLevels, nVertices)
    layerThicknessVertex::FV2

    # var: divergence of horizonal velocity [s^{-1}]
    # dim: (nVertLevels, nCells)
    velocityDivCell::FV2

    # var: curl of horizontal velocity [s^{-1}]
    # dim: (nVertLevels, nVertices)
    relativeVorticityVertex::FV2

    norm_rel_vort_vertex::FV2
    norm_rel_vort_edge::FV2

    norm_planet_vort_vertex::FV2
    norm_planet_vort_edge::FV2

    #= UNUSED FOR NOW:
    # var: flux divergence [m s^{-1}] ?
    # dim: (nVertLevels, nCells)
    div_hu::Array{F,2}

    # var: Gradient of sea surface height at edges. [-]
    # dim: (nEdges), Time)?
    gradSSH::Array{F,1}

    # var: horizontal velocity, tangential to an edge [m s^{-1}]
    # dim: (nVertLevels, nEdges)
    tangentialVelocity::Array{F, 2}

    # var: kinetic energy of horizonal velocity on cells [m^{2} s^{-2}]
    # dim: (nVertLevels, nCells)
    kineticEnergyCell::Array{F, 2}

    =#

    function DiagnosticVars(layerThicknessEdge::AT2D,
                            layerThicknessVertex::AT2D,
                            velocityDivCell::AT2D,
                            relativeVorticityVertex::AT2D,
                            norm_rel_vort_vertex::AT2D,
                            norm_rel_vort_edge::AT2D,
                            norm_planet_vort_vertex::AT2D,
                            norm_planet_vort_edge::AT2D) where {AT2D}
        # pack all the arguments into a tuple for type and backend checking
        args = (layerThicknessEdge, layerThicknessVertex,
                velocityDivCell, relativeVorticityVertex,
                norm_rel_vort_vertex, norm_rel_vort_vertex,
                norm_planet_vort_vertex, norm_rel_vort_vertex)

        # check the type names; irrespective of type parameters
        # (e.g. `Array` instead of `Array{Float64, 1}`)
        check_typeof_args(args)
        # check that all args are on the same backend
        check_args_backend(args)
        # check that all args have the same `eltype` and get that type
        type = check_eltype_args(args)

        new{type, AT2D}(layerThicknessEdge,
                        layerThicknessVertex,
                        velocityDivCell,
                        relativeVorticityVertex,
                        norm_rel_vort_vertex,
                        norm_rel_vort_edge,
                        norm_planet_vort_vertex,
                        norm_planet_vort_edge)
    end
end

function DiagnosticVars(config::GlobalConfig, Mesh::Mesh; backend=KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    nEdges = Edges.nEdges
    nCells = PrimaryCells.nCells
    nVertices = DualCells.nVertices
    nVertLevels = VertMesh.nVertLevels

    # Here in the init function is where some sifting through will
    # need to be done, such that only diagnostic variables required by
    # the `Config` or requested by the `streams` will be activated.

    FT = Float64

    # create zero vectors to store diagnostic variables, on desired backend
    velocityDivCell = KA.zeros(backend, FT, nVertLevels, nCells)
    # initialize to Inf to avoid divide by zero and NaN problems
    layerThicknessEdge = KA.ones(backend, FT, nVertLevels, nEdges) * -typemax(FT)
    layerThicknessVertex = KA.ones(backend, FT, nVertLevels, nVertices) * -typemax(FT)
    relativeVorticityVertex = KA.zeros(backend, FT, nVertLevels, nVertices)
    norm_rel_vort_vertex = KA.zeros(backend, FT, nVertLevels, nVertices)
    norm_rel_vort_edge = KA.zeros(backend, FT, nVertLevels, nEdges)
    norm_planet_vort_vertex = KA.zeros(backend, FT, nVertLevels, nVertices)
    norm_planet_vort_edge = KA.zeros(backend, FT, nVertLevels, nEdges)


    DiagnosticVars(layerThicknessEdge,
                   layerThicknessVertex,
                   velocityDivCell,
                   relativeVorticityVertex,
                   norm_rel_vort_vertex,
                   norm_rel_vort_edge,
                   norm_planet_vort_vertex,
                   norm_planet_vort_edge)
end

function Adapt.adapt_structure(to, x::DiagnosticVars)
    return DiagnosticVars(Adapt.adapt(to, x.layerThicknessEdge),
                          Adapt.adapt(to, x.layerThicknessVertex),
                          Adapt.adapt(to, x.velocityDivCell),
                          Adapt.adapt(to, x.relativeVorticityVertex),
                          Adapt.adapt(to, x.norm_rel_vort_vertex),
                          Adapt.adapt(to, x.norm_rel_vort_edge),
                          Adapt.adapt(to, x.norm_planet_vort_vertex),
                          Adapt.adapt(to, x.norm_planet_vort_edge))
end

function diagnostic_compute!(Mesh::Mesh,
                             Diag::DiagnosticVars,
                             Prog::PrognosticVars;
                             backend = KA.CPU())

    calculate_layerThicknessEdge!(Diag, Prog, Mesh; backend = backend)
    calculate_VorticityDiags!(Diag, Prog, Mesh; backend = backend)
    calculate_velocityDivCell!(Diag, Prog, Mesh; backend = backend)
end

function calculate_layerThicknessEdge!(Diag::DiagnosticVars,
                                       Prog::PrognosticVars,
                                       Mesh::Mesh;
                                       backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack nEdges, cellsOnEdge = Edges
    @unpack nVertLevels, maxLevelEdge = VertMesh

    # get the current timelevel of layerThickness
    layerThickness = Prog.layerThickness[end]
    # unpack the layer thickness edge diagnostic term
    @unpack layerThicknessEdge = Diag

    nthreads = 100
    kernel! = compute_layerThicknessEdge!(backend, nthreads)
    # use kernel to compute diagnostic field
    kernel!(layerThicknessEdge,
            layerThickness,
            cellsOnEdge,
            maxLevelEdge.Top,
            nEdges, nVertLevels,
            ndrange = nEdges)

    # sync the backend
    KA.synchronize(backend)

    # pack the diagnostic field back into the struct for further computation
    @pack! Diag = layerThicknessEdge
end

@kernel function compute_layerThicknessEdge!(layerThicknessEdge,
                                             @Const(layerThickness),
                                             @Const(cellsOnEdge),
                                             @Const(maxLevelEdgeTop),
                                             @Const(nEdges),
                                             @Const(nVertLevels))

    iEdge = @index(Global, Linear)

    if iEdge < nEdges + 1

        # initialize to avoid divide by zero and NaN problems
        @inbounds for k in 1:nVertLevels
            @inbounds layerThicknessEdge[k, iEdge] = -1.0e34
        end

        @inbounds for k in 1:maxLevelEdgeTop[iEdge]

            @inbounds @private iCell1 = cellsOnEdge[1,iEdge]
            @inbounds @private iCell2 = cellsOnEdge[2,iEdge]

            @inbounds layerThicknessEdge[k, iEdge] = 0.5 *
                (layerThickness[k, iCell1] + layerThickness[k, iCell2])
        end
    end

    @synchronize()
end

function calculate_velocityDivCell!(Diag::DiagnosticVars,
                                    Prog::PrognosticVars,
                                    Mesh::Mesh;
                                    backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    nEdges = Edges.nEdges
    nVertLevels = VertMesh.nVertLevels

    normalVelocity = Prog.normalVelocity[end]
    velocityDivCell = Diag.velocityDivCell

    # allocate a scratch array
    scratch = KA.zeros(backend, eltype(normalVelocity), nVertLevels, nEdges)

    DivergenceOnCell!(velocityDivCell, normalVelocity, scratch, Mesh; backend=backend)

    @pack! Diag = velocityDivCell
end

@kernel function compute_VorticityDiagsVertex!(relativeVorticityVertex,
                                               norm_rel_vort_vertex,
                                               norm_planet_vort_vertex,
                                               layerThicknessVertex,
                                               @Const(normalVelocity),
                                               @Const(layerThickness),
                                               @Const(fVertex),
                                               @Const(cellsOnVertex),
                                               @Const(edgesOnVertex),
                                               @Const(dcEdge),
                                               @Const(edgeSignOnVertex),
                                               @Const(areaTriangle),
                                               @Const(kiteAreasOnVertex),
                                               @Const(vertexDegree),
                                               @Const(nVertLevels))

    # global indicies over nVertices
    iVertex = @index(Global, Linear)

    #layerThicknessVertex = @localmem eltype(relativeVorticityVertex) (nVertLevels)

    @inbounds @private invAreaTriangle = 1.0 / areaTriangle[iVertex]

    for j in 1:vertexDegree
        @inbounds iCell = cellsOnVertex[j, iVertex]
        @inbounds iEdge = edgesOnVertex[j, iVertex]

        for k in 1:nVertLevels
            layerThicknessVertex[k, iVertex] += invAreaTriangle *
                                                kiteAreasOnVertex[j, iVertex] *
                                                layerThickness[k, iCell]

            # TODO: Add support for free-slip and partial slip
            relativeVorticityVertex[k, iVertex] += dcEdge[iEdge] *
                                                   invAreaTriangle *
                                                   normalVelocity[k, iEdge] *
                                                   edgeSignOnVertex[j, iVertex]
        end
    end

    for k in 1:nVertLevels
        @inbounds @private invLayerThicknessVertex = 1.0 / layerThicknessVertex[k, iVertex]

        norm_rel_vort_vertex[k, iVertex] =
            relativeVorticityVertex[k, iVertex] * invLayerThicknessVertex

        norm_planet_vort_vertex[k, iVertex] =
            fVertex[iVertex] * invLayerThicknessVertex
    end

    @synchronize()
end

@kernel function compute_VorticityDiagsEdge!(norm_rel_vort_edge,
                                             norm_planet_vort_edge,
                                             @Const(norm_rel_vort_vertex),
                                             @Const(norm_planet_vort_vertex),
                                             @Const(verticesOnEdge),
                                             @Const(nVertLevels))

    iEdge = @index(Global, Linear)

    jVertex1 = verticesOnEdge[1, iEdge]
    jVertex2 = verticesOnEdge[2, iEdge]

    for k in 1:nVertLevels
        norm_rel_vort_edge[k, iEdge] = 0.5 *
        (norm_rel_vort_vertex[k, jVertex1] + norm_rel_vort_vertex[k, jVertex2])

        norm_planet_vort_edge[k, iEdge] = 0.5 *
        (norm_planet_vort_vertex[k, jVertex1] + norm_planet_vort_vertex[k, jVertex2])
    end

    @synchronize()
end

function calculate_VorticityDiags!(Diag::DiagnosticVars,
                                   Prog::PrognosticVars,
                                   Mesh::Mesh;
                                   backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh
    @unpack DualCells, Edges = HorzMesh

    @unpack nVertLevels = VertMesh
    @unpack nEdges, dcEdge, verticesOnEdge = Edges
    @unpack nVertices, vertexDegree = DualCells
    @unpack areaTriangle, kiteAreasOnVertex, fᵛ = DualCells
    @unpack edgeSignOnVertex, edgesOnVertex, cellsOnVertex = DualCells

    # get the current timelevel of normalVelocity
    normalVelocity = Prog.normalVelocity[end]
    layerThickness = Prog.layerThickness[end]
    # unpack the relativeVorticity diagnostic term
    @unpack relativeVorticityVertex = Diag
    @unpack norm_rel_vort_edge, norm_planet_vort_edge = Diag
    @unpack norm_rel_vort_vertex, norm_planet_vort_vertex = Diag

    layerThicknessVertex = KA.zeros(backend, eltype(normalVelocity), nVertLevels, nVertices)

    relativeVorticityVertex .= 0.0

    #nthreads = 50
    kernel1!  = compute_VorticityDiagsVertex!(backend)#, nthreads)
    # use kernel to compute diagnostic field
    kernel1!(relativeVorticityVertex,
             norm_rel_vort_vertex,
             norm_planet_vort_vertex,
             layerThicknessVertex,
             normalVelocity,
             layerThickness,
             fᵛ,
             cellsOnVertex,
             edgesOnVertex,
             dcEdge,
             edgeSignOnVertex,
             areaTriangle,
             kiteAreasOnVertex,
             vertexDegree,
             nVertLevels,
             ndrange=nVertices)

    # pack the diagnostic field back into the struct for further computation
    kernel2! = compute_VorticityDiagsEdge!(backend)#, nthreads)
    # use kernel to compute diagnostic field
    kernel2!(norm_rel_vort_edge,
             norm_planet_vort_edge,
             norm_rel_vort_vertex,
             norm_planet_vort_vertex,
             verticesOnEdge,
             nVertLevels,
             ndrange=nEdges)

    # sync the backend
    KA.synchronize(backend)

    @pack! Diag = relativeVorticityVertex
    @pack! Diag = layerThicknessVertex
    @pack! Diag = norm_rel_vort_edge, norm_planet_vort_edge
    @pack! Diag = norm_rel_vort_vertex, norm_planet_vort_vertex
end
