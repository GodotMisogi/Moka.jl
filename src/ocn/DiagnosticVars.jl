import Adapt

using CUDA
using KernelAbstractions

mutable struct DiagnosticVars{F <: AbstractFloat, FV2 <: AbstractArray{F,2}}

    # var: layer thickness averaged from cell centers to edges [m]
    # dim: (nVertLevels, nEdges)
    layerThicknessEdge::FV2

    # var: divergence of horizonal velocity [s^{-1}]
    # dim: (nVertLevels, nCells)
    velocityDivCell::FV2

    # var: curl of horizontal velocity [s^{-1}]
    # dim: (nVertLevels, nVertices)
    relativeVorticity::FV2

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
                            velocityDivCell::AT2D,
                            relativeVorticity::AT2D) where {AT2D}
        # pack all the arguments into a tuple for type and backend checking
        args = (layerThicknessEdge, velocityDivCell, relativeVorticity)

        # check the type names; irrespective of type parameters
        # (e.g. `Array` instead of `Array{Float64, 1}`)
        check_typeof_args(args)
        # check that all args are on the same backend
        check_args_backend(args)
        # check that all args have the same `eltype` and get that type
        type = check_eltype_args(args)

        new{type, AT2D}(layerThicknessEdge,
                        velocityDivCell,
                        relativeVorticity)
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
    relativeVorticity = KA.zeros(backend, FT, nVertLevels, nVertices)
    # initialize to Inf to avoid divide by zero and NaN problems
    layerThicknessEdge = KA.ones(backend, FT, nVertLevels, nEdges) * -typemax(FT)

    DiagnosticVars(layerThicknessEdge,
                   velocityDivCell,
                   relativeVorticity)
end

function Adapt.adapt_structure(to, x::DiagnosticVars)
    return DiagnosticVars(Adapt.adapt(to, x.layerThicknessEdge),
                          Adapt.adapt(to, x.velocityDivCell),
                          Adapt.adapt(to, x.relativeVorticity))
end

function diagnostic_compute!(Mesh::Mesh,
                             Diag::DiagnosticVars,
                             Prog::PrognosticVars;
                             backend = KA.CPU())

    calculate_layerThicknessEdge!(Diag, Prog, Mesh; backend = backend)
    calculate_relativeVorticity!(Diag, Prog, Mesh; backend = backend)
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

@kernel function compute_relativeVorticity!(relativeVorticity,
                                            @Const(normalVelocity),
                                            @Const(edgesOnVertex),
                                            @Const(dcEdge),
                                            @Const(edgeSignOnVertex),
                                            @Const(areaTriangle),
                                            @Const(vertexDegree),
                                            @Const(maxLevelVertexBot))

    # global indicies over nVertices
    iVertex = @index(Global, Linear)

    @inbounds @private invAreaTriangle = 1.0 / areaTriangle[iVertex]

    for j in 1:vertexDegree
        #@inbounds 
        iEdge = edgesOnVertex[j, iVertex]
        
        # padded iEdge array would probably be better
        if iEdge > 0 break end

        for k in 1:maxLevelVertexBot[iVertex]
            # TODO: Add support for free-slip and partial slip
            relativeVorticity[k, iVertex] += dcEdge[iEdge] *
                                             invAreaTriangle *
                                             normalVelocity[k, iEdge] *
                                             edgeSignOnVertex[j, iVertex]
        end
    end
end

function calculate_relativeVorticity!(Diag::DiagnosticVars,
                                      Prog::PrognosticVars,
                                      Mesh::Mesh;
                                      backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh
    @unpack DualCells, Edges = HorzMesh

    @unpack nEdges, dcEdge = Edges
    @unpack maxLevelVertex = VertMesh
    @unpack nVertices, vertexDegree = DualCells
    @unpack areaTriangle, edgeSignOnVertex, edgesOnVertex = DualCells

    # get the current timelevel of normalVelocity
    normalVelocity = Prog.normalVelocity[end]
    # unpack the relativeVorticity diagnostic term
    @unpack relativeVorticity = Diag

    relativeVorticity .= 0.0

    #nthreads = 50
    kernel!  = compute_relativeVorticity!(backend)#, nthreads)
    # use kernel to compute diagnostic field
    kernel!(relativeVorticity,
            normalVelocity,
            edgesOnVertex,
            dcEdge,
            edgeSignOnVertex,
            areaTriangle,
            vertexDegree,
            maxLevelVertex.Bot,
            ndrange=nVertices)

    # sync the backend
    KA.synchronize(backend)

    # pack the diagnostic field back into the struct for further computation
    @pack! Diag = relativeVorticity
end
