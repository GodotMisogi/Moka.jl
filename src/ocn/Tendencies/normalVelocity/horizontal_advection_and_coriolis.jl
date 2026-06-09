"""
methods for calculating tendencies from the horizontal advection and
coriolis force using KernelAbstractions
"""

# might need some help determining the way to abstract the vorticity options
abstract type Coriolis end

# define the supported coriolis formulations to dispatch on
abstract type linearCoriolis <: Coriolis end

function horizontal_advection_and_coriolis_tendency!(Tend::TendencyVars,
                                                     Prog::PrognosticVars,
                                                     Diag::DiagnosticVars,
                                                     Mesh::Mesh,
                                                     ::Type{linearCoriolis};
                                                     backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh
    @unpack nEdges, nEdgesOnEdge, edgeMask = Edges
    @unpack weightsOnEdge, fᵉ, cellsOnEdge, edgesOnEdge = Edges

    @unpack layerThicknessEdge = Diag
    @unpack norm_rel_vort_edge, norm_planet_vort_edge = Diag
    # get the current timelevel of normalVelocity
    normalVelocity = Prog.normalVelocity[end]
    # unpack the normal velocity tendency term
    @unpack tendNormalVelocity = Tend

    # initialize the kernel
    nthreads = 50
    kernel!  = coriolis_force_tendency_kernel!(backend, nthreads)
    # use kernel to compute coriolis and horizontal advection
    kernel!(tendNormalVelocity,
            normalVelocity,
            layerThicknessEdge,
            norm_rel_vort_edge,
            norm_planet_vort_edge,
            nEdgesOnEdge,
            edgesOnEdge,
            maxLevelEdge.Top,
            weightsOnEdge,
            edgeMask,
            ndrange = nEdges)
    # sync the backend
    KA.synchronize(backend)

    # pack the tendecy pack into the struct for further computation
    @pack! Tend = tendNormalVelocity
end

@kernel function coriolis_force_tendency_kernel!(tendency,
                                                 @Const(normalVelocity),
                                                 @Const(layerThicknessEdge),
                                                 @Const(norm_rel_vort_edge),
                                                 @Const(norm_planet_vort_edge),
                                                 @Const(nEdgesOnEdge),
                                                 @Const(edgesOnEdge),
                                                 @Const(maxLevelEdgeTop),
                                                 @Const(weightsOnEdge),
                                                 @Const(edgeMask))

    # global indices over nEdges
    iEdge = @index(Global, Linear)

    @inbounds for j in 1:nEdgesOnEdge[iEdge]

        @inbounds @private jEdge = edgesOnEdge[j, iEdge]

        @inbounds for k in 1:maxLevelEdgeTop[iEdge]
            @inbounds @private normVorticity = 0.5 *
            (norm_rel_vort_edge[k, iEdge] + norm_planet_vort_edge[k, iEdge] +
             norm_rel_vort_edge[k, jEdge] + norm_planet_vort_edge[k, jEdge] )

            @inbounds tendency[k, iEdge] += weightsOnEdge[j, iEdge] *
                                            normalVelocity[k, jEdge] *
                                            layerThicknessEdge[k, jEdge] *
                                            normVorticity * edgeMask[k, iEdge]
        end
    end
end
