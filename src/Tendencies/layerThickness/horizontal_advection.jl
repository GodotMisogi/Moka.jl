
"""
    thickness_flux_divergence!(tendency, h, u, mesh)

Compute the thickness flux divergence and store within tendency Field.
"""
function thickness_flux_divergence!(tendency, hᵉ, u, mesh)

    @unpack HorizontalMesh, VerticalMesh = mesh    
    @unpack PrimaryCells, DualCells, Edges = HorizontalMesh
    
    @unpack dvEdge = Edges
    @unpack maxLevelEdge = VerticalMesh 
    @unpack nCells, nEdgesOnCell = PrimaryCells
    @unpack edgesOnCell, edgeSignOnCell, areaCell = PrimaryCells

    arch = architecture(mesh)
    kernel_args = (h, u, nEdgesOnCell, edgesOnCell, maxLevelEdge, edgeSignOnCell, dvEdge, areaCell)

    launch!(arch, mesh, thickness_flux_divergence!, tendency, kernel_args...)
end

@kernel function thickness_flux_divergence!(tendency, 
                                            @Const(hᵉ),
                                            @Const(u),
                                            @Const(nEdgesOnCell),     
                                            @Const(edgesOnCell),
                                            @Const(maxLevelEdge),
                                            @Const(edgeSignOnCell),
                                            @Const(dvEdge),
                                            @Const(areaCell))

    iCell = @index(Global, Linear)

    # get inverse cell area
    @inbounds invArea = 1. / areaCell[iCell]

    # loop over number of edges in primary cell
    @inbounds for i in 1:nEdgesOnCell[iCell]
        @inbounds iEdge = edgesOnCell[i,iCell]
        # loop over the number of (active) vertical layers
        @inbounds for k in 1:maxLevelEdge[iEdge]
            @inbounds tendency[k, iCell] += hᵉ[k, iCell] * u[k, iEdge] *
                                            dvEdge[iEdge] * edgeSignOnCell[i, iCell] * invArea
        end
    end

    @synchronize()
end
