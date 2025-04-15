"""
methods for calculating tendencies from the horizontal advection and
coriolis force using KernelAbstractions
"""

# might need some help determining the way to abstract the vorticity options
abstract type Coriolis end

# define the supported coriolis formulations to dispatch on
abstract type linearCoriolis <: Coriolis end

function horizontal_advection_and_coriolis_tendency!(tendency, u, mesh)

    @unpack HorizontalMesh, VerticalMesh = mesh    
    @unpack PrimaryCells, DualCells, Edges = HorizontalMesh

    @unpack maxLevelEdge = VerticalMesh 
    @unpack nEdges, nEdgesOnEdge, edgeMask = Edges
    @unpack weightsOnEdge, cellsOnEdge, edgesOnEdge = Edges
    
    fᵉ = coriolis(mesh, Edge)

    arch = architecture(mesh)
    kernel_args = (u, fᵉ, nEdgesOnEdge, edgesOnEdge, maxLevelEdgeTop, weightsOnEdge, edgeMask)

    launch!(arch, mesh, coriolis_force_tendency_kernel!, tendency, kernel_args...)
end

@kernel function coriolis_force_tendency_kernel!(tendency,
                                                 @Const(u), 
                                                 @Const(fᵉ), 
                                                 @Const(nEdgesOnEdge),
                                                 @Const(edgesOnEdge),
                                                 @Const(maxLevelEdgeTop),
                                                 @Const(weightsOnEdge), 
                                                 @Const(edgeMask))
    
    # global indices over nEdges
    iEdge = @index(Global, Linear)

    @inbounds for i in 1:nEdgesOnEdge[iEdge]
        
        @inbounds eoe = edgesOnEdge[i, iEdge]
        
        if eoe == 0 continue end 

        @inbounds for k in 1:maxLevelEdgeTop[iEdge]
            @inbounds tendency[k, iEdge] += u[k, eoe] * weightsOnEdge[i, iEdge] *
                                            fᵉ[eoe] * edgeMask[k, iEdge]
        end
    end

    @synchronize()
end 
