
struct FluxLayerThickness
end

function FluxLayerThickness(mesh::Mesh, upwind=false)

    field((Cell, Layer), flux_layer_thickness_center!, mesh)
end

function compute!(f::FluxLayerThickness, model)

    normal_velocity = model.solution.u
    layer_thick_cell = model.solution.h

    cells_on_edge = f.mesh.HorizontalMesh.Edges.cellsOnEdge
    max_level_edge = f.mesh.VerticalMesh.maxLevelEdge

    args = 
    f.Func()
end

@kernel function flux_layer_thickness_center!(flux_layer_thick,
                                              @Const(layer_thick_cell),
                                              @Const(normal_velocity),
                                              @Const(cells_on_edge),
                                              @Const(max_level_edge))

    iEdge = @index(Global, Linear)

    for k in 1:max_level_edge[iEdge]
        @inbounds @private iCell1 = cells_on_edge[1, iEdge]
        @inbounds @private iCell2 = cells_on_edge[2, iEdge]

        @inbounds flux_layer_thick[k, iEdge] = 0.5 *
            (layer_thick_cell[k, iCell1] + layer_thick_cell[k, iCell2])
    end

    @synchronize()
end

@kernel function flux_layer_thickness_upwind!(flux_layer_thick,
                                              @Const(layer_thick_cell),
                                              @Const(normal_velocity),
                                              @Const(cells_on_edge),
                                              @Const(max_level_edge))

    iEdge = @index(Global, Linear)

    for k in 1:max_level_edge[iEdge]
        @inbounds @private iCell1 = cells_on_edge[1, iEdge]
        @inbounds @private iCell2 = cells_on_edge[2, iEdge]
        
        if normal_velocity[k, iEdge] > 0
            @inbounds flux_layer_thick[k, iEdge] = layer_thick_cell[k, iCell1]
        elseif normal_velocity[k, iEdge] < 0
            @inbounds flux_layer_thick[k, iEdge] = layer_thick_cell[k, iCell2]
        else
            @inbounds flux_layer_thick[k, iEdge] = max(
                layer_thick_cell[k, iCell1], layer_thick_cell[k, iCell2]
            )
        end
    end

    @synchronize()
end
