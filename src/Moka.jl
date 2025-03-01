module Moka

    export
        # Architectures
        CPU, GPU,

        # Mesh
        Cell, Edge, Vertex, Layer,
        PrimaryCells, DualCells, Edges,
        Mesh, HorizontalMesh, VerticalMesh,

        # Fields and field manipulation
        Field, CellField, EdgeField, VertexField

    #####
    ##### Include all the submodules
    #####
    include("Architectures.jl")
    include("Meshs/Meshs.jl")
    include("Fields/Fields.jl")

    #####
    ##### Needed so we can export names from sub-modules at the top level
    #####
    using .Architectures
    using .Meshs
    using .Fields

end
