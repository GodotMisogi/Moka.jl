module Moka

    export
        # Architectures
        CPU, GPU,

        # Mesh
        Cell, Edge, Vertex, Layer,
        PrimaryCells, DualCells, Edges,
        Mesh, HorizontalMesh, VerticalMesh

    #####
    ##### Include all the submodules
    #####
    include("Architectures.jl")
    include("Meshs/Meshs.jl")

    #####
    ##### Needed so we can export names from sub-modules at the top level
    #####
    using .Architectures
    using .Meshs
end
