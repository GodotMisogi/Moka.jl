module Moka

    export
        # Architectures
        CPU, GPU,

        # Mesh
        Cell, Edge, Vertex, Layer,
        PrimaryCells, DualCells, Edges,
        Mesh, HorizontalMesh, VerticalMesh,

        # Fields and field manipulation
        Field, CellField, EdgeField, VertexField,

        # Models
        ShallowWaterModel, #fields

        # Time stepping
        Clock, time_step!


    #####
    ##### Abstract types
    #####

    """
        AbstractModel

    Abstract supertype for models.
    """
    abstract type AbstractModel{TS} end

    """
        AbstractOutputWriter

    Abstract supertype for output writers that write data to disk.
    """
    abstract type AbstractOutputWriter end

    # Callsites for Callbacks
    struct TimeStepCallsite end
    struct TendencyCallsite end
    struct UpdateStateCallsite end

    #####
    ##### Place-holder functions
    #####

    function write_output! end
    function initialize! end # for initializing models, simulations, etc
    function location end
    function fields end
    function prognostic_fields end

    #####
    ##### Include all the submodules
    #####

    # Basics
    include("Architectures.jl")
    include("Meshs/Meshs.jl")
    include("Fields/Fields.jl")
    include("TimeSteppers/TimeSteppers.jl")

    # Physics, forcings, and models
    include("Models/Models.jl")

    #####
    ##### Needed so we can export names from sub-modules at the top level
    #####

    using .Architectures
    using .Meshs
    using .Fields
    using .Models
    using .TimeSteppers

end
