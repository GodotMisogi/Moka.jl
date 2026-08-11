#abstract type StateVar end 
#normalVelocity <: StateVar
#layerThickness <: StateVar
#temperature <: StateVar
#salinty <: StateVar

mutable struct TendencyVars{F<:AbstractFloat, FV2 <: AbstractArray{F,2}, FV3 <: AbstractArray{F,3}}

    # var: time tendency of normal component of velociy [m s^{-2}]
    # dim: (nVertLevels, nEdges), Time?)
    tendNormalVelocity::FV2

    # var: time tendency of layer thickness [m s^{-1}]
    # dim: (nVertLevels, nCells), Time?)
    tendLayerThickness::FV2

    # var: time tendency of tracers [tracer-unit s^{-1}]
    # dim: (nTracers, nVertLevels, nCells). Zero-tracer default is (0, nVL, nCells).
    tendTracers::FV3

    function TendencyVars(tendNormalVelocity::AT2D,
                          tendLayerThickness::AT2D;
                          tendTracers::Union{Nothing,AT3D}=nothing) where {AT2D, AT3D}

        # pack the 2-D args for type and backend checking
        args = (tendNormalVelocity, tendLayerThickness)

        # check the type names; irrespective of type parameters
        # (e.g. `Array` instead of `Array{Float64, 1}`)
        check_typeof_args(args)
        # check that all args are on the same backend
        check_args_backend(args)
        # check that all args have the same `eltype` and get that type
        type = check_eltype_args(args)

        # zero-tracer default matching the array type/backend of tendLayerThickness
        nVertLevels, nCells = size(tendLayerThickness)
        tend0 = tendTracers === nothing ?
            similar(tendLayerThickness, (0, nVertLevels, nCells)) : tendTracers

        new{type, AT2D, typeof(tend0)}(tendNormalVelocity, tendLayerThickness, tend0)
    end
end

function TendencyVars(Mesh::Mesh; backend=KernelAbstractions.CPU(), nTracers::Int=0)

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, Edges = HorzMesh

    nEdges = Edges.nEdges
    nCells = PrimaryCells.nCells
    nVertLevels = VertMesh.nVertLevels

    # create zero vectors to store tendecy vars on the desired backend
    tendNormalVelocity = zeros(Float64, nVertLevels, nEdges)
    tendLayerThickness = KA.zeros(backend, Float64, nVertLevels, nCells)
    tendTracers        = KA.zeros(backend, Float64, nTracers, nVertLevels, nCells)

    TendencyVars(Adapt.adapt(backend, tendNormalVelocity), tendLayerThickness;
                 tendTracers=tendTracers)
end

function axb!(a::Array{T,2}, x::T, b::Array{T,2}) where {T<:AbstractFloat}
    m,n = size(a)

    @boundscheck (m,n) == size(b) || throw(BoundsError())

    @inbounds for j ∈ 1:n
        for i ∈ 1:m
           a[i,j] += x*b[i,j]
        end
    end
end 

