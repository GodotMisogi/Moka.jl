# int vector and float matrix type aliases for supported array types
const IV = Union{Array{Int32, 1}, CuArray{Int32, 1}, OffsetArray{Int32, 1}}
const MT = Union{Array{FT, 2}, CuArray{FT, 2}, OffsetArray{FT, 2}} where FT

"""
    VerticalMesh

A strucutre describing our vertical mesh of a layered ocean model
"""
mutable struct VerticalMesh{FT, Arch, IV, FM <: MT{FT}}
           architecture :: Arch
            nVertLevels :: Int32
           maxLevelCell :: IV
           maxLevelEdge :: IV
         maxLevelVertex :: IV
       restingThickness :: FM
    restingThicknessSum :: FM
end

"""
    architecture(mesh::HorizontalMesh)

Return the architecture (CPU or GPU) that the vertical mesh element lives on.
"""
@inline architecture(mesh::VerticalMesh) = mesh.architecture

Base.eltype(::VerticalMesh{FT}) where FT = FT
Base.eps(::VerticalMesh{FT}) where FT = eps(FT)

Base.length(v::VerticalMesh, ::Layer) = v.nVertLevels
Base.length(v::VerticalMesh, loc::Type{Layer}) = length(v, loc())

dimsize(v::VerticalMesh) = (nVertLevels=v.nVertLevels,)

@inline Base.size(v::VerticalMesh) = (v.nVertLevels,)

Base.size(v::VerticalMesh, ::Layer) = (v.nVertLevels,)
Base.size(v::VerticalMesh, loc::Type{Layer}) = size(v, loc())

"""
    VerticalMesh(ds::NCDataset, horizontal_mesh::HorizontalMesh, FT=Float64)

Construct a `VerticalMesh`
"""
function VerticalMesh(ds::NCDataset,
                      horizontal_mesh::HorizontalMesh,
                      FT::DataType = eltype(horizontal_mesh))

    # if no vertical info is present, then create a single layered mesh
    if !haskey(ds.dim, "nVertLevels")
        return VerticalMesh(mesh)
    else
        nVertLevels = Int32(ds.dim["nVertLevels"])
    end

    nCells = length(horizontal_mesh, Cell)
    # Pre-allocate zero indexed offsetarrays
    maxLevelCell = padded_index_array(nCells)
    # Read in the maximum level for all interior indices
    maxLevelCell[1:end] = ds["maxLevelCell"][:]

    # check that the vertical mesh is stacked
    if !all(maxLevelCell[1:end] .== nVertLevels)
        @error """ (Vertical Mesh Initializaton)\n
               Vertical Mesh is not stacked. Must implement vertical masking
               before this mesh can be used
               """
    end

    active_levels_edge = active_levels(maxLevelCell, horizontal_mesh, Edge)
    active_levels_vertex = active_levels(maxLevelCell, horizontal_mesh, Vertex)

    restingThickness = convert(Array{FT, 2}, ds["restingThickness"][:,:,1])
    restingThicknessSum = sum(restingThickness; dims=1)

    return VerticalMesh(CPU(),
                        nVertLevels,
                        maxLevelCell,
                        active_levels_edge,
                        active_levels_vertex,
                        restingThickness,
                        restingThicknessSum)
end

#=
"""
Constructor for an (n) layer stacked vertical mesh. Only valid when paired
with a *periodic* horizontal mesh.

This function is handy for unit test that read in purely horizontal meshes.

NOTE: Not to be used for real simualtions, only for unit testing.
"""
function VerticalMesh(mesh; nVertLevels=1)

    nCells = mesh.PrimaryCells.nCells

    maxLevelCell = ones(Int32, nCells) .* Int32(nVertLevels)
    # unit thickness water column, irrespective of how many vertical levels
    restingThickness    = ones(Float64, nCells)
    restingThicknessSum = ones(Float64, nCells) # MIGHT NEED TO CHANGE THIS

    ActiveLevelsEdge = ActiveLevels{Edge}(maxLevelCell, mesh)
    ActiveLevelsVertex = ActiveLevels{Vertex}(maxLevelCell, mesh)

    # All array have been allocated on the requested backend,
    # so no need to call methods from Adapt
    VerticalMesh(nVertLevels,
                 maxLevelCell,
                 ActiveLevelsEdge,
                 ActiveLevelsVertex)
                 #restingThickness,
                 #restingThicknessSum)
end

function Adapt.adapt_structure(backend, x::ActiveLevels)
    return ActiveLevels(Adapt.adapt(backend, x.Top),
                        Adapt.adapt(backend, x.Bot))
end
=#

function padded_index_array(dimLength; eltype=Int32)
    OffsetArray(zeros(eltype, dimLength + 1), 0:dimLength)
end

"""
ActiveLevels constructor for a nVertLevel stacked *periodic* meshes
"""
active_levels(dim, eltype) = ones(eltype, dim)

""" TODO: write some documentation"""
function active_levels(maxLevelCell, horizontal_mesh, ::Type{Edge})

    @unpack nEdges, cellsOnEdge = horizontal_mesh.Edges

    # top is the minimum (shallowest) of the surrounding cells
    top = padded_index_array(nEdges)

    for iEdge in 1:nEdges
        @inbounds iCell1 = cellsOnEdge[1, iEdge]
        @inbounds iCell2 = cellsOnEdge[2, iEdge]

        top[iEdge] = min(maxLevelCell[iCell1], maxLevelCell[iCell2])
    end

    return top
end

""" TODO: write some documentation"""
function active_levels(maxLevelCell, horizontal_mesh, ::Type{Vertex})

    @unpack nVertices, cellsOnVertex, vertexDegree = horizontal_mesh.DualCells

    # top is the minimum (shallowest) of the surrounding cells
    top = padded_index_array(nVertices)

    for iVertex in 1:nVertices
        # get vector indices of the cellsOnVertex (e.g. (3,))
        cellsOnVertex_i = [cellsOnVertex[i, iVertex] for i in 1:vertexDegree]

        top[iVertex] = minimum(maxLevelCell[cellsOnVertex_i])
    end

    return top
end

#####
##### Utilities
#####

function Adapt.adapt_structure(device, x::VerticalMesh)
    return VerticalMesh(Architectures.architecture(device),
                        x.nVertLevels,
                        Adapt.adapt(device, x.maxLevelCell),
                        Adapt.adapt(device, x.maxLevelEdge),
                        Adapt.adapt(device, x.maxLevelVertex),
                        Adapt.adapt(device, x.restingThickness),
                        Adapt.adapt(device, x.restingThicknessSum))
end

"""
    on_architecture(architecture, vertical_mesh)

Return a horizontal mesh that's identical to `vertial_mesh` but on `architecture`.
"""
function on_architecture(arch::AbstractSerialArchitecture, mesh::VerticalMesh)
    if arch == architecture(mesh)
        return grid
    end
    return Adapt.adapt_structure(Architectures.device(arch), mesh)
end

#####
##### Showing grids
#####

function Base.summary(mesh::VerticalMesh)
    FT = eltype(mesh)

    return string(" VerticalMesh{$FT} on ", summary(architecture(mesh)),
                  " with ", dimsize(mesh))
end

function Base.show(io::IO, mesh::VerticalMesh, withsummary=true)
        print(io, summary(mesh), "\n")
end
