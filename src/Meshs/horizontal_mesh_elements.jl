"""
    AbstractMeshElement{FT, Arch}

Abstract supertype for MPAS mesh elements with a (floating point) type `FT` on architecture `Arch`.
"""
abstract type AbstractMeshElement{FT, Arch} end

"""
    architecture(element::AbstractMeshElement)

Return the architecture (CPU or GPU) that the mesh element lives on.
"""
@inline architecture(element::AbstractMeshElement) = element.architecture

Base.eltype(::AbstractMeshElement{FT}) where FT = FT
Base.eps(::AbstractMeshElement{FT}) where FT = eps(FT)

# vector and matrix type alias for supported array types
const VT = Union{Array{FT, 1}, CuArray{FT, 1}, OffsetArray{FT, 1}} where FT
const MT = Union{Array{FT, 2}, CuArray{FT, 2}, OffsetArray{FT, 2}} where FT

@kwdef struct Edges{FT, FV <: VT{FT}, FM <: MT{FT}, IV, IM, Arch} <: AbstractMeshElement{FT, Arch}
      architecture :: Arch
            nEdges :: Int  # The number of edges in the mesh.
             xEdge :: FV   # x axis position of all edge locations
             yEdge :: FV   # y axis position of all edge locations
             zEdge :: FV   # z axis position of all edge locations
             #fEdge :: FV   # coriolis parameter
      nEdgesOnEdge :: IV   # Number of edges on a given edge. Used to reconstruct tangential velocities.
       cellsOnEdge :: IM   # Cell indices that saddle a given edge.
    verticesOnEdge :: IM   # Vertex indices that saddle a given edge.
       edgesOnEdge :: IM   # Edge indices that are used to reconstruct tangential velocities.
     weightsOnEdge :: FM   # Weights used to reconstruct tangential velocities.
         angleEdge :: FV   # Angle in radians an edge's normal vector makes with the local eastward direction.
          edgeMask :: IM   # mask to determine if computation should be done on edge
            dvEdge :: FV   # Distance in meters between the vertices that saddle a given edge.
            dcEdge :: FV   # Distance in meters between the cells that saddle a given edge.
end

@kwdef struct PrimaryCells{FT, FV <: VT{FT}, IV, IM, Arch} <: AbstractMeshElement{FT, Arch}
      architecture :: Arch
            nCells :: Int  # The number of primary cells in the mesh.
          maxEdges :: Int  # The maximum number of vertices and edges on any primary cell.
             xCell :: FV   # x axis position of all cell centers.
             yCell :: FV   # y axis position of all cell centers.
             zCell :: FV   # z axis position of all cell centers.
             #fCell :: FV
      nEdgesOnCell :: IV   # Number of edges on a given cell.
       edgesOnCell :: IM   # Edge indices that surround a given cell.
    verticesOnCell :: IM   # Vertex indices that surround a given cell.
       cellsOnCell :: IM   # Cell indices that surround a given cell.
    edgeSignOnCell :: IM   #
          areaCell :: FV   # Area in square meters for a given cell of the primary mesh.
end

@kwdef struct DualCells{FT, FV <: VT{FT}, IM, Arch} <: AbstractMeshElement{FT, Arch}
        architecture :: Arch
           nVertices :: Int  # The number of vertices in the mesh, or the number cells in the dual mesh.
        vertexDegree :: Int  # The max number of primary cells connected to a dual cell (i.e. number of corners in a dual cell).
             xVertex :: FV   # x axis position of all cell vertices.
             yVertex :: FV   # y axis position of all cell vertices.
             zVertex :: FV   # z axis position of all cell vertices.
             fVertex :: FV
       edgesOnVertex :: IM   # Edge indices that radiate from a given vertex.
       cellsOnVertex :: IM   # Cell indices that radiate from a given vertex.
    edgeSignOnVertex :: IM   #
        areaTriangle :: FV   # Area in square meters for a given triangle of the dual mesh.
end

@inline Base.size(edges::Edges) = (edges.nEdges,)
@inline Base.size(cells::PrimaryCells) = (cells.nCells,)
@inline Base.size(vertices::DualCells) = (vertices.nVertices,)

dimsize(edges::Edges) = (nEdges=edges.nEdges,)
dimsize(cells::PrimaryCells) = (nCells=cells.nCells, maxEdges=cells.maxEdges)
dimsize(vertices::DualCells) = (nVertices=vertices.nVertices, vertexDegree=vertices.vertexDegree)

#####
##### Constructors for reading mesh from disk
#####

function PrimaryCells(ds::NCDataset)
    field_names = fieldnames(PrimaryCells)[2:end]
    field_values = map(f -> read_field(ds, f), field_names)
    kwargs = NamedTuple{field_names}(field_values)
    @reset kwargs.edgeSignOnCell = zeros(Int32, (kwargs.maxEdges, kwargs.nCells))
    return PrimaryCells(; architecture=CPU(), kwargs...)
end

function DualCells(ds::NCDataset)
    field_names = fieldnames(DualCells)[2:end]
    field_values = map(f -> read_field(ds, f), field_names)
    kwargs = NamedTuple{field_names}(field_values)
    if :vertexMask ∈ field_names
        # dummy value to be overwritten by `setBoundaryMask!`
        kwargs.vertexMask = zeros(Int32, (1, kwargs.nVertices))
    end
    @reset kwargs.edgeSignOnVertex = zeros(Int32, (ds.dim["maxEdges"], kwargs.nVertices))
    return DualCells(; architecture=CPU(), kwargs...)
end

function Edges(ds::NCDataset)
    field_names = fieldnames(Edges)[2:end]
    field_values = map(f -> read_field(ds, f), field_names)
    kwargs = NamedTuple{field_names}(field_values)
    # dummy value to be overwritten by `setBoundaryMask!`
    @reset kwargs.edgeMask = zeros(Int32, (1, kwargs.nEdges))
    return Edges(; architecture=CPU(), kwargs...)
end

function _read_from_disk(ds::NCDataset, element::AbstractMeshElement)
    field_names = fieldnames(typeof(element))[2:end]
    field_values = map(f -> read_field(ds, f), field_names)
    return NamedTuple{field_names}(field_values)
end

PrimaryCells(filepath::AbstractString) = PrimaryCells(NCDataset(filepath))
DualCells(filepath::AbstractString) = DualCells(NCDataset(filepath))
Edges(filepath::AbstractString) = Edges(NCDataset(filepath))

function read_field(ds::NCDataset, field::Symbol)

    field = string(field)

    if haskey(ds.dim, field)
        return ds.dim[field]
    elseif occursin(r"^(edgeSignOn.*|.*Mask)$", field)
        return nothing
    elseif !haskey(ds, field)
        @error ""
    end

    if ndims(ds[field]) == 2
        return ds[field][:, :]
    else
        return ds[field][:]
    end
end

#####
##### Utilities
#####

adapt_field(to, field) = typeof(field) <: Union{VT, MT} ? Adapt.adapt(to, field) : field

Adapt.adapt_structure(to, edges::Edges) = Edges(;_adapt_structure(to, edges)...)
Adapt.adapt_structure(to, cells::PrimaryCells) = PrimaryCells(;_adapt_structure(to, cells)...)
Adapt.adapt_structure(to, vertices::DualCells) = DualCells(;_adapt_structure(to, vertices)...)

function _adapt_structure(device, element::AbstractMeshElement)
    field_names = fieldnames(typeof(element))
    field_values = map(f -> getfield(element, f), field_names[2:end])
    adapted_values = map(f -> adapt_field(device, f), field_values)
    architecture = Architectures.architecture(device)
    return NamedTuple{field_names}((architecture, adapted_values...))
end


"""
    on_architecture(architecture, [PrimaryCells | DualCells | Edges])

Return a mesh_element that's identical to `grid` but on `architecture`.
"""
function on_architecture(arch::AbstractSerialArchitecture, element::AbstractMeshElement)
    if arch == architecture(element)
        return grid
    end
    return Adapt.adapt_structure(Architectures.device(arch), element)
end

