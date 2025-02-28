const AE = AbstractMeshElement{FT, Arch} where {FT, Arch}

"""
    HorizontalMesh

A structure describing a 2-D (horizontal) TRiSK mesh on a particular architecture
"""
struct HorizontalMesh{FT, Arch, PC <: AE{FT, Arch}, DC <: AE{FT, Arch}, E <: AE{FT, Arch}}
    architecture :: Arch
    PrimaryCells :: PC
       DualCells :: DC
           Edges :: E
end

"""
    architecture(mesh::HorizontalMesh)

Return the architecture (CPU or GPU) that the horizontal mesh element lives on.
"""
@inline architecture(mesh::HorizontalMesh) = mesh.architecture

Base.eltype(::HorizontalMesh{FT}) where FT = FT
Base.eps(::HorizontalMesh{FT}) where FT = eps(FT)

dimsize(h::HorizontalMesh) = (nCells=dimsize(h.PrimaryCells).nCells,
                              nVertices=dimsize(h.DualCells).nVertices,
                              nEdges=dimsize(h.Edges).nEdges)

@inline Base.size(h::HorizontalMesh) = (dimsize(h).nCells, dimsize(h).nVertices, dimsize(h).nEdges)

const HT = Union{Type{Cell}, Type{Edge}, Type{Vertex}}

Base.size(h::HorizontalMesh, ::Cell) = dimsize(h).nCells
Base.size(h::HorizontalMesh, ::Edge) = dimsize(h).nEdges
Base.size(h::HorizontalMesh, ::Vertex) = dimsize(h).nVertices
Base.size(h::HorizontalMesh, loc::HT) = size(h, loc())

"""
    HorizontalMesh(filepath;)

TODO: Write description for HorizontalMesh constructor
"""
function HorizontalMesh(meshDataset::NCDataset)

    primary_cells = PrimaryCells(meshDataset)
    dual_cells = DualCells(meshDataset)
    edges = Edges(meshDataset)

    # set the edgeSignOn[Cell|Vertex] fields
    signIndexField!(primary_cells, edges)
    signIndexField!(dual_cells, edges)

    return HorizontalMesh(CPU(), primary_cells, dual_cells, edges)
end

HorizontalMesh(filepath::AbstractString) = HorizontalMesh(NCDataset(filepath))

function signIndexField!(primaryCells::PrimaryCells, edges::Edges)

    @unpack cellsOnEdge = edges
    @unpack nCells, edgeSignOnCell, nEdgesOnCell, edgesOnCell = primaryCells

    @inbounds for iCell in 1:nCells, i in 1:nEdgesOnCell[iCell]

        iEdge = edgesOnCell[i, iCell]

        # vector points from cell 1 to cell 2
        if iCell == cellsOnEdge[1, iEdge]
            edgeSignOnCell[i, iCell] = -1
        else
            edgeSignOnCell[i, iCell] = 1
        end
    end

    # PrimaryCell struct is immutable so need to use Accessor package,
    @reset primaryCells.edgeSignOnCell = edgeSignOnCell
end

function signIndexField!(dualMesh::DualCells, edges::Edges)

    @unpack verticesOnEdge = edges
    @unpack vertexDegree, nVertices, edgeSignOnVertex, edgesOnVertex = dualMesh

    for iVertex in 1:nVertices, i in 1:vertexDegree

        @inbounds iEdge = edgesOnVertex[i, iVertex]

        # if edge missing from vertex (i.e. vertex is on a culled boundary),
        # then leave edgeSignOnVertex as undefined (i.e. zero)
        if iEdge == 0 continue end

        # vector points from cell 1 to cell 2
        if iVertex == verticesOnEdge[1, iEdge]
            @inbounds edgeSignOnVertex[i, iVertex] = -1
        else
            @inbounds edgeSignOnVertex[i, iVertex] = 1
        end
    end

    # DualCell struct is immutable so need to use Accessor package,
    @reset dualMesh.edgeSignOnVertex = edgeSignOnVertex
end

#####
##### Utilities
#####

Adapt.adapt_structure(device, mesh::HorizontalMesh) =
    HorizontalMesh(Architectures.architecture(device),
                   Adapt.adapt_structure(device, mesh.PrimaryCells),
                   Adapt.adapt_structure(device, mesh.DualCells),
                   Adapt.adapt_structure(device, mesh.Edges))


"""
    on_architecture(architecture, horizontal_mesh)

Return a horizontal mesh that's identical to `horizontal_mesh` but on `architecture`.
"""
function on_architecture(arch::AbstractSerialArchitecture, mesh::HorizontalMesh)
    if arch == architecture(mesh)
        return grid
    end
    return Adapt.adapt_structure(Architectures.device(arch), mesh)
end

#####
##### Showing grids
#####

function Base.summary(mesh::HorizontalMesh)
    FT = eltype(mesh)

    return string(" HorizontalMesh{$FT} on ", summary(architecture(mesh)),
                  " with ", dimsize(mesh))
end

function Base.show(io::IO, mesh::HorizontalMesh, withsummary=true)
        print(io, summary(mesh), "\n")
end
