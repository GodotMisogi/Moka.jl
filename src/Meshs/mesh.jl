"""
    AbstractMesh{FT, Arch}

Abstract supertype for meshes with elements of type `FT` on architecures `Arch`.
"""
abstract type AbstractMesh{FT, Arch} end

const VM = VerticalMesh{FT, Arch} where {FT, Arch}
const HM = HorizontalMesh{FT, Arch} where {FT, Arch}

"""
    Mesh

A structure describing a 3-D TRiSK mesh on a particular architecture
"""
struct Mesh{FT, Arch, H <: HM{FT, Arch}, V <: VM{FT, Arch}}
      architecture :: Arch
    HorizontalMesh :: H
      VerticalMesh :: V
end

Mesh{FT}(a::Arch, h::HM, v::VM) where {FT, Arch, HM, VM} = Mesh{FT, Arch, HM, VM}(a, h, v)

"""
    architecture(mesh::Mesh)

Return the architecture (CPU or GPU) that the mesh element lives on.
"""
@inline architecture(mesh::Mesh) = mesh.architecture

Base.eltype(::Mesh{FT}) where FT = FT
Base.eps(::Mesh{FT}) where FT = eps(FT)

dimsize(m::Mesh) = (nCells=dimsize(m.HorizontalMesh).nCells,
                    nVertices=dimsize(m.HorizontalMesh).nVertices,
                    nEdges=dimsize(m.HorizontalMesh).nEdges,
                    nVertLevels=dimsize(m.VerticalMesh).nVertLevels)

@inline Base.size(m::Mesh) = (dimsize(m.HorizontalMesh).nCells,
                              dimsize(m.HorizontalMesh).nVertices,
                              dimsize(m.HorizontalMesh).nEdges,
                              dimsize(m.VerticalMesh).nVertLevels)

const LT = Union{Type{Cell}, Type{Edge}, Type{Vertex}, Type{Layer}}

Base.size(m::Mesh, loc::Cell) = size(m.HorizontalMesh, loc)
Base.size(m::Mesh, loc::Edge) = size(m.HorizontalMesh, loc)
Base.size(m::Mesh, loc::Vertex) = size(m.HorizontalMesh, loc)
Base.size(m::Mesh, loc::Layer) = size(m.VerticalMesh, loc)
Base.size(m::Mesh, loc::LT) = size(m, loc())

Base.size(m::Mesh, loc_tuple::Tuple) = Tuple(size(m, loc) for loc in loc_tuple)

"""
    Mesh(filepath::AbstractString, architecture=CPU(), FT=Float64; kwargs...)

Construct a `Mesh` with data type `FT` by reading a MPAS mesh (netcdf) file
from disk onto `architecture` (CPU() or GPU()). The `Mesh` structure contains
both the `HorizontalMesh` and `VerticalMesh` as fields.

Keyword arguments
=================

- `nVertLevels :: Int`: The number of vertical layers to create the mesh with.
  If `nothing` the vertical layer information will be read from the input
  `filepath`. If the input `filepath` does not contain an vertical mesh
  information then the vertical mesh will be created with one layer.
"""
function Mesh(filepath::AbstractString,
              architecture::AbstractArchitecture = CPU(),
              FT::DataType = Float64; kwargs...)

    ds = NCDataset(filepath, "r", format=:netcdf4)
    Mesh(ds, architecture, FT; kwargs...)
end

function Mesh(ds::NCDataset, arch::AbstractArchitecture, FT::DataType; nVertLevels=nothing)

    # Read horizontal mesh onto CPU with `FT` eltype
    horizontal_mesh = HorizontalMesh(ds, FT)

    if isnothing(nVertLevels)
        # Create a vertical mesh on the CPU, using the horizontal mesh
        vertical_mesh = VerticalMesh(ds, horizontal_mesh, FT)
    else
        # check kwarg nVertLevels is consitent with the input mesh file
        nVertLevels = validate_vertical_mesh_args(ds, nVertLevels)
        # create a stacked vertical mesh with (n) vertical levels
        vertical_mesh = VerticalMesh(horizontal_mesh; nVertLevels=nVertLevels)
    end
    # With both a horizontal and vertical mesh; initalize the boundary mask
    horizontal_mesh = set_boundary_mask(horizontal_mesh, vertical_mesh)

    # Create the full Mesh strucutre on the CPU
    return Mesh{FT}(arch, horizontal_mesh, vertical_mesh)
end

""" Initialize the Boundary Mask """
function set_boundary_mask(HorzMesh, VertMesh)

    nEdges = HorzMesh.Edges.nEdges
    nVertLevels = VertMesh.nVertLevels
    @unpack maxLevelEdge = VertMesh

    # allocate a new edgeMask array, now with the proper numbe of vert levels
    edgeMask = zeros(Int32, (nVertLevels, nEdges))
    for iEdge in 1:nEdges, k in 1:maxLevelEdge[iEdge]
        edgeMask[k, iEdge] = 1
    end

    # Edges struct is immutable so need to use Accessor package,
    @reset HorzMesh.Edges.edgeMask = edgeMask

    # For some reason we cannot manipulate this in place, and must
    # retrun in order to register the updated nested attributes
    return HorzMesh
end

""" Validate the """
function validate_vertical_mesh_args(ds::NCDataset, nVertLevels::Int)
    # parse nVertLevels from the NetCDF file
    _nVertLevels = has_vertical_dim(ds) ? ds.dim["nVertLevels"] : nothing

    if all(.!isnothing.([nVertLevels,_nVertLevels]))
        if nVertLevels != _nVertLevels
            err = DimensionMismatch("")
            @error sprint(showerror, err)
            throw(err)
        end
    else
        return nVertLevels
    end
end

has_vertical_dim(ds::NCDataset) = haskey(ds.dim, "nVertLevels")

#####
##### Utilities
#####

function Adapt.adapt_structure(device, x::Mesh)
    return Mesh(architecture(device),
                Adapt.adapt(device, x.HorizontalMesh),
                Adapt.adapt(device, x.VerticalMesh))
end

"""
    on_architecture(architecture, mesh::Mesh)

Return a mesh that's identical to `mesh` but on `architecture`.
"""
function on_architecture(arch::AbstractSerialArchitecture, mesh::Mesh)
    if arch == architecture(mesh)
        return grid
    end
    return Adapt.adapt_structure(Architectures.device(arch), mesh)
end
