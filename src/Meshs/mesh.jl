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

#=
function Mesh(Config::GlobalConfig; backend=KA.CPU())
    # get mesh section of the streams file
    meshConfig = ConfigGet(Config.streams, "mesh")
    # get mesh filepath from streams section
    meshPath = ConfigGet(meshConfig, "filename_template")
    # read the mesh file once
    mesh_ds = NCDataset(meshPath, "r", format=:netcdf4)
    # checks nVertLevels from config file to ensure consitency with mesh file
    nVertLevels = validate_vertical_mesh_args(mesh_ds, meshConfig)

    return Mesh(mesh_ds; nVertLevels=nVertLevels, backend=backend)
end
=#

"""
    Mesh(filepath::AbstractString; FT=Float64, Arch=CPU(), nVertLevels=nothing)

Constuctor function to read an MPAS mesh from disk and initialize both a horizontal
and vertical mesh, with element types `FT` and on architecutre `Arch`
"""
function Mesh(ds::NCDataset; FT=Float64, nVertLevels=nothing)

    # Read in the purely horizontal mesh on the CPU
    horizontal_mesh = HorizontalMesh(ds)

    if isnothing(nVertLevels)
        # Create a vertical mesh on the CPU, using the horizontal mesh
        vertical_mesh = VerticalMesh(ds, horizontal_mesh)
    else
        # check kwarg nVertLevels is consitent with the input mesh file
        nVertLevels = validate_vertical_mesh_args(ds, nVertLevels)
        # create a stacked vertical mesh with (n) vertical levels
        vertical_mesh = VerticalMesh(horizontal_mesh; nVertLevels=nVertLevels)
    end
    # With both a horizontal and vertical mesh; initalize the boundary mask
    horizontal_mesh = setBoundaryMask(horizontal_mesh, vertical_mesh)

    # Create the full Mesh strucutre on the CPU
    return Mesh{FT}(CPU(), horizontal_mesh, vertical_mesh)
end

function Mesh(mesh_fp::String; kwargs...)
    Mesh(NCDataset(mesh_fp, "r", format=:netcdf4); kwargs...)
end

function Adapt.adapt_structure(backend, x::Mesh)
    return Mesh(Adapt.adapt(backend, x.HorzMesh),
                Adapt.adapt(backend, x.VertMesh))
end

function setBoundaryMask(HorzMesh, VertMesh)

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
