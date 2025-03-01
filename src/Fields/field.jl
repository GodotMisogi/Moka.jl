using Adapt
using Base: @propagate_inbounds

struct Field{LH, LV, M, D, T} <: AbstractField{LH, LV, M, T, 2}
    mesh :: M
    data :: D

    # Inner constructor that does not validate _anything_! 
    function Field{LH, LV}(mesh::M, data::D) where {LH, LV, M, D}
        T = eltype(data)
        return new{LH, LV, M, D, T}(mesh, data)
    end
end

#####
##### Constructor utilities
#####

function validate_field_data(loc, data, mesh)
    Fh, Fv = size(mesh, loc)

    if size(data) != (Fh, Fv)
        LH, LV = loc
        e = "Cannot construct field at ($LH, $LV) with size(data)=$(size(data)). " *
            "`data` must have size ($Fh, $Fv)."
        throw(ArgumentError(e))
    end

    return nothing
end

#####
##### Some basic constructors
#####

# Common outer constructor that performs input validation
function Field(loc::Tuple, mesh::Mesh, data)
    validate_field_data(loc, data, mesh)
    LH, LV = loc
    return Field{LH, LV}(mesh, data)
end

"""
    Field{LH, LV}(mesh::Mesh, T::DataType=eltype(mesh); kw...) where {LH, LV}

Construct a `Field` on `mesh` with data type `T` at the location `(LH, LV)`.
`LH` is either `Cell`, `Edge`, or `Vertex` and determines the field's location
on the horixontal mesh. `LH` is always `Layer` corresponding to layer midpoints.

Keyword arguments
=================

- `data :: OffsetArray`: An offset array with the fields data. If nothing is provided the
  field is filled with zeros.
"""
function Field{LH, LV}(mesh::Mesh, T::DataType=eltype(mesh); kw...) where {LH, LV}
    return Field((LH, LV), grid, T; kw...)
end

function Field(loc::Tuple,
               mesh::Mesh,
               T::DataType = eltype(mesh);
               data = zeros(T, size(mesh, loc)))

    return Field(loc, mesh, data)
end

"""
    CellField(mesh, T=eltype(mesh); kw...)

Return a `Field{Cell, Layer}` on `mesh`.
Additional keyword arguments are passed to the `Field` constructor.
"""
CellField(mesh::Mesh, T::DataType=eltype(mesh); kw...) = Field((Cell, Layer), mesh, T; kw...)

"""
    EdgeField(grid, T=eltype(grid); kw...)

Return a `Field{Edge, Layer}` on `mesh`.
Additional keyword arguments are passed to the `Field` constructor.
"""
EdgeField(mesh::Mesh, T::DataType=eltype(mesh); kw...) = Field((Edge, Layer), mesh, T; kw...)

"""
    VertexField(grid, T=eltype(grid); kw...)

Return a `Field{Vertex, Layer}` on `mesh`.
Additional keyword arguments are passed to the `Field` constructor.
"""
VertexField(mesh::Mesh, T::DataType=eltype(mesh); kw...) = Field((Vertex, Layer), mesh, T; kw...)

#####
##### Field utils
#####

data(field::Field) = field.data

# Don't use axes(f) to checkbounds; use axes(f.data)
Base.checkbounds(f::Field, I...) = Base.checkbounds(f.data, I...)

@propagate_inbounds Base.getindex(f::Field, inds...) = getindex(f.data, inds...)
@propagate_inbounds Base.setindex!(f::Field, val, i, k) = setindex!(f.data, val, i, k)
@propagate_inbounds Base.lastindex(f::Field) = lastindex(f.data)
@propagate_inbounds Base.lastindex(f::Field, dim) = lastindex(f.data, dim)

Adapt.adapt_structure(to, f::Field) = Adapt.adapt(to, f.data)

#####
##### Move Fields between architectures
#####

on_architecture(arch, field::Field{LH, LV}) where {LH, LV} =
    Field{LH, LV}(on_architecture(arch, field.mesh),
                  on_architecture(arch, field.data))
