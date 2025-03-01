
"""
    AbstractField{LH, LV, M, T, N}

Abstract supertype for fields located at horizontal `LH` and vertical `LV` location,
defined on a mesh `M` with eltype `T` and `N` dimensions.

Note: we need the parameter `T` to subtype AbstractArray.
"""
abstract type AbstractField{LH, LV, M, T, N} <: AbstractArray{T, N} end

Base.IndexStyle(::Type{<:AbstractField}) = IndexCartesian()

#####
##### AbstractField functionality
#####

@inline location(::AbstractField{LH, LV}) where {LH, LV} = (LH, LV) # note no instantiation
@inline instantiated_location(::AbstractField{LH, LV}) where {LH, LV} = (LH(), LV())
Base.eltype(::AbstractField{<:Any, <:Any, <:Any, T}) where T = T

"Returns the architecture of on which `f` is defined."
architecture(f::AbstractField) = architecture(f.mesh)

"""
    size(f::AbstractField)

Returns the size of an `AbstractField{LX, LY, LZ}` located at `LX, LY, LZ`.
This is a 3-tuple of integers corresponding to the number of interior nodes
of `f` along `x, y, z`.
"""
Base.size(f::AbstractField) = size(f.mesh, location(f))
Base.length(f::AbstractField) = prod(size(f))

const Abstract2DField = AbstractField{<:Any, <:Any, <:Any, <:Any, 2}
const Abstract3DField = AbstractField{<:Any, <:Any, <:Any, <:Any, 3}
