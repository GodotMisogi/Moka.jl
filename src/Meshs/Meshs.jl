module Meshs

export Cell, Edge, Vertex, Layer
export Mesh, HorizontalMesh, VerticalMesh
export on_architecture

using Adapt
using Accessors
using CUDA
using NCDatasets
using OffsetArrays
using UnPack

using Moka
using Moka.Architectures

import Base: size, eltype

#####
##### Abstract types
#####

"""
    Cell

A type describing the location at the center of Primary Cell
"""
struct Cell end

"""
    Vertex

A type describing the location at the center of Dual Cells
"""
struct Vertex end

"""
    Edge

A type describing the location edge points where velocity is defined
"""
struct Edge end

"""
    Layer
A type describing the location of the vertical layer midpoints
"""
struct Layer end

include("horizontal_mesh_elements.jl")
include("horizontal_mesh.jl")
include("vertical_mesh.jl")
include("mesh.jl")

end
