module Fields

export location
export AbstractField, Field
export CellField, EdgeField, VertexField
#export set!, compute!, @compute, regrid!

using Moka.Architectures
using Moka.Meshs

import Moka.Architectures: on_architecture
import Moka: location

include("abstract_field.jl")
include("function_fields.jl")
include("field.jl")

@inline field(loc, a::Function, grid) = FunctionField(loc, a, grid)

end
