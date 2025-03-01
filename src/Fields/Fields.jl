module Fields

export AbstractField, Field
export CellField, EdgeField, VertexField
#export set!, compute!, @compute, regrid!

using Moka.Architectures
using Moka.Meshs

import Moka.Architectures: on_architecture

include("abstract_field.jl")
include("field.jl")

end
