using Moka
using Moka.Models: ShallowWaterModel

mesh = Mesh("/global/u2/a/anolan/Moka.jl/examples/initial_state.nc", CPU())

model = ShallowWaterModel(; mesh)

