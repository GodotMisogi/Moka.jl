module Models

export ShallowWaterModel

using Moka: AbstractModel

iteration(model::AbstractModel) = model.clock.iteration
Base.time(model::AbstractModel) = model.clock.time
Base.eltype(model::AbstractModel) = eltype(model.grid)
architecture(model::AbstractModel) = model.grid.architecture
initialize!(model::AbstractModel) = nothing
timestepper(model::AbstractModel) = model.timestepper

include("ShallowWaterModels/ShallowWaterModels.jl")

using .ShallowWaterModels: ShallowWaterModel

end
