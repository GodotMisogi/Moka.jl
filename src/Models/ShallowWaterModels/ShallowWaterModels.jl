module ShallowWaterModels

export ShallowWaterModel

using Dates: DateTime
####
#### ShallowWaterModel definition
####

include("shallow_water_model.jl")
#include("set_shallow_water_model.jl")
#include("show_shallow_water_model.jl")

#####
##### Time-stepping ShallowWaterModels
#####

"""
    fields(model::ShallowWaterModel)

Return a flattened `NamedTuple` of the fields in `model.solution` and `model.tracers` for
a `ShallowWaterModel` model.
"""
fields(model::ShallowWaterModel) = merge(model.solution, model.tracers)

"""
    prognostic_fields(model::HydrostaticFreeSurfaceModel)

Return a flattened `NamedTuple` of the prognostic fields associated with `ShallowWaterModel`.
"""
prognostic_fields(model::ShallowWaterModel) = fields(model)

end
