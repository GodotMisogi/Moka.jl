import Moka.TimeSteppers: compute_tendencies!

"""
	compute_tendencies!(model::ShallowWaterModel)

"""
function compute_tendencies!(model::ShallowWaterModel, callbacks)

    u_solution_tendency!(...)
    h_solution_tendency!(...)

    # TODO: Build out callback infrastrucutre
    #[callback(model) for callback in callbacks if isa(callback.callsite, TendencyCallsite)]

    return nothing
end

#####
##### Tendency calculators for the normal velocity and height: u, h
#####

"""
Compute the tendency for the layer thickness, h.
"""
function thickness_solution_tendency!(model::ShallowWaterModel)
    thickness_flux_divergence!(model.timestepper.Gⁿ[1]
                               model.solution.u,
                               model.solution.h,
                               model.mesh)
end
