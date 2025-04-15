"""
    struct ForwardEulerTimeStepper{TG} <: AbstractTimeStepper

Holds tendency fields for a first-order forward backwards time-stepping
scheme.
"""
mutable struct ForwardEulerTimeStepper{TG} <: AbstractTimeStepper
    Gⁿ :: TG
    G⁻ :: TG
end

"""
    ForwardEulerTimeStepper(grid, prognostic_fields)

Return a 1st-order Forward Euler time stepper
"""
function ForwardEulerTimeStepper(mesh, prognostic_fields;
                                 Gⁿ::TG = map(similar, prognostic_fields),
                                 G⁻     = map(similar, prognostic_fields)) where {TG}
    return ForwardEulerTimeStepper{TG}(Gⁿ, G⁻)
end 

#####
##### Time steppping
#####

"""
    time_step!(model::AbstractModel{<:ForwardEulerTimeStepper}, Δt)

Step forward `model` one time step `Δt` with a 1st-order forward Euler method.
"""
function time_step!(model::AbstractModel{<:ForwardEulerTimeStepper}, Δt; callbacks=[])
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"

    tⁿ⁺¹ = next_time(model.clock, Δt)
    
    if model.clock.iteration == 0
        update_state!(model, callbacks; compute_tendencies = true)
    end

    FE_substep!(model, Δt)

    tick!(model.clock, Δt)
    model.clock.last_Δt = Δt
    
    update_state!(model, callbacks; compute_tendencies = true)

    return nothing
end

####
#### Time stepping in each substep
####

function FE_substep(model, Δt)

    mesh = model.mesh
    arch = architecture(mesh)
    model_fields = prognostic_fields(model)

    for (i, field) in enumerate(model_fields)
        # TODO: get the ndrange of the ith fields for update kernel 
        loc = location(field)
        kernel_args = (field, Δt, model.timestepper.Gⁿ[i])
        launch!(arch, grid, loc, FE_substep_field!, kernel_args...)
    end

    return nothing
end

"""
Time step example (i.e. velocity) field via 1st-order Forward-Euler Method

```
Uᵐ⁺¹ = Uᵐ + Δt * Gᵐ
```

where `m` denotes the substage.
"""
@kernel function FE_substep_field!(U, Δt, G¹, FT)
    i, k = @index(Global, NTuple)

    @inbounds begin
        U[i, k] += convert(FT, Δt) * G¹[i, k]
    end
end
