import Moka.TimeSteppers: update_state!

"""
    update_state!(model::ShallowWaterModel, callbacks=[]; compute_tendencies=true)

Update the diagnostic state of `ShallowWaterModel`.

1. update model time series,
2. compute auxiliary fields used in tendency calculations
3. `callbacks` are executed
4. tendencies are computed if `compute_tendencies=true`
"""
function update_state!(model::ShallowWaterModel, callbacks=[]; compute_tendencies=true)

    # Update possible FieldTimeSeries used in the model
    update_model_field_time_series!(model, model.clock)

    # Compute auxiliary fields
    for aux_field in model.auxiliary_fields
        compute!(aux_field)
    end
    
    # TODO: Build out callback infrastrucutre
    #for callback in callbacks
    #    callback.callsite isa UpdateStateCallsite && callback(model)
    #end

    compute_tendencies && compute_tendencies!(model, callbacks)

    return nothing
end
