module TimeSteppers

export
    ForwardEulerTimeStepper,
    Clock

using KernelAbstractions
using Moka: AbstractModel

"""
    abstract type AbstractTimeStepper

Abstract supertype for time steppers.
"""
abstract type AbstractTimeStepper end

"""
    TimeStepper(name::Symbol, args...; kwargs...)

Returns a timestepper with name `name`, instantiated with `args...` and `kwargs...`.

Example
=======

```julia
julia> stepper = TimeStepper(:ForwardBackwards, CPU(), mesh, tracernames)
```
"""
function TimeStepper(name::Symbol, args...; kwargs...)
    fullname = Symbol(name, :TimeStepper)
    TS = getglobal(@__MODULE__, fullname)
    return TS(args...; kwargs...)
end


# Fallback
TimeStepper(stepper::AbstractTimeStepper, args...; kwargs...) = stepper

function update_state! end
function compute_tendencies! end

reset!(timestepper) = nothing

include("clock.jl")
include("forward_euler.jl")
end
