using Dates: AbstractTime, DateTime, Nanosecond, Millisecond

"""
    mutable struct Clock{TT<:AbstractTime, DT}

Keeps track of the current `time`, `last_Δt`, `iteration` number,
and time-stepping `stage`. The `stage` is updated only for
multi-stage time-stepping methods.
"""
mutable struct Clock{TT<:AbstractTime, DT}
    time :: TT
    last_Δt :: DT
    last_stage_Δt :: DT
    iteration :: Int
    stage :: Int
end

"""
    Clock(; time, last_Δt=Inf, last_stage_Δt=Inf, iteration=0, stage=1)

Returns a `Clock` object. By default, `Clock` is initialized to the zeroth `iteration`
and first time step `stage` with `last_Δt=last_stage_Δt=Inf`.
"""
function Clock(; time,
               last_Δt = Inf,
               last_stage_Δt = Inf,
               iteration = 0,
               stage = 1)

    TT = typeof(time)
    DT = typeof(last_Δt)
    last_stage_Δt = convert(DT, last_Δt)
    return Clock{TT, DT}(time, last_Δt, last_stage_Δt, iteration, stage)
end

function Base.summary(clock::Clock)
    TT = typeof(clock.time)
    DT = typeof(clock.last_Δt)
    return string("Clock{", TT, ", ", DT, "}",
                  "(time=", clock.time,
                  ", iteration=", clock.iteration,
                  ", last_Δt=", clock.last_Δt, ")")
end

function Base.show(io::IO, clock::Clock)
    return print(io, summary(clock), '\n',
                 "├── stage: ", clock.stage, '\n',
                 "└── last_stage_Δt: ", clock.last_stage_Δt)
end

# Assumed Δt to have units be in seconds
next_time(clock, Δt) = clock.time + Nanosecond(round(Int, 1e9 * Δt))
tick_time!(clock, Δt) = clock.time += Nanosecond(round(Int, 1e9 * Δt))

Time(clock::Clock) = Time(clock.time)

function tick!(clock, Δt; stage=false)

    tick_time!(clock, Δt)

    if stage # tick a stage update
        clock.stage += 1
    else # tick an iteration and reset stage
        clock.iteration += 1
        clock.stage = 1
    end

    return nothing
end
