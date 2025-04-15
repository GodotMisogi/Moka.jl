using Moka: AbstractModel

using Moka.Architectures: AbstractArchitecture, CPU
using Moka.Fields: Field, CellField, EdgeField, VertexField
using Moka.TimeSteppers: Clock, TimeStepper

import Moka.Architectures: architecture

function shallow_water_tendency_fields(mesh, prognostic_names)
    u = EdgeField(mesh)
    h = CellField(mesh)
    return NamedTuple{prognostic_names}((u, h))
end

function shallow_water_solution_fields(mesh)
    u = EdgeField(mesh)
    h = CellField(mesh)
    return NamedTuple{(:u, :h)}((u, h))
end

mutable struct ShallowWaterModel{M, A <: AbstractArchitecture, GR, Q, C, TS} <: AbstractModel{TS}
                          mesh :: M         # Mesh of physical points on which `Model` is solved
                  architecture :: A         # Computer `Architecture` on which `Model` is run
                         clock :: Clock     # Tracks iteration number and simulation time of `Model`
    gravitational_acceleration :: GR        # Gravitational acceleration
                      solution :: Q         # Container for normal velocity `u` and thickness `h`
                       tracers :: C         # Container for tracer fields
                   timestepper :: TS        # Object containing timestepper fields and parameters
end

function ShallowWaterModel(;
                           mesh,
                           gravitational_acceleration = 9.80665,
                           clock = Clock(time=DateTime(0)),
                           timestepper::Symbol = :ForwardEuler,
                           tracers = (),)

    arch = architecture(mesh)

    @show arch
    prognostic_field_names = (:u, :h, tracers...)

    solution = shallow_water_solution_fields(mesh)

    # Instantiate timestepper if not already instantiated
    timestepper = TimeStepper(timestepper, mesh, solution;
                              Gⁿ = shallow_water_tendency_fields(mesh, prognostic_field_names),
                              G⁻ = shallow_water_tendency_fields(mesh, prognostic_field_names))

    model = ShallowWaterModel(mesh,
                              arch,
                              clock,
                              eltype(mesh)(gravitational_acceleration),
                              solution,
                              tracers,
                              timestepper)

    #update_state!(model; compute_tendencies = false)
end

architecture(model::ShallowWaterModel) = model.architecture
