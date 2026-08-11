module Tracer

export compute_tracer_tendency!

using UnPack
using KernelAbstractions
using MOKA: TendencyVars, PrognosticVars, DiagnosticVars, Mesh, DEFAULT_NTHREADS

const KA = KernelAbstractions

include("horizontal_advection.jl")
include("horizontal_mixing.jl")

"""
    compute_tracer_tendency!(Tend, Prog, Diag, Mesh; tracerDel2=0.0, nthreads=...)

Assemble the tracer tendency `(nTracers, nVertLevels, nCells)`: horizontal
advection of every tracer by the thickness flux, plus optional del2 (Laplacian)
tracer mixing. A no-op when there are no tracers (`nTracers == 0`). The tendency
is zeroed first, then the terms are summed in.
"""
function compute_tracer_tendency!(Tend::TendencyVars,
                                  Prog::PrognosticVars,
                                  Diag::DiagnosticVars,
                                  Mesh::Mesh;
                                  tracerDel2::Float64=0.0,
                                  nthreads=DEFAULT_NTHREADS)
    nTracers = size(Tend.tendTracers, 1)
    nTracers == 0 && return nothing

    backend = KA.get_backend(Tend.tendTracers)
    nCells      = Mesh.HorzMesh.PrimaryCells.nCells
    nVertLevels = Mesh.VertMesh.nVertLevels

    # zero the tracer tendency accumulator
    z! = zero_tracer_tendency!(backend, nthreads)
    z!(Tend.tendTracers, nTracers, nCells, ndrange=(nCells, nVertLevels, nTracers))

    tracer_horizontal_advection_tendency!(Tend, Prog, Diag, Mesh; nthreads=nthreads)

    if tracerDel2 != 0.0
        tracer_horizontal_mixing_tendency!(Tend, Prog, Diag, Mesh;
                                           tracerDel2=tracerDel2, nthreads=nthreads)
    end

    return nothing
end

@kernel function zero_tracer_tendency!(tend, nTracers, arrayLength)
    iCell, k, t = @index(Global, NTuple)
    if iCell < arrayLength + 1
        @inbounds tend[t, k, iCell] = 0.0
    end
end

end # module Tracer
