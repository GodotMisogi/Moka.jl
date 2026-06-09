using Dates, CUDA, MOKA, KernelAbstractions
using CUDA: @allowscalar
const KA = KernelAbstractions

function ocn_run(config_fp)
    backend = CUDABackend(prefer_blocks=false, always_inline=false)
    Setup, Diag, Tend, Prog, Forcing, IO_writer = ocn_init(config_fp; backend=backend)
    clock, simulationAlarm, outputAlarm = ocn_init_alarms(Setup)
    timestep = KA.zeros(backend, Float64, (1,))
    @allowscalar timestep[1] = convert(Float64, Dates.value(Second(Setup.timeManager.timeStep)))
    ocn_run_loop(timestep, Prog, Diag, Tend, Forcing, Setup, RungeKutta4,
                 clock, simulationAlarm, outputAlarm, IO_writer; backend=backend)
    println("Moka.jl ran on $(typeof(backend) <: KA.GPU ? "GPU" : "CPU")")
    println(clock.currTime)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) > 0 && isfile(ARGS[1]) ? ocn_run(ARGS[1]) : error("Usage: julia mpas_ocean.jl <config.yml>")
end
