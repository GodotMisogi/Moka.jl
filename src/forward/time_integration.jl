# define our parent abstract type
abstract type timeStepper end
# define the supported timeStepper types to dispatch on.
abstract type ForwardEuler <: timeStepper end
abstract type RungeKutta4  <: timeStepper end

using CUDA: @allowscalar
using KernelAbstractions

function advanceTimeLevels!(Prog::PrognosticVars; backend=CUDABackend())

    nthreads = 100

    kernel2d! = advance_2d_array(backend, nthreads)
    kernel3d! = advance_3d_array(backend, nthreads)

    for field_name in propertynames(Prog)

        ndim = field_name == :ssh ? 1 : 2

        field = getproperty(Prog, field_name)

        if length(field) > 2 error("nTimeLevels must be <= 2") end

        # Here: set first entry of Vector{Array} equal to second

        # some short hand for this would be nice
        if ndim == 1
            #field[:,end-1] .= field[:,end]
            #@show size(field), size(field)[1]
            kernel2d!(field[1], field[2], size(field[1])[1], ndrange=size(field[1])[1])
        else
            #field[:,:,end-1] .= field[:,:,end]
            #kernel3d!(field, ndrange=(size(field)[1],size(field)[2]))
            kernel3d!(field[1], field[2], size(field[1])[2], ndrange=size(field[1])[2])
        end

        setproperty!(Prog, field_name, field)
    end
end

@kernel function advance_2d_array(fieldPrev, @Const(fieldNext), arrayLength)
    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds fieldPrev[j] = fieldNext[j]
    end
    @synchronize()
end

@kernel function advance_3d_array(fieldPrev, @Const(fieldNext), arrayLength)
    #i, j = @index(Global, NTuple)
    #@inbounds fieldPrev[i, j] = fieldNext[i, j]

    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds fieldPrev[1, j] = fieldNext[1, j]
    end
    @synchronize()
end

function ocn_timestep(_timestep,
                      Prog::PrognosticVars,
                      Diag::DiagnosticVars,
                      Tend::TendencyVars,
                      Forcing::ForcingVars,
                      S::ModelSetup,
                      ::Type{RungeKutta4};
                      backend = KA.CPU())

    Mesh = S.mesh
    Clock = S.timeManager
    Config = S.config

    advanceTimeLevels!(Prog; backend=backend)

    dt = convert(Float64, Dates.value(Second(Clock.timeStep)))

    a = [dt/2., dt/2., dt]
    b = [dt/6., dt/3., dt/3., dt/6.]

    normalVelocityCurr   = @view Prog.normalVelocity[:,:,end-1]
    layerThicknessCurr   = @view Prog.layerThickness[:,:,end-1]
    normalVelocityProvis = @view Prog.normalVelocity[:,:,end]
    layerThicknessProvis = @view Prog.layerThickness[:,:,end]

    @unpack ssh, normalVelocity, layerThickness = Prog

    # accumulation arrays — copies, not views, so substep overwrites don't corrupt them
    normalVelocityNew  = normalVelocity[:,:,end]
    layerThicknessNew  = layerThickness[:,:,end]

    for RK_step in 1:4
        computeNormalVelocityTendency!(Tend, Prog, Diag, Mesh, Forcing, Config; backend=backend)
        computeLayerThicknessTendency!(Tend, Prog, Diag, Mesh, Config; backend=backend)

        @unpack tendNormalVelocity, tendLayerThickness = Tend

        if RK_step < 4
            normalVelocityProvis  .= normalVelocityCurr  .+ a[RK_step] .* tendNormalVelocity
            layerThicknessProvis  .= layerThicknessCurr  .+ a[RK_step] .* tendLayerThickness
            diagnostic_compute!(Mesh, Diag, Prog; backend=backend)
        end

        normalVelocityNew .= normalVelocityNew .+ b[RK_step] .* tendNormalVelocity
        layerThicknessNew .= layerThicknessNew .+ b[RK_step] .* tendLayerThickness
    end

    normalVelocity[:,:,end] = normalVelocityNew
    layerThickness[:,:,end] = layerThicknessNew

    @pack! Prog = ssh, normalVelocity, layerThickness

    diagnostic_compute!(Mesh, Diag, Prog; backend=backend)
end

function ocn_timestep(timestep,
                      Prog::PrognosticVars,
                      Diag::DiagnosticVars,
                      Tend::TendencyVars,
                      Forcing::ForcingVars,
                      S::ModelSetup,
                      ::Type{ForwardEuler};
                      backend = CUDABackend())

    Mesh = S.mesh
    Clock = S.timeManager
    Config = S.config

    # advance the timelevels within the state strcut
    advanceTimeLevels!(Prog; backend=backend)

    # unpack the state variable arrays
    @unpack ssh, normalVelocity, layerThickness = Prog

    # compute the diagnostics
    diagnostic_compute!(Mesh, Diag, Prog; backend = backend)

    # compute normalVelocity tenedency
    computeNormalVelocityTendency!(Tend, Prog, Diag, Mesh, Forcing, Config;
                                   backend = backend)

    # compute layerThickness tendency
    computeLayerThicknessTendency!(Tend, Prog, Diag, Mesh, Config;
                                   backend = backend)

    # update the state variables by the tendencies
    nthreads = 50
    tendKernel! = UpdateStateVariable!(backend, nthreads)

    tendKernel!(normalVelocity[end], Tend.tendNormalVelocity, timestep, Mesh.HorzMesh.Edges.nEdges, ndrange=Mesh.HorzMesh.Edges.nEdges)
    tendKernel!(layerThickness[end], Tend.tendLayerThickness, timestep, Mesh.HorzMesh.PrimaryCells.nCells, ndrange=Mesh.HorzMesh.PrimaryCells.nCells)

    ssh_length = size(ssh[end])[1]

    kernel! = Update_ssh!(backend, nthreads)
    kernel!(ssh[end], Prog.layerThickness[end], Mesh.VertMesh.restingThicknessSum, ssh_length, ndrange=ssh_length)

    @pack! Prog = ssh, normalVelocity, layerThickness

end

# Zeros out a vector along its entire length
@kernel function UpdateStateVariable!(var, @Const(tendVar), @Const(dt), arrayLength)
    j = @index(Global, Linear)
    if j < arrayLength + 1
        var[1,j] = var[1,j] + dt[1] * tendVar[1, j]
    end
    @synchronize()
end


@kernel function Update_ssh!(ssh, @Const(layerThickness), @Const(restingThicknessSum), arrayLength)

    j = @index(Global, Linear)
    if j < arrayLength + 1
        @inbounds ssh[j] = layerThickness[1,j] - restingThicknessSum[j]
    end
    @synchronize()
end
