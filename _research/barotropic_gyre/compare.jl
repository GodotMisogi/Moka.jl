# %% Dependencies
using NCDatasets
using CairoMakie
using Statistics
using Dates
using LinearAlgebra
using DataFrames

# %% Exact Solution Struct
"""
Struct to compute the exact solution for the barotropic gyre test case.
"""
struct ExactSolution
    xCell          :: Vector{Float64}
    yCell          :: Vector{Float64}
    fCell          :: Vector{Float64}
    L_x            :: Float64
    L_y            :: Float64
    layerThickness :: Vector{Float64}
    tau_0          :: Float64
    rho            :: Float64
    g              :: Float64
    nu_2           :: Float64
    beta           :: Float64
    f_0            :: Float64
end

"""
    ExactSolution(ds::NCDataset) -> ExactSolution

Construct an ExactSolution from an MPAS mesh NCDataset for barotropic gyre.
"""
function ExactSolution(ds::NCDataset)
    xCell = Array(ds["xCell"][:])
    yCell = Array(ds["yCell"][:])
    fCell = Array(ds["fCell"][:])

    L_x = maximum(xCell) - minimum(xCell)
    L_y = maximum(yCell) - minimum(yCell)
    
    # Check if restingThickness exists, else fallback to bottomDepth
    if haskey(ds, "restingThickness")
        layerThickness = Array(ds["restingThickness"][:, 1])
    else
        layerThickness = Array(ds["bottomDepth"][:])
    end

    # Default parameters based on Python setup.py
    tau_0 = 0.1
    rho   = 1.e3
    g     = 9.81
    nu_2  = 4.e2
    beta  = 1e-11
    f_0   = 1e-4

    return ExactSolution(xCell, yCell, fCell, L_x, L_y, layerThickness,
                         tau_0, rho, g, nu_2, beta, f_0)
end

# %% Exact Solution Methods
"""
    ssh(exact::ExactSolution) -> Vector{Float64}

Exact sea surface height solution on cells for linear Munk layer.
"""
function ssh(exact::ExactSolution)
    pi_val = π
    sqrt3 = sqrt(3.0)
    delta_m = (exact.nu_2 / exact.beta)^(1.0 / 3.0)
    gamma = (sqrt3 .* exact.xCell) ./ (2.0 .* delta_m)

    ssh_vals = (exact.tau_0 ./ (exact.rho .* exact.g .* exact.layerThickness)) .* 
               (exact.fCell ./ exact.beta) .*
               (1.0 .- exact.xCell ./ exact.L_x) .* pi_val .* sin.(pi_val .* exact.yCell ./ exact.L_y) .*
               (1.0 .- exp.(-1.0 .* exact.xCell ./ (2.0 .* delta_m)) .*
               (cos.(gamma) .+ (1.0 ./ sqrt3) .* sin.(gamma)))

    return ssh_vals
end

# %% MPAS / MOJO Preprocessing
const MPAS_DT_fmt = dateformat"yyyy-mm-dd_HH:MM:SS"

"""
    preprocess_MPAS(ds::NCDataset) -> (dt, T_f, ssh_num)

Parse MPAS output: extract timestep, final time, SSH.
"""
function preprocess_MPAS(ds::NCDataset)
    xtime_raw = ds["xtime"][:]
    time_strs = [strip(String(collect(s))) for s in eachcol(xtime_raw)]

    times = [DateTime(s, MPAS_DT_fmt) for s in time_strs]
    T_f   = Float64(Dates.value(times[end] - times[1])) / 1000.0  # ms -> s

    # ssh is (nCells,), take the last time step if there's a time dimension
    if ndims(ds["ssh"]) > 1
        ssh_num = Array(ds["ssh"][:, end])
    else
        ssh_num = Array(ds["ssh"][:])
    end

    dt_str = split(ds.attrib["config_dt"], "_")

    if parse(Int, dt_str[1]) == 0
        t  = Time(dt_str[2], dateformat"HH:MM:SS")
        dt = Float64(
            Dates.value(
                Millisecond(Hour(t) + Minute(t) + Second(t))
            )
        ) / 1000.0
    else
        error("Problem parsing: " * join(dt_str, "_"))
    end

    return dt, T_f, ssh_num
end

"""
    preprocess_MOJO(ds::NCDataset) -> (dt, T_f, ssh_num)

Parse MOJO output: extract timestep, final time, SSH.
"""
function preprocess_MOJO(ds::NCDataset)
    # time is a variable, not an attribute
    T_f = Float64(Array(ds["time"][:])[end])
    dt = Float64(ds.attrib["dt"])

    if ndims(ds["ssh"]) > 1
        ssh_num = Array(ds["ssh"][:, end])
    else
        ssh_num = Array(ds["ssh"][:])
    end

    return dt, T_f, ssh_num
end


# %% Read and Compare
"""
    read_and_compare(ds_fp, mesh_fp) -> (results::Dict, mesh_ds::NCDataset)

Load numerical results, compute analytical solution, and return comparison dict.
"""
function read_and_compare(ds_fp::String, mesh_fp::String)
    num_ds = NCDataset(ds_fp)

    if haskey(num_ds.attrib, "model_name")
        mesh_ds = NCDataset(mesh_fp)
        exact   = ExactSolution(mesh_ds)
        dt, T_f, ssh_num = preprocess_MPAS(num_ds)
        dcEdge  = mean(Array(mesh_ds["dcEdge"][:])) / 1e3
        close(mesh_ds)
    else
        exact   = ExactSolution(num_ds)
        dt, T_f, ssh_num = preprocess_MOJO(num_ds)
        dcEdge  = mean(Array(num_ds["dcEdge"][:])) / 1e3
    end

    ssh_ext = ssh(exact)

    results = Dict(
        "ssh_num" => ssh_num,
        "ssh_ext" => ssh_ext,
        "ssh_err" => ssh_ext .- ssh_num,
        "dt"      => dt,
        "time"    => T_f,
        "dcEdge"  => dcEdge,
    )

    close(num_ds)
    return results, exact
end

# %% Error Logging
"""
    log_error(results::Dict) -> DataFrame

Compute RMSE for SSH and return as a one-row DataFrame.
"""
function log_error(results::Dict)
    T_f = results["time"]
    dc  = results["dcEdge"]
    dt  = results["dt"]

    ssh_error = sqrt(mean(results["ssh_err"] .^ 2))

    return DataFrame(
        T_f      = T_f,
        dt       = dt,
        dc       = dc,
        rmse_ssh = ssh_error,
    )
end

# %% Plot Fields
function plot_fields(results::Dict, exact::ExactSolution, res_name::String)
    fig = Figure(size = (1200, 400))
    
    ax1 = Axis(fig[1, 1], title="Numerical SSH ($(res_name))")
    sc1 = scatter!(ax1, exact.xCell ./ 1e3, exact.yCell ./ 1e3, color=results["ssh_num"], colormap=:viridis, markersize=5)
    Colorbar(fig[1, 2], sc1)

    ax2 = Axis(fig[1, 3], title="Exact SSH")
    sc2 = scatter!(ax2, exact.xCell ./ 1e3, exact.yCell ./ 1e3, color=results["ssh_ext"], colormap=:viridis, markersize=5)
    Colorbar(fig[1, 4], sc2)

    ax3 = Axis(fig[1, 5], title="Error SSH")
    sc3 = scatter!(ax3, exact.xCell ./ 1e3, exact.yCell ./ 1e3, color=results["ssh_err"], colormap=:RdBu, colorrange=(-maximum(abs.(results["ssh_err"])), maximum(abs.(results["ssh_err"]))), markersize=5)
    Colorbar(fig[1, 6], sc3)
    
    save("ssh_comparison_$(res_name).png", fig; px_per_unit = 2)
end

# %% Main Script: Convergence Plot
# Usually runs on different resolutions like 240km, 120km, 60km, 30km or something equivalent for gyre. 
# We'll put generic folders here, adjusting based on actual simulated data.

res_dirs = ["40km", "20km", "10km"]
df_list = DataFrame[]

for dir_ in reverse(res_dirs)
    mesh_fp   = joinpath(dir_, "initial_state.nc")
    output_fp = joinpath(dir_, "output.nc")

    if isfile(mesh_fp) && isfile(output_fp)
        results, exact = read_and_compare(output_fp, mesh_fp)
        push!(df_list, log_error(results))
        plot_fields(results, exact, dir_)
    else
        println("Skipping $dir_ as output.nc or initial_state.nc is missing")
    end
end

if !isempty(df_list)
    df = vcat(df_list...)

    # --- Polynomial fit in log10 space ---
    log_dc  = log10.(df.dc)
    log_ssh = log10.(df.rmse_ssh)

    A    = hcat(log_dc, ones(length(log_dc)))
    poly = A \ log_ssh

    convergence = poly[1]
    conv_round  = round(convergence; digits=3)

    fit     = df.dc .^ poly[1] .* 10 .^ poly[2]
    order1  = 0.5 .* df.rmse_ssh[end] .* (df.dc ./ df.dc[end])
    order2  = 0.5 .* df.rmse_ssh[end] .* (df.dc ./ df.dc[end]) .^ 2

    # --- CairoMakie convergence plot ---
    fig = Figure(size = (500, 400))
    ax  = Axis(fig[1, 1],
        xscale      = log10,
        yscale      = log10,
        title       = "Barotropic Gyre Convergence",
        xlabel      = "Resolution (km)",
        ylabel      = "SSH RMSE",
    )

    lines!(ax, df.dc, order1;
        color     = :black,
        linestyle = :dash,
        alpha     = 0.3,
        label     = "First order",
    )

    lines!(ax, df.dc, order2;
        color     = :black,
        linestyle = :solid,
        alpha     = 0.3,
        label     = "Second order",
    )

    lines!(ax, df.dc, fit;
        color = :black,
        label = "Linear fit (order = $(conv_round))",
    )

    scatter!(ax, df.dc, df.rmse_ssh;
        marker = :circle,
        label  = "RMSE SSH",
    )

    axislegend(ax; position = :lt)

    save("convergence_barotropic_gyre.png", fig; px_per_unit = 3)
    println("Convergence plot saved successfully.")
end
