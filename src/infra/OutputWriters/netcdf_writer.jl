using Dates
using NCDatasets
using OffsetArrays

import MOKA: yaml_config, ConfigGet
import MOKA: PrognosticVars, DiagnosticVars

import MOKA.MPASMesh: Mesh

const NCVar = Union{NCDatasets.Variable, NCDatasets.CFVariable}

mutable struct NetCDFWriter{M, D, O} <: AbstractOutputWriter
        mesh :: M
    filepath :: String
     dataset :: D
     outputs :: O
    interval :: Int
end

function NetCDFWriter(mesh::Mesh, filepath::String, output_variables)
    # This creates a new NetCDF file (clobber)
    ds = NCDataset(filepath, "c")

    # Add the dimensions to the dataset
    initialize_nc_dims!(ds, mesh)

    # Make sure requested output variables are supported
    validate_output_variables(output_variables)

    # Add the output variables to dataset
    outputs = NamedTuple(Symbol(var) => initialize_nc_var!(ds, var)
                         for var in output_variables)

    return NetCDFWriter(mesh, filepath, ds, outputs, 1)
end

function NetCDFWriter(mesh::Mesh, config::yaml_config)
    filepath = ConfigGet(config, "filename_template")
    variables = ConfigGet(config, "contents")
    # remove xtime from IO list if it's present, handled seperately
    deleteat!(variables, variables .== "xtime")
    return NetCDFWriter(mesh, filepath, variables)
end

Base.open(ow::NetCDFWriter) = NCDataset(ow.filepath, "a")
Base.close(ow::NetCDFWriter) = close(ow.dataset)

advance!(w::NetCDFWriter) = w.interval += 1

function write_output!(writer::NetCDFWriter,
                       Prog::PrognosticVars,
                       Diag::DiagnosticVars,
                       Time::DateTime)

    # write to the time cordinate
    #writer["time"][writer.interval] = Time

    # itterate over the ouput variables
    for (var, output) in zip(keys(writer.outputs), writer.outputs)

        # Get the data from the appropriate structure
        if String(var) ∈ keys(streams["prognostics"])
            data = getfield(Prog, var)[end]
        elseif String(var) ∈ keys(streams["diagnostics"])
            data = getfield(Diag, var)
        else
            @error "Unable to find $(var)"
        end

        write_array!(output, data, writer.interval)
    end

    # increment to the next IO level
    advance!(writer)

    return nothing
end

write_array!(var::NCVar, array::Array{N, 1}, i::Int) where {N} = var[:, i] = array[:]
write_array!(var::NCVar, array::Array{N, 2}, i::Int) where {N} = var[:, :, i] = array[:, :]
write_array!(var::NCVar, array::OffsetArray{N, 1, Array{N, 1}}, i::Int) where {N} = var[:, i] = array[1:end]
write_array!(var::NCVar, array::OffsetArray{N, 2, Array{N, 2}}, i::Int) where {N} = var[:, :, i] = array[:, 1:end]

function initialize_nc_var!(ds::NCDataset, io_var)
    # not the most efficent to be loop over this everytime, ohwell
    for (field_type, fields) in streams
        for (var, dict) in fields
            if var == io_var
                v = defVar(ds, var, Float32, dict["dimensions"])
                v.attrib["units"] = dict["units"]
                return v
             end
        end
     end
end

function initialize_nc_dims!(ds::NCDataset, mesh::Mesh)
    nEdges = mesh.HorzMesh.Edges.nEdges
    nCells = mesh.HorzMesh.PrimaryCells.nCells
    nVertices = mesh.HorzMesh.DualCells.nVertices
    nVertLevels = mesh.VertMesh.nVertLevels

    # Define horizontal dimensions
    defDim(ds, "nCells", nCells)
    defDim(ds, "nEdges", nEdges)
    defDim(ds, "nVertices", nVertices)
    # Define vertical dimensions
    defDim(ds, "nVertLevels", nVertLevels)
    # Define an unlimited time dimension
    defDim(ds, "time", Inf)
    # Define time coordinate
    #defVar(ds, "time", Float64, ("time",), attrib=Dict("units" => "days since 0000-01-01"))
end
