module OutputWriters

export NetCDFWriter, write_output!

abstract type AbstractOutputWriter end

Base.open(ow::AbstractOutputWriter) = nothing
Base.close(ow::AbstractOutputWriter) = nothing

include("output_writer_utils.jl")
include("netcdf_writer.jl")

end # module
