using TOML

# static nested dict of supported vars and their dims and attrs
const streams = TOML.parsefile(joinpath(@__DIR__, "Streams.toml"))
# static vector of supported variables for convience
const supported_variables = begin
    reduce(vcat, [collect(keys(streams[k])) for k in keys(streams)])
end

function validate_output_variables(output_variables)
    for var in output_variables
        if String(var) ∉ supported_variables
            @warn "$(var) is not a supported output variable"
        end
    end
end
