using CUDA

#####
##### set!
#####

set!(obj, ::Nothing) = nothing

# interface
set!(u::Field, f::Function) = set_to_function!(u, f)
set!(u::Field, a::Union{Array, CuArray, OffsetArray}) = set_to_array!(u, a)
set!(u::Field, v::Field) = set_to_field!(u, v)

function set!(u::Field, v)
    u .= v # fallback
    return u
end

### 
### Set to specific things
###

function set_to_function(u, f)
    # Only supports serial 
    arch = architecture(u)
