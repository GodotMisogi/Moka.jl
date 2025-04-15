using Moka.Architectures
using Moka.Meshs

import Moka
    
function launch!(arch, mesh, kernel!, first_kernel_arg, other_kernel_args...;
                 exclude_vertical_loop=true)

    # TODO: Dynamically determine this
    nthreads = 50
    loc = Moka.location(first_kernel_arg)
    dev = Architectures.device(arch)

    loop! = kernel!(dev, nthreads)

    if !exclude_vertical_loop
        ndrange = size(mesh, loc)
    else
        # just get legnth of horizontal dimension
        ndrange = length(mesh, loc[0])
    end

    loop!(first_kernel_arg, other_kernel_args...; ndrange=ndrange)
 end
