# heavily lifted from: https://github.com/CliMA/Oceananigans.jl/blob/main/src/Architectures.jl
module Architectures

export AbstractArchitecture, AbstractSerialArchitecture
export CPU, GPU
export device, architecture, unified_array, device_copy_to!
export array_type, on_architecture, arch_array

using CUDA
using KernelAbstractions
using Adapt
using OffsetArrays

"""
    AbstractArchitecture

Abstract supertype for architectures supported by Moka.
"""
abstract type AbstractArchitecture end

"""
    AbstractSerialArchitecture

Abstract supertype for serial architectures supported by Moka.
"""
abstract type AbstractSerialArchitecture <: AbstractArchitecture end

"""
    CPU <: AbstractArchitecture

Run Moka on one CPU node. Uses multiple threads if the environment
variable `JULIA_NUM_THREADS` is set.
"""
struct CPU <: AbstractSerialArchitecture end

"""
    GPU(device)

Return a GPU architecture using `device`.
`device` defauls to CUDA.CUDABackend(always_inline=false)
"""
struct GPU{D} <: AbstractSerialArchitecture 
    device :: D
end

const CUDAGPU = GPU{<:CUDA.CUDABackend}
CUDAGPU() = GPU(CUDA.CUDABackend(always_inline=true))
Base.summary(::CUDAGPU) = "CUDAGPU"

function GPU()
    if CUDA.has_cuda_gpu()
        return CUDAGPU()
    else
        msg = """We cannot make a GPU with the CUDA backend:
                 a CUDA GPU was not found!"""
        throw(ArgumentError(msg))
    end
end


device(a::CPU) = KernelAbstractions.CPU()
device(a::GPU) = a.device

architecture() = nothing
architecture(::Number) = nothing
architecture(::Array) = CPU()
architecture(::CuArray) = CUDAGPU()
architecture(a::OffsetArray) = architecture(parent(a))

architecture(::CUDABackend) = CUDAGPU()
architecture(::KernelAbstractions.CPU) = CPU()

array_type(::CPU) = Array
array_type(::GPU) = CuArray

# Fallback 
on_architecture(arch, a) = a

# Tupled implementation
on_architecture(arch::AbstractSerialArchitecture, t::Tuple) = Tuple(on_architecture(arch, elem) for elem in t)
on_architecture(arch::AbstractSerialArchitecture, nt::NamedTuple) = NamedTuple{keys(nt)}(on_architecture(arch, Tuple(nt)))

# On architecture for array types
on_architecture(::CPU, a::Array) = a
on_architecture(::CPU, a::CuArray) = Array(a)

on_architecture(::CUDAGPU, a::Array) = CuArray(a)
on_architecture(::CUDAGPU, a::CuArray) = a

on_architecture(arch::AbstractSerialArchitecture, a::OffsetArray) = OffsetArray(on_architecture(arch, a.parent), a.offsets...)

cpu_architecture(::CPU) = CPU()
cpu_architecture(::GPU) = CPU()

unified_array(::CPU, a) = a
unified_array(::GPU, a) = a

end #module
