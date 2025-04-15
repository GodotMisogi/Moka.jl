struct FunctionField{LH, LV, C, P, F, M, T} <: AbstractField{LH, LV, M, T, 2}
          func :: F
          mesh :: M
         clock :: C
    parameters :: P

    """
       FunctionField{LH, LV}(func, mesh)

    """

    function FunctionField{LH, LV}(func::F,
                                   mesh::M,
                                   clock::C=nothing,
                                   parameters::P=nothing) where {LH, LV, F, M, C, P}
        FT = eltype(mesh)
        return new{LH, LV, C, P, F, M, FT}(func, mesh, clock, parameters)
    end
end

# This is a convenience form with `L` as positional argument.
@inline FunctionField(L::Tuple, func, mesh) = FunctionField{L[1], L[2]}(func, mesh)

