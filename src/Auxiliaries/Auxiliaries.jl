module Auxiliaries


function compute!(aux::AbstractAuxiliaryField, model::AbstractModel)
    compute!(aux, model.solution)
end
