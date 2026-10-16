module OutputSignaturesLogExpFunctionsExt

using OutputSignatures: ValueInfo, _machine, _floating, _scalar_function_call
import OutputSignatures: _infer_call, _scalar_type
using LogExpFunctions: logaddexp, logistic

# Metadata for LogExpFunctions' real scalar methods, and through _scalar_type
# their broadcasts. logistic is floating like exp. logaddexp promotes its two
# operands, then returns their floating type (Bool and integers give Float64).
const _LogExp = Union{typeof(logaddexp),typeof(logistic)}
_scalar_type(f::_LogExp, ts::Vector{Type}) = all(_machine, ts) ? _logexp_type(f, ts) : nothing
_logexp_type(::typeof(logistic), ts::Vector{Type}) = length(ts) == 1 ? _floating(ts[1]) : nothing
_logexp_type(::typeof(logaddexp), ts::Vector{Type}) =
    length(ts) == 2 ? _floating(promote_type(ts[1], ts[2])) : nothing
_infer_call(f::_LogExp, args::Vector{ValueInfo}) = _scalar_function_call(f, args)

end # module
