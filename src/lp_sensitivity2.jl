#  Copyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

import SparseArrays
import LinearAlgebra

"""
    SensitivityReport

See [`lp_sensitivity`](@ref).
"""
struct SensitivityReport
    rhs::Dict{ConstraintRef, Tuple{Float64, Float64}}
    objective::Dict{VariableRef, Tuple{Float64, Float64}}
end

Base.getindex(s::SensitivityReport, c::ConstraintRef) = s.rhs[c]
Base.getindex(s::SensitivityReport, x::VariableRef) = s.objective[x]

"""
    lp_sensitivity(model::Model; atol::Float64 = 1e-8)::SensitivityReport

Given a linear program `model` containing a current optimal basis, return a
[`SensitivityReport`](@ref) object, which maps:

 - every constraint reference to a range over which the right-hand side of the
    corresponding constraint can vary such that the basis remains optimal.
    ```julia
    model = lp_sensitivity(model)
    dRHS_lo, dRHS_hi = model[c]
    ```
    Note: interval constraints are NOT supported.

 - every variable reference to a range over which the objective coefficient of
    the corresponding variable can vary such that the basis remains optimal.
    ```julia
    model = lp_sensitivity(model)
    dx_lo, dx_hi = model[x]
    ```

`atol` is the primal/dual optimality tolerance, and should match the tolerance
of the solver use to compute the basis.
"""
function lp_sensitivity(model::Model; atol::Float64 = 1e-8)
    if !_is_lp(model)
        error(
            "Unable to compute LP sensitivity because model is not a linear " *
            "program (or it contains interval constraints)."
        )
    elseif !has_values(model)
        error("Unable to compute LP sensitivity: no primal solution available.")
    elseif !has_duals(model)
        error("Unable to compute LP sensitivity: no dual solution available.")
    end

    prob = _standard_form_matrix(model)
    basis = _standard_form_basis(model, prob)
    B = prob.A[:, basis.basic_cols]
    @assert size(B, 1) == size(B, 2)

    n = length(prob.columns)
    is_min = objective_sense(model) == MOI.MIN_SENSE

    x = vcat(value.(all_variables(model)), value.(prob.constraints))
    x_B = @view x[basis.basic_cols]
    l_B = @view prob.lower[basis.basic_cols]
    u_B = @view prob.upper[basis.basic_cols]

    B_fact = LinearAlgebra.factorize(B)
    d = Dict{Int, Vector{Float64}}(
        # We call `collect` here because some Julia versions are missing sparse
        # matrix \ sparse vector fallbacks.
        j => B_fact \ collect(prob.A[:, j])
        for j = 1:length(basis.basic_cols) if basis.basic_cols[j] == false
    )

    report = SensitivityReport(
        Dict{ConstraintRef, Tuple{Float64, Float64}}(),
        Dict{VariableRef, Tuple{Float64, Float64}}(),
    )

    ###
    ### Compute RHS sensitivity
    ###

    # There is an easy case to consider: a constraint is basic, so we can just
    # take the distance between the value of the constraint and the
    # corresponding bound. Otherwise, we need to compute a search direction as
    # in `_compute_rhs_range`. This is just the negative of the search direction
    # computed above. Moreover, we have to be careful with doubly-bounded
    # variables, because our computed range doesn't take into account the
    # inactive bound.

    for (i, con) in enumerate(prob.constraints)
        if basis.constraints[i] == MOI.BASIC
            report.rhs[con] = _basic_range(con, constraint_object(con).set)
        else
            report.rhs[con] = _compute_rhs_range(-d[i + n], x_B, l_B, u_B, atol)
        end
    end
    for (i, con) in enumerate(prob.bounds)
        con_obj = constraint_object(con)
        if basis.bounds[i] == MOI.BASIC
            report.rhs[con] = _basic_range(con, con_obj.set)
        else
            col = prob.columns[con_obj.func]
            t_lo, t_hi = _compute_rhs_range(-d[col], x_B, l_B, u_B, atol)
            if basis.bounds[i] == MOI.NONBASIC_AT_UPPER
                t_lo = max(t_lo, prob.lower[col] - x[col])
            elseif basis.bounds[i] == MOI.NONBASIC_AT_LOWER
                t_hi = min(t_hi, prob.upper[col] - x[col])
            end
            report.rhs[con] = (t_lo, t_hi)
        end
    end

    ###
    ### Compute objective sensitivity
    ###

    π = Dict{Int, Float64}(
        i => reduced_cost(var)
        for (var, i) in prob.columns if basis.variables[i] != MOI.BASIC
    )
    for (i, c) in enumerate(prob.constraints)
        if basis.constraints[i] != MOI.BASIC
            π[n + i] = is_min ? dual(c) : -dual(c)
        end
    end

    for (var, i) in prob.columns
        if basis.variables[i] == MOI.NONBASIC_AT_LOWER && is_min
            @assert π[i] > -atol
            # We are minimizing and variable `i` is nonbasic at lower bound.
            # (δ⁻, δ⁺) = (-πᵢ, ∞) because increasing the objective coefficient
            # will only keep it at the bound.
            report.objective[var] = (-π[i], Inf)
        elseif basis.variables[i] == MOI.NONBASIC_AT_UPPER && !is_min
            @assert π[i] > -atol
            # We are maximizing and variable `i` is nonbasic at upper bound.
            # (δ⁻, δ⁺) = (-πᵢ, ∞) because increasing the objective coefficient
            # will only keep it at the bound.
            report.objective[var] = (-π[i], Inf)
        elseif basis.variables[i] != MOI.BASIC && prob.lower[i] < prob.upper[i]
            @assert π[i] < atol
            # The variable is nonbasic with nonfixed bounds. This is the
            # reverse of the above two cases because the ariable is at the
            # opposite bound
            report.objective[var] = (-Inf, -π[i])
        elseif basis.variables[i] != MOI.BASIC
            # The variable is nonbasic with fixed bounds. Therefore, (δ⁻, δ⁺) =
            # (-∞, ∞) because the variable can be effectively substituted out.
            # TODO(odow): is this correct?
            report.objective[var] = (-Inf, Inf)
        else
            # The variable `i` is basic. Given an optimal basis B, the reduced
            # costs are:
            #   c_bar = π = c_N - c_Bᵀ(B⁻¹N)
            # To maintain optimality, we want to find a δ such that (if
            # minimizing):
            #     c_N - (c_B + δeᵢ)ᵀ(B⁻¹N) ≥ 0
            #     c_N - c_BᵀB⁻¹N - δ(eᵢ)ᵀ(B⁻¹N) ≥ 0
            #     π_N ≥ δ * (eᵢ)ᵀ(B⁻¹N)
            # To do so, we can loop through every nonbasic variable `j`, and
            # compute
            #     dᵢⱼ = (eᵢ)ᵀB⁻¹aⱼ
            # Then, depending on the sign of dᵢⱼ, we can compute bounds on δ.
            @assert basis.variables[i] == MOI.BASIC
            t_lo, t_hi = -Inf, Inf
            e_i = sum(basis.basic_cols[ii] for ii = 1:i)
            for j = 1:length(basis.basic_cols)
                if basis.basic_cols[j]
                    continue  # Ignore basic components.
                elseif isapprox(prob.lower[j], prob.upper[j]; atol = atol)
                    continue  # Fixed variables can be ignored.
                elseif abs(d[j][e_i]) <= atol
                    continue  # Direction is ≈0 so any value of δ is okay.
                end
                # There are three confusing sign switch opportunities. We
                # usually want to be:
                # - minimizing (switch if maximizing)
                # - with d[j][e_i] ≥ 0 (switch if ≤ 0)
                # - and nonbasic at the lower bound (switch if upper)
                # If an odd number of these things is true, then the ratio
                # forms an upper bound for δ. Otherwise, it forms a lower bound.
                stat = j <= n ? basis.variables[j] : basis.constraints[j - n]
                if xor(is_min, d[j][e_i] > atol, stat == MOI.NONBASIC_AT_LOWER)
                    t_hi = min(t_hi, π[j] / d[j][e_i])
                else
                    t_lo = max(t_lo, π[j] / d[j][e_i])
                end
            end
            report.objective[var] = (t_lo, t_hi)
        end
    end

    return report
end

_basic_range(con, set::MOI.LessThan) = (value(con) - set.upper, Inf)
_basic_range(con, set::MOI.GreaterThan) = (-Inf, value(con) - set.lower)
_basic_range(con, set) = (0.0, 0.0)

"""
    _compute_rhs_range(d_B, x_B, l_B, u_B, atol)

Assume we start with the optimal solution `x_old`, we want to compute a step
size `t` in a direction `d` such that `x_new = x_old + t * d` is still
represented by the same optimal basis. This can be computed a la primal simplex
where we use an artificial entering variable.

    A * x_new = A * (x_old + t * d)
                = A * x_old + t * A * d
                = 0         + t * A * d  # Since A * x_old = 0
    =>  A * d = 0
    => B * d_B + N * d_N = 0
    => d_B = B \\ -(N * d_N)

Note we only have to compute the basic component of the direction vector,
because `d_N` is just zeros with a `1` in the component associated with the
artificial entering variable. Therefore, all that remains is to compute the
associated column of `N`.

If we are increasing the bounds associated with the `i`th decision variable,
then our artificial entering variable is a duplicate of the `i`th variable, and
`N * d_N = A[:, i]`.

If we are increasing the bounds associated with the `i`th affine constraint,
then our artificial entering variable is a duplicate of the slack variable
associated with the `i`th constraint, i.e., a `-1` in the `i`th row and zeros
everywhere else.

In either case:

    d_B = -(B \\ A[:, i])

Now, having computed a direction such that `x_new = x_old + t * d`. By ensuring
that `A * d = 0`, we maintained structural feasibility. Now we need to compute
bounds on `t` such that `x_new` maintains bound feasibility. That is, compute
bounds on t such that:

    l_B[j] <= x_B[j] + t * d_B[j] <= u_B[j].
"""
function _compute_rhs_range(d_B, x_B, l_B, u_B, atol)
    t_lo, t_hi = -Inf, Inf
    for j = 1:length(l_B)
        if d_B[j] > atol
            t_hi = min(t_hi, (u_B[j] - x_B[j]) / d_B[j])
            t_lo = max(t_lo, (l_B[j] - x_B[j]) / d_B[j])
        elseif d_B[j] < -atol
            t_hi = min(t_hi, (l_B[j] - x_B[j]) / d_B[j])
            t_lo = max(t_lo, (u_B[j] - x_B[j]) / d_B[j])
        else
            continue  # d_B[j] ≈ 0.0
        end
    end
    return t_lo, t_hi
end

"""
    _is_lp(model::Model)

Return `true` if `model` is a linear program.
"""
function _is_lp(model::Model)
    for (F, S) in list_of_constraint_types(model)
        # TODO(odow): support Interval constraints.
        if !(S <: Union{MOI.LessThan, MOI.GreaterThan, MOI.EqualTo})
            return false
        elseif !(F <: Union{VariableRef, GenericAffExpr})
            return false
        end
    end
    return true
end

"""
    _standard_form_matrix(model::Model)

Given a problem:

    r_l <= Ax <= r_u
    c_l <=  x <= c_u

Return the standard form:

           [A -I] [x, y] = 0
    [c_l, r_l] <= [x, y] <= [c_u, r_u]

`columns` maps the variable references to column indices.
"""
function _standard_form_matrix(model::Model)
    columns = Dict(var => i for (i, var) in enumerate(all_variables(model)))
    n = length(columns)
    c_l, c_u = fill(-Inf, n), fill(Inf, n)
    r_l, r_u = Float64[], Float64[]
    I, J, V = Int[], Int[], Float64[]
    bound_constraints = ConstraintRef[]
    affine_constraints = ConstraintRef[]
    for (F, S) in list_of_constraint_types(model)
        _fill_standard_form(
            model,
            columns,
            bound_constraints,
            affine_constraints,
            F,
            S,
            c_l,
            c_u,
            r_l,
            r_u,
            I,
            J,
            V,
        )
    end
    return (
        columns = columns,
        lower = vcat(c_l, r_l),
        upper = vcat(c_u, r_u),
        A = SparseArrays.sparse(I, J, V, length(r_l), n + length(r_l)),
        bounds = bound_constraints,
        constraints = affine_constraints,
    )
end

function _fill_standard_form(
    model::Model,
    x::Dict{VariableRef, Int},
    bound_constraints::Vector{ConstraintRef},
    ::Vector{ConstraintRef},
    F::Type{VariableRef},
    S::Type,
    c_l::Vector{Float64},
    c_u::Vector{Float64},
    ::Vector{Float64},
    ::Vector{Float64},
    ::Vector{Int},
    ::Vector{Int},
    ::Vector{Float64},
)
    for c in all_constraints(model, F, S)
        push!(bound_constraints, c)
        c_obj = constraint_object(c)
        i = x[c_obj.func]
        set = MOI.Interval(c_obj.set)
        c_l[i] = max(c_l[i], set.lower)
        c_u[i] = min(c_u[i], set.upper)
    end
    return
end

function _fill_standard_form(
    model::Model,
    x::Dict{VariableRef, Int},
    ::Vector{ConstraintRef},
    affine_constraints::Vector{ConstraintRef},
    F::Type{<:GenericAffExpr},
    S::Type,
    ::Vector{Float64},
    ::Vector{Float64},
    r_l::Vector{Float64},
    r_u::Vector{Float64},
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{Float64},
)
    for c in all_constraints(model, F, S)
        push!(affine_constraints, c)
        c_obj = constraint_object(c)
        @assert iszero(c_obj.func.constant)
        row = length(r_l) + 1
        set = MOI.Interval(c_obj.set)
        push!(r_l, set.lower)
        push!(r_u, set.upper)
        for (var, coef) in c_obj.func.terms
            push!(I, row)
            push!(J, x[var])
            push!(V, coef)
        end
        push!(I, row)
        push!(J, length(x) + row)
        push!(V, -1.0)
    end
    return
end

_convert_nonbasic_status(::MOI.LessThan) = MOI.NONBASIC_AT_UPPER
_convert_nonbasic_status(::MOI.GreaterThan) = MOI.NONBASIC_AT_LOWER
_convert_nonbasic_status(::Any) = MOI.NONBASIC

function _standard_form_basis(model::Model, prob)
    variable_status = fill(MOI.BASIC, length(prob.columns))
    bound_status = fill(MOI.BASIC, length(prob.bounds))
    constraint_status = fill(MOI.BASIC, length(prob.constraints))
    for (i, c) in enumerate(prob.bounds)
        status = MOI.get(model, MOI.ConstraintBasisStatus(), c)
        c_obj = constraint_object(c)
        if status == MOI.NONBASIC
            status = _convert_nonbasic_status(c_obj.set)
        end
        if status != MOI.BASIC
            col = prob.columns[c_obj.func]
            variable_status[col] = status
        end
        bound_status[i] = status
    end
    for (i, c) in enumerate(prob.constraints)
        status = MOI.get(model, MOI.ConstraintBasisStatus(), c)
        if status == MOI.NONBASIC
            status = _convert_nonbasic_status(constraint_object(c).set)
        end
        constraint_status[i] = status
    end
    return (
        variables = variable_status,
        bounds = bound_status,
        constraints = constraint_status,
        basic_cols = vcat(variable_status, constraint_status) .== Ref(MOI.BASIC)
    )
end
