#  Copyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

import SparseArrays

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
function lp_sensitivity(model::Model; atol::Float64 = 1e-6)
    if !_is_lp(model)
        error(
            "Unable to compute LP sensitivity because model is not a linear " *
            "program (or it contains interval constraints)."
        )
    elseif !has_values(model)
        error("Unable to compute LP sensitivity no primal solution available.")
    elseif !has_duals(model)
        error("Unable to compute LP sensitivity no dual solution available.")
    end
    columns = Dict(var => i for (i, var) in enumerate(all_variables(model)))
    n = length(columns)
    l, u, A, bound_constraints, affine_constraints = _standard_form_matrix(
        model, columns
    )
    basis_status = _standard_form_basis(
        model, columns, bound_constraints, affine_constraints
    )
    x = vcat(value.(all_variables(model)), value.(affine_constraints))
    basis = basis_status .== MOI.BASIC
    B = A[:, basis]
    @assert size(B, 1) == size(B, 2)
    is_min = objective_sense(model) == MOI.MIN_SENSE

    ###
    ### Compute RHS sensitivity
    ###
    #
    # There is an easy case to consider:
    #   1) A constraint is basic, so we can just take the distance between the
    #       value of the constraint and the corresponding bound.
    # Otherwise, we need to compute a search direction as in _search_direction
    #   and _compute_rhs_range. However, we have to be careful with doubly-
    # bounded variables, because our computed range doesn't take into account
    # the inactive bound.

    rhs_output = Dict{ConstraintRef, Tuple{Float64, Float64}}()
    for (i, con) in enumerate(affine_constraints)
        if basis_status[n + i] == MOI.BASIC
            set = constraint_object(con).set
            if set isa MOI.LessThan
                rhs_output[con] = (value(con) - set.upper, Inf)
            elseif set isa MOI.GreaterThan
                rhs_output[con] = (-Inf, value(con) - set.lower)
            else
                rhs_output[con] = (0.0, 0.0)
            end
        else
            d = _search_direction(A, B, n + i)
            rhs_output[con] = @views _compute_rhs_range(
                d, x[basis], l[basis], u[basis], atol
            )
        end
    end
    for con in bound_constraints
        i = columns[constraint_object(con).func]
        set = constraint_object(con).set
        if basis_status[i] == MOI.BASIC
            if set isa MOI.LessThan
                rhs_output[con] = (value(con) - set.upper, Inf)
            elseif set isa MOI.GreaterThan
                rhs_output[con] = (-Inf, value(con) - set.lower)
            else
                rhs_output[con] = (0.0, 0.0)
            end
        else
            d = _search_direction(A, B, i)
            t_lo, t_hi = @views _compute_rhs_range(
                d, x[basis], l[basis], u[basis], atol
            )
            if basis_status[i] == MOI.NONBASIC_AT_UPPER
                if set isa MOI.LessThan
                    t_lo = max(t_lo, l[i] - x[i])
                elseif set isa MOI.GreaterThan
                    t_hi = u[i] - l[i]
                end
            elseif basis_status[i] == MOI.NONBASIC_AT_LOWER
                if set isa MOI.LessThan
                    t_lo = l[i] - u[i]
                elseif set isa MOI.GreaterThan
                    t_hi = min(t_hi, u[i] - x[i])
                end
            end
            rhs_output[con] = (t_lo, t_hi)
        end
    end

    ###
    ### Compute objective sensitivity
    ###
    #
    # Given an optimal basis B, the reduced costs are:
    #
    #     c_bar = π = c_N - c_Bᵀ(B⁻¹N)
    #
    # Case 1) we are minimizing and variable `i` is nonbasic at lower bound, or
    #   we are maximizing and variable `i` is nonbasic at upper bound:
    #   - (δ⁻, δ⁺) = (-πᵢ, ∞) because increasing the objective coefficient will
    #     only keep it at the bound.
    # Case 2) variable is nonbasic with nonfixed bounds:
    #   - The reverse of Case 1). Variable is at the opposite bound.
    #   - (δ⁻, δ⁺) = (-∞, -πᵢ)
    # Case 3) variable is nonbasic with fixed bounds:
    #   - (δ⁻, δ⁺) = (-∞, ∞) because the variable can be effectively
    #     substituted out.
    # Case 4) variable `i` is basic.
    #   - We want to find a δ such that (if minimizing):
    #       c_N - (c_B + δeᵢ)ᵀ(B⁻¹N) ≥ 0
    #       c_N - c_BᵀB⁻¹N - δ(eᵢ)ᵀ(B⁻¹N) ≥ 0
    #       π_N - δ * (eᵢ)ᵀ(B⁻¹N) ≥ 0
    #     To do so, we can loop through every nonbasic variable `j`, and compute
    #       dᵢⱼ = (eᵢ)ᵀB⁻¹aⱼ
    #     Then, depending on the sign of dᵢⱼ, we can compute bounds on δ.
    π = vcat(
        reduced_cost.(all_variables(model)),
        (is_min ? 1.0 : -1.0) .* dual.(affine_constraints)
    )
    d = Dict{Int, Vector{Float64}}(
        j => B \ collect(A[:, j]) for j = 1:length(basis) if basis[j] == false
    )
    obj_output = Dict{VariableRef, Tuple{Float64, Float64}}()
    for (i, var) in enumerate(all_variables(model))
        if basis_status[i] == MOI.NONBASIC_AT_LOWER && is_min
            obj_output[var] = (-π[i], Inf)   # Case 1)
        elseif basis_status[i] == MOI.NONBASIC_AT_UPPER && !is_min
            obj_output[var] = (-π[i], Inf)   # Case 1)
        elseif basis_status[i] != MOI.BASIC && l[i] < u[i]
            obj_output[var] = (-Inf, -π[i])  # Case 2)
        elseif basis_status[i] != MOI.BASIC
            obj_output[var] = (-Inf, Inf)    # Case 3)
        else
            @assert basis_status[i] == MOI.BASIC  # Case 4)
            t_lo, t_hi = -Inf, Inf
            e_i = sum(basis[ii] for ii = 1:i)
            for j = 1:length(basis)
                if basis[j] || isapprox(l[j], u[j]; atol = atol) || abs(d[j][e_i]) <= atol
                    continue
                end
                in_lb = isequal(
                    d[j][e_i] > atol,
                    basis_status[j] == MOI.NONBASIC_AT_UPPER
                )
                if is_min == in_lb
                    t_lo = max(t_lo, π[j] / d[j][e_i])
                else
                    t_hi = min(t_hi, π[j] / d[j][e_i])
                end
            end
            obj_output[var] = (t_lo, t_hi)
        end
    end

    return SensitivityReport(rhs_output, obj_output)
end

"""
    _search_direction(A, B, i, is_variable)

Compute a search direction based on the bound that is changing.

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

If `is_variable`, then we are increasing the bounds associated with the `i`th
variable. Therefore, our artificial entering variable is a duplicate of the
`i`th variable, and `N * d_N = A[:, i]`.

If `!is_variable`, then we are increasing the bounds associated with the `i`th
affine constraint. Therefore, our artificial entering variable is a duplicate of
the slack variable associated with the `i`th constraint, i.e., a `-1` in the
`i`th row and zeros everywhere else.
"""
_search_direction(A, B, i) = -(B \ collect(A[:, i]))

"""
    _compute_rhs_range(d_B, x_B, l_B, u_B)

In `_search_direction`, we computed a direction such that
`x_new = x_old + t * d`. By ensuring that `A * d = 0`, we maintained structural
feasibility. Now we need to compute bounds on `t` such that `x_new` maintains
bound feasibility. That is, compute bounds on t such that:

    l_B[j] <= x_B[j] + t * d_B[j] <= u_B[j].

See also: `_search_direction`.
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
            continue
        end
    end
    return t_lo, t_hi
end

"""
    _is_lp(model::Model)

Return `true` if `model` is a linear program.

TODO: support Interval constraints.
"""
function _is_lp(model::Model)
    for (F, S) in list_of_constraint_types(model)
        if !(S <: Union{MOI.LessThan, MOI.GreaterThan, MOI.EqualTo})
            return false
        elseif !(F <: Union{VariableRef, GenericAffExpr})
            return false
        end
    end
    return true
end

"""
    _standard_form_matrix(model)

Given a problem

    r_l <= Ax <= r_u
    c_l <=  x <= c_u

Return the standard form:

           [A -I] [x, y] = 0
    [c_l, r_l] <= [x, y] <= [c_u, r_u]
"""
function _standard_form_matrix(model::Model, columns::Dict{VariableRef, Int})
    n = length(columns)
    # Initialize storage
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
    A = SparseArrays.sparse(I, J, V, length(r_l), n + length(r_l))
    return vcat(c_l, r_l), vcat(c_u, r_u), A, bound_constraints, affine_constraints
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
        if !iszero(c_obj.func.constant)
            error("Constraint constant not zero.")
        end
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
end

function _standard_form_basis(
    model::Model,
    x::Dict{VariableRef, Int},
    bound_constraints::Vector{ConstraintRef},
    affine_constraints::Vector{ConstraintRef},
)
    variable_basis_status = fill(MOI.BASIC, length(x))
    for c in bound_constraints
        c_obj = constraint_object(c)
        status = MOI.get(model, MOI.ConstraintBasisStatus(), c)
        S = typeof(constraint_object(c).set)
        if S <: MOI.LessThan && status == MOI.NONBASIC
            status = MOI.NONBASIC_AT_UPPER
        elseif S <: MOI.GreaterThan && status == MOI.NONBASIC
            status = MOI.NONBASIC_AT_LOWER
        end
        variable_basis_status[x[c_obj.func]] = status
    end
    affine_basis_status = fill(MOI.BASIC, length(affine_constraints))
    for (i, c) in enumerate(affine_constraints)
        status = MOI.get(model, MOI.ConstraintBasisStatus(), c)
        S = typeof(constraint_object(c).set)
        if S <: MOI.LessThan && status == MOI.NONBASIC
            status = MOI.NONBASIC_AT_UPPER
        elseif S <: MOI.GreaterThan && status == MOI.NONBASIC
            status = MOI.NONBASIC_AT_LOWER
        end
        affine_basis_status[i] = status
    end
    return vcat(variable_basis_status, affine_basis_status)
end
