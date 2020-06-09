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
    columns = Dict(var => i for (i, var) in enumerate(all_variables(model)))
    n = length(columns)
    l, u, A, bound_constraints, affine_constraints = _standard_form_matrix(
        model, columns
    )
    variable_status, bound_status, affine_status = _standard_form_basis(
        model, columns, bound_constraints, affine_constraints
    )
    x = vcat(value.(all_variables(model)), value.(affine_constraints))
    basis = vcat(variable_status, affine_status) .== Ref(MOI.BASIC)
    B = A[:, basis]
    @assert size(B, 1) == size(B, 2)
    is_min = objective_sense(model) == MOI.MIN_SENSE

    ###
    ### Compute RHS sensitivity
    ###

    # There is an easy case to consider: a constraint is basic, so we can just
    # take the distance between the value of the constraint and the
    # corresponding bound. Otherwise, we need to compute a search direction as
    # in `_compute_rhs_range`. However, we have to be careful with
    # doubly-bounded variables, because our computed range doesn't take into
    # account the inactive bound.

    rhs_output = Dict{ConstraintRef, Tuple{Float64, Float64}}()
    for (i, con) in enumerate(affine_constraints)
        if affine_status[i] == MOI.BASIC
            set = constraint_object(con).set
            rhs_output[con] = _basic_range(con, set)
        else
            rhs_output[con] = @views _compute_rhs_range(
                A, B, i + n, x[basis], l[basis], u[basis], atol
            )
        end
    end
    for (i, con) in enumerate(bound_constraints)
        set = constraint_object(con).set
        if bound_status[i] == MOI.BASIC
            rhs_output[con] = _basic_range(con, set)
        else
            col = columns[constraint_object(con).func]
            t_lo, t_hi = @views _compute_rhs_range(
                A, B, col, x[basis], l[basis], u[basis], atol
            )
            if bound_status[i] == MOI.NONBASIC_AT_UPPER
                t_lo = max(t_lo, l[col] - x[col])
            elseif bound_status[i] == MOI.NONBASIC_AT_LOWER
                t_hi = min(t_hi, u[col] - x[col])
            end
            rhs_output[con] = (t_lo, t_hi)
        end
    end

    ###
    ### Compute objective sensitivity
    ###

    π = vcat(
        reduced_cost.(all_variables(model)),
        (is_min ? 1.0 : -1.0) .* dual.(affine_constraints)
    )
    d = Dict{Int, Vector{Float64}}(
        j => B \ collect(A[:, j]) for j = 1:length(basis) if basis[j] == false
    )
    obj_output = Dict{VariableRef, Tuple{Float64, Float64}}()
    for (i, var) in enumerate(all_variables(model))
        if variable_status[i] == MOI.NONBASIC_AT_LOWER && is_min
            @assert π[i] > -atol
            # We are minimizing and variable `i` is nonbasic at lower bound.
            # (δ⁻, δ⁺) = (-πᵢ, ∞) because increasing the objective coefficient
            # will only keep it at the bound.
            obj_output[var] = (-π[i], Inf)
        elseif variable_status[i] == MOI.NONBASIC_AT_UPPER && !is_min
            @assert π[i] > -atol
            # We are maximizing and variable `i` is nonbasic at upper bound.
            # (δ⁻, δ⁺) = (-πᵢ, ∞) because increasing the objective coefficient
            # will only keep it at the bound.
            obj_output[var] = (-π[i], Inf)
        elseif variable_status[i] != MOI.BASIC && l[i] < u[i]
            @assert π[i] < atol
            # The variable is nonbasic with nonfixed bounds. This is the
            # reverse of the above two cases because the ariable is at the
            # opposite bound
            obj_output[var] = (-Inf, -π[i])
        elseif variable_status[i] != MOI.BASIC
            # The variable is nonbasic with fixed bounds. Therefore, (δ⁻, δ⁺) =
            # (-∞, ∞) because the variable can be effectively substituted out.
            # TODO(odow): is this correct?
            obj_output[var] = (-Inf, Inf)
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
            @assert variable_status[i] == MOI.BASIC
            t_lo, t_hi = -Inf, Inf
            e_i = sum(basis[ii] for ii = 1:i)
            for j = 1:length(basis)
                if basis[j]
                    continue  # Ignore basic components.
                elseif isapprox(l[j], u[j]; atol = atol)
                    continue  # Fixed variables can be ignored.
                elseif abs(d[j][e_i]) <= atol
                    continue  # Direction is ≈0 so any value of δ is okay.
                end
                # There are three confusing sign switch opportunities. We
                # usually want to be:
                # - minimizing (switch if maximizing)
                # - with d[j][e_i] ≥ 0 (switch if ≤ 0)
                # - and nonbasic at the lower bound (switch if upper)
                # If an odd number of these switches is true, then the ratio
                # forms an upper bound for δ. Otherwise, it forms a lower bound.
                st = j <= n ? variable_status[j] : affine_status[j - n]
                if isodd(
                    is_min + (d[j][e_i] > atol) + (st == MOI.NONBASIC_AT_LOWER)
                )
                    t_hi = min(t_hi, π[j] / d[j][e_i])
                else
                    t_lo = max(t_lo, π[j] / d[j][e_i])
                end
            end
            obj_output[var] = (t_lo, t_hi)
        end
    end

    return SensitivityReport(rhs_output, obj_output)
end

_basic_range(con, set::MOI.LessThan) = (value(con) - set.upper, Inf)
_basic_range(con, set::MOI.GreaterThan) = (-Inf, value(con) - set.lower)
_basic_range(con, set) = (0.0, 0.0)

"""
    _compute_rhs_range(A, B, i, x_B, l_B, u_B, atol)

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

In either case:

    d_B = -(B \\ A[:, i])

Now, having computed a direction such that `x_new = x_old + t * d`. By ensuring
that `A * d = 0`, we maintained structural feasibility. Now we need to compute
bounds on `t` such that `x_new` maintains bound feasibility. That is, compute
bounds on t such that:

    l_B[j] <= x_B[j] + t * d_B[j] <= u_B[j].
"""
function _compute_rhs_range(A, B, i, x_B, l_B, u_B, atol)
    d_B = -(B \ collect(A[:, i]))  # We call `collect` because `A` is sparse.
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
    _standard_form_matrix(model, columns::Dict{VariableRef, Int})

Given a problem:

    r_l <= Ax <= r_u
    c_l <=  x <= c_u

Return the standard form:

           [A -I] [x, y] = 0
    [c_l, r_l] <= [x, y] <= [c_u, r_u]

`columns` maps the variable references to column indices.
"""
function _standard_form_matrix(model::Model, columns::Dict{VariableRef, Int})
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
    return
end

_convert_nonbasic_status(::MOI.LessThan) = MOI.NONBASIC_AT_UPPER
_convert_nonbasic_status(::MOI.GreaterThan) = MOI.NONBASIC_AT_LOWER
_convert_nonbasic_status(::Any) = MOI.NONBASIC

function _standard_form_basis(
    model::Model,
    x::Dict{VariableRef, Int},
    bound_constraints::Vector{ConstraintRef},
    affine_constraints::Vector{ConstraintRef},
)
    variable_basis_status = fill(MOI.BASIC, length(x))
    bound_basis_status = fill(MOI.BASIC, length(bound_constraints))
    for (i, c) in enumerate(bound_constraints)
        c_obj = constraint_object(c)
        status = MOI.get(model, MOI.ConstraintBasisStatus(), c)
        if status == MOI.NONBASIC
            status = _convert_nonbasic_status(constraint_object(c).set)
        end
        if status != MOI.BASIC
            variable_basis_status[x[c_obj.func]] = status
        end
        bound_basis_status[i] = status
    end
    affine_basis_status = fill(MOI.BASIC, length(affine_constraints))
    for (i, c) in enumerate(affine_constraints)
        status = MOI.get(model, MOI.ConstraintBasisStatus(), c)
        if status == MOI.NONBASIC
            status = _convert_nonbasic_status(constraint_object(c).set)
        end
        affine_basis_status[i] = status
    end
    return variable_basis_status, bound_basis_status, affine_basis_status
end
