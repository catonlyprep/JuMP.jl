#  Copyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

using Test

struct TestSensitivtySolution
    primal::Float64
    dual::Float64
    basis::Union{Nothing, MOI.BasisStatusCode}
    range::Tuple{Float64, Float64}
end

function _test_sensitivity(model_string, solution)
    m = MOIU.MockOptimizer(
        MOIU.Model{Float64}(), eval_variable_constraint_dual=false
    )
    model = direct_model(m)
    MOI.Utilities.loadfromstring!(m, model_string)
    optimize!(model)
    MOI.set(m, MOI.TerminationStatus(), MOI.OPTIMAL)
    MOI.set(m, MOI.PrimalStatus(), MOI.FEASIBLE_POINT)
    MOI.set(m, MOI.DualStatus(), MOI.FEASIBLE_POINT)
    obj_map = Dict{String, Any}()
    for (key, val) in solution
        var = variable_by_name(model, key)
        if var !== nothing
            obj_map[key] = var
            MOI.set(model, MOI.VariablePrimal(), var, val.primal)
            continue
        end
        c = constraint_by_name(model, key)
        @assert c !== nothing
        obj_map[key] = c
        MOI.set(model, MOI.ConstraintDual(), c, val.dual)
        MOI.set(model, MOI.ConstraintBasisStatus(), c, val.basis)
    end
    sens = lp_sensitivity(model)
    @testset "$(s_key)" for (s_key, val) in solution
        key = obj_map[s_key]
        @test sens[key][1] ≈ val.range[1]
        @test sens[key][2] ≈ val.range[2]
    end
end

@testset "Problem I" begin
    _test_sensitivity(
        """
        variables: x, y, z, w
        maxobjective: 1.1 * x + y
        xlb: x >= -1.0
        xub: x <= 1.0
        ylb: y >= 0.0
        zfx: z == 1.0
        c1: x + y + z + w == 1.0
        c2: x + y         <= 2.0
        """,
        Dict(
            "x" => TestSensitivtySolution(1.0, NaN, nothing, (-0.1, Inf)),
            "y" => TestSensitivtySolution(1.0, NaN, nothing, (-1, 0.1)),
            "z" => TestSensitivtySolution(1.0, NaN, nothing, (-Inf, Inf)),
            "w" => TestSensitivtySolution(-2.0, NaN, nothing, (-Inf, 1.0)),
            "xlb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 2.0)),
            "xub" => TestSensitivtySolution(NaN, -0.1, MOI.NONBASIC, (-2, 1.0)),
            "ylb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 1.0)),
            "c1" => TestSensitivtySolution(NaN, 0.0, MOI.NONBASIC, (-Inf, Inf)),
            "c2" => TestSensitivtySolution(NaN, -1.0, MOI.NONBASIC, (-1.0, Inf)),
            "zfx" => TestSensitivtySolution(NaN, 0.0, MOI.NONBASIC, (-Inf, Inf)),
        ),
    )
    _test_sensitivity(
        """
        variables: x, y, z, w
        minobjective: -1.1 * x + -1.0 * y
        xlb: x >= -1.0
        xub: x <= 1.0
        ylb: y >= 0.0
        zfx: z == 1.0
        c1: x + y + z + w == 1.0
        c2: x + y         <= 2.0
        """,
        Dict(
            "x" => TestSensitivtySolution(1.0, NaN, nothing, (-Inf, 0.1)),
            "y" => TestSensitivtySolution(1.0, NaN, nothing, (-0.1, 1)),
            "z" => TestSensitivtySolution(1.0, NaN, nothing, (-Inf, Inf)),
            "w" => TestSensitivtySolution(-2.0, NaN, nothing, (-1.0, Inf)),
            "xlb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 2.0)),
            "xub" => TestSensitivtySolution(NaN, -0.1, MOI.NONBASIC, (-2, 1.0)),
            "ylb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 1.0)),
            "c1" => TestSensitivtySolution(NaN, 0.0, MOI.NONBASIC, (-Inf, Inf)),
            "c2" => TestSensitivtySolution(NaN, -1.0, MOI.NONBASIC, (-1.0, Inf)),
            "zfx" => TestSensitivtySolution(NaN, 0.0, MOI.NONBASIC, (-Inf, Inf)),
        ),
    )
    _test_sensitivity(
        """
        variables: x, y
        maxobjective: -1.0 * x + -1.0 * y
        xlb: x >= 0.0
        ylb: y >= 0.0
        c1l: x + 2 * y >= -1.0
        c1u: x + 2 * y <= 2.0
        c2: x + y      >= 0.5
        c3: 2 * x + y  <= 2.0
        """,
        Dict(
            "x" => TestSensitivtySolution(0.5, NaN, nothing, (0, 1)),
            "y" => TestSensitivtySolution(0.0, NaN, nothing, (-Inf, 0)),
            "xlb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 0.5)),
            "ylb" => TestSensitivtySolution(NaN, 0.0, MOI.NONBASIC, (-1, 0.5)),
            "c1l" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 1.5)),
            "c1u" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-1.5, Inf)),
            "c2" => TestSensitivtySolution(NaN, 1.0, MOI.NONBASIC, (-0.5, 0.5)),
            "c3" => TestSensitivtySolution(NaN, 1.0, MOI.BASIC, (-1.0, Inf)),
        ),
    )
    _test_sensitivity(
        """
        variables: x, y
        minobjective: 1.0 * x + 1.0 * y
        xlb: x >= 0.0
        ylb: y >= 0.0
        c1l: x + 2 * y >= -1.0
        c1u: x + 2 * y <= 2.0
        c2: x + y      >= 0.5
        c3: 2 * x + y  <= 2.0
        """,
        Dict(
            "x" => TestSensitivtySolution(0.5, NaN, nothing, (-1, 0)),
            "y" => TestSensitivtySolution(0.0, NaN, nothing, (0, Inf)),
            "xlb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 0.5)),
            "ylb" => TestSensitivtySolution(NaN, 0.0, MOI.NONBASIC, (-1, 0.5)),
            "c1l" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 1.5)),
            "c1u" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-1.5, Inf)),
            "c2" => TestSensitivtySolution(NaN, 1.0, MOI.NONBASIC, (-0.5, 0.5)),
            "c3" => TestSensitivtySolution(NaN, 1.0, MOI.BASIC, (-1.0, Inf)),
        ),
    )
    _test_sensitivity(
        """
        variables: x, y
        maxobjective: 6.0 * x + 4.0 * y
        xlb: x >= 0.0
        ylb: y >= 0.0
        c1: 1 * x + 1 * y <=  6.0
        c2: 2 * x + 1 * y <=  9.0
        c3: 2 * x + 3 * y <= 16.0
        """,
        Dict(
            "x" => TestSensitivtySolution(3.0, NaN, nothing, (-2, 2)),
            "y" => TestSensitivtySolution(3.0, NaN, nothing, (-1, 2)),
            "xlb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 3)),
            "ylb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 3)),
            "c1" => TestSensitivtySolution(NaN, -2.0, MOI.NONBASIC, (-1.5, 0.25)),
            "c2" => TestSensitivtySolution(NaN, -2.0, MOI.NONBASIC, (-1, 3)),
            "c3" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-1, Inf)),
        ),
    )
    _test_sensitivity(
        """
        variables: x, y
        minobjective: -6.0 * x + -4.0 * y
        xlb: x >= 0.0
        ylb: y >= 0.0
        c1: 1 * x + 1 * y <=  6.0
        c2: 2 * x + 1 * y <=  9.0
        c3: 2 * x + 3 * y <= 16.0
        """,
        Dict(
            "x" => TestSensitivtySolution(3.0, NaN, nothing, (-2, 2)),
            "y" => TestSensitivtySolution(3.0, NaN, nothing, (-2, 1)),
            "xlb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 3)),
            "ylb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 3)),
            "c1" => TestSensitivtySolution(NaN, -2.0, MOI.NONBASIC, (-1.5, 0.25)),
            "c2" => TestSensitivtySolution(NaN, -2.0, MOI.NONBASIC, (-1, 3)),
            "c3" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-1, Inf)),
        ),
    )
    _test_sensitivity(
        """
        variables: x, y
        maxobjective: 1.0 * x + 1.0 * y
        xlb: x >= 0.0
        ylb: y >= 0.0
        c1l: x + 2 * y >= -1.0
        c1u: x + 2 * y <= 2.0
        c2: x + y      >= 0.5
        c3: 2 * x + y  <= 2.0
        """,
        Dict(
            "x" => TestSensitivtySolution(2/3, NaN, nothing, (-0.5, 1)),
            "y" => TestSensitivtySolution(2/3, NaN, nothing, (-0.5, 1)),
            "xlb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 2/3)),
            "ylb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 2/3)),
            "c1l" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 3)),
            "c1u" => TestSensitivtySolution(NaN, -1/3, MOI.NONBASIC, (-1, 2)),
            "c2" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 5/6)),
            "c3" => TestSensitivtySolution(NaN, -1/3, MOI.NONBASIC, (-1.0, 2)),
        ),
    )
    _test_sensitivity(
        """
        variables: x, y
        minobjective: -1.0 * x + -1.0 * y
        xlb: x >= 0.0
        ylb: y >= 0.0
        c1l: x + 2 * y >= -1.0
        c1u: x + 2 * y <= 2.0
        c2: x + y      >= 0.5
        c3: 2 * x + y  <= 2.0
        """,
        Dict(
            "x" => TestSensitivtySolution(2/3, NaN, nothing, (-1, 0.5)),
            "y" => TestSensitivtySolution(2/3, NaN, nothing, (-1, 0.5)),
            "xlb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 2/3)),
            "ylb" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 2/3)),
            "c1l" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 3)),
            "c1u" => TestSensitivtySolution(NaN, -1/3, MOI.NONBASIC, (-1, 2)),
            "c2" => TestSensitivtySolution(NaN, 0.0, MOI.BASIC, (-Inf, 5/6)),
            "c3" => TestSensitivtySolution(NaN, -1/3, MOI.NONBASIC, (-1.0, 2)),
        ),
    )
end
