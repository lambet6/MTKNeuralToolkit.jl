
#        +--- clamp.p ---+--- channel.p ---+
#        |                                 |
#     [ v ~ E ]                   [ i ~ g*m^p*(v - E_rev) ]
#        |                                 |
#        +--- clamp.n ---+--- channel.n ---+--- gnd (v = 0)
#
# 
#   1. Hold at `V_hold` until the gates reach steady state (done once).
#   2. Step to `V_test`, record the current for `t_step`.
#
#   I_inst   current in the first instant after the step, before the gates have
#            moved. This is the open-channel (instantaneous) relation; an ohmic
#            channel gives a straight line here.
#   I_peak   the largest excursion during the step. This is the textbook curve
#            for a transient current like Na+, where activation is fast and
#            inactivation eats the current before steady state.
#   I_ss     current at the end of the step: the steady-state curve, gates fully
#            relaxed to their voltage-dependent equilibrium.
#

module IVCurves

using MTKNeuralToolkit
using MTKNeuralToolkit: FixedReversal, SymbolicT 
using ModelingToolkit: System, Equation, mtkcompile, connect,
                       unknowns, @named, t_nounits as t
using OrdinaryDiffEq
using Plots
using Plots.PlotMeasures: px

export iv_curve, plot_iv

const DEFAULT_VOLTAGES = -99.5:1.0:60.5

"""
    iv_curve(channel; voltages, V_hold, t_hold, t_step, solver, reltol, abstol)

Run a voltage-clamp protocol on a single ion `channel` and return its I-V data.

`channel` is any already-named MTK system built on a OnePort, e.g.
`@named na = GenericChannel(g=120.0, E_rev=50.0, gates=sodium_gates)`.

Returns a NamedTuple with fields `V`, `I_inst`, `I_peak`, `I_ss`, the raw
`times` and `traces` per step, and the compiled system `sys` (useful for
plotting gating variables afterwards).
"""
function iv_curve(channel;
                  voltages = DEFAULT_VOLTAGES,
                  V_hold   = -80.0,
                  t_hold   = 500.0,
                  t_step   = 100.0,
                  solver   = Rosenbrock23(),
                  reltol   = 1e-8,
                  abstol   = 1e-8)

    voltages = collect(float.(voltages))

    # 1. Build the clamp circuit, once 
    @named clamp = FixedReversal(E = V_hold)
    @named gnd   = Ground()

    eqs = Equation[
        connect(clamp.p, channel.p),
        connect(clamp.n, channel.n, gnd.g),
    ]

    @named circuit = System(eqs, t, SymbolicT[], SymbolicT[];
                            systems = [clamp, channel, gnd])
    sys = mtkcompile(circuit)

    # Symbolic handles: the parameter swept, and the current measured.
    E_clamp = sys.clamp.E
    I_ch    = getproperty(sys, nameof(channel)).i
    states  = unknowns(sys) 

    # 2. Hold phase 
    # Let the gates settle at V_hold. Identical for every test voltage, so
    # only do it once and reuse the settled state as the initial condition.
    prob_hold = ODEProblem(sys, [E_clamp => V_hold], (0.0, t_hold))
    sol_hold  = solve(prob_hold, solver; reltol = reltol, abstol = abstol)
    sol_hold.retcode == ReturnCode.Success || error(
        "Holding phase failed to solve ($(sol_hold.retcode)). " *
        "Check the channel's gating functions at V = $V_hold mV.")
    u_settled = sol_hold.u[end]

   # 3. Step phase: one clamp step per test voltage 
    n = length(voltages)
    I_inst = zeros(n)
    I_peak = zeros(n)
    I_ss   = zeros(n)
    I_late = zeros(n)                       
    times  = Vector{Vector{Float64}}(undef, n)
    traces = Vector{Vector{Float64}}(undef, n)

    # Define a base problem 
    # Use voltages[1] as a placeholder for the parameter, 
    # and lock in u_settled as the starting state for all steps.
    base_prob = ODEProblem(sys, [states .=> u_settled; E_clamp => voltages[1]],
                           (0.0, t_step))

    for (k, V) in enumerate(voltages)
        # remake creates a lightweight copy with only the specified parameter changed.
        prob = remake(base_prob, p = [E_clamp => V])
        sol = solve(prob, solver; reltol = reltol, abstol = abstol)
        sol.retcode == ReturnCode.Success ||
            error("Step to $V mV failed to solve ($(sol.retcode)).")

        i = sol[I_ch]

        all(isfinite, i) || error("""
            Got a non-finite current at V = $V mV. The usual cause is a \
            removable singularity in one of the channel's rate functions being \
            hit exactly (classic HH alpha_m is 0/0 at v = -40, alpha_n at \
            v = -55). Nudge the voltage grid off that point.
            """)

        times[k]  = sol.t
        traces[k] = i
        I_inst[k] = i[1]                    
        I_peak[k] = i[argmax(abs.(i))]      
        I_ss[k]   = i[end]
        I_late[k] = sol(0.9 * t_step; idxs = I_ch)
    end

    # 4. Did the steps actually reach steady state? 
    scale = maximum(abs, I_ss)
    if scale > 0
        drifting = findall(k -> abs(I_ss[k] - I_late[k]) > 0.01 * scale,
                           eachindex(I_ss))
        isempty(drifting) || @warn(
            "Current was still drifting at the end of the step, so I_ss is " *
            "not a true steady state at these voltages. Increase t_step.",
            voltages = voltages[drifting], t_step)
    end

    return (V = voltages, I_inst = I_inst, I_peak = I_peak, I_ss = I_ss,
            times = times, traces = traces, sys = sys,
            channel_name = nameof(channel), V_hold = V_hold)
end

"""
    plot_iv(res)

Plot the three I-V curves returned by [`iv_curve`](@ref).
"""
function plot_iv(res)
    plt = plot(res.V, res.I_ss;
               lw = 2, label = "steady state",
               xlabel = "Clamp voltage (mV)",
               ylabel = "Current (outward positive, µA)",
               title  = "I-V: $(res.channel_name)",
               legend = :topleft)
    plot!(plt, res.V, res.I_peak; lw = 2, ls = :dash, label = "peak")
    plot!(plt, res.V, res.I_inst; lw = 2, ls = :dot,  label = "instantaneous")
    hline!(plt, [0.0]; c = :black, lw = 1, label = false)
    return plt
end


const HH = HodgkinHuxley

function demo()
    @named na   = HH.SodiumChannel()
    @named k    = HH.PotassiumChannel()
    @named leak = HH.LeakChannel()

    na_res   = iv_curve(na, V_hold=-40.5)
    k_res   = iv_curve(k)
    leak_res = iv_curve(leak)

    return plot(plot_iv(na_res), plot_iv(k_res), plot_iv(leak_res);
                layout = (1, 3), size = (1000, 420), margin=20px)
end

end