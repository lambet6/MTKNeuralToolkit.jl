# =============================================================================
# iv_curve.jl -- voltage-clamp I-V curves for a single MTKNeuralToolkit channel
# =============================================================================
#
# THE IDEA
# --------
# A normal simulation builds a compartment: a capacitor in parallel with some
# channels. The capacitor equation `D(v) ~ i/C` makes v a state variable, so it
# evolves and you watch it spike.
#
# An I-V curve wants the opposite. We want to *impose* the voltage and *measure*
# the current. So we delete the capacitor and put an ideal voltage source in its
# place: `FixedReversal`, which is nothing more than `v ~ E`. Voltage is now
# imposed algebraically, and the only states left are the channel's own gates.
#
#        +--- clamp.p ---+--- channel.p ---+
#        |                                 |
#     [ v ~ E ]                   [ i ~ g*m^p*(v - E_rev) ]
#        |                                 |
#        +--- clamp.n ---+--- channel.n ---+--- gnd (v = 0)
#
# THE PROTOCOL
# ------------
# For each test voltage we do what an electrophysiologist does:
#   1. Hold at `V_hold` until the gates reach steady state (done once).
#   2. Step to `V_test`, record the current for `t_step`.
#
# From each step we take three numbers, because "the" I-V curve is really three
# curves and people rarely say which one they mean:
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
# SIGN CONVENTION
# ---------------
# `channel.i` flows p -> n. Here p is the "inside" node and n is ground, so
# positive i = outward current, matching the usual physiology convention. The
# curve crosses zero at the channel's reversal potential -- that zero crossing
# is the cheapest check that everything is wired up correctly.
#
# SCOPE
# -----
# Works for any channel that is *just* a OnePort: `GenericChannel`, or anything
# built with the `extend(..., oneport)` pattern from the README. It does NOT
# work for `CaVChannel` / `KCaChannel`, which expose a `ca_port` and need a
# `CalciumPool` wired in to close their feedback loop. Scalar topology only.
#
# USAGE
# -----
# Including this file defines the module and nothing else -- it deliberately
# runs no top-level code, so you need to bring the package into scope yourself:
#
#   include("scripts/iv_curve.jl")
#   using .IVCurves
#   using MTKNeuralToolkit
#   using ModelingToolkit: @named
#
#   @named na = HodgkinHuxley.SodiumChannel()
#   res = iv_curve(na)
#   plot_iv(res)
#
# Or, needing nothing else in scope:
#
#   include("scripts/iv_curve.jl")
#   IVCurves.demo()   # sodium + leak, end to end
# =============================================================================

module IVCurves

using MTKNeuralToolkit
using MTKNeuralToolkit: FixedReversal, SymbolicT 
using ModelingToolkit: System, Equation, mtkcompile, connect,
                       unknowns, @named, t_nounits as t
using OrdinaryDiffEq
using Plots

export iv_curve, plot_iv

# The half-millivolt offset is deliberate. Standard HH rate functions have
# removable singularities -- alpha_m is 0/0 at exactly v = -40, alpha_n at
# v = -55 -- which Julia evaluates to NaN. A grid on round numbers lands on
# them; this one steps over.
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

    # --- 1. Build the clamp circuit, once ------------------------------------
    @named clamp = FixedReversal(E = V_hold)
    @named gnd   = Ground()

    eqs = Equation[
        connect(clamp.p, channel.p),
        connect(clamp.n, channel.n, gnd.g),
    ]

    @named circuit = System(eqs, t, SymbolicT[], SymbolicT[];
                            systems = [clamp, channel, gnd])
    sys = mtkcompile(circuit)

    # Symbolic handles: the parameter we sweep, and the current we measure.
    E_clamp = sys.clamp.E
    I_ch    = getproperty(sys, nameof(channel)).i
    states  = unknowns(sys) 

    # --- 2. Hold phase -------------------------------------------------------
    # Let the gates settle at V_hold. Identical for every test voltage, so we
    # only do it once and reuse the settled state as the initial condition.
    prob_hold = ODEProblem(sys, [E_clamp => V_hold], (0.0, t_hold))
    sol_hold  = solve(prob_hold, solver; reltol = reltol, abstol = abstol)
    sol_hold.retcode == ReturnCode.Success || error(
        "Holding phase failed to solve ($(sol_hold.retcode)). " *
        "Check the channel's gating functions at V = $V_hold mV.")
    u_settled = sol_hold.u[end]

    # --- 3. Step phase: one clamp step per test voltage ----------------------
    n = length(voltages)
    I_inst = zeros(n)
    I_peak = zeros(n)
    I_ss   = zeros(n)
    I_late = zeros(n)                       # used for the settling check below
    times  = Vector{Vector{Float64}}(undef, n)
    traces = Vector{Vector{Float64}}(undef, n)

    for (k, V) in enumerate(voltages)
        prob = ODEProblem(sys, [states .=> u_settled; E_clamp => V],
                          (0.0, t_step))
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
        I_inst[k] = i[1]                     # gates still at their hold values
        I_peak[k] = i[argmax(abs.(i))]      
        I_ss[k]   = i[end]
        I_late[k] = sol(0.9 * t_step; idxs = I_ch)
    end

    # --- 4. Did the steps actually reach steady state? -----------------------
    # Tolerance is scaled by the largest current in the sweep rather than each
    # point's own value, or points where I_ss is near zero (sodium's window
    # current) would warn spuriously.
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
               ylabel = "Current (outward positive)",
               title  = "I-V: $(res.channel_name)",
               legend = :topleft)
    plot!(plt, res.V, res.I_peak; lw = 2, ls = :dash, label = "peak")
    plot!(plt, res.V, res.I_inst; lw = 2, ls = :dot,  label = "instantaneous")
    hline!(plt, [0.0]; c = :black, lw = 1, label = false)
    return plt
end

# Demo. Expected results, useful as a sanity check:
#   leak:   all three curves the same straight line, crossing zero at -54.4 mV
#   sodium: instantaneous curve straight through +50 mV; peak curve the textbook
#           inverted-N, deeply inward between about -40 and +50 mV; steady state
#           nearly flat, because h inactivates almost completely
#

const HH = HodgkinHuxley

function demo()
    @named na   = HH.SodiumChannel()
    @named k    = HH.PotassiumChannel()
    @named leak = HH.LeakChannel()

    na_res   = iv_curve(na, V_hold=-40.5)
    k_res   = iv_curve(k)
    leak_res = iv_curve(leak)

    return plot(plot_iv(na_res), plot_iv(k_res), plot_iv(leak_res);
                layout = (1, 3), size = (1000, 420))
end

end # module IVCurves