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
using MTKNeuralToolkit: FixedReversal, FixedCalcium, SymbolicT
using ModelingToolkit: System, Equation, mtkcompile, connect,
                       unknowns, @named, t_nounits as t
using OrdinaryDiffEq
using Plots
using Plots.PlotMeasures: px

export iv_curve, plot_iv, plot_iv_group

const DEFAULT_VOLTAGES = -99.5:1.0:60.5

"""
    iv_curve(channel; voltages, V_hold, t_hold, t_step, Ca_hold, solver, reltol, abstol)

Run a voltage-clamp protocol on a single ion `channel` and return its I-V data.

`channel` is any already-named MTK system built on a OnePort, e.g.
`@named na = GenericChannel(g=120.0, E_rev=50.0, gates=sodium_gates)`.

If `channel` also exposes a `ca_port` (i.e. it's a `CaVChannel` or
`KCaChannel`), intracellular Ca2+ is clamped to the fixed concentration
`Ca_hold` for the whole protocol via `FixedCalcium` -- the same idea as the
electrical `V_hold`/`voltages` clamp, but for the chemical domain. This
mirrors buffering intracellular Ca2+ with a chelator (EGTA/BAPTA) in a real
patch pipette so the `CaVChannel`'s Nernst potential and any Ca-dependent
gating (`KCaChannel`) stay well-defined and don't drift during the step.

Returns a NamedTuple with fields `V`, `I_inst`, `I_peak`, `I_ss`, the raw
`times` and `traces` per step, and the compiled system `sys` (useful for
plotting gating variables afterwards).

This is a thin wrapper: it runs the shared group protocol (see
[`iv_curve(::AbstractVector)`](@ref)) on a one-element `[channel]` vector,
then unwraps the result back down to scalar-style fields and drops the
resulting "neuron" total column, which for a single channel would just
duplicate the channel's own current.
"""
function iv_curve(channel; kwargs...)
    res = _iv_curve_impl([channel]; kwargs...)
    return (V = res.V,
            I_inst = res.I_inst[:, 1],
            I_peak = res.I_peak[:, 1],
            I_ss   = res.I_ss[:, 1],
            times  = res.times,
            traces = [tr[:, 1] for tr in res.traces],
            sys    = res.sys,
            channel_name = nameof(channel),
            V_hold = res.V_hold)
end

"""
    iv_curve(channels::AbstractVector; voltages, V_hold, t_hold, t_step, Ca_hold, solver, reltol, abstol)

Run a voltage-clamp protocol on a whole "neuron" made of several ion `channels`
wired in parallel (all `p` pins tied together to the clamp, all `n` pins tied
to ground) -- i.e. the same circuit as [`iv_curve`](@ref) but with a
`FixedReversal` clamp doing the job of the capacitor, so there is no membrane
capacitance to charge.

Any `channels` that expose a `ca_port` (`CaVChannel`, `KCaChannel`) share a
single `FixedCalcium` clamp holding intracellular Ca2+ at `Ca_hold` for the
whole protocol -- see the single-channel [`iv_curve`](@ref) docstring for why.

Returns a NamedTuple with the same fields as the single-channel [`iv_curve`](@ref),
except `I_inst`, `I_peak` and `I_ss` are `length(voltages) x (length(channels)+1)`
matrices: one column per channel (in `channels` order), plus a final "neuron"
column holding the total clamp current (`-sys.clamp.i`, i.e. the sum of all
channel currents). `channel_names` lists the corresponding column names.
"""
iv_curve(channels::AbstractVector; kwargs...) = _iv_curve_impl(channels; kwargs...)

# Shared simulation core for both `iv_curve` methods above. Always takes a
# vector of channels (even a single one, via the scalar `iv_curve` wrapper
# above) and always returns the matrix-with-"neuron"-column shape. Not
# exported -- call `iv_curve` instead.
function _iv_curve_impl(channels::AbstractVector;
                  voltages = DEFAULT_VOLTAGES,
                  V_hold   = -80.0,
                  t_hold   = 500.0,
                  t_step   = 100.0,
                  Ca_hold  = 0.05,
                  solver   = Rosenbrock23(),
                  reltol   = 1e-8,
                  abstol   = 1e-8)

    voltages = collect(float.(voltages))

    # 1. Build the clamp circuit: channels wired in parallel, no capacitor.
    @named clamp = FixedReversal(E = V_hold)
    @named gnd   = Ground()

    eqs = Equation[
        connect(clamp.p, (ch.p for ch in channels)...),
        connect(clamp.n, gnd.g, (ch.n for ch in channels)...),
    ]
    systems = System[clamp; channels; gnd]

    # Any channels with a ca_port (CaVChannel, KCaChannel) share one fixed
    # Ca2+ clamp -- same reasoning as the single-channel method above.
    ca_channels = filter(ch -> hasproperty(ch, :ca_port), channels)
    if !isempty(ca_channels)
        @named ca_clamp = FixedCalcium(Ca = Ca_hold)
        push!(eqs, connect(ca_clamp.port, (ch.ca_port for ch in ca_channels)...))
        push!(systems, ca_clamp)
    end

    @named circuit = System(eqs, t, SymbolicT[], SymbolicT[]; systems = systems)
    sys = mtkcompile(circuit)

    # Symbolic handles: the parameter swept, and the currents measured.
    E_clamp       = sys.clamp.E
    channel_names = nameof.(channels)
    I_chs         = [getproperty(sys, name).i for name in channel_names]
    I_total       = -sys.clamp.i               # sum of all channel currents
    all_I         = [I_chs; I_total]
    col_names     = [channel_names; :neuron]
    states        = unknowns(sys)

    # 2. Hold phase (shared initial condition for every test voltage)
    prob_hold = ODEProblem(sys, [E_clamp => V_hold], (0.0, t_hold))
    sol_hold  = solve(prob_hold, solver; reltol = reltol, abstol = abstol)
    sol_hold.retcode == ReturnCode.Success || error(
        "Holding phase failed to solve ($(sol_hold.retcode)). " *
        "Check the channels' gating functions at V = $V_hold mV.")
    u_settled = sol_hold.u[end]

    # 3. Step phase: one clamp step per test voltage
    n    = length(voltages)
    n_ch = length(all_I)
    I_inst = zeros(n, n_ch)
    I_peak = zeros(n, n_ch)
    I_ss   = zeros(n, n_ch)
    I_late = zeros(n, n_ch)
    times  = Vector{Vector{Float64}}(undef, n)
    traces = Vector{Matrix{Float64}}(undef, n)

    base_prob = ODEProblem(sys, [states .=> u_settled; E_clamp => voltages[1]],
                           (0.0, t_step))

    for (k, V) in enumerate(voltages)
        prob = remake(base_prob, p = [E_clamp => V])
        sol = solve(prob, solver; reltol = reltol, abstol = abstol)
        sol.retcode == ReturnCode.Success ||
            error("Step to $V mV failed to solve ($(sol.retcode)).")

        times[k]  = sol.t
        trace_k   = Matrix{Float64}(undef, length(sol.t), n_ch)

        for (c, I_ch) in enumerate(all_I)
            i = sol[I_ch]

            all(isfinite, i) || error("""
                Got a non-finite current ($(col_names[c])) at V = $V mV. The \
                usual cause is a removable singularity in one of the \
                channel's rate functions being hit exactly (classic HH \
                alpha_m is 0/0 at v = -40, alpha_n at v = -55). Nudge the \
                voltage grid off that point.
                """)

            trace_k[:, c]   = i
            I_inst[k, c]    = i[1]
            I_peak[k, c]    = i[argmax(abs.(i))]
            I_ss[k, c]      = i[end]
            I_late[k, c]    = sol(0.9 * t_step; idxs = I_ch)
        end
        traces[k] = trace_k
    end

    # 4. Did the steps actually reach steady state?
    scale = maximum(abs, I_ss)
    if scale > 0
        drifting = findall(k -> maximum(abs.(I_ss[k, :] .- I_late[k, :])) > 0.01 * scale,
                           1:n)
        isempty(drifting) || @warn(
            "Current was still drifting at the end of the step, so I_ss is " *
            "not a true steady state at these voltages. Increase t_step.",
            voltages = voltages[drifting], t_step)
    end

    return (V = voltages, I_inst = I_inst, I_peak = I_peak, I_ss = I_ss,
            times = times, traces = traces, sys = sys,
            channel_names = col_names, V_hold = V_hold)
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

"""
    plot_iv_group(res; which = :I_ss)

Plot the per-channel I-V curves returned by the group [`iv_curve`](@ref)
(`which` is one of `:I_inst`, `:I_peak`, `:I_ss`), overlaid with the total
"neuron" current in bold black.
"""
function plot_iv_group(res; which = :I_ss)
    data = getproperty(res, which)
    plt = plot(xlabel = "Clamp voltage (mV)",
               ylabel = "Current (outward positive, µA)",
               title  = "I-V by channel ($(which))",
               legend = :topleft)
    for (c, name) in enumerate(res.channel_names)
        name === :neuron && continue
        plot!(plt, res.V, data[:, c]; lw = 2, label = String(name))
    end
    neuron_col = findfirst(==(:neuron), res.channel_names)
    plot!(plt, res.V, data[:, neuron_col];
          lw = 3, c = :black, label = "neuron (total)")
    hline!(plt, [0.0]; c = :black, lw = 1, ls = :dot, label = false)
    return plt
end




const HH = HodgkinHuxley

function demo()
    @named na   = HH.SodiumChannel()
    @named k    = HH.PotassiumChannel()
    @named leak = HH.LeakChannel()
    V_hold = -40.5
    na_res   = iv_curve(na, V_hold=V_hold)
    k_res   = iv_curve(k, V_hold=V_hold)
    leak_res = iv_curve(leak, V_hold=V_hold)

    return plot(plot_iv(na_res), plot_iv(k_res), plot_iv(leak_res);
                layout = (1, 3), size = (1000, 420), margin=20px)
end

function demo_calcium()
    CaV_m_inf(v) = 1.0 ./ (1.0 .+ exp.(-(v .+ 20.0) ./ 5.0))
    CaV_tau_m(v) = 5.0 .+ 10.0 ./ (1.0 .+ exp.((v .+ 20.0) ./ 10.0))
    cav_gates = [GateSpec(:mCaV, 3, 0.0, InfTau(CaV_m_inf, CaV_tau_m))]

    KCa_m_inf(v, ca) = (ca ./ (ca .+ 3.0)) ./ (1.0 .+ exp.(-(v .+ 20.0) ./ 5.0))
    KCa_tau_m(v) = 20.0
    kca_gates = [GateSpec(:mKCa, 4, 0.0, InfTauCa(KCa_m_inf, KCa_tau_m))]

    # CaVChannel derives E_rev from the Nernst equation using ca_port.Ca, and
    # KCaChannel's gate itself depends on ca_port.Ca -- iv_curve clamps that
    # to Ca_hold for both, since neither is wired to a CalciumPool here.
    @named cav = CaVChannel(g = 2.0, gates = cav_gates, Ca_out = 3000.0,
                            nernst_factor = 13.0, conversion_factor = 0.047)
    @named kca = KCaChannel(g = 5.0, E_rev = -80.0, gates = kca_gates)

    cav_res = iv_curve(cav; V_hold = -70.0)
    kca_res = iv_curve(kca; V_hold = -70.0, Ca_hold = 1.0, t_step = 300.0)

    return plot(plot_iv(cav_res), plot_iv(kca_res);
                layout = (1, 2), size = (700, 420), margin = 20px)
end

function demo_group()
    @named na   = HH.SodiumChannel()
    @named k    = HH.PotassiumChannel()
    @named leak = HH.LeakChannel()

    res = iv_curve([na, k, leak])

    return plot(plot_iv_group(res, which = :I_ss),
                plot_iv_group(res, which = :I_peak);
                layout = (1, 2), size = (900, 420), margin = 20px)
end


end