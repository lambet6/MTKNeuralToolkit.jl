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
                       unknowns, parameters, getdefault, @named, t_nounits as t
using OrdinaryDiffEq
using BifurcationKit
using Plots
using Plots.PlotMeasures: px

export iv_curve, plot_iv,
       iv_curve_continuation, plot_iv_continuation

const DEFAULT_VOLTAGES = -99.5:1.0:60.5

# Shared clamp-circuit builder: channels wired in parallel to a FixedReversal
# clamp (no capacitor), any ca_port-bearing channels sharing one FixedCalcium
# clamp. Used by both `iv_curve` and `iv_curve_continuation` below -- the
# circuit itself doesn't depend on how the resulting system gets solved.
function _build_clamp_circuit(channels::AbstractVector; V_hold, Ca_hold)
    @named clamp = FixedReversal(E = V_hold)
    @named gnd   = Ground()

    eqs = Equation[
        connect(clamp.p, (ch.p for ch in channels)...),
        connect(clamp.n, gnd.g, (ch.n for ch in channels)...),
    ]
    systems = System[clamp; channels; gnd]

    # Any channels with a ca_port (CaVChannel, KCaChannel) share one fixed
    # Ca2+ clamp -- see the iv_curve docstring for why.
    ca_channels = filter(ch -> hasproperty(ch, :ca_port), channels)
    if !isempty(ca_channels)
        @named ca_clamp = FixedCalcium(Ca = Ca_hold)
        push!(eqs, connect(ca_clamp.port, (ch.ca_port for ch in ca_channels)...))
        push!(systems, ca_clamp)
    end

    @named circuit = System(eqs, t, SymbolicT[], SymbolicT[]; systems = systems)
    return mtkcompile(circuit)
end

"""
    iv_curve(channels::AbstractVector; voltages, V_hold, t_hold, t_step, Ca_hold, solver, reltol, abstol)

Run a voltage-clamp protocol on a "neuron" made of one or more ion `channels`
wired in parallel (all `p` pins tied together to the clamp, all `n` pins tied
to ground) -- a `FixedReversal` clamp does the job of the capacitor, so there
is no membrane capacitance to charge. Pass a single-element vector to study
one channel on its own, e.g. `iv_curve([na])`.

`channels` are already-named MTK systems built on a OnePort, e.g.
`@named na = GenericChannel(g=120.0, E_rev=50.0, gates=sodium_gates)`. Any
that expose a `ca_port` (i.e. `CaVChannel` or `KCaChannel`) share a single
`FixedCalcium` clamp holding intracellular Ca2+ at the fixed concentration
`Ca_hold` for the whole protocol -- the same idea as the electrical
`V_hold`/`voltages` clamp, but for the chemical domain. This mirrors
buffering intracellular Ca2+ with a chelator (EGTA/BAPTA) in a real patch
pipette so the `CaVChannel`'s Nernst potential and any Ca-dependent gating
(`KCaChannel`) stay well-defined and don't drift during the step.

Returns a NamedTuple with fields `V`, `I_inst`, `I_peak`, `I_ss`, the raw
`times` and `traces` per step, and the compiled system `sys` (useful for
plotting gating variables afterwards). `I_inst`, `I_peak` and `I_ss` are
`length(voltages) x (length(channels)+1)` matrices: one column per channel
(in `channels` order), plus a final "neuron" column holding the total clamp
current (`-sys.clamp.i`, i.e. the sum of all channel currents) -- for a
single channel this column just duplicates the channel's own current.
`channel_names` lists the corresponding column names.
"""
function iv_curve(channels::AbstractVector;
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
    sys = _build_clamp_circuit(channels; V_hold = V_hold, Ca_hold = Ca_hold)

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
    plot_iv(res; which = :I_ss)

Plot the I-V curves returned by [`iv_curve`](@ref).

For a single channel (`res.channel_names` holding just one channel plus the
duplicate `:neuron` column), all three relations -- instantaneous, peak,
steady state -- are overlaid for that channel and `which` is ignored. For
several channels, only `which` (one of `:I_inst`, `:I_peak`, `:I_ss`) is
plotted, one line per channel, with the total "neuron" current overlaid in
bold black.
"""
function plot_iv(res; which = :I_ss)
    chs = findall(!=(:neuron), res.channel_names)

    if length(chs) == 1
        c = only(chs)
        plt = plot(res.V, res.I_ss[:, c];
                   lw = 2, label = "steady state",
                   xlabel = "Clamp voltage (mV)",
                   ylabel = "Current (outward positive, µA)",
                   title  = "I-V: $(res.channel_names[c])",
                   legend = :topleft)
        plot!(plt, res.V, res.I_peak[:, c]; lw = 2, ls = :dash, label = "peak")
        plot!(plt, res.V, res.I_inst[:, c]; lw = 2, ls = :dot,  label = "instantaneous")
    else
        data = getproperty(res, which)
        plt = plot(xlabel = "Clamp voltage (mV)",
                   ylabel = "Current (outward positive, µA)",
                   title  = "I-V by channel ($(which))",
                   legend = :topleft)
        for c in chs
            plot!(plt, res.V, data[:, c]; lw = 2, label = String(res.channel_names[c]))
        end
        neuron_col = findfirst(==(:neuron), res.channel_names)
        plot!(plt, res.V, data[:, neuron_col];
              lw = 3, c = :black, label = "neuron (total)")
    end
    hline!(plt, [0.0]; c = :black, lw = 1, label = false)
    return plt
end

"""
    iv_curve_continuation(channels::AbstractVector; voltages, V_hold, t_hold, Ca_hold, solver, reltol, abstol, ds, dsmax)

Compute the steady-state I-V curve of one or more voltage-clamped ion
`channels` by numerical continuation (BifurcationKit.jl via MTK's
`BifurcationProblem`), rather than by forward-simulating a step at every test
voltage like [`iv_curve`](@ref) does. This traces `I_ss` as an equilibrium
branch of the clamp circuit directly, which is both faster and more accurate
very close to removable singularities in the gating rate functions (e.g.
classic HH `alpha_m` at v = -40) than integrating through them at loose
tolerance. Pass a single-element vector to study one channel on its own,
e.g. `iv_curve_continuation([na])`.

Same clamp circuit as `iv_curve` (built by [`_build_clamp_circuit`](@ref)):
`channels` are wired in parallel with a `FixedReversal` clamp holding voltage
at `V_hold` (no membrane capacitor). The continuation is seeded from a real
steady state -- the gates are simulated to equilibrium at `V_hold` (same
hold-then-settle pattern as `iv_curve`), then that state is continued in the
clamp voltage `E` across the full span of `voltages`, in both directions
(`bothside = true`, since `V_hold` sits in the interior of the range).

Requires `channels` to compile down to at least one differential unknown
(i.e. at least one gate) between them. Gateless channels only (e.g. a plain
leak) have no continuation state to track -- their `I_ss` is a straight line
obtainable directly from `i ~ g*(v - E_rev)`, so use [`iv_curve`](@ref) for
that case.

Returns a NamedTuple with fields `V`, `I_ss`, `stable` (a `BitMatrix`, from
BifurcationKit's per-point unstable-eigenvalue count), the compiled system
`sys`, and `channel_names`. `I_ss` and `stable` are
`length(voltages) x (length(channels)+1)` matrices -- one column per channel
(in `channels` order) plus a final "neuron" column holding the total clamp
current -- `channel_names` lists the corresponding column names, and
`branches` is a vector of the per-column `ContResult`s. Sorted by `V`
(continuation runs in both directions from `V_hold`, so branch order isn't
guaranteed).
"""
function iv_curve_continuation(channels::AbstractVector;
                  voltages = DEFAULT_VOLTAGES,
                  V_hold   = -80.0,
                  t_hold   = 500.0,
                  Ca_hold  = 0.05,
                  solver   = Rosenbrock23(),
                  reltol   = 1e-8,
                  abstol   = 1e-8,
                  ds       = 1.0,
                  dsmax    = 2.0)

    voltages = collect(float.(voltages))
    p_min, p_max = extrema(voltages)

    # 1. Build the clamp circuit (same as iv_curve).
    sys = _build_clamp_circuit(channels; V_hold = V_hold, Ca_hold = Ca_hold)

    E_clamp = sys.clamp.E
    states  = unknowns(sys)

    isempty(states) && error(
        "iv_curve_continuation needs at least one differential unknown " *
        "(gate variable) once the clamp circuit is compiled, but " *
        "$(nameof.(channels)) has none. A gateless channel's I_ss is a " *
        "straight line -- use `iv_curve` instead.")

    channel_names = nameof.(channels)
    I_chs         = [getproperty(sys, name).i for name in channel_names]

    # 2. Seed continuation from a real steady state: hold at V_hold and
    # settle, exactly like iv_curve's hold phase.
    prob_hold = ODEProblem(sys, [E_clamp => V_hold], (0.0, t_hold))
    sol_hold  = solve(prob_hold, solver; reltol = reltol, abstol = abstol)
    sol_hold.retcode == ReturnCode.Success || error(
        "Holding phase failed to solve ($(sol_hold.retcode)). " *
        "Check the channels' gating functions at V = $V_hold mV.")
    u0_guess = sol_hold.u[end]

    # BifurcationProblem needs a value for every parameter, not just the
    # bifurcation parameter -- take the rest from their compiled defaults.
    ps = filter(!isequal(E_clamp), parameters(sys))
    p_start = [ps .=> getdefault.(ps); E_clamp => V_hold]

    opts_br = ContinuationPar(
        p_min = p_min, p_max = p_max,
        ds = ds, dsmax = dsmax,
        detect_bifurcation = 1, nev = length(states),
        max_steps = 2000)

    # 3. One continuation run per current of interest. The branch geometry
    # (voltage grid, stability) is set by the underlying dynamics alone, not
    # by which observable gets recorded as `plot_var`, so the per-channel
    # and total runs land on exactly the same voltage grid.
    run_branch(plot_var) = continuation(
        BifurcationProblem(sys, states .=> u0_guess, p_start, E_clamp;
                            plot_var = plot_var, jac = false),
        PALC(), opts_br; bothside = true)

    branches  = [run_branch(I_ch) for I_ch in I_chs]
    br_clamp  = run_branch(sys.clamp.i)   # total = -clamp.i

    n    = length(branches[1].branch)
    n_ch = length(channels) + 1
    V      = branches[1].branch.param
    I_ss   = zeros(n, n_ch)
    stable = falses(n, n_ch)
    for (c, br) in enumerate(branches)
        I_ss[:, c]   = br.branch.x
        stable[:, c] = br.branch.stable
    end
    I_ss[:, end]   = -br_clamp.branch.x
    stable[:, end] = br_clamp.branch.stable

    order = sortperm(V)
    return (V = V[order], I_ss = I_ss[order, :], stable = stable[order, :],
            sys = sys, branches = [branches; br_clamp],
            channel_names = [channel_names; :neuron])
end


# Draws one V/I branch as alternating solid/dashed segments (solid where
# `stable`), only labelling the first segment so a channel gets one legend
# entry no matter how many stability switches its branch has.
function _plot_branch!(plt, V, I, stable; c, lw, label)
    i = firstindex(V)
    first_seg = true
    while i <= lastindex(V)
        j = i
        while j < lastindex(V) && stable[j+1] == stable[i]
            j += 1
        end
        ls = stable[i] ? :solid : :dash
        plot!(plt, V[i:j], I[i:j]; lw = lw, ls = ls, c = c,
              label = first_seg ? label : false)
        first_seg = false
        i = j + 1
    end
    return plt
end

"""
    plot_iv_continuation(res)

Plot the `I_ss` branch(es) returned by [`iv_curve_continuation`](@ref): solid
where `stable`, dashed otherwise.

For a single channel, that one branch is drawn without a legend. For several
channels, each branch is drawn in its own color with a legend, and the total
"neuron" branch is overlaid in bold black.
"""
function plot_iv_continuation(res)
    chs    = findall(!=(:neuron), res.channel_names)
    single = length(chs) == 1

    plt = plot(xlabel = "Clamp voltage (mV)",
               ylabel = "Current (outward positive, µA)",
               title  = single ? "I-V (continuation): $(res.channel_names[only(chs)])" :
                                  "I-V by channel (continuation)",
               legend = single ? false : :topleft)

    for c in chs
        _plot_branch!(plt, res.V, res.I_ss[:, c], res.stable[:, c];
                      c = c, lw = 2, label = String(res.channel_names[c]))
    end
    neuron_col = findfirst(==(:neuron), res.channel_names)
    _plot_branch!(plt, res.V, res.I_ss[:, neuron_col], res.stable[:, neuron_col];
                  c = single ? 1 : :black, lw = single ? 2 : 3, label = "neuron (total)")

    hline!(plt, [0.0]; c = :black, lw = 1, label = false)
    return plt
end


const HH = HodgkinHuxley

function demo()
    @named na   = HH.SodiumChannel()
    @named k    = HH.PotassiumChannel()
    @named leak = HH.LeakChannel()
    V_hold = -40.5
    na_res   = iv_curve([na], V_hold=V_hold)
    k_res   = iv_curve([k], V_hold=V_hold)
    # leak_res = iv_curve([leak], V_hold=V_hold)

    # return plot(plot_iv(na_res), plot_iv(k_res), plot_iv(leak_res);
    return plot(plot_iv(na_res), plot_iv(k_res),;
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

    cav_res = iv_curve([cav]; V_hold = -70.0)
    kca_res = iv_curve([kca]; V_hold = -70.0, Ca_hold = 1.0, t_step = 300.0)

    return plot(plot_iv(cav_res), plot_iv(kca_res);
                layout = (1, 2), size = (700, 420), margin = 20px)
end

function demo_group()
    @named na   = HH.SodiumChannel()
    @named k    = HH.PotassiumChannel()
    @named leak = HH.LeakChannel()

    res = iv_curve([na, k, leak])

    return plot(plot_iv(res, which = :I_ss),
                plot_iv(res, which = :I_peak);
                layout = (1, 2), size = (900, 420), margin = 20px)
end


function demo_continuation()
    @named na = HH.SodiumChannel()
    @named k  = HH.PotassiumChannel()

    V_hold = -40.5
    na_res = iv_curve_continuation([na]; V_hold = V_hold)
    k_res  = iv_curve_continuation([k]; V_hold = V_hold)

    return plot(plot_iv_continuation(na_res),
                plot_iv_continuation(k_res);
                layout = (1, 2), size = (700, 420), margin = 20px)
end


function demo_group_continuation()
    @named na   = HH.SodiumChannel()
    @named k    = HH.PotassiumChannel()
    @named leak = HH.LeakChannel()

    res = iv_curve_continuation(
        [na, k, leak];
        V_hold = -40.5,
        ds = 1.0,
        dsmax = 2.0,
    )

    return plot_iv_continuation(res)
end


end