# ==========================================
# Voltage-Clamp I-V Curve Protocol
# ==========================================
module IVCurves

using MTKNeuralToolkit
using MTKNeuralToolkit: FixedReversal, FixedCalcium, SymbolicT
using ModelingToolkit: System, Equation, mtkcompile, connect,
                       unknowns, @named, t_nounits as t
using OrdinaryDiffEq
using Plots
using Plots.PlotMeasures: px

export iv_curve, plot_iv


"""
    _build_clamp_circuit(channels; V_hold, Ca_hold)

Internal helper that wires `channels` straight to a `FixedReversal` voltage
clamp - bypassing `build_compartment` and the membrane capacitor entirely -
plus a `FixedCalcium` clamp at `Ca_hold` for any channel exposing a
`ca_port`. This is the compiled circuit `iv_curve`/`iv_curve_continuation`
step through.
"""
function _build_clamp_circuit(channels::AbstractVector; V_hold, Ca_hold)
    @named clamp = FixedReversal(E = V_hold)
    @named gnd   = Ground()

    eqs = Equation[
        connect(clamp.p, (ch.p for ch in channels)...),
        connect(clamp.n, gnd.g, (ch.n for ch in channels)...),
    ]
    systems = System[clamp; channels; gnd]

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
    iv_curve(channels; voltages=DEFAULT_VOLTAGES, V_hold=-80.0, t_hold=500.0, t_step=100.0, Ca_hold=0.05, solver=Rosenbrock23(), reltol=1e-8, abstol=1e-8)

Runs a voltage-clamp protocol on `channels`: hold at `V_hold` for
`t_hold` ms to let the gates settle, then step to each voltage in `voltages`
in turn for `t_step` ms, recording the instantaneous, peak, and steady-state
current through every channel (plus their sum, `:neuron`).

`channels` can be a single channel or a vector of channels sharing the
clamp, giving the combined (and per-channel) I-V relationship of a whole
"neuron" built from those channels. Any channel exposing a `ca_port` (e.g.
`CaVChannel`, `KCaChannel`) is tied off with a `FixedCalcium` clamp at
`Ca_hold` for the duration.

# Arguments
- `channels`: A channel system, or vector of channel systems, to clamp.
- `voltages`: The grid of step voltages to sweep (mV).
- `V_hold`: The holding potential before each step (mV).
- `t_hold`: Duration of the initial holding phase (ms).
- `t_step`: Duration of each voltage step (ms).
- `Ca_hold`: Intracellular Ca2+ concentration held fixed for `ca_port`-bearing channels.
- `solver`, `reltol`, `abstol`: Passed through to `solve`.

# Returns
- A `NamedTuple` with `V`, `I_inst`, `I_peak`, `I_ss` (each a `voltages x channels`
  matrix), the raw per-step `times`/`traces`, the compiled `sys`, `channel_names`
  (including `:neuron` for the summed current), and `V_hold`. `I_inst` is the
  open-channel current the instant the step lands (a straight line for an
  ohmic channel); `I_peak` is the the biggest spike in current during the step; 
  `I_ss` is the steady-state current once the gates have fully relaxed.
"""
function iv_curve(channels::AbstractVector;
                  voltages = -99.5:1.0:60.5,
                  V_hold   = -80.0,
                  t_hold   = 500.0,
                  t_step   = 100.0,
                  Ca_hold  = 0.05,
                  solver   = Rosenbrock23(),
                  reltol   = 1e-8,
                  abstol   = 1e-8)

    voltages = collect(float.(voltages))


    sys = _build_clamp_circuit(channels; V_hold = V_hold, Ca_hold = Ca_hold)


    E_clamp       = sys.clamp.E
    channel_names = nameof.(channels)
    I_chs         = [getproperty(sys, name).i for name in channel_names]
    I_total       = -sys.clamp.i              
    all_I         = [I_chs; I_total]
    col_names     = [channel_names; :neuron]
    states        = unknowns(sys)


    prob_hold = ODEProblem(sys, [E_clamp => V_hold], (0.0, t_hold))
    sol_hold  = solve(prob_hold, solver; reltol = reltol, abstol = abstol)
    sol_hold.retcode == ReturnCode.Success || error(
        "Holding phase failed to solve ($(sol_hold.retcode)). " *
        "Check the channels' gating functions at V = $V_hold mV.")
    u_settled = sol_hold.u[end]


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
    plot_iv(res; which=:I_ss)

Plots the I-V relationship from an `iv_curve` result `res`. For a single
channel, overlays its instantaneous, peak, and steady-state currents. For a
group of channels, plots the `which` current (`:I_inst`, `:I_peak`, or
`:I_ss`) per channel, alongside the total `:neuron` current as a dashed,
semi-transparent black line.
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
              lw = 1, ls = :dash, alpha = 0.6, c = :black, label = "neuron (total)")
    end
    hline!(plt, [0.0]; c = :black, lw = 1, label = false)
    return plt
end



# ==========================================
# Demos
# ==========================================

const HH = HodgkinHuxley

function demo()
    @named na   = HH.SodiumChannel()
    @named k    = HH.PotassiumChannel()
    @named leak = HH.LeakChannel()
    V_hold = -30.0
    na_res   = iv_curve([na], V_hold=V_hold)
    k_res   = iv_curve([k], V_hold=V_hold)

    return plot(plot_iv(na_res), plot_iv(k_res),;
                layout = (1, 3), size = (1000, 420), margin=20px)
end

function demo_calcium()
    # Made-up activation/inactivation curves, just to exercise
    # CaVChannel/KCaChannel's Ca2+-dependent E_rev and gating through iv_curve.
    CaV_m_inf(v) = 1.0 ./ (1.0 .+ exp.(-(v .+ 20.0) ./ 5.0))
    CaV_tau_m(v) = 5.0 .+ 10.0 ./ (1.0 .+ exp.((v .+ 20.0) ./ 10.0))
    cav_gates = [GateSpec(:mCaV, 3, 0.0, InfTau(CaV_m_inf, CaV_tau_m))]

    KCa_m_inf(v, ca) = (ca ./ (ca .+ 3.0)) ./ (1.0 .+ exp.(-(v .+ 20.0) ./ 5.0))
    KCa_tau_m(v) = 20.0
    kca_gates = [GateSpec(:mKCa, 4, 0.0, InfTauCa(KCa_m_inf, KCa_tau_m))]

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

    res = iv_curve([na, k, leak], V_hold = -65.0)

    return plot(plot_iv(res, which = :I_ss),
                plot_iv(res, which = :I_peak);
                layout = (1, 2), size = (900, 420), margin = 20px)
end


end