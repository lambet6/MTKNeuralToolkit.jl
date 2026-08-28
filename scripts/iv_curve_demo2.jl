# =============================================================================
# iv_curve_demo2.jl -- exercising iv_curve() across a menagerie of channels
# =============================================================================
#
# The original iv_curve.jl demo only tries sodium + leak. This one deliberately
# picks channels that stress different parts of the protocol:
#
#   custom: persistent Na-like   -- one activation gate, NO inactivation.
#                                    I_peak and I_ss should coincide exactly,
#                                    since nothing ever shuts the current off.
#   custom: window current       -- activation and inactivation curves tuned
#                                    to overlap over a broad voltage range, so
#                                    I_ss stays visibly nonzero across a chunk
#                                    of the sweep instead of the usual narrow
#                                    sliver -- an exaggerated version of the
#                                    "sodium window current" the original
#                                    script's own settling-check comment
#                                    name-checks.
#   Morris-Lecar: Ca2+ (fast)    -- gating tau = 0.1, i.e. "instantaneous" in
#                                    all but name. I_peak and I_ss should sit
#                                    right on top of each other; only I_inst,
#                                    which is pinned to the *hold* voltage's
#                                    gate value, will differ.
#   Morris-Lecar: K+ (slow)      -- an ordinary non-inactivating recovery
#                                    current (tau_n = 10). I_peak == I_ss again,
#                                    but now the approach to it is slow enough
#                                    to see.
#   FitzHugh-Nagumo (not a channel!) -- built on a bare OnePort with a cubic
#                                    i-v relation and its own internal recovery
#                                    state w. This isn't biophysical at all;
#                                    it's here to confirm iv_curve() only cares
#                                    that something exposes .p/.n, not that it
#                                    came from GenericChannel. Needs its own
#                                    voltage range -- see note below.
#   Prinz STG: Na+                -- a transient current like HH sodium, but
#                                    different kinetics/temperature regime.
#                                    Good side-by-side check against the
#                                    HH sodium curve from the original demo.
#   Prinz STG: A-type K+ (Ka)     -- transient, inactivating, but on the K+
#                                    side: outward instead of inward, but the
#                                    same textbook peak-vs-steady-state split.
#   Prinz STG: Ih                 -- hyperpolarization-ACTIVATED: m_inf falls
#                                    as v rises, the mirror image of every
#                                    other gate here. Good check that nothing
#                                    in iv_curve assumes gates open on
#                                    depolarization.
#   Prinz STG: CaS, CaT           -- the two Ca2+ currents, built via
#                                    CaVChannel. Their E_rev comes from the
#                                    Nernst equation over ca_port.Ca rather
#                                    than a fixed parameter, so these only
#                                    work now that iv_curve() ties ca_port off
#                                    with a FixedCalcium clamp (Ca_hold).
#                                    CaT inactivates faster than CaS -- look
#                                    for the bigger I_peak/I_ss gap on CaT.
#   Prinz STG: KCa                -- Ca2+-activated K+ current (KCaChannel):
#                                    its gate itself is a function of
#                                    ca_port.Ca, so its I-V curve actually
#                                    depends on the clamped Ca_hold level, not
#                                    just voltage. See demo_kca_ca_sensitivity()
#                                    below for that dependence on its own.
#
# USAGE
# -----
#   include("scripts/iv_curve.jl"); using .IVCurves
#   include("scripts/iv_curve_demo2.jl")
#   demo2()                      # single channels
#   demo_neurons()               # groups of channels ("neurons"), via iv_curve(channels::Vector)
#   demo_kca_ca_sensitivity()    # one KCa channel, swept over several Ca_hold levels
#
# ContinuousSpikers and PrinzNeuron are submodules of MTKNeuralToolkit.
# `using MTKNeuralToolkit` alone won't bring bare names like `PrinzNeuron`
# into scope unless they're on MTKNeuralToolkit's export list (that's what
# the `UndefVarError: PrinzNeuron not defined` means if you hit it) -- and
# as of writing, PrinzNeuron's submodule isn't actually exported correctly.
# Rather than touch the package, we just reference the submodules by their
# fully-qualified path below (MTKNeuralToolkit.PrinzNeuron, etc.), which
# works regardless of what is or isn't exported.
# =============================================================================

using MTKNeuralToolkit
using MTKNeuralToolkit: GateSpec, GenericChannel, InfTau, CaVChannel, KCaChannel
using ModelingToolkit: @named
using Plots
using Plots.PlotMeasures: px

# -----------------------------------------------------------------------------
# Custom channel 1: persistent (non-inactivating) sodium-like current.
# -----------------------------------------------------------------------------
persistent_m_inf(v) = 1.0 / (1.0 + exp((v + 40.0) / -6.0))
persistent_tau_m(v) = 0.15 + 0.9 / (1.0 + exp((v + 40.0) / 10.0))
const persistent_gates = [GateSpec(:mNaP, 1, 0.0, InfTau(persistent_m_inf, persistent_tau_m))]

# -----------------------------------------------------------------------------
# Custom channel 2: "window current" channel. Activation (~-45 mV) and
# inactivation (~-35 mV) are placed close together with a shallow inactivation
# slope, so there's a wide band where m is partly open and h hasn't fully
# closed -- current never returns to zero in that band, even at steady state.
# -----------------------------------------------------------------------------
window_m_inf(v) = 1.0 / (1.0 + exp((v + 45.0) / -5.0))
window_tau_m(v) = 0.3
window_h_inf(v) = 1.0 / (1.0 + exp((v + 35.0) / 8.0))
window_tau_h(v) = 25.0
const window_gates = [GateSpec(:mWin, 2, 0.0, InfTau(window_m_inf, window_tau_m)),
                       GateSpec(:hWin, 1, 1.0, InfTau(window_h_inf, window_tau_h))]

# Nernst prefactor PrinzNeuron uses for its Ca2+ channels (R*T/(z*F) at 283.15 K,
# z=2): CaVChannel's default E_rev = nernst_factor * log(Ca_out / ca_port.Ca).
const PRINZ_NERNST_FACTOR = 500.0 * 8.6174e-5 * 283.15

# -----------------------------------------------------------------------------
# Grid plotting helper: run iv_curve on each (label => channel) pair, using
# whatever protocol keyword overrides are supplied per-channel, and lay the
# resulting plots out together.
# -----------------------------------------------------------------------------
function demo_grid(named_channels; ncols = 4)
    plots = map(enumerate(named_channels)) do (idx, (label, channel, kwargs))
        res = iv_curve(channel; kwargs...)
        plt = plot_iv(res)
        plot!(plt; title = label, legend = (idx == 1 ? :topleft : false))
        plt
    end
    nrows = cld(length(plots), ncols)
    plot(plots...; layout = (nrows, ncols), size = (380 * ncols, 320 * nrows), margin = 30px)
end

function demo2()
    # --- Custom channels, default mV-scale protocol -------------------------
    @named nap_ch = GenericChannel(g = 15.0, E_rev = 50.0, gates = persistent_gates)
    @named win_ch = GenericChannel(g = 5.0,  E_rev = 50.0, gates = window_gates)

    # --- Morris-Lecar's two gated currents (leak is skipped -- it's the same
    #     flat-line story as the HH leak already shown in the original demo)
    ca_ch, k_ch, _ = MTKNeuralToolkit.ContinuousSpikers.MorrisLecar(name = :ml)

    # --- FitzHugh-Nagumo: a bare OnePort, not built via GenericChannel at
    #     all. Its state is dimensionless (v ~ O(1), not mV), so it needs its
    #     own voltage range and timescale rather than the physiological
    #     defaults -- otherwise the cubic i-v term blows up to meaningless
    #     magnitudes over a +/-100 mV sweep.
    @named fhn = MTKNeuralToolkit.ContinuousSpikers.FitzHughNagumo()

    # --- Three channels built straight from PrinzNeuron's exported gate sets,
    #     each via plain GenericChannel -- exactly the pattern from the
    #     iv_curve docstring.
    @named prinz_na = GenericChannel(g = 100.0, E_rev = 50.0,  gates = MTKNeuralToolkit.PrinzNeuron.na_gates)
    @named prinz_ka = GenericChannel(g = 50.0,  E_rev = -80.0, gates = MTKNeuralToolkit.PrinzNeuron.ka_gates)
    @named prinz_h  = GenericChannel(g = 0.5,   E_rev = -20.0, gates = MTKNeuralToolkit.PrinzNeuron.h_gates)

    # --- Prinz STG's two Ca2+ currents, built via CaVChannel -- their E_rev
    #     comes from the Nernst equation over ca_port.Ca, which iv_curve now
    #     clamps to Ca_hold (default 0.05, matching PrinzNeuron's resting
    #     Ca_inf) via FixedCalcium.
    @named prinz_cas = CaVChannel(g = 6.0, gates = MTKNeuralToolkit.PrinzNeuron.cas_gates,
                                   Ca_out = 3000.0, nernst_factor = PRINZ_NERNST_FACTOR)
    @named prinz_cat = CaVChannel(g = 2.5, gates = MTKNeuralToolkit.PrinzNeuron.cat_gates,
                                   Ca_out = 3000.0, nernst_factor = PRINZ_NERNST_FACTOR)

    # --- Prinz STG's Ca2+-activated K+ current -- KCa_m_inf's ca/(ca+3) term
    #     is tiny at the default Ca_hold = 0.05, so bump Ca_hold here just to
    #     make the curve visible in this grid; demo_kca_ca_sensitivity() below
    #     is where the Ca2+-dependence itself is the point.
    @named prinz_kca = KCaChannel(g = 5.0, E_rev = -80.0, gates = MTKNeuralToolkit.PrinzNeuron.kca_gates)

    named_channels = [
        ("custom: persistent Na-like",       nap_ch,  (;)),
        ("custom: window current",           win_ch,  (t_step = 300.0,)),   # tau_h = 25 ms constant;
                                                                              # 100 ms leaves ~1.8% unsettled
        ("Morris-Lecar: Ca2+ (fast)",        ca_ch,   (;)),
        ("Morris-Lecar: K+ (slow)",          k_ch,    (;)),
        ("FitzHugh-Nagumo (not a channel!)", fhn,     (voltages = -3.0:0.05:3.0,
                                                          V_hold = -1.0, t_hold = 50.0,
                                                          t_step = 50.0)),
        ("Prinz STG: Na+",                   prinz_na, (;)),
        ("Prinz STG: A-type K+ (Ka)",        prinz_ka, (t_step = 600.0,)),   # tau_h up to ~77 ms;
                                                                              # 100 ms is only ~1.3 tau
        ("Prinz STG: Ih",                    prinz_h,  (;)),
        ("Prinz STG: Ca2+ (CaS, slow)",      prinz_cas, (t_step = 2800.0,)), # tau_h up to ~353 ms
        ("Prinz STG: Ca2+ (CaT, transient)", prinz_cat, (t_step = 1600.0,)), # tau_h up to ~210 ms
        ("Prinz STG: KCa",                   prinz_kca, (t_step = 1500.0,   # tau_m up to ~180 ms
                                                          Ca_hold = 1.0)),
    ]

    demo_grid(named_channels)
end

# -----------------------------------------------------------------------------
# Grid plotting helper for the *group* iv_curve() method: run it on each
# (label => channels) list and show both the I_ss and I_peak per-channel
# breakdowns (plot_iv_group overlays each channel's contribution together
# with the total "neuron" current, in bold black).
# -----------------------------------------------------------------------------
function demo_group_grid(named_groups)
    plots = mapreduce(vcat, named_groups) do (label, channels, kwargs)
        res = iv_curve(channels; kwargs...)
        ss_plt   = plot_iv(res; which = :I_ss)
        peak_plt = plot_iv(res; which = :I_peak)
        plot!(ss_plt; title = "$label (I_ss)")
        plot!(peak_plt; title = "$label (I_peak)")
        [ss_plt, peak_plt]
    end
    nrows = length(named_groups)
    plot(plots...; layout = (nrows, 2), size = (500 * 2, 380 * nrows), margin = 30px)
end

function demo_neurons()
    # --- Textbook HH neuron: Na+ + K+ + leak, wired in parallel (no
    #     capacitor -- same clamp-circuit topology as the single-channel
    #     demos above, just with several channels sharing the clamp).
    @named hh_na   = MTKNeuralToolkit.HodgkinHuxley.SodiumChannel()
    @named hh_k    = MTKNeuralToolkit.HodgkinHuxley.PotassiumChannel()
    @named hh_leak = MTKNeuralToolkit.HodgkinHuxley.LeakChannel()

    # --- Morris-Lecar neuron: its two gated currents plus the leak that was
    #     skipped in demo2() above.
    ml_ca, ml_k, ml_leak = MTKNeuralToolkit.ContinuousSpikers.MorrisLecar(name = :ml)

    # --- Prinz STG-style neuron, now with its two Ca2+ currents (CaS/CaT,
    #     via CaVChannel) and its Ca2+-activated K+ current (KCa) included:
    #     iv_curve(channels) shares a single FixedCalcium clamp (at Ca_hold,
    #     default 0.05, matching PrinzNeuron's resting Ca_inf) across every
    #     channel here that exposes a ca_port, so CaS/CaT's Nernst E_rev and
    #     KCa's Ca-dependent gate are both well-defined without needing a
    #     full CalciumTracker pool. Note this is a *fixed* Ca2+ clamp, not
    #     the free-running Ca2+ feedback loop the real Prinz neuron has, so
    #     it shows these channels' I-V behavior at a buffered internal Ca2+
    #     level rather than the emergent bursting dynamics.
    @named stg_na   = GenericChannel(g = 100.0, E_rev = 50.0,  gates = MTKNeuralToolkit.PrinzNeuron.na_gates)
    @named stg_cas  = CaVChannel(g = 6.0, gates = MTKNeuralToolkit.PrinzNeuron.cas_gates,
                                  Ca_out = 3000.0, nernst_factor = PRINZ_NERNST_FACTOR)
    @named stg_cat  = CaVChannel(g = 2.5, gates = MTKNeuralToolkit.PrinzNeuron.cat_gates,
                                  Ca_out = 3000.0, nernst_factor = PRINZ_NERNST_FACTOR)
    @named stg_ka   = GenericChannel(g = 50.0,  E_rev = -80.0, gates = MTKNeuralToolkit.PrinzNeuron.ka_gates)
    @named stg_kca  = KCaChannel(g = 5.0, E_rev = -80.0, gates = MTKNeuralToolkit.PrinzNeuron.kca_gates)
    @named stg_kdr  = GenericChannel(g = 50.0,  E_rev = -80.0, gates = MTKNeuralToolkit.PrinzNeuron.kdr_gates)
    @named stg_h    = GenericChannel(g = 0.5,   E_rev = -20.0, gates = MTKNeuralToolkit.PrinzNeuron.h_gates)
    @named stg_leak = GenericChannel(g = 0.3,   E_rev = -50.0, gates = GateSpec[])

    # --- A made-up "neuron" combining the two custom channels from demo2():
    #     the persistent current sets a depolarizing floor and the window
    #     current adds its own steady-state bump on top of it.
    # @named custom_nap = GenericChannel(g = 15.0, E_rev = 50.0, gates = persistent_gates)
    # @named custom_win = GenericChannel(g = 5.0,  E_rev = 50.0, gates = window_gates)

    named_groups = [
        ("HH neuron (Na+K+leak)",           [hh_na, hh_k, hh_leak],                    (;)),
        ("Morris-Lecar neuron (Ca+K+leak)", [ml_ca, ml_k, ml_leak],                     (;)),
        ("Prinz STG neuron",                [stg_na, stg_cas, stg_cat, stg_ka, stg_kca,
                                              stg_kdr, stg_h, stg_leak],                 (t_step = 2800.0,)),  # CaS's tau_h up to ~353 ms
        # ("custom: persistent + window",     [custom_nap, custom_win],                  (t_step = 300.0,)),  # window's tau_h = 25 ms
    ]

    demo_group_grid(named_groups)
end

# -----------------------------------------------------------------------------
# KCa Ca2+-sensitivity sweep: the same KCa channel run at several different
# Ca_hold clamp levels. Unlike every other demo in this file, the point here
# isn't the channel -- it's iv_curve's Ca_hold parameter itself. KCa_m_inf's
# ca/(ca+3) Hill term means the channel barely opens at Ca_hold = 0.05 (the
# resting level PrinzNeuron's CalciumTracker settles to) and opens
# progressively more as the clamped intracellular Ca2+ rises, exactly like a
# real Ca2+-activated K+ current responding to a Ca2+ load -- something a
# voltage-only sweep could never show.
# -----------------------------------------------------------------------------
function demo_kca_ca_sensitivity()
    @named kca = KCaChannel(g = 5.0, E_rev = -80.0, gates = MTKNeuralToolkit.PrinzNeuron.kca_gates)

    Ca_levels = [0.05, 0.5, 1.0, 3.0, 10.0]
    plt = plot(xlabel = "Clamp voltage (mV)", ylabel = "Current (outward positive, µA)",
               title = "KCa I-V vs. clamped intracellular Ca2+ (steady state)",
               legend = :topleft)
    for Ca_hold in Ca_levels
        res = iv_curve(kca; Ca_hold = Ca_hold, t_step = 1500.0)   # tau_m up to ~180 ms
        plot!(plt, res.V, res.I_ss; lw = 2, label = "Ca_hold = $Ca_hold µM")
    end
    hline!(plt, [0.0]; c = :black, lw = 1, label = false)
    return plt
end

