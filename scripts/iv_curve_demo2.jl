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
#
# USAGE
# -----
#   include("scripts/iv_curve.jl"); using .IVCurves
#   include("scripts/iv_curve_demo2.jl")
#   demo2()
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
using MTKNeuralToolkit: GateSpec, GenericChannel, InfTau
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
    ]

    demo_grid(named_channels)
end