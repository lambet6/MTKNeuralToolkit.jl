# ============================================================
# Figure 5.7: KCa I-V family at fixed [Ca]i.
#
# One steady-state I-V curve per clamped calcium concentration for the
# Ca2+-dependent potassium channel of `IVCurves.demo_calcium`. The gate
#
#     mKCa_inf(v, ca) = (ca / (ca + 3)) * sigmoid((v + 20) / 5)
#
# factorises into a voltage term and a saturating calcium term, so the
# calcium concentration sets the ceiling on the conductance while the
# voltage dependence keeps the same shape. Sweeping Ca_hold therefore
# scales the family rather than shifting it, which is the point of the
# figure: at 0.05 microM the channel is essentially silent, and by
# 10 microM it is close to its saturated maximum.
#
# The KCa gate has tau = 20 ms, so the step must be long enough for it
# to relax; t_step = 300 ms (15 tau) clears the drift warning that
# `iv_curve` emits when I_ss is not yet a true steady state.
#
# Output:    figures/kca_iv_calcium.pdf (and .png)
# ============================================================

include("../iv_curve.jl")

using .IVCurves: iv_curve
using MTKNeuralToolkit: KCaChannel, GateSpec, InfTauCa
using ModelingToolkit: @named
using Plots
using Plots.PlotMeasures: px
using Printf: @sprintf

const VOLTAGES = -99.5:1.0:60.5
const V_HOLD   = -70.0
const T_STEP   = 300.0
const CA_HOLDS = [0.05, 0.5, 1.0, 3.0, 10.0]   # microM

# Same made-up gate as IVCurves.demo_calcium.
KCa_m_inf(v, ca) = (ca ./ (ca .+ 3.0)) ./ (1.0 .+ exp.(-(v .+ 20.0) ./ 5.0))
KCa_tau_m(v) = 20.0
kca_gates = [GateSpec(:mKCa, 4, 0.0, InfTauCa(KCa_m_inf, KCa_tau_m))]

@named kca = KCaChannel(g = 5.0, E_rev = -80.0, gates = kca_gates)

results = [iv_curve([kca]; voltages = VOLTAGES, V_hold = V_HOLD,
                    t_step = T_STEP, Ca_hold = ca) for ca in CA_HOLDS]

# Colour-graded light-to-dark with increasing calcium.
colours = cgrad(:viridis, length(CA_HOLDS); categorical = true)

plt = plot(; xlabel = "Clamp voltage (mV)",
             ylabel = "Current (outward positive, µA)",
             title  = "KCa steady-state I-V vs [Ca]i (hold $(V_HOLD) mV)",
             xlims  = (-100.0, 60.0), grid = false, legend = :topleft,
             size = (760, 500), margin = 20px, dpi = 300)

for (n, (res, ca)) in enumerate(zip(results, CA_HOLDS))
    c = findfirst(!=(:neuron), res.channel_names)
    plot!(plt, res.V, res.I_ss[:, c];
          lw = 2.5, c = colours[n],
          label = @sprintf("[Ca]i = %.2f µM", ca))
end

hline!(plt, [0.0]; c = :black, lw = 1, label = false)
vline!(plt, [-80.0]; c = :black, lw = 1, ls = :dot, label = "E_K = -80 mV")

mkpath("figures")
savefig(plt, joinpath("figures", "kca_iv_calcium.pdf"))
savefig(plt, joinpath("figures", "kca_iv_calcium.png"))
println("Saved figures/kca_iv_calcium.pdf and .png")
