# ============================================================
# Figure 5.4: Per-channel I-V relationships for the
# Hodgkin-Huxley sodium, potassium and leak channels.
#
# Each panel is `plot_iv`'s single-channel branch, which overlays the
# instantaneous, peak and steady-state currents. The x-axis is shared
# (-100 to +60 mV); the y-axes are deliberately independent, since the
# leak current is two orders of magnitude smaller than the others.
#
# Output:    figures/iv_per_channel.pdf (and .png)
# ============================================================

include("../iv_curve.jl")

using .IVCurves: iv_curve, plot_iv
using MTKNeuralToolkit: HodgkinHuxley
using ModelingToolkit: @named
using Plots
using Plots.PlotMeasures: px

const HH = HodgkinHuxley

# alpha_m is 0/0 at v = -40 and alpha_n at v = -55; the half-mV grid
# offset keeps the sweep clear of both removable singularities.
const VOLTAGES = -99.5:1.0:60.5
const V_HOLD   = -80.0
const XLIMS    = (-100.0, 60.0)

@named na   = HH.SodiumChannel()
@named k    = HH.PotassiumChannel()
@named leak = HH.LeakChannel()

results = [iv_curve([ch]; voltages = VOLTAGES, V_hold = V_HOLD)
           for ch in (na, k, leak)]

panels = [plot!(plot_iv(res); xlims = XLIMS, grid = false) for res in results]

plt = plot(panels...;
           layout = (1, 3), size = (1300, 430), margin = 20px, dpi = 300)

mkpath("figures")
savefig(plt, joinpath("figures", "iv_per_channel.pdf"))
savefig(plt, joinpath("figures", "iv_per_channel.png"))
println("Saved figures/iv_per_channel.pdf and .png")
