# ============================================================
# Figure 5.4: Per-channel I-V relationships for the
# Hodgkin-Huxley sodium, potassium and leak channels.
#
# `plot_iv`'s single-channel branch overlays all three measures on one
# y-axis, which hides two of them: at the holding potential the gates are
# nearly shut (m_inf(-80)^3 ~ 1e-5 for sodium), so I_inst is a fraction of
# a µA against a several-thousand-µA peak/steady-state axis and renders as
# a flat line on zero. The plotting is therefore done here instead: a top
# row for peak and steady state, and a bottom row giving I_inst its own
# panel (and its own scale) per channel.
#
# The x-axis is shared (-100 to +60 mV); the y-axes are deliberately
# independent, since the leak current is two orders of magnitude smaller
# than the others.
#
# Output:    figures/iv_per_channel.pdf (and .png)
# ============================================================

include("../iv_curve.jl")

using .IVCurves: iv_curve
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
const YLABEL   = "Current (outward positive, µA)"

@named na   = HH.SodiumChannel()
@named k    = HH.PotassiumChannel()
@named leak = HH.LeakChannel()

channels = (na, k, leak)
results  = [iv_curve([ch]; voltages = VOLTAGES, V_hold = V_HOLD)
            for ch in channels]

"Peak and steady-state currents for one channel, sharing a y-axis."
function panel_gated(res)
    name = String(res.channel_names[1])
    plt = plot(res.V, res.I_ss[:, 1];
               lw = 2, label = "steady state",
               xlabel = "Clamp voltage (mV)", ylabel = YLABEL,
               title = "I-V: $name", legend = :topleft,
               xlims = XLIMS, grid = false)
    plot!(plt, res.V, res.I_peak[:, 1]; lw = 2, ls = :dash, label = "peak")
    hline!(plt, [0.0]; c = :black, lw = 1, label = false)
    return plt
end

"Instantaneous current for one channel, on its own scale."
function panel_inst(res)
    name = String(res.channel_names[1])
    plt = plot(res.V, res.I_inst[:, 1];
               lw = 2, ls = :dot, c = 3, label = "instantaneous",
               xlabel = "Clamp voltage (mV)", ylabel = YLABEL,
               title = "$name: instantaneous",
               legend = :topleft, xlims = XLIMS, grid = false)
    hline!(plt, [0.0]; c = :black, lw = 1, label = false)
    return plt
end

plt = plot(panel_gated.(results)..., panel_inst.(results)...;
           layout = (2, 3), size = (1300, 860), margin = 20px, dpi = 300)

mkpath("figures")
savefig(plt, joinpath("figures", "iv_per_channel.pdf"))
savefig(plt, joinpath("figures", "iv_per_channel.png"))
println("Saved figures/iv_per_channel.pdf and .png")
