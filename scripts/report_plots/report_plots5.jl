# ============================================================
# Figure 5.5: Sodium I-V at three holding potentials.
#
# The same `iv_curve` sweep run on the Hodgkin-Huxley sodium channel
# alone, from V_hold = -100, -80 and -50 mV. Peak current (solid) and
# instantaneous current (dashed) are overlaid per hold.
#
# At -100 mV, h is close to 1 but m is closed, so almost no conductance
# is open the instant the step lands and I_inst sits flat near zero. By
# -50 mV inactivation has already removed much of the available h and
# the whole family shifts. This is what makes V_hold a keyword of
# `iv_curve` rather than a fixed constant.
#
# Output:    figures/na_iv_holds.pdf (and .png)
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
const V_HOLDS = [-80.0, -65.0, -50.0]
const COLOURS = [:steelblue, :seagreen, :firebrick]

@named na = HH.SodiumChannel()

results = [iv_curve([na]; voltages = VOLTAGES, V_hold = Vh) for Vh in V_HOLDS]

peak_plt = plot(; xlabel = "Clamp voltage (mV)",
                  ylabel = "Peak current (outward positive, µA)",
                  title = "Peak sodium current",
                  xlims = (-100.0, 60.0), grid = false, legend = :bottomleft)

inst_plt = plot(; xlabel = "Clamp voltage (mV)",
                  ylabel = "Instantaneous current (outward positive, µA)",
                  title = "Instantaneous sodium current",
                  xlims = (-100.0, 60.0), grid = false, legend = :bottomleft)

for (res, Vh, col) in zip(results, V_HOLDS, COLOURS)
    c = findfirst(!=(:neuron), res.channel_names)
    label = "hold $(Vh) mV"

    plot!(peak_plt, res.V, res.I_peak[:, c]; lw = 2.5, c = col, label)
    plot!(inst_plt, res.V, res.I_inst[:, c]; lw = 2.5, c = col, label)
end

hline!(peak_plt, [0.0]; c = :black, lw = 1, label = false)
hline!(inst_plt, [0.0]; c = :black, lw = 1, label = false)

plt = plot(peak_plt, inst_plt;
           layout = (1, 2), size = (1_080, 450), margin = 20px, dpi = 300)

mkpath("figures")
savefig(plt, joinpath("figures", "na_iv_holds.pdf"))
savefig(plt, joinpath("figures", "na_iv_holds.png"))
println("Saved figures/na_iv_holds.pdf and .png")
