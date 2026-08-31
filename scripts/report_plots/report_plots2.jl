# ============================================================
# Figure 5.2: Voltage-clamp current families for the
# Hodgkin-Huxley sodium and potassium channels.
#
# Output:    figures/vclamp_families.pdf (and .png)
# ============================================================

include("../iv_curve.jl")

using .IVCurves: iv_curve
using MTKNeuralToolkit: HodgkinHuxley
using ModelingToolkit: @named
using Plots
using Plots.PlotMeasures: px

const HH = HodgkinHuxley

# alpha_m is 0/0 at v = -40 and alpha_n at v = -55; the 0.137 offset
# keeps the step grid clear of both removable singularities.
const V_STEPS = collect(-60.0:20.0:40.0) .+ 0.137
const V_HOLD  = -80.0
const T_STEP  = 20.0

@named na = HH.SodiumChannel()
@named k  = HH.PotassiumChannel()

na_res = iv_curve([na]; voltages = V_STEPS, V_hold = V_HOLD, t_step = T_STEP)
k_res  = iv_curve([k];  voltages = V_STEPS, V_hold = V_HOLD, t_step = T_STEP)

clims = (minimum(V_STEPS), maximum(V_STEPS))

"""Plot the I(t) family for one `iv_curve` result, colour-graded by step voltage."""
function family_panel(res, title)
    c = findfirst(!=(:neuron), res.channel_names)
    plt = plot(; xlabel = "Time from step onset (ms)",
                 ylabel = "Current (outward positive, µA)",
                 title  = title, legend = false, grid = false,
                 colorbar = true, clims = clims,
                 colorbar_title = "Step voltage (mV)")
    for kk in eachindex(res.V)
        t = res.times[kk]
        plot!(plt, t, res.traces[kk][:, c];
              lw = 2, line_z = fill(res.V[kk], length(t)),
              c = cgrad(:viridis), clims = clims, colorbar_entry = (kk == 1))
    end
    hline!(plt, [0.0]; c = :black, lw = 1, alpha = 0.5, colorbar_entry = false)
    return plt
end

p1 = family_panel(na_res, "(a) Sodium")
p2 = family_panel(k_res,  "(b) Potassium")

plt = plot(p1, p2; layout = (1, 2), size = (950, 380), margin = 20px, dpi = 300)

mkpath("figures")
savefig(plt, joinpath("figures", "vclamp_families.pdf"))
savefig(plt, joinpath("figures", "vclamp_families.png"))
println("Saved figures/vclamp_families.pdf and .png")
