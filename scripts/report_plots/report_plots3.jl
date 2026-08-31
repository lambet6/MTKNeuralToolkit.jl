# ============================================================
# Figure 5.3: Annotated voltage-clamp measurement schematic.
#
# A single sodium current trace for a step from -80 mV to 0 mV,
# marking the three quantities `iv_curve` records per step:
# I_inst (t = 0+), I_peak (argmax |i|), and I_ss (end of step).
#
# Output:    figures/measurement_schematic.pdf (and .png)
# ============================================================

include("../iv_curve.jl")

using .IVCurves: iv_curve
using MTKNeuralToolkit: HodgkinHuxley
using ModelingToolkit: @named
using Plots
using Plots.PlotMeasures: px

const HH = HodgkinHuxley

# alpha_m is 0/0 at v = -40 and alpha_n at v = -55; 0 mV is clear of both.
const V_STEP = 0.0
const V_HOLD = -80.0
const T_STEP = 20.0

@named na = HH.SodiumChannel()

res = iv_curve([na]; voltages = [V_STEP], V_hold = V_HOLD, t_step = T_STEP)

c = findfirst(!=(:neuron), res.channel_names)
t = res.times[1]
i = res.traces[1][:, c]

k_inst = 1
k_peak = argmax(abs.(i))
k_ss   = length(i)

marks = [(t[k_inst], i[k_inst], "\$I_{inst}\$", :topright),
         (t[k_peak], i[k_peak], "\$I_{peak}\$", :bottomright),
         (t[k_ss],   i[k_ss],   "\$I_{ss}\$",   :bottomleft)]

plt = plot(t, i;
           lw = 2.5, c = :steelblue, label = false, grid = false,
           xlabel = "Time from step onset (ms)",
           ylabel = "Current (outward positive, µA)",
           title  = "Sodium current, step $(V_HOLD) → $(V_STEP) mV",
           size = (700, 460), margin = 20px, dpi = 300)

hline!(plt, [0.0]; c = :black, ls = :dot, lw = 1, label = false)

span = maximum(i) - minimum(i)
for (tk, ik, lab, pos) in marks
    vline!(plt, [tk]; c = :grey, ls = :dash, lw = 1, alpha = 0.5, label = false)
    scatter!(plt, [tk], [ik]; c = :firebrick, ms = 6, label = false)
    dy = pos === :bottomright || pos === :bottomleft ? -0.06span : 0.06span
    dx = pos === :bottomleft ? -0.04 * (t[end] - t[1]) : 0.04 * (t[end] - t[1])
    halign = pos === :bottomleft ? :right : :left
    annotate!(plt, tk + dx, ik + dy, text(lab, 11, halign, :firebrick))
end

mkpath("figures")
savefig(plt, joinpath("figures", "measurement_schematic.pdf"))
savefig(plt, joinpath("figures", "measurement_schematic.png"))
println("Saved figures/measurement_schematic.pdf and .png")
