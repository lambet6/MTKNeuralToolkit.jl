# ============================================================
# Figure 5.10: Liu STG bursting neuron currentscape.
#
# Reproduces `Currentscape.demo_liu()` as written: the eight-channel
# Liu STG neuron (na, cas, cat, ih, ka, kca, kdr, leak) with no
# injected drive, simulated for 750 ms, with the voltage trace
# panel on top of the currentscape.
#
# Output:    figures/liu_currentscape.pdf (and .png)
# ============================================================

include("../currentscape.jl")

using .Currentscape: demo_liu
using Plots

plt = demo_liu()
plot!(plt; dpi = 300)

mkpath("figures")
savefig(plt, joinpath("figures", "liu_currentscape.pdf"))
savefig(plt, joinpath("figures", "liu_currentscape.png"))
println("Saved figures/liu_currentscape.pdf and .png")
