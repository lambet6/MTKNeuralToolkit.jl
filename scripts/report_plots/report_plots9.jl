# ============================================================
# Figure 5.9: Hodgkin-Huxley currentscape.
#
# Reproduces `Currentscape.demo()` as written: an HH soma
# (Na, K, leak) driven with a 10.0 constant current for 75 ms,
# with the voltage trace panel on top of the currentscape.
#
# Output:    figures/hh_currentscape.pdf (and .png)
# ============================================================

include("../currentscape.jl")

using .Currentscape: demo
using Plots

plt = demo()
plot!(plt; dpi = 300)

mkpath("figures")
savefig(plt, joinpath("figures", "hh_currentscape.pdf"))
savefig(plt, joinpath("figures", "hh_currentscape.png"))
println("Saved figures/hh_currentscape.pdf and .png")
