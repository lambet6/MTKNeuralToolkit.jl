# ============================================================
# Figure 5.8: Hodgkin-Huxley equilibrium branch (V-I curve).
#
# The applied current is used as a continuation parameter and the
# resting membrane voltage is traced as an equilibrium branch
# V*(I_app), exactly as in `VICurves.demo`. The branch is drawn solid
# where the equilibrium is stable and dashed where it is unstable, and
# the detected bifurcations (a pair of Hopf points for the classical HH
# parameters) are marked.
#
# The sweep is seeded at I_app = -20 microA, where the cell is quiet and
# a genuine resting state exists to start the continuation from. Between
# the two Hopf currents the equilibrium is unstable and the cell fires
# repetitively; outside them it settles back to rest.
#
# The located bifurcation currents and voltages are printed to stdout so
# they can be quoted in the report text.
#
# Output:    figures/hh_vi_curve.pdf (and .png)
# ============================================================

include("../vi_curve.jl")

using .VICurves: vi_curve, plot_vi
using MTKNeuralToolkit: Capacitor, HodgkinHuxley, build_compartment
using ModelingToolkit: @named
using Plots
using Plots.PlotMeasures: px

const HH      = HodgkinHuxley
const CURRENTS = -20.0:200.0
const I_START  = -20.0

@named soma_cap = Capacitor(C = 1.0)
@named na_ch    = HH.SodiumChannel()
@named k_ch     = HH.PotassiumChannel()
@named leak     = HH.LeakChannel()

soma = build_compartment(soma_cap, [na_ch, k_ch, leak];
                         name = :soma, V_init = -65.0)

res = vi_curve(soma; currents = CURRENTS, I_start = I_START)

println("Hodgkin-Huxley equilibrium branch: detected bifurcations")
for p in res.points
    println("  $(p.type) at I_app = $(round(p.I, digits=3)) µA, " *
            "V = $(round(p.V, digits=3)) mV")
end

plt = plot_vi(res; title = "Hodgkin-Huxley equilibrium branch")
plot!(plt; grid = false, size = (760, 500), margin = 20px, dpi = 300)

mkpath("figures")
savefig(plt, joinpath("figures", "hh_vi_curve.pdf"))
savefig(plt, joinpath("figures", "hh_vi_curve.png"))
println("Saved figures/hh_vi_curve.pdf and .png")
