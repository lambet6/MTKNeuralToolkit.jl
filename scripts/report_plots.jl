# ============================================================
# Figure 2.2: Hodgkin-Huxley steady-state gating curves and
# time constants, as implemented in MTKNeuralToolkit.jl.
#
# Run with:  julia --project hh_gating_curves.jl
# Output:    figures/hh_gating_curves.pdf
# ============================================================

using MTKNeuralToolkit: HodgkinHuxley
using Plots
using Plots.PlotMeasures: px

const HH = HodgkinHuxley

# The rate functions are stored as v -> (alpha, beta); convert to the
# (x_inf, tau) form the figure reports.
inf_tau(rate, v) = let (a, b) = rate(v)
    (a ./ (a .+ b), 1.0 ./ (a .+ b))
end

# alpha_m is 0/0 at v = -40 and alpha_n at v = -55 (removable
# singularities). The 0.137 offset keeps the grid off both points.
V = collect(-100.0:0.1:60.0) .+ 0.137

m_inf, tau_m = inf_tau(HH.na_m, V)
h_inf, tau_h = inf_tau(HH.na_h, V)
n_inf, tau_n = inf_tau(HH.k_n,  V)

# --- Left panel: steady-state activation/inactivation ---
p1 = plot(V, m_inf; lw = 2, label = raw"$m_\infty$", c = 1,
          xlabel = "Membrane potential (mV)",
          ylabel = "Steady-state open probability",
          title  = "Steady-state gating",
          legend = :right, ylims = (-0.02, 1.02), grid = false)
plot!(p1, V, h_inf; lw = 2, label = raw"$h_\infty$", c = 2)
plot!(p1, V, n_inf; lw = 2, label = raw"$n_\infty$", c = 3)

# --- Right panel: time constants ---
p2 = plot(V, tau_m; lw = 2, label = raw"$\tau_m$", c = 1,
          xlabel = "Membrane potential (mV)",
          ylabel = "Time constant (ms)",
          title  = "Kinetics",
          legend = :topright, grid = false)
plot!(p2, V, tau_h; lw = 2, label = raw"$\tau_h$", c = 2)
plot!(p2, V, tau_n; lw = 2, label = raw"$\tau_n$", c = 3)

plt = plot(p1, p2; layout = (1, 2), size = (900, 380), margin = 20px,
           dpi = 300)

mkpath("figures")
savefig(plt, joinpath("figures", "hh_gating_curves.pdf"))
savefig(plt, joinpath("figures", "hh_gating_curves.png"))
println("Saved figures/hh_gating_curves.pdf and .png")

plt