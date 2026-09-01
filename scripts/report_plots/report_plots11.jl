# ============================================================
# Figure 5.11: Composite, Hodgkin-Huxley neuron, three panels.
#
# (a) The summed steady-state I-V curve of Fig 5.6 (I_ss for
#     [na, k, leak], hold -65 mV). Its zero crossing is the voltage at
#     which the total membrane current vanishes with every gate settled:
#     the predicted resting potential.
#
# (b) The equilibrium branch of Fig 5.8, V*(I_app), traced by
#     continuation. The low-current end of this branch is the same
#     resting state that (a) predicts - the two are marked with the same
#     dashed guide so they can be read against each other. The lower Hopf
#     point is where that rest loses stability: the predicted onset of
#     repetitive firing.
#
# (c) A currentscape of the same neuron driven just above that Hopf
#     current (I_HOPF_MARGIN microA above it), rather than the arbitrary
#     10 microA of `Currentscape.demo`. The drive is read straight off
#     panel (b), so the figure is one argument: (a) predicts where the
#     cell sits, (b) predicts where it starts firing, and (c) is run at
#     that prediction and shows the currents that carry the resulting
#     spikes.
#
# `plot_currentscape` returns its own multi-panel grid, and Plots.jl
# cannot nest a grid inside another layout, so the composite is saved as
# two files: panels (a)+(b) together, and panel (c) alongside.
#
# Output:    figures/hh_composite_iv_vi.pdf (and .png)
#            figures/hh_composite_currentscape.pdf (and .png)
# ============================================================

include("../iv_curve.jl")
include("../vi_curve.jl")
include("../currentscape.jl")

using .IVCurves: iv_curve, plot_iv
using .VICurves: vi_curve, plot_vi
using .Currentscape: channel_currents, plot_currentscape
using MTKNeuralToolkit: Capacitor, HodgkinHuxley, build_compartment,
                        build_acausal_network
using ModelingToolkit: @named, mtkcompile
using OrdinaryDiffEq
using Plots
using Plots.PlotMeasures: px
using Printf: @sprintf

const HH = HodgkinHuxley

# alpha_m is 0/0 at v = -40 and alpha_n at v = -55; the half-mV grid
# offset keeps the clamp sweep clear of both removable singularities.
const VOLTAGES       = -99.5:1.0:60.5
const V_HOLD         = -65.0
const CURRENTS       = -20.0:200.0
const I_START        = -20.0
const I_HOPF_MARGIN  = 1.0     # microA above the lower Hopf current
const T_END          = 100.0   # ms of simulation for the currentscape

# ------------------------------------------------------------
# (a) Summed steady-state I-V, and its zero crossing.
# ------------------------------------------------------------

@named na   = HH.SodiumChannel()
@named k    = HH.PotassiumChannel()
@named leak = HH.LeakChannel()

iv_res = iv_curve([na, k, leak]; voltages = VOLTAGES, V_hold = V_HOLD)

"""
    zero_crossings(V, I)

Voltages where `I` changes sign, found by linear interpolation between
neighbouring grid points (plus any grid point sitting exactly on zero).
Sign changes across a jump larger than 5% of the curve's full scale are
rejected as metric artefacts rather than genuine crossings.
"""
function zero_crossings(V, I)
    jump = 0.05 * maximum(abs, I)
    roots = Float64[]
    for j in 1:(length(V) - 1)
        a, b = I[j], I[j + 1]
        if a == 0.0
            push!(roots, V[j])
        elseif sign(a) != sign(b) && abs(b - a) <= jump
            push!(roots, V[j] - a * (V[j + 1] - V[j]) / (b - a))
        end
    end
    I[end] == 0.0 && push!(roots, V[end])
    return roots
end

neuron_col = findfirst(==(:neuron), iv_res.channel_names)
roots      = zero_crossings(iv_res.V, iv_res.I_ss[:, neuron_col])
isempty(roots) && error("No zero crossing in the summed I_ss curve; " *
                        "nothing to call a resting potential.")
V_rest = only(roots)

# ------------------------------------------------------------
# (b) Equilibrium branch, and the lower Hopf current.
# ------------------------------------------------------------

@named soma_cap = Capacitor(C = 1.0)
@named na_ch    = HH.SodiumChannel()
@named k_ch     = HH.PotassiumChannel()
@named leak_ch  = HH.LeakChannel()
channels = [na_ch, k_ch, leak_ch]

soma = build_compartment(soma_cap, channels; name = :soma, V_init = -65.0)

vi_res = vi_curve(soma; currents = CURRENTS, I_start = I_START)

hopfs = filter(p -> p.type == :hopf, vi_res.points)
isempty(hopfs) && error("No Hopf point on the branch; there is no " *
                        "predicted firing onset to drive panel (c) with.")
hopf  = argmin(p -> p.I, hopfs)      # onset = the lower of the pair
I_app = hopf.I + I_HOPF_MARGIN

println("Predicted resting potential (I_ss zero crossing): ",
        @sprintf("%.2f mV", V_rest))
println("Hodgkin-Huxley equilibrium branch: detected bifurcations")
for p in vi_res.points
    println("  $(p.type) at I_app = $(round(p.I, digits=3)) µA, " *
            "V = $(round(p.V, digits=3)) mV")
end
println("Firing onset (lower Hopf) at ", @sprintf("%.3f µA", hopf.I),
        "; currentscape driven at ", @sprintf("%.3f µA", I_app))

# ------------------------------------------------------------
# Panels (a) and (b), sharing the resting-potential guide line.
# ------------------------------------------------------------

rest_label = @sprintf("predicted rest %.1f mV", V_rest)

panel_a = plot_iv(iv_res; which = :I_ss)
plot!(panel_a; title = "(a) Summed I-V, steady state (hold $(V_HOLD) mV)",
      xlims = (-100.0, 60.0), ylims = (-200.0, 800.0),
      grid = false, legend = :topleft)
vline!(panel_a, [V_rest]; c = :black, ls = :dot, lw = 1.5, label = rest_label)
scatter!(panel_a, [V_rest], [0.0]; m = :circle, ms = 6, c = :black,
         msw = 1.5, msc = :white, label = "zero crossing")
annotate!(panel_a, V_rest, 0.0,
          text(@sprintf("  %.1f mV", V_rest), 8, :left, :bottom))

panel_b = plot_vi(vi_res; title = "(b) Equilibrium branch V*(I_app)")
plot!(panel_b; grid = false, legend = :bottomright)
hline!(panel_b, [V_rest]; c = :black, ls = :dot, lw = 1.5, label = rest_label)
vline!(panel_b, [I_app]; c = :darkorange, ls = :dash, lw = 1.5,
       label = @sprintf("drive for (c): %.1f µA", I_app))
annotate!(panel_b, hopf.I, hopf.V,
          text(@sprintf("firing onset %.1f µA  ", hopf.I), 8, :right, :bottom))

plt_ab = plot(panel_a, panel_b; layout = (1, 2), size = (1150, 500),
              margin = 20px, dpi = 300,
              plot_title = "HH neuron: predicted rest and firing onset")

# ------------------------------------------------------------
# (c) Currentscape at the drive read off panel (b).
# ------------------------------------------------------------

# Fresh channel systems: the ones above are already wired into the
# continuation network built for panel (b).
@named cs_cap  = Capacitor(C = 1.0)
@named na_cs   = HH.SodiumChannel()
@named k_cs    = HH.PotassiumChannel()
@named leak_cs = HH.LeakChannel()
cs_channels = [na_cs, k_cs, leak_cs]

cs_soma = build_compartment(cs_cap, cs_channels; name = :soma, V_init = -65.0)
net = build_acausal_network([cs_soma]; drivers = [(1, I_app)],
                            name = :hh_onset)
sys  = mtkcompile(net.sys)
prob = ODEProblem(sys, [], (0.0, T_END))
sol  = solve(prob, Rosenbrock23(); reltol = 1e-8, abstol = 1e-8)
sol.retcode == ReturnCode.Success ||
    error("Currentscape simulation failed ($(sol.retcode)) at I_app = $I_app.")

currents, _ = channel_currents(sys, :soma, cs_channels)
# The panel-(c) channels carry `_cs` suffixes to keep them distinct from the
# ones already wired into panel (b); the legend gets the plain names.
labels = ["na", "k", "leak"]
plt_c = plot_currentscape(sol, currents, labels; V = sys.soma.cs_cap.v,
                          size = (800, 700), dpi = 300,
                          plot_title = @sprintf(
                              "(c) Currentscape: I_app = %.1f µA", I_app))

mkpath("figures")
savefig(plt_ab, joinpath("figures", "hh_composite_iv_vi.pdf"))
savefig(plt_ab, joinpath("figures", "hh_composite_iv_vi.png"))
savefig(plt_c,  joinpath("figures", "hh_composite_currentscape.pdf"))
savefig(plt_c,  joinpath("figures", "hh_composite_currentscape.png"))
println("Saved figures/hh_composite_iv_vi.pdf and .png")
println("Saved figures/hh_composite_currentscape.pdf and .png")
