# ============================================================
# Figure 5.12: Voltage traces of the free Hodgkin-Huxley neuron at
# three applied currents, one per regime of the equilibrium branch of
# Fig 5.8.
#
#   (a) I_app = 5 microA    - below the first Hopf point.
#   (b) I_app = 10.8 microA - between the two Hopf points.
#   (c) I_app = 170 microA  - above the second Hopf point.
#
# The same compartment definition is used for all three; only the
# driver current changes. The panels share an x-axis and y-limits so
# the amplitudes are directly comparable.
#
# On each panel the equilibrium voltage V*(I_app) predicted by the
# branch is drawn as a dotted line. It is *read off* the `vi_curve`
# result by interpolating the continuation branch at that current
# rather than hardcoded, which is what makes this figure a test of
# Fig 5.8: in (a) and (c) the trace should converge onto the dotted
# line (stable equilibrium), while in (b) it should oscillate around
# it (unstable equilibrium, repetitive firing).
#
# Output:    figures/hh_voltage_regimes.pdf (and .png)
# ============================================================

include("../vi_curve.jl")

using .VICurves: vi_curve
using MTKNeuralToolkit: Capacitor, HodgkinHuxley, build_compartment,
                        build_acausal_network
using ModelingToolkit: @named, mtkcompile
using OrdinaryDiffEq
using Plots
using Plots.PlotMeasures: px
using Printf: @sprintf

const HH       = HodgkinHuxley
const CURRENTS = -20.0:200.0
const I_START  = -20.0
const V_INIT   = -65.0
const T_END    = 300.0            # ms; long enough for (a) and (c) to settle
const DRIVES   = (5.0, 10.8, 170.0)
const TAGS     = ("(a)", "(b)", "(c)")

# ------------------------------------------------------------
# The equilibrium branch of Fig 5.8, recomputed here so the guide
# lines come from the continuation rather than from quoted numbers.
# ------------------------------------------------------------

function build_soma(name)
    cap  = Capacitor(C = 1.0, name = Symbol(name, :_cap))
    na   = HH.SodiumChannel(name = Symbol(name, :_na))
    k    = HH.PotassiumChannel(name = Symbol(name, :_k))
    leak = HH.LeakChannel(name = Symbol(name, :_leak))
    return build_compartment(cap, [na, k, leak]; name = name, V_init = V_INIT)
end

vi_res = vi_curve(build_soma(:soma_vi); currents = CURRENTS, I_start = I_START)

println("Hodgkin-Huxley equilibrium branch: detected bifurcations")
for p in vi_res.points
    println("  $(p.type) at I_app = $(round(p.I, digits=3)) µA, " *
            "V = $(round(p.V, digits=3)) mV")
end

"""
    branch_voltage(res, I_target)

The equilibrium voltage the branch predicts at `I_target`, together with
whether that equilibrium is stable. The branch is stored in continuation
order (not sorted by current), so every consecutive pair of steps that
brackets `I_target` is a candidate; for the Hodgkin-Huxley branch, which
has no folds, there is exactly one. The voltage is linearly interpolated
between the bracketing steps and the equilibrium is called stable only if
both of them are.
"""
function branch_voltage(res, I_target)
    hits = NamedTuple{(:V, :stable), Tuple{Float64, Bool}}[]
    for j in 1:(length(res.I) - 1)
        a, b = res.I[j], res.I[j + 1]
        (min(a, b) <= I_target <= max(a, b)) || continue
        w = a == b ? 0.0 : (I_target - a) / (b - a)
        push!(hits, (V = res.V[j] + w * (res.V[j + 1] - res.V[j]),
                     stable = res.stable[j] && res.stable[j + 1]))
    end
    isempty(hits) && error("The branch does not reach I_app = $I_target µA, " *
                           "so it predicts no equilibrium voltage there.")
    length(hits) > 1 && @warn "Branch is multivalued at I_app = $I_target µA; " *
                              "using the first equilibrium found." hits
    return first(hits)
end

# ------------------------------------------------------------
# One free-running simulation per drive, same compartment throughout.
# ------------------------------------------------------------

"""
    trace(I_app, tag)

Simulate the free compartment for `T_END` ms at the constant applied
current `I_app` and return `(t, V)`. A fresh set of channel systems is
built for each run: the ones already wired into a network cannot be
reused in another.
"""
function trace(I_app, tag)
    soma = build_soma(Symbol(:soma_, tag))
    net  = build_acausal_network([soma]; drivers = [(1, I_app)],
                                 name = Symbol(:hh_, tag))
    sys  = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, T_END))
    sol  = solve(prob, Rosenbrock23(); reltol = 1e-8, abstol = 1e-8)
    sol.retcode == ReturnCode.Success ||
        error("Simulation failed ($(sol.retcode)) at I_app = $I_app µA.")
    V_sym = getproperty(getproperty(sys, nameof(soma.sys)),
                        soma.interfaces.cap_name).v
    return sol.t, sol[V_sym]
end

runs = [(I = I, tag = tag, eq = branch_voltage(vi_res, I),
         data = trace(I, Symbol(:p, i)))
        for (i, (I, tag)) in enumerate(zip(DRIVES, TAGS))]

for r in runs
    println(r.tag, " I_app = ", @sprintf("%.1f µA", r.I),
            ": branch predicts V* = ", @sprintf("%.2f mV", r.eq.V),
            r.eq.stable ? " (stable)" : " (unstable)")
end

# Shared y-limits, padded so the guide lines are never on the frame.
V_all = reduce(vcat, [r.data[2] for r in runs])
V_lo, V_hi = extrema(V_all)
pad   = 0.05 * (V_hi - V_lo)
ylims = (V_lo - pad, V_hi + pad)

panels = map(enumerate(runs)) do (i, r)
    last_panel = i == length(runs)
    plt = plot(r.data[1], r.data[2];
               c = 1, lw = 1.4, label = "membrane voltage",
               title = @sprintf("%s I_app = %.1f µA", r.tag, r.I),
               titlelocation = :left, titlefontsize = 10,
               ylabel = "V (mV)",
               xlabel = last_panel ? "Time (ms)" : "",
               xformatter = last_panel ? :auto : (_ -> ""),
               xlims = (0.0, T_END), ylims = ylims,
               grid = false, legend = :topright, legendfontsize = 7)
    hline!(plt, [r.eq.V]; c = :black, ls = :dot, lw = 1.5,
           label = @sprintf("branch V* = %.1f mV (%s)", r.eq.V,
                            r.eq.stable ? "stable" : "unstable"))
    plt
end

plt = plot(panels...;
           layout = (3, 1), size = (760, 850),
           link = :x, margin = 12px, dpi = 300,
           plot_title = "HH voltage traces across the equilibrium branch")

mkpath("figures")
savefig(plt, joinpath("figures", "hh_voltage_regimes.pdf"))
savefig(plt, joinpath("figures", "hh_voltage_regimes.png"))
println("Saved figures/hh_voltage_regimes.pdf and .png")
