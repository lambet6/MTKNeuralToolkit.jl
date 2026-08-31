# ============================================================
# Figure 5.6: Summed I-V for the Hodgkin-Huxley neuron.
#
# `iv_curve([na, k, leak]; V_hold = -65.0)` plotted through `plot_iv`
# twice: peak current on the left, steady-state current on the right.
#
# The two panels answer different questions. The peak curve is taken
# before the gates have relaxed, so it still carries the fast sodium
# current: that is where the negative-slope region - the regenerative
# stretch that makes a spike possible - shows up. The steady-state
# curve is taken once every gate has settled, so the sodium current has
# largely inactivated and what is left is the balance that sets the
# resting potential. Neither panel alone tells the whole story, so both
# are shown.
#
# Zero crossings of the total ("neuron") curve are marked in each panel;
# they are the voltages at which the summed membrane current vanishes.
#
# Output:    figures/iv_summed.pdf (and .png)
# ============================================================

include("../iv_curve.jl")

using .IVCurves: iv_curve, plot_iv
using MTKNeuralToolkit: HodgkinHuxley
using ModelingToolkit: @named
using Plots
using Plots.PlotMeasures: px
using Printf: @sprintf

const HH = HodgkinHuxley

# alpha_m is 0/0 at v = -40 and alpha_n at v = -55; the half-mV grid
# offset keeps the sweep clear of both removable singularities.
const VOLTAGES = -99.5:1.0:60.5
const V_HOLD   = -65.0

@named na   = HH.SodiumChannel()
@named k    = HH.PotassiumChannel()
@named leak = HH.LeakChannel()

res = iv_curve([na, k, leak]; voltages = VOLTAGES, V_hold = V_HOLD)

"""
    zero_crossings(V, I)

Voltages where `I` changes sign, found by linear interpolation between
neighbouring grid points (plus any grid point sitting exactly on zero).

Sign changes across a jump discontinuity are rejected. `I_peak` picks the
largest-magnitude sample of the step, so once the outward potassium
plateau overtakes the inward sodium transient the peak curve steps
straight from a large negative value to a large positive one. That jump
crosses zero but is an artefact of the peak metric, not a voltage at
which the membrane current vanishes.
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

neuron_col = findfirst(==(:neuron), res.channel_names)

"""
    panel(which, title)

One `plot_iv` panel with the total curve's zero crossings marked.
"""
function panel(which, title; ylims = :auto)
    plt = plot_iv(res; which = which)
    plot!(plt; title = title, xlims = (-100.0, 60.0), ylims = ylims,
               grid = false, legend = :topleft)

    roots = zero_crossings(res.V, getproperty(res, which)[:, neuron_col])
    isempty(roots) && return plt

    scatter!(plt, roots, zeros(length(roots));
             m = :circle, ms = 2, c = :black, msw = 1.5, msc = :white,
             label = "zero crossing")
    for (n, r) in enumerate(roots)
        annotate!(plt, r, 0.0,
                  text(@sprintf("  %.1f mV", r), 8, :left,
                       n == 1 ? :bottom : :top))
    end
    return plt
end

# The potassium current runs to several thousand microamps at the top of
# the sweep, which flattens everything near rest. Both panels are clipped
# to the range where the crossings and the negative-slope region live.
plt = plot(
    panel(:I_peak, "Peak current (I_peak)"; ylims = (-1600.0, 800.0)),
    panel(:I_ss, "Steady-state current (I_ss)"; ylims = (-200.0, 800.0)),
    panel(:I_ss, "Steady-state zoom"; ylims = (-60.0, 60.0)),
    layout = (1, 3),
    size = (1550, 500),
    margin = 20px,
    dpi = 300,
    plot_title = "Summed I-V, HH neuron (hold $(V_HOLD) mV)",
)

plot!(plt[3]; xlims = (-80.0, -30.0))

mkpath("figures")
savefig(plt, joinpath("figures", "iv_summed.pdf"))
savefig(plt, joinpath("figures", "iv_summed.png"))
println("Saved figures/iv_summed.pdf and .png")

for which in (:I_peak, :I_ss)
    roots = zero_crossings(res.V, getproperty(res, which)[:, neuron_col])
    println("$(which) zero crossings (mV): ",
            join((@sprintf("%.2f", r) for r in roots), ", "))
end
