# =============================================================================
# currentscape.jl -- per-channel %% current contribution over time
# =============================================================================
#
# A "currentscape" (see https://github.com/openbraininstitute/Currentscape)
# answers: at each instant during a spiking/current-clamp simulation, what
# fraction of the inward current comes from each channel, and what fraction
# of the outward current comes from each channel?
#
# This is deliberately a pure post-processing/plotting tool: it does not build
# or drive a network. Build and solve your compartment/network exactly as in
# docs/examples/01_single_compartment_HH.jl, then hand the solved `sol` here.
#
# USAGE
# -----
#   include("scripts/currentscape.jl"); using .Currentscape
#   Currentscape.demo()
#
#   Or on your own model:
#     channels = [na_ch, k_ch, leak]
#     soma = build_compartment(cap, channels; name=:soma, topology=top)
#     net  = build_acausal_network([soma]; drivers=[(1, 10.0)])
#     sys  = mtkcompile(net.sys)
#     sol  = solve(ODEProblem(sys, [], (0.0, 50.0)), Rosenbrock23())
#
#     currents, labels = channel_currents(sys, :soma, channels)
#     plot_currentscape(sol, currents, labels; V = sys.soma.cap.v)
# =============================================================================

module Currentscape

using Plots

export channel_currents, plot_currentscape, demo, demo_liu

"""
    channel_currents(sys, compartment_name, channels)

Convenience helper: given the compiled `sys`, the `name::Symbol` of a
compartment built via `build_compartment`, and the same `channels` vector
passed to that call (e.g. `channels = [na_ch, k_ch, leak]`), return
`(currents, labels)` -- the per-channel current symbolic expressions and their
String labels -- ready to pass to [`plot_currentscape`](@ref).
"""
function channel_currents(sys, compartment_name::Symbol, channels)
    comp_sys = getproperty(sys, compartment_name)
    currents = [getproperty(comp_sys, nameof(ch)).i for ch in channels]
    labels   = String.(nameof.(channels))
    return currents, labels
end

"""
    channel_currents(sys, compartment_name, channel_names::AbstractVector{Symbol}; labels=String.(channel_names))

Same idea as [`channel_currents`](@ref), but for neuron-builder functions
(e.g. `LiuCalciumNeuron.build_liu_neuron`) that only return the finished
compartment and not the individual channel components. Pass the `@named`
names given to each channel inside the builder (e.g. `:na_ch`, `:cas_ch`,
...) and this looks up `comp_sys.<name>.i` directly.
"""
function channel_currents(sys, compartment_name::Symbol,
                          channel_names::AbstractVector{Symbol};
                          labels = String.(channel_names))
    comp_sys = getproperty(sys, compartment_name)
    currents = [getproperty(getproperty(comp_sys, cn), :i) for cn in channel_names]
    return currents, String.(labels)
end

"""
    plot_currentscape(sol, currents, labels; V=nothing, t=nothing, colors=palette(:tab10), kwargs...)

Plot the classic Currentscape figure for an already-solved `sol`: an optional
membrane voltage panel on top, followed by a single mirrored stacked-area
panel showing each channel's %% share of the total inward current (stacked
upward from the zero line) and %% share of the total outward current
(stacked downward from the same zero line), the way the original Python
Currentscape package does it.

`currents` is a vector of per-channel current symbolic expressions (see
[`channel_currents`](@ref) for a shortcut), and `labels` their names.

The solution is resampled onto a uniform time grid `t` (default: 500 points
spanning `sol.t`) since adaptive-solver time points are non-uniform and area
stacking needs a common grid.
"""
function plot_currentscape(sol, currents, labels;
                           V = nothing, t = nothing,
                           colors = palette(:tab10), kwargs...)
    ts = t === nothing ? range(sol.t[1], sol.t[end]; length = 500) : t

    I = reduce(hcat, [sol(ts; idxs = c).u for c in currents])  # n_t x n_ch

    # This toolkit's channel convention is i ~ g*(v - E_rev) (see README section
    # 4), and D(v) = -sum(i_channels)/C -- so a channel current is inward
    # (depolarizing) when negative, outward (hyperpolarizing) when positive.
    inward  = max.(-I, 0.0)
    outward = max.(I, 0.0)
    inward_frac  = inward  ./ max.(sum(inward;  dims = 2), eps())
    outward_frac = outward ./ max.(sum(outward; dims = 2), eps())

    panels = []

    if V !== nothing
        Vt = sol(ts; idxs = V).u
        push!(panels, plot(ts, Vt; ylabel = "V (mV)", legend = false,
                           lw = 1.5, c = :black))
    end

    # One mirrored panel instead of two stacked boxes: inward is stacked
    # upward from 0, outward is stacked downward (negated) from the same 0,
    # so both share one continuous axis with the zero line in the middle --
    # matching the Python Currentscape package's layout.
    push!(panels, currentscape_panel(ts, inward_frac, outward_frac, labels,
                                     colors; legend = false))

    n = length(panels)

    # Only the bottom panel needs an x label; repeating it on every panel is
    # just visual noise once the axes are stacked. Pin xlims explicitly
    # (rather than using Plots' `link = :x`) so the shared range can't get
    # clobbered by the dummy NaN data in the legend swatch below.
    xl = extrema(ts)
    for (i, p) in enumerate(panels)
        plot!(p; xlabel = i == n ? "Time (ms)" : "", xlims = xl,
              left_margin = 6Plots.mm)
    end

    # Every row gets its own right-hand cell: a truly blank one for every
    # panel except the last (currentscape), which gets Plots' own native
    # legend (box, color swatches, automatic compact spacing -- no manual
    # positioning math needed). Since it's a separate cell only in that row,
    # it can't stretch up alongside the V panel the way a single
    # full-height legend column would.
    cells = Any[]
    for (i, p) in enumerate(panels)
        push!(cells, p)
        push!(cells, i == n ? legend_swatch(labels, colors) : blank_panel())
    end

    l = grid(n, 2; widths = [0.86, 0.14])

    return plot(cells...; layout = l, kwargs...)
end

"""
    legend_swatch(labels, colors)

Internal helper: Plots.jl's own native legend (box, color line-swatch, and
label per channel), rendered on an otherwise-blank subplot. Using the real
legend machinery -- rather than hand-placing text -- means it gets a proper
border and compact, auto-sized spacing for free.
"""
function legend_swatch(labels, colors)
    plt = plot(; framestyle = :none, legend = :left, grid = false,
               ticks = nothing)
    for (i, l) in enumerate(labels)
        plot!(plt, [NaN], [NaN]; label = l, c = colors[i], lw = 4)
    end
    return plt
end

"""
    blank_panel()

Internal helper: a completely empty, invisible subplot -- used as filler in
every row of the layout except the one carrying the legend, so the legend
only occupies its own row instead of a column spanning the whole figure.
"""
function blank_panel()
    return plot(; framestyle = :none, legend = false, grid = false,
               ticks = nothing)
end

"""
    currentscape_panel(ts, inward_frac, outward_frac, labels, colors; kwargs...)

Internal helper: draw the classic mirrored Currentscape panel. `inward_frac`
and `outward_frac` are each `n_t x n_ch` matrices of per-channel fractions
(rows summing to 1) at each time in `ts`. Inward channels are stacked upward
from `y = 0`; outward channels are stacked downward (negated) from the same
`y = 0`, using Plots.jl's native `fillrange` -- so no extra plotting
dependency (e.g. StatsPlots) is required. Channel `c` gets the same color
(`colors[c]`) whether it appears above or below the line, matching the
original Python package's convention.
"""
function currentscape_panel(ts, inward_frac, outward_frac, labels, colors; kwargs...)
    n_ch = size(inward_frac, 2)
    cum_in  = cumsum(inward_frac;  dims = 2)
    cum_out = cumsum(outward_frac; dims = 2)

    plt = plot(; ylabel = "% inward / outward", ylims = (-1.05, 1.05),
               yticks = (-1:0.5:1, ["100", "50", "0", "50", "100"]),
               kwargs...)

    prev = zeros(length(ts))
    for c in 1:n_ch
        plot!(plt, ts, cum_in[:, c]; fillrange = prev, label = labels[c],
              c = colors[c], lw = 0)
        prev = cum_in[:, c]
    end

    prev = zeros(length(ts))
    for c in 1:n_ch
        plot!(plt, ts, -cum_out[:, c]; fillrange = -prev, label = "",
              c = colors[c], lw = 0)
        prev = cum_out[:, c]
    end

    hline!(plt, [0.0]; c = :black, lw = 1, label = "")
    return plt
end


# -----------------------------------------------------------------------------
# Demo: standard HH single-compartment neuron, step-current driven.
# -----------------------------------------------------------------------------
using MTKNeuralToolkit
using MTKNeuralToolkit: HodgkinHuxley
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq

function demo()
    top = Scalar()

    @named cap  = Capacitor(topology = top, C = 1.0)
    @named na   = HodgkinHuxley.SodiumChannel()
    @named k    = HodgkinHuxley.PotassiumChannel()
    @named leak = HodgkinHuxley.LeakChannel()
    channels = [na, k, leak]

    soma = build_compartment(cap, channels; name = :soma, V_init = -65.0,
                             topology = top)
    net  = build_acausal_network([soma]; drivers = [(1, 10.0)],
                                 name = :currentscape_demo)

    sys  = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 50.0))
    sol  = solve(prob, Rosenbrock23(); reltol = 1e-6, abstol = 1e-6)

    currents, labels = channel_currents(sys, :soma, channels)
    V = sys.soma.cap.v

    return plot_currentscape(sol, currents, labels; V = V, size = (800, 700))
end

# -----------------------------------------------------------------------------
# Demo 2: Liu AB neuron -- 8 channels (Na, CaS, CaT, Ih, Ka, KCa, Kdr, leak)
# plus calcium tracking, so a much busier currentscape than the plain HH demo.
# It's a crustacean-STG pacemaker cell, so it bursts on its own -- no driver
# current needed.
# -----------------------------------------------------------------------------
using MTKNeuralToolkit: LiuCalciumNeuron

function demo_liu()
    comp = LiuCalciumNeuron.build_liu_neuron(name = :soma)
    net  = build_acausal_network([comp]; drivers = [(1, 0.0)],
                                 name = :currentscape_liu_demo)

    sys  = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 1000.0))  # ms -- long enough to see a few burst cycles
    sol  = solve(prob, Rosenbrock23(); reltol = 1e-6, abstol = 1e-6)

    channel_names = [:na_ch, :cas_ch, :cat_ch, :ih_ch, :ka_ch, :kca_ch, :kdr_ch, :leak]
    labels         = ["na", "cas", "cat", "ih", "ka", "kca", "kdr", "leak"]
    currents, labels = channel_currents(sys, :soma, channel_names; labels = labels)
    V = sys.soma.cap.v

    return plot_currentscape(sol, currents, labels; V = V, size = (900, 700))
end

end