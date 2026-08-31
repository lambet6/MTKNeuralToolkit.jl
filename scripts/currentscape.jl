# ==========================================
# Currentscape Plotting
# ==========================================
module Currentscape

using Plots
using Printf: @sprintf

export channel_currents, plot_currentscape, demo, demo_liu


"""
    channel_currents(sys, compartment_name, channels)

Pulls out the `.i` (current) variable of each channel in `channels` from a
compiled network `sys`, ready to hand to `plot_currentscape`.

# Arguments
- `sys`: The compiled (`mtkcompile`d) network system.
- `compartment_name::Symbol`: The name of the compartment the channels live in.
- `channels`: The channel systems (e.g. as passed to `build_compartment`).

# Returns
- `(currents, labels)`: a vector of symbolic current variables and a matching
  vector of `String` names.
"""
function channel_currents(sys, compartment_name::Symbol, channels)
    comp_sys = getproperty(sys, compartment_name)
    currents = [getproperty(comp_sys, nameof(ch)).i for ch in channels]
    labels   = String.(nameof.(channels))
    return currents, labels
end


"""
    channel_currents(sys, compartment_name, channel_names; labels=String.(channel_names))

Variant that looks channels up by `Symbol` name instead of passing the
channel systems themselves -- handy when you only have the compiled `sys`.
"""
function channel_currents(sys, compartment_name::Symbol,
                          channel_names::AbstractVector{Symbol};
                          labels = String.(channel_names))
    comp_sys = getproperty(sys, compartment_name)
    currents = [getproperty(getproperty(comp_sys, cn), :i) for cn in channel_names]
    return currents, String.(labels)
end


"""
    plot_currentscape(sol, currents, labels; V=nothing, colors=palette(:tab10),
                      units="[µA/cm²]", kwargs...)

Plots a "currentscape": at every timepoint, each channel's contribution to
the total inward current and to the total outward current is stacked as a
fraction (0-100%) of that total, so you can see which channels *dominate*
the membrane current at each moment, independent of its absolute size. 
Above and below are two black, log-scale area panels showing the *absolute* 
total outward (above) and total inward (below) current.

# Arguments
- `sol`: An `ODESolution` from a compiled network.
- `currents`: Symbolic current variables, e.g. from `channel_currents`.
- `labels`: `String` names for `currents`, used in the legend.
- `V`: Optional symbolic voltage variable, plotted in a panel above the currentscape.
- `colors`: A color palette, one color per channel.
- `units`: Unit label for the total-current panels.

# Returns
- A `Plots.Plot` combining the voltage panel (if any), the total outward-current panel,
  the currentscape, the total inward-current panel, and a legend.
"""
function plot_currentscape(sol, currents, labels;
                           V = nothing,
                           colors = palette(:tab10), units = "[µA/cm²]", kwargs...)
    ts = sol.t

    I = reduce(hcat, [sol[c] for c in currents])

    # Channels follow toolkit's convention, so a channel current
    # is inward when negative, outward when positive.
    inward  = max.(-I, 0.0)
    outward = max.(I, 0.0)
    inward_frac  = inward  ./ max.(sum(inward;  dims = 2), eps())
    outward_frac = outward ./ max.(sum(outward; dims = 2), eps())

    inward_sum  = vec(sum(inward;  dims = 2))
    outward_sum = vec(sum(outward; dims = 2))
    sum_ylim, sum_ticks = autoscale_current_ticks(outward_sum, inward_sum)

    panels = []

    if V !== nothing
        Vt = sol[V]
        push!(panels, plot(ts, Vt; ylabel = "V (mV)", legend = false,
                           lw = 1.5, c = :black, grid = false))
    end

    push!(panels, current_sum_panel(ts, outward_sum, sum_ylim, sum_ticks;
                                    flip = false, units = units))

    legend_row = length(panels) + 1
    push!(panels, currentscape_panel(ts, inward_frac, outward_frac, labels,
                                     colors; legend = false, grid = false))

    push!(panels, current_sum_panel(ts, inward_sum, sum_ylim, sum_ticks;
                                    flip = true, units = "-" * units))

    n = length(panels)

    xl = extrema(ts)
    for (i, p) in enumerate(panels)
        is_last = i == n
        plot!(p; xlabel = is_last ? "Time (ms)" : "", xlims = xl,
              xaxis = is_last, left_margin = 6Plots.mm)
    end

    cells = Any[]
    for (i, p) in enumerate(panels)
        push!(cells, p)
        push!(cells, i == legend_row ? legend_swatch(labels, colors) : blank_panel())
    end

    heights = V === nothing ? [1 / 6, 4 / 6, 1 / 6] : [2 / 8, 1 / 8, 4 / 8, 1 / 8]
    l = grid(n, 2; widths = [0.86, 0.14], heights = heights)

    return plot(cells...; layout = l, kwargs...)
end

# ==========================================
# Internal Plotting Helpers
# ==========================================

"""
Internal helper: builds a frameless panel holding only a color-coded legend
for `labels`, placed alongside a currentscape panel in the plot grid.
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
Internal helper: an empty, frameless panel used as a spacer in the plot grid.
"""
function blank_panel()
    return plot(; framestyle = :none, legend = false, grid = false,
               ticks = nothing)
end

"""
Internal helper: rounds `n` down to its leading significant digit
(e.g. 723 -> 700, 0.0456 -> 0.04), matching the tick/ylim heuristic used by
the python `currentscape` package's `round_down_sig_digit`.
"""
function round_down_sig_digit(n)
    n <= 0 && return 0.0
    mag = 10.0^floor(log10(n))
    return mag * floor(n / mag)
end

"""
Internal helper: picks a log-scale ylim and three decade-spaced ticks for
the total inward/outward current panels from the largest current magnitude
seen across `pos_sum` and `neg_sum`, mirroring the python `currentscape`
package's `autoscale_ticks_and_ylim`.
"""
function autoscale_current_ticks(pos_sum, neg_sum)
    maxi = max(maximum(pos_sum), maximum(neg_sum), eps())
    ylim = (5.0 * maxi / 1e7, 5.0 * maxi)
    sig = round_down_sig_digit(maxi)
    ticks = [sig * 1e-5, sig * 1e-3, sig * 1e-1]
    return ylim, ticks
end

"""
Internal helper: formats a tick value the way python's `"%g"` formatter
would (e.g. `400`, `4`, `0.04`), trimming trailing zeros.
"""
function format_tick(x)
    s = @sprintf("%g", x)
    return s
end

"""
Internal helper: draws one of the black, log-scale total-current panels
(outward above the currentscape, inward below), matching the style of the
python `currentscape` package's `plot_sum`. `flip=true` mirrors the y-axis
so that small values sit next to the currentscape stack and large values
sit away from it (used for the inward/bottom panel).
"""
function current_sum_panel(ts, curr_sum, ylim, ticks; flip = false, units = "[µA/cm²]", kwargs...)
    clamped = max.(curr_sum, ylim[1])

    plt = plot(; yscale = :log10, ylims = ylim, yflip = flip,
               ylabel = units, legend = false, grid = false,
               yticks = (ticks, format_tick.(ticks)), kwargs...)

    hline!(plt, ticks; c = :black, ls = :dot, lw = 1, label = "")
    plot!(plt, ts, clamped; fillrange = ylim[1], c = :black, lw = 0,
          fillalpha = 1, label = "")

    return plt
end

"""
Internal helper: draws the stacked-area currentscape itself from precomputed
per-channel inward/outward fractions (`inward_frac`, `outward_frac`), inward
stacked above zero and outward stacked below.
"""
function currentscape_panel(ts, inward_frac, outward_frac, labels, colors; kwargs...)
    n_ch = size(inward_frac, 2)
 
    plt = plot(; ylabel = "% inward / outward", ylims = (-1.05, 1.05),
               yticks = (-1:0.5:1, ["100", "50", "0", "50", "100"]),
               kwargs...)
 
    order = n_ch:-1:1
 
    # Outward currentsstacked above zero.
    prev = zeros(length(ts))
    cum  = zeros(length(ts))
    for c in order
        cum = cum .+ outward_frac[:, c]
        plot!(plt, ts, cum; fillrange = prev, label = labels[c],
              c = colors[c], lw = 0)
        prev = cum
    end
 
    # Inward currents stacked below zero.
    prev = zeros(length(ts))
    cum  = zeros(length(ts))
    for c in order
        cum = cum .+ inward_frac[:, c]
        plot!(plt, ts, -cum; fillrange = -prev, label = "",
              c = colors[c], lw = 0)
        prev = cum
    end
 
    hline!(plt, [0.0]; c = :black, lw = 1, label = "")
    return plt
end

# ==========================================
# Demos
# ==========================================

using MTKNeuralToolkit
using MTKNeuralToolkit: HodgkinHuxley
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq

function demo()
    @named cap  = Capacitor()
    @named na   = HodgkinHuxley.SodiumChannel()
    @named k    = HodgkinHuxley.PotassiumChannel()
    @named leak = HodgkinHuxley.LeakChannel()
    channels = [na, k, leak]

    soma = build_compartment(cap, channels; name = :soma, V_init = -65.0)
    net  = build_acausal_network([soma]; drivers = [(1, 10.0)],
                                 name = :currentscape_demo)

    # === Compile and simulate ===
    sys  = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 75.0))
    sol  = solve(prob, Rosenbrock23(); reltol = 1e-6, abstol = 1e-6)

    # === Extract channel currents and plot the currentscape ===
    currents, labels = channel_currents(sys, :soma, channels)
    V = sys.soma.cap.v

    return plot_currentscape(sol, currents, labels; V = V, size = (800, 700))
end


using MTKNeuralToolkit: LiuCalciumNeuron
function demo_liu()
    comp = LiuCalciumNeuron.build_liu_neuron(name = :soma)
    net  = build_acausal_network([comp]; drivers = [(1, 0.0)],
                                 name = :currentscape_liu_demo)

    sys  = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 750.0))
    sol  = solve(prob, Rosenbrock23(); reltol = 1e-6, abstol = 1e-6)

    channel_names = [:na_ch, :cas_ch, :cat_ch, :ih_ch, :ka_ch, :kca_ch, :kdr_ch, :leak]
    # labels used for legend in the currentscape plot
    labels         = ["na", "cas", "cat", "ih", "ka", "kca", "kdr", "leak"]                 
    currents, labels = channel_currents(sys, :soma, channel_names; labels = labels)
    V = sys.soma.cap.v

    return plot_currentscape(sol, currents, labels; V = V, size = (900, 700))
end

end