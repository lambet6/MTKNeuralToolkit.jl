# ==========================================
# Currentscape Plotting
# ==========================================
module Currentscape

using Plots

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
    plot_currentscape(sol, currents, labels; V=nothing, t=nothing, colors=palette(:tab10), kwargs...)

Plots a "currentscape": at every timepoint, each channel's contribution to
the total inward current and to the total outward current is stacked as a
fraction (0-100%) of that total, so you can see which channels *dominate*
the membrane current at each moment, independent of its absolute size.

# Arguments
- `sol`: An `ODESolution` from a compiled network.
- `currents`: Symbolic current variables, e.g. from `channel_currents`.
- `labels`: `String` names for `currents`, used in the legend.
- `V`: Optional symbolic voltage variable, plotted in a panel above the currentscape.
- `t`: Optional time grid to sample `sol` on (defaults to 500 points over `sol`'s span).
- `colors`: A color palette, one color per channel.

# Returns
- A `Plots.Plot` combining the voltage panel (if any), the currentscape, and a legend.
"""
function plot_currentscape(sol, currents, labels;
                           V = nothing, t = nothing,
                           colors = palette(:tab10), kwargs...)
    ts = t === nothing ? range(sol.t[1], sol.t[end]; length = 500) : t

    I = reduce(hcat, [sol(ts; idxs = c).u for c in currents])

    # Channels follow toolkit's convention, so a channel current
    #  is inward when negative, outward when positive.
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

    push!(panels, currentscape_panel(ts, inward_frac, outward_frac, labels,
                                     colors; legend = false))

    n = length(panels)

    xl = extrema(ts)
    for (i, p) in enumerate(panels)
        plot!(p; xlabel = i == n ? "Time (ms)" : "", xlims = xl,
              left_margin = 6Plots.mm)
    end

    cells = Any[]
    for (i, p) in enumerate(panels)
        push!(cells, p)
        push!(cells, i == n ? legend_swatch(labels, colors) : blank_panel())
    end

    l = grid(n, 2; widths = [0.86, 0.14])

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
    # === Build a standard HH compartment ===
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

    # === Compile and simulate ===
    sys  = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 50.0))
    sol  = solve(prob, Rosenbrock23(); reltol = 1e-6, abstol = 1e-6)

    # === Extract channel currents and plot the currentscape ===
    currents, labels = channel_currents(sys, :soma, channels)
    V = sys.soma.cap.v

    return plot_currentscape(sol, currents, labels; V = V, size = (800, 700))
end


using MTKNeuralToolkit: LiuCalciumNeuron

# Same as demo(), but for the prebuilt Liu et al. crustacean stomatogastric
# neuron model (8 channels), to show a currentscape on a busier cell.
function demo_liu()
    comp = LiuCalciumNeuron.build_liu_neuron(name = :soma)
    net  = build_acausal_network([comp]; drivers = [(1, 0.0)],
                                 name = :currentscape_liu_demo)

    sys  = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 750.0))
    sol  = solve(prob, Rosenbrock23(); reltol = 1e-6, abstol = 1e-6)

    channel_names = [:na_ch, :cas_ch, :cat_ch, :ih_ch, :ka_ch, :kca_ch, :kdr_ch, :leak]
    labels         = ["na", "cas", "cat", "ih", "ka", "kca", "kdr", "leak"]
    currents, labels = channel_currents(sys, :soma, channel_names; labels = labels)
    V = sys.soma.cap.v

    return plot_currentscape(sol, currents, labels; V = V, size = (900, 700))
end

end