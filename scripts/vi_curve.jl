# ==========================================
# Current-Clamp V-I Curve
# ==========================================
module VICurves

using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, unknowns, parameters, getdefault, @named
using ModelingToolkitStandardLibrary.Blocks: Constant
using OrdinaryDiffEq
using BifurcationKit
using Plots
using Plots.PlotMeasures: px

export vi_curve, plot_vi


"""
    vi_curve(comp; currents=-10.0:200.0, I_start=minimum(currents), t_settle=2000.0, solver=Rosenbrock23(), reltol=1e-8, abstol=1e-8, ds=0.1, dsmax=1.0, max_steps=2000, jac=false)

Trace the steady-state V-I curve of a current-clamped `Compartment` by numerical
continuation: the applied current `I_app` is the bifurcation parameter and the
membrane voltage is the recorded variable, so the result is the equilibrium
branch `V*(I_app)`, each point tagged stable or unstable, with every fold and
Hopf point located along the way.

`comp` is whatever `build_compartment` gives you (or a prebuilt neuron such as
`LiuCalciumNeuron.build_liu_neuron()`).

To begin the V-I sweep, we first find a real resting state by running the model 
for t_settle ms at I_start. Choose an I_start where the cell is at rest (usually 
the more negative end of currents). If the cell is already spiking there, there 
is no stable resting state to start from, and the calculation will fail.

# Arguments
- `comp`: The `Compartment` to current-clamp.
- `currents`: The applied-current range to sweep; only its extrema are used, as
  the continuation's `p_min`/`p_max`.
- `I_start`: Where the sweep is seeded (continuation runs in both directions).
- `t_settle`: How long to integrate to find the seeding steady state (ms).
- `solver`, `reltol`, `abstol`: Passed through to the settling `solve`.
- `ds`, `dsmax`, `max_steps`: Continuation step size and budget.
- `jac`: Pass `true` to hand BifurcationKit a symbolic Jacobian. The default
  `false` lets it use `ForwardDiff`, which is more robust for the exponentials
  in typical gating functions and still gives eigenvalues (hence stability).

# Returns
A `NamedTuple` with `I`, `V` and `stable` (parallel vectors, one entry per
continuation step, **in branch order** -- deliberately not sorted by `I`, since
a folded branch has several voltages at the same current), `points` (the
detected bifurcations as `(type, I, V)`), plus the raw BifurcationKit result
`br` and the compiled system `sys`.
"""
function vi_curve(comp::Compartment;
                  currents  = -10.0:200.0,
                  I_start   = minimum(currents),
                  t_settle  = 2000.0,
                  solver    = Rosenbrock23(),
                  reltol    = 1e-8,
                  abstol    = 1e-8,
                  ds        = 0.1,
                  dsmax     = 1.0,
                  max_steps = 2000,
                  jac       = false)

    I_min, I_max = extrema(float.(currents))
    I_min <= I_start <= I_max ||
        error("I_start = $I_start lies outside currents = [$I_min, $I_max].")

    # 1. We must wrap the injected current in a named `Constant` block so the current
    # value stays as a real model parameter after compilation. Without this, the
    # network cannot expose `I_app` as a parameter for continuation, and
    # BifurcationKit has no parameter to vary, so it cannot trace the V-I branch.
    @named I_app = Constant(k = I_start)
    net = build_acausal_network([comp]; drivers = [(1, I_app)], name = :vi_net)
    sys = mtkcompile(net.sys)

    I_par  = sys.I_app.k
    V_sym  = getproperty(getproperty(sys, nameof(comp.sys)), comp.interfaces.cap_name).v
    states = unknowns(sys)

    # 2. Seed from a real steady state at I_start.
    prob = ODEProblem(sys, [], (0.0, t_settle))
    sol  = solve(prob, solver; reltol = reltol, abstol = abstol)
    sol.retcode == ReturnCode.Success || error(
        "Settling phase failed to solve ($(sol.retcode)) at I_app = $I_start.")

    u0 = sol.u[end]

    # 3. Continue the equilibrium in I_app across the whole current range.
    # BifurcationProblem wants a value for every parameter, not just the
    # bifurcation one - take the rest from their compiled defaults.
    ps      = filter(!isequal(I_par), parameters(sys))
    p_start = [ps .=> getdefault.(ps); I_par => I_start]

    opts = ContinuationPar(
        p_min = I_min, p_max = I_max,
        ds = ds, dsmax = dsmax,
        detect_bifurcation = 3,          # 3 = bisect to *locate* folds/Hopfs
        nev = length(states),
        max_steps = max_steps)

    br = continuation(
        BifurcationProblem(sys, states .=> u0, p_start, I_par;
                           plot_var = V_sym, jac = jac),
        PALC(), opts; bothside = true)

    points = [(type = sp.type, I = sp.param, V = sp.printsol.x)
              for sp in br.specialpoint if sp.type != :endpoint]

    return (I = br.branch.param, V = br.branch.x, stable = br.branch.stable,
            points = points, br = br, sys = sys)
end


# Draws one branch as alternating solid/dashed segments (solid where `stable`),
# labelling only the first segment so the branch gets a single legend entry no
# matter how many times it changes stability.
function _plot_branch!(plt, x, y, stable; c, lw, label)
    i = firstindex(x)
    first_seg = true
    while i <= lastindex(x)
        j = i
        while j < lastindex(x) && stable[j+1] == stable[i]
            j += 1
        end
        plot!(plt, x[i:j], y[i:j];
              lw = lw, ls = stable[i] ? :solid : :dash, c = c,
              label = first_seg ? label : false)
        first_seg = false
        i = j + 1
    end
    return plt
end

const _POINT_STYLE = Dict(
    :hopf => (:circle,  :red,   "Hopf"),
    :fold => (:diamond, :blue,  "fold (saddle-node)"),
    :bp   => (:diamond, :blue,  "branch point (saddle-node)"),
)

"""
    plot_vi(res; title="V-I curve (current clamp)")

Plot the equilibrium branch from a `vi_curve` result: applied current on x,
membrane voltage on y, solid where the equilibrium is stable and dashed where it
is unstable. Detected bifurcations are marked, one legend entry per type.
"""
function plot_vi(res; title = "V-I curve (current clamp)")
    plt = plot(xlabel = "Applied current (µA/cm²)",
               ylabel = "Steady-state voltage (mV)",
               title  = title,
               legend = :bottomright)

    _plot_branch!(plt, res.I, res.V, res.stable;
                  c = 1, lw = 2, label = "equilibrium (solid = stable)")

    for type in unique(p.type for p in res.points)
        pts = filter(p -> p.type == type, res.points)
        marker, colour, name = get(_POINT_STYLE, type, (:utriangle, :purple, String(type)))
        scatter!(plt, [p.I for p in pts], [p.V for p in pts];
                 marker = marker, ms = 6, c = colour, label = name)
    end
    return plt
end


# ==========================================
# Demos
# ==========================================

const HH = HodgkinHuxley
function demo()
    @named soma_cap = Capacitor(C = 1.0)
    @named na_ch    = HH.SodiumChannel()
    @named k_ch     = HH.PotassiumChannel()
    @named leak     = HH.LeakChannel()

    soma = build_compartment(soma_cap, [na_ch, k_ch, leak];
                             name = :soma, V_init = -65.0)

    res = vi_curve(soma; currents = -20.0:200.0, I_start = -20.0)
    for p in res.points
        println("  $(p.type) at I_app = $(round(p.I, digits=3)) µA/cm², " *
                "V = $(round(p.V, digits=3)) mV")
    end
    return plot_vi(res; title = "Hodgkin-Huxley V-I curve")
end

function demo_morris_lecar()
    @named ml_cap = Capacitor(C = 20.0)
    # `MorrisLecar`'s default g_L = 0.1 leaves the cell almost leak-free, which
    # makes the resting branch so steep it runs off to hundreds of millivolts.
    # g_L = 2.0 is the usual Morris-Lecar value and puts it in the bistable regime.
    ca_ch, k_ch, leak = ContinuousSpikers.MorrisLecar(name = :ml, g_L = 2.0)

    cell = build_compartment(ml_cap, [ca_ch, k_ch, leak];
                             name = :ml_cell, V_init = -60.0)

    res = vi_curve(cell; currents = -60.0:10.0, I_start = -60.0,
                   ds = 0.05, dsmax = 0.5)
    for p in res.points
        println("  $(p.type) at I_app = $(round(p.I, digits=3)), " *
                "V = $(round(p.V, digits=3)) mV")
    end
    return plot_vi(res; title = "Morris-Lecar V-I curve")
end

using MTKNeuralToolkit: LiuCalciumNeuron

function demo_liu()
    comp = LiuCalciumNeuron.build_liu_neuron(name = :soma)

    res = vi_curve(comp; currents = -5.0:0.2, I_start = -5.0,
                   ds = 0.02, dsmax = 0.1)
    for p in res.points
        println("  $(p.type) at I_app = $(round(p.I, digits=3)), " *
                "V = $(round(p.V, digits=3)) mV")
    end
    return plot_vi(res; title = "Liu et al. crustacean STG neuron V-I curve")
end

end
