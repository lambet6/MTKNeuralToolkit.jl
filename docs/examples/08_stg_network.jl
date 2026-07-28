# # Example 8: Stomatogastric Ganglion (STG) Network
# 
# This example reconstructs a simplified 3-neuron network (AB, LP, PY) based on 
# the classic stomatogastric ganglion models (e.g., Prinz et al., 2004). 
# It demonstrates the use of Calcium channels, Calcium trackers, custom geometries, 
# and multiple synapse types (Cholinergic and Glutamatergic) in a single network.

using MTKNeuralToolkit
using MTKNeuralToolkit.PrinzNeuron
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq, Plots

# ## Network Parameters & Geometry
# Prinz uses a custom geometry to scale capacitance, conductances, and calcium 
# flow. We instantiate it here and define shared Calcium parameters.
# Note that the custom geometry occupies < 10 lines of code in the src/library/PrinzCalciumNeuron.jl file. Geometries are easy to make.
const geom = PrinzGeometry(area=0.0628, C_m=10.0)
const tauCa = 200.0
const Ca_inf = 0.05
const prinz_ion_config = CalciumTracker(decay=ca -> (Ca_inf .- ca) ./ tauCa, Ca_init=Ca_inf)
const nernst_factor = 500.0 * 8.6174e-5 * 283.15

# ## Local Channel Builders
# We use closures to neatly attach the geometry, reversal potentials, and tauCa 
# to the pre-built Prinz gate definitions. This keeps the neuron builder clean.
NaCh(g; name)   = GenericChannel(name=name, g=g, E_rev=50.0, gates=PrinzNeuron.na_gates, geometry=geom)
CaSCh(g; name)  = CaVChannel(name=name, g=g, gates=PrinzNeuron.cas_gates, Ca_out=3000.0, 
                             nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
CaTCh(g; name)  = CaVChannel(name=name, g=g, gates=PrinzNeuron.cat_gates, Ca_out=3000.0, 
                             nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
HCh(g; name)    = GenericChannel(name=name, g=g, E_rev=-20.0, gates=PrinzNeuron.h_gates, geometry=geom)
KaCh(g; name)   = GenericChannel(name=name, g=g, E_rev=-80.0, gates=PrinzNeuron.ka_gates, geometry=geom)
KCaCh(g; name)  = KCaChannel(name=name, g=g, E_rev=-80.0, gates=PrinzNeuron.kca_gates, geometry=geom)
KdrCh(g; name)  = GenericChannel(name=name, g=g, E_rev=-80.0, gates=PrinzNeuron.kdr_gates, geometry=geom)
LeakCh(g; name) = GenericChannel(name=name, g=g, E_rev=-50.0, gates=GateSpec[], geometry=geom)

# ## Build Neurons
function build_AB()
    @named cap  = Capacitor(geometry=geom)
    @named na   = NaCh(100.0); @named cas  = CaSCh(6.0);  @named cat = CaTCh(2.5)
    @named h    = HCh(0.01);   @named ka   = KaCh(50.0);  @named kca = KCaCh(5.0)
    @named kdr  = KdrCh(100.0)
    return build_compartment(cap, [na, cas, cat, h, ka, kca, kdr]; 
                             name=:AB, V_init=-60.0, ion_config=prinz_ion_config)
end

function build_PY()
    @named cap  = Capacitor(geometry=geom)
    @named na   = NaCh(100.0); @named cas  = CaSCh(2.0);  @named cat = CaTCh(2.4)
    @named h    = HCh(0.05);   @named ka   = KaCh(50.0);  @named kdr = KdrCh(125.0)
    @named leak = LeakCh(0.01)
    return build_compartment(cap, [na, cas, cat, h, ka, kdr, leak]; 
                             name=:PY, V_init=-55.0, ion_config=prinz_ion_config)
end

function build_LP()
    @named cap  = Capacitor(geometry=geom)
    @named na   = NaCh(100.0); @named cas  = CaSCh(4.0)
    @named h    = HCh(0.05);   @named ka   = KaCh(20.0);  @named kdr = KdrCh(25.0)
    @named leak = LeakCh(0.03)
    return build_compartment(cap, [na, cas, h, ka, kdr, leak]; 
                             name=:LP, V_init=-65.0, ion_config=prinz_ion_config)
end

# Assign to variables so STG_synapses() can reference them in the global scope
AB = build_AB()
PY = build_PY()
LP = build_LP()

neurons = [AB, PY, LP]


# ## Define Synapses & Network
function STG_synapses()
    @named ABLP_chol = CholSynapse(g_max=30.0, geometry=geom)
    @named ABPY_chol = CholSynapse(g_max=3.0 , geometry=geom)
    @named ABLP_glut = GlutSynapse(g_max=30.0, geometry=geom)
    @named ABPY_glut = GlutSynapse(g_max=10.0, geometry=geom)
    @named LPAB_glut = GlutSynapse(g_max=30.0, geometry=geom)
    @named LPPY_glut = GlutSynapse(g_max=1.0 , geometry=geom)
    @named PYLP_glut = GlutSynapse(g_max=30.0, geometry=geom)

    return [
        SynapseSpec(LP.interfaces.V, AB.interfaces.V, AB.interfaces.I_syn, LPAB_glut),
        SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_chol),
        SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_glut),
        SynapseSpec(LP.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, LPPY_glut),
        SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_chol),
        SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_glut),
        SynapseSpec(PY.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, PYLP_glut)
    ]
end

net = build_acausal_network(neurons; synapse_specs=STG_synapses(), name=:stg)
println("Compiling STG network...")
sys = mtkcompile(net.sys)

# ## Simulate & Plot
# STG networks often need a few seconds to settle into their characteristic 
# alternating rhythm (pyloric rhythm). We simulate for 10000 ms.
tspan = (0.0, 10000.0)
prob = ODEProblem(sys, [], tspan, jac=true, sparse=true)

println("Solving STG network...")
sol = solve(prob, Rosenbrock23())

p1 = plot(sol, idxs=[sys.AB.cap.v], title="AB Neuron", legend=false, ylabel="V (mV)")
p2 = plot(sol, idxs=[sys.LP.cap.v], title="LP Neuron", legend=false, ylabel="V (mV)")
p3 = plot(sol, idxs=[sys.PY.cap.v], title="PY Neuron", legend=false, ylabel="V (mV)", xlabel="Time (ms)")

plot(p1, p2, p3, layout=(3,1), size=(800,600))
