This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: src/**/*
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
src/
  components/
    calcium.jl
    channels.jl
    electrical.jl
    synapses.jl
  library/
    ContinuousSpikers.jl
    HodgkinHuxley.jl
    LiuCalciumNeuron.jl
    PrinzCalciumNeuron.jl
  training/
    pem.jl
  geometry.jl
  MTKNeuralToolkit.jl
  network.jl
  topology.jl
```

# Files

## File: src/components/calcium.jl
```julia
@connector function CaPort(; name, topology=Scalar())
    if topology isa Scalar
        vars = @variables begin
            Ca(t)
            J_Ca(t), [connect = Flow]
        end
    else
        vars = @variables begin
            Ca(t)[1:topology.N]
            J_Ca(t)[1:topology.N], [connect = Flow]
        end
    end
    return System(Equation[], t, vars, SymbolicT[]; name=name)
end

@component function CalciumPool(; name, decay=100.0, Ca_init=0.0, topology=Scalar())
    @named port = CaPort(topology=topology)
    
    @parameters tau_Ca = (decay isa Function ? 0.0 : decay)
    
    if topology isa Scalar
        @variables Ca(t)=Ca_init
        vars = SymbolicT[Ca]
        init_conds = Dict(Ca => Ca_init)
    else
        @variables Ca(t)[1:topology.N] = fill(Ca_init, topology.N)
        vars = SymbolicT[Ca]
        init_conds = Dict(Ca => fill(Ca_init, topology.N))
    end

    if decay isa Function
        decay_term = decay(Ca)
    else
        decay_term = .-Ca ./ tau_Ca
    end
    
    eqs = Equation[
        D(Ca) ~ decay_term .+ port.J_Ca,
        port.Ca ~ Ca
    ]
    
    params = decay isa Function ? SymbolicT[] : SymbolicT[tau_Ca]
    
    return System(eqs, t, vars, params; systems=[port], initial_conditions=init_conds, name=name)
end


@component function CaVChannel(; name, g, gates::Vector{<:GateSpec}, topology=Scalar(), 
                               conversion_factor=1.0, E_rev=nothing, Ca_out=3000.0, nernst_factor=13.0,
                               geometry=NoGeometry(), tauCa=nothing)
    g_val = get_conductance(g, geometry)
    conv_val = get_ca_conversion_factor(conversion_factor, geometry, tauCa)
    
    if topology isa Scalar
        @named oneport = OnePort()
        @named ca_port = CaPort(topology=topology)
        @parameters g=g_val conversion_factor=conv_val
    else
        @named oneport = VectorizedOnePort(N=topology.N)
        @named ca_port = CaPort(topology=topology)
        if g_val isa AbstractArray
            @parameters begin
                g[1:topology.N] = g_val
            end
        else
            @parameters g=g_val
        end
        if conv_val isa AbstractArray
            @parameters begin
                conversion_factor[1:topology.N] = conv_val
            end
        else
            @parameters conversion_factor=conv_val
        end
    end
    @unpack v, i = oneport
    
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{SymbolicT, Any}()
    
    params = SymbolicT[g, conversion_factor]
    
    if isnothing(E_rev)
        @parameters Ca_out=Ca_out nernst_factor=nernst_factor
        E_rev_expr = nernst_factor .* log.(Ca_out ./ ca_port.Ca)
        push!(params, Ca_out, nernst_factor)
    else
        if topology isa Scalar
            @parameters E_rev=E_rev
        else
            if E_rev isa AbstractArray
                @parameters begin
                    E_rev[1:topology.N] = E_rev
                end
            else
                @parameters E_rev=E_rev
            end
        end
        E_rev_expr = E_rev
        push!(params, E_rev)
    end
    
    conductance_factor = true
    for gate in gates
        if topology isa Scalar
            gate_var = only(@variables $(gate.name)(t))
            alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t))
            beta_var = only(@variables $(Symbol(gate.name, :_beta))(t))
            init_conds[gate_var] = gate.ic
        else
            gate_var = only(@variables $(gate.name)(t)[1:topology.N])
            alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t)[1:topology.N])
            beta_var = only(@variables $(Symbol(gate.name, :_beta))(t)[1:topology.N])
            init_conds[gate_var] = fill(gate.ic, topology.N)
        end
        
        push!(vars, gate_var, alpha_var, beta_var)
        alpha_expr, beta_expr = gate.dynamics(v)
        
        push!(eqs, alpha_var ~ alpha_expr)
        push!(eqs, beta_var ~ beta_expr)
        push!(eqs, D(gate_var) ~ alpha_expr .* (1.0 .- gate_var) .- beta_expr .* gate_var)
        conductance_factor = conductance_factor .* (gate_var .^ gate.power)
    end
    
    push!(eqs, i ~ g .* conductance_factor .* (v .- E_rev_expr))
    push!(eqs, ca_port.J_Ca ~ conversion_factor .* i)
    
    return extend(System(eqs, t, vars, params; 
                       systems=[ca_port], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end

@component function KCaChannel(; name, g, E_rev, gates::Vector{<:GateSpec}, topology=Scalar(), geometry=NoGeometry())
    g_val = get_conductance(g, geometry)
    
    if topology isa Scalar
        @named oneport = OnePort()
        @named ca_port = CaPort(topology=topology)
        @parameters g=g_val E_rev=E_rev
    else
        @named oneport = VectorizedOnePort(N=topology.N)
        @named ca_port = CaPort(topology=topology)
        if g_val isa AbstractArray
            @parameters begin
                g[1:topology.N] = g_val
            end
        else
            @parameters g=g_val
        end
        if E_rev isa AbstractArray
            @parameters begin
                E_rev[1:topology.N] = E_rev
            end
        else
            @parameters E_rev=E_rev
        end
    end
    @unpack v, i = oneport
    
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{SymbolicT, Any}()
    
    push!(eqs, ca_port.J_Ca ~ ground_current(topology))
    
    conductance_factor = true
    for gate in gates
        if topology isa Scalar
            gate_var = only(@variables $(gate.name)(t))
            alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t))
            beta_var = only(@variables $(Symbol(gate.name, :_beta))(t))
            init_conds[gate_var] = gate.ic
        else
            gate_var = only(@variables $(gate.name)(t)[1:topology.N])
            alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t)[1:topology.N])
            beta_var = only(@variables $(Symbol(gate.name, :_beta))(t)[1:topology.N])
            init_conds[gate_var] = fill(gate.ic, topology.N)
        end
        
        push!(vars, gate_var, alpha_var, beta_var)
        
        alpha_expr, beta_expr = gate.dynamics(v, ca_port.Ca)
        
        push!(eqs, alpha_var ~ alpha_expr)
        push!(eqs, beta_var ~ beta_expr)
        push!(eqs, D(gate_var) ~ alpha_expr .* (1.0 .- gate_var) .- beta_expr .* gate_var)
        conductance_factor = conductance_factor .* (gate_var .^ gate.power)
    end
    
    push!(eqs, i ~ g .* conductance_factor .* (v .- E_rev))
    
    return extend(System(eqs, t, vars, [g, E_rev]; 
                       systems=[ca_port], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end
```

## File: src/components/channels.jl
```julia
struct GateSpec{I<:Integer, T<:AbstractFloat, F<:Function}
    name::Symbol
    power::I
    ic::T
    dynamics::F 
end

# Convert (inf, tau) -> (alpha, beta) where alpha = inf/tau and beta = (1-inf)/tau

InfTau(inf_fn, tau_fn) = v -> (inf_fn(v) ./ tau_fn(v), (1.0 .- inf_fn(v)) ./ tau_fn(v))
InfTauCa(inf_fn, tau_fn) = (v, ca) -> (inf_fn(v, ca) ./ tau_fn(v), (1.0 .- inf_fn(v, ca)) ./ tau_fn(v))

@component function GenericChannel(; name, g, E_rev, gates::Vector{<:GateSpec}, topology=Scalar(), geometry=NoGeometry())
    g_val = get_conductance(g, geometry)
    
    if topology isa Scalar
        @named oneport = OnePort()
        @parameters g=g_val E_rev=E_rev
    else
        N = topology.N
        @named oneport = VectorizedOnePort(N=N)
        
        # Properly handle array (heterogeneous) parameters in MTK v11
        if g_val isa AbstractArray
            @parameters begin
                g[1:N] = g_val
            end
        else
            @parameters g=g_val
        end
        
        if E_rev isa AbstractArray
            @parameters begin
                E_rev[1:N] = E_rev
            end
        else
            @parameters E_rev=E_rev
        end
    end
    
    @unpack v, i = oneport
    
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{Any, Any}()
    
    if isempty(gates)
        push!(eqs, i ~ g .* (v .- E_rev))
    else
        conductance_factor = true
        
        for gate in gates
            if topology isa Scalar
                gate_var = only(@variables $(gate.name)(t))
                alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t))
                beta_var = only(@variables $(Symbol(gate.name, :_beta))(t))
                init_conds[gate_var] = gate.ic
            else
                gate_var = only(@variables $(gate.name)(t)[1:topology.N])
                alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t)[1:topology.N])
                beta_var = only(@variables $(Symbol(gate.name, :_beta))(t)[1:topology.N])
                init_conds[gate_var] = fill(gate.ic, topology.N)
            end
            
            push!(vars, gate_var, alpha_var, beta_var)
            alpha_expr, beta_expr = gate.dynamics(v)
            
            push!(eqs, alpha_var ~ alpha_expr)
            push!(eqs, beta_var ~ beta_expr)
            push!(eqs, D(gate_var) ~ alpha_expr .* (1.0 .- gate_var) .- beta_expr .* gate_var)
            
            conductance_factor = conductance_factor .* (gate_var .^ gate.power)
        end
        
        push!(eqs, i ~ g .* conductance_factor .* (v .- E_rev))
    end
    
    return extend(System(eqs, t, vars, [g, E_rev]; 
                       systems=System[], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end



@component function ContinuousLIFChannel(; name, g_L=0.1, E_L=-70.0, V_th=-50.0, Δ_T=2.0, topology=Scalar())
    if topology isa Scalar
        @named oneport = OnePort()
        @unpack v, i = oneport
        
        @parameters g_L=g_L E_L=E_L V_th=V_th Δ_T=Δ_T
        params = SymbolicT[g_L, E_L, V_th, Δ_T]
        
        vars = SymbolicT[]
        
        # Standard scalar math
        reset_current = g_L * Δ_T * exp((v - V_th) / Δ_T)
        eqs = Equation[
            i ~ g_L * (v - E_L) + reset_current
        ]
        
        return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
    else
        N = topology.N
        @named oneport = VectorizedOnePort(N=N)
        @unpack v, i = oneport
        
        @parameters g_L=g_L E_L=E_L V_th=V_th Δ_T=Δ_T
        params = SymbolicT[g_L, E_L, V_th, Δ_T]
        
        vars = SymbolicT[]
        
        # Use scalar * array (g_L * ...) instead of broadcast (g_L .* ...) 
        # to avoid Symbolics BroadcastBuffer errors.
        diff = v .- V_th
        leak_current = g_L * (v .- E_L)
        reset_current = (g_L * Δ_T) * exp.(diff ./ Δ_T)
        
        eqs = Equation[
            i ~ leak_current .+ reset_current
        ]
        
        return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
    end
end
```

## File: src/components/synapses.jl
```julia
# ==========================================
# Synapse Components
# ==========================================

"""
    SynapsePort

A boundary connector that exposes the postsynaptic current variable (`I_syn`) 
and binds it to the positive pin (`p.i`) of a standard electrical port. 

This component is typically used internally by compartment builders to route 
synaptic currents into a postsynaptic compartment's `CurrentSource`.
"""
@component function SynapsePort(; name, topology=Scalar())
    if topology isa Scalar
        @named p = Pin()
        @variables I_syn(t)
        vars = SymbolicT[I_syn]
        eqs = Equation[p.i ~ I_syn]
    else
        @named p = VectorizedPin(N=topology.N)
        @variables I_syn(t)[1:topology.N]
        vars = SymbolicT[I_syn]
        eqs = Equation[p.i ~ I_syn]
    end
    return System(eqs, t, vars, SymbolicT[]; systems=[p], name=name)
end

"""
    CholSynapse(; name, g_max=30.0, E_rev=-80.0, k_minus=0.01, V_th=-35.0, delta=5.0, geometry=NoGeometry())

A continuous cholinergic synapse model. The synaptic state variable `s` represents 
the fraction of open receptors. It rises towards a steady-state `s_inf` governed by 
the presynaptic voltage, and decays exponentially.

The synaptic current is calculated as the current injected into the postsynaptic membrane:
`I_syn = g_max * s * (E_rev - V_post)`

# Arguments
- `g_max`: Maximum synaptic conductance (scaled by geometry if provided).
- `E_rev`: Reversal potential of the synapse (e.g., -80 mV for inhibitory).
- `k_minus`: Rate constant for receptor unbinding (controls decay time).
- `V_th`: Half-activation voltage for the presynaptic sigmoid.
- `delta`: Slope of the presynaptic sigmoid activation.
- `geometry`: AbstractGeometry struct for scaling `g_max`.
"""
@component function CholSynapse(; name, g_max=30.0, E_rev=-80.0, k_minus=0.01, V_th=-35.0, delta=5.0, geometry=NoGeometry())
    g_max_val = get_synaptic_conductance(g_max, geometry)
    
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max_val E_rev=E_rev k_minus=k_minus V_th=V_th delta=delta
    
    s_inf = 1.0 / (1.0 + exp((V_th - V_pre) / delta))
    tau_s = (1.0 - s_inf) / k_minus
    
    eqs = Equation[
        D(s) ~ (s_inf - s) / tau_s,
        I_syn ~ g_max * s * (E_rev - V_post)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, E_rev, k_minus, V_th, delta]; systems=System[], name=name)
end

"""
    GlutSynapse(; name, g_max=30.0, E_rev=-70.0, k_minus=0.025, V_th=-35.0, delta=5.0, geometry=NoGeometry())

A continuous glutamatergic synapse model. Behaves identically to `CholSynapse` but uses 
default parameters typical for fast excitatory glutamatergic receptors.

# Arguments
- `g_max`: Maximum synaptic conductance (scaled by geometry if provided).
- `E_rev`: Reversal potential of the synapse (e.g., -70 mV or higher for excitatory).
- `k_minus`: Rate constant for receptor unbinding.
- `V_th`: Half-activation voltage for the presynaptic sigmoid.
- `delta`: Slope of the presynaptic sigmoid activation.
- `geometry`: AbstractGeometry struct for scaling `g_max`.
"""
@component function GlutSynapse(; name, g_max=30.0, E_rev=-70.0, k_minus=0.025, V_th=-35.0, delta=5.0, geometry=NoGeometry())
    g_max_val = get_synaptic_conductance(g_max, geometry)
    
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max_val E_rev=E_rev k_minus=k_minus V_th=V_th delta=delta
    
    s_inf = 1.0 / (1.0 + exp((V_th - V_pre) / delta))
    tau_s = (1.0 - s_inf) / k_minus
    
    eqs = Equation[
        D(s) ~ (s_inf - s) / tau_s,
        I_syn ~ g_max * s * (E_rev - V_post)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, E_rev, k_minus, V_th, delta]; systems=System[], name=name)
end

"""
    ExpSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)

A simple exponential decay synapse. The synaptic gating variable `s` is driven by a 
continuous sigmoidal function of the presynaptic voltage and decays exponentially with time constant `τ`.

The current injected into the postsynaptic compartment is:
`I_syn = g_max * s * (E_rev - V_post)`

# Arguments
- `g_max`: Maximum synaptic conductance.
- `τ`: Decay time constant of the synapse.
- `E_rev`: Reversal potential of the synapse.
- `V_th`: Threshold voltage for presynaptic activation.
- `slope`: Slope of the presynaptic sigmoid activation.
"""
@component function ExpSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    
    eqs = [
        D(s) ~ -s / τ + σ(V_pre - V_th),
        I_syn ~ g_max * s * (E_rev - V_post)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, τ, E_rev, V_th, slope]; 
                  systems=System[], name=name)
end

"""
    AlphaSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)

An alpha-function synapse implemented via a cascade of two first-order filters (`s1` and `s2`). 
This produces the classic unimodal alpha-function response in synaptic conductance following 
a sustained presynaptic depolarization.

The current injected into the postsynaptic compartment is:
`I_syn = g_max * s2 * (E_rev - V_post)`

# Arguments
- `g_max`: Maximum synaptic conductance.
- `τ`: Time constant for both cascaded filters.
- `E_rev`: Reversal potential of the synapse.
- `V_th`: Threshold voltage for presynaptic activation.
- `slope`: Slope of the presynaptic sigmoid activation.
"""
@component function AlphaSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)
    @variables s1(t)=0.0 s2(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    
    eqs = [
        D(s1) ~ -s1 / τ + σ(V_pre - V_th),
        D(s2) ~ -s2 / τ + s1,
        I_syn ~ g_max * s2 * (E_rev - V_post)
    ]
    return System(eqs, t, [s1, s2, I_syn, V_pre, V_post], 
                  [g_max, τ, E_rev, V_th, slope]; systems=System[], name=name)
end

"""
    NMDASynapse(; name, g_max=1.0, τ=100.0, E_rev=0.0, V_th=-20.0, Mg_conc=1.0, slope=2.0)

An N-Methyl-D-Aspartate (NMDA) receptor synapse. It includes the classic voltage-dependent 
Magnesium block that reduces conductance at hyperpolarized potentials. The gating variable `s` 
decays with a slow time constant `τ`.

The current injected into the postsynaptic compartment is:
`I_syn = g_max * s * mg_block(V_post) * (E_rev - V_post)`

# Arguments
- `g_max`: Maximum synaptic conductance.
- `τ`: Slow decay time constant typical of NMDA receptors.
- `E_rev`: Reversal potential of the synapse (usually near 0 mV).
- `V_th`: Threshold voltage for presynaptic activation.
- `Mg_conc`: Extracellular Magnesium concentration determining block strength.
- `slope`: Slope of the presynaptic sigmoid activation.
"""
@component function NMDASynapse(; name, g_max=1.0, τ=100.0, E_rev=0.0, V_th=-20.0, 
                                  Mg_conc=1.0, slope=2.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th Mg_conc=Mg_conc slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    mg_block(V) = 1.0 / (1.0 + Mg_conc * exp(-0.062 * V))
    
    eqs = [
        D(s) ~ -s / τ + σ(V_pre - V_th),
        I_syn ~ g_max * s * mg_block(V_post) * (E_rev - V_post)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], 
                  [g_max, τ, E_rev, V_th, Mg_conc, slope]; systems=System[], name=name)
end

"""
    VectorizedExpSynapse(; name, N_pre, N_post, W, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)

A vectorized block of exponential synapses representing a dense `N_post` by `N_pre` projection. 
It accepts an entire weight matrix `W` mapping presynaptic gating variables to postsynaptic currents.

The synaptic state `s` is a vector of length `N_pre`. The postsynaptic current vector is computed via:
`I_syn[i] = g_max * sum_j(W[i, j] * s[j]) * (E_rev - V_post[i])`

# Arguments
- `N_pre`: Number of presynaptic elements.
- `N_post`: Number of postsynaptic elements.
- `W`: A matrix of connection weights (dimensions `N_post` x `N_pre`).
- `g_max`: Maximum global synaptic conductance.
- `τ`: Decay time constant.
- `E_rev`: Reversal potential of the synapse.
- `V_th`: Threshold voltage for presynaptic activation.
- `slope`: Slope of the presynaptic sigmoid activation.
"""
@component function VectorizedExpSynapse(; name, N_pre, N_post, W,
                                            g_max=1.0, τ=5.0, E_rev=0.0,
                                            V_th=-20.0, slope=2.0)
    @variables s(t)[1:N_pre] I_syn(t)[1:N_post] V_pre(t)[1:N_pre] V_post(t)[1:N_post]
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    # Make W a symbolic parameter!
    @parameters W[1:N_post, 1:N_pre]=W

    σ(V) = 1.0 ./ (1.0 .+ exp.(-(V .- V_th) ./ slope))
    synaptic_drive = W * s
    
    eqs = [
        D(s) ~ -s ./ τ .+ σ(V_pre),
        I_syn ~ g_max .* (E_rev .- V_post) .* synaptic_drive
    ]
    
    init_conds = Dict(s => zeros(N_pre))
    
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, τ, E_rev, V_th, slope, W];
                  systems=System[], 
                  initial_conditions=init_conds, 
                  name=name)
end


"""
    STDPSynapse(; name, g_max=1.0, E_rev=0.0, V_th=0.0, slope=2.0, τ_s=5.0, 
                τ_plus=20.0, τ_minus=20.0, A_plus=0.1, A_minus=0.1, 
                w_init=0.5, w_max=1.0, w_min=0.0)

A continuous, smooth approximation of Spike-Timing-Dependent Plasticity (STDP) with soft bounds.
It uses a continuous spike-detector function (sigmoid) and trace variables (`x` for pre, `y` for post) 
to approximate the relative timing of spikes without requiring discrete event handling.

The weight `w` evolves continuously according to:
`dw/dt = A_plus * (w_max - w) * x * σ(V_post) - A_minus * (w - w_min) * y * σ(V_pre)`

where `x` and `y` are exponentially decaying traces, and `σ(V)` is a sigmoid acting as a 
continuous spike detector. This formulation is purely acausal and ODE-based, making it incredibly 
robust for standard differential equation solvers while demonstrating classic STDP behavior.
"""
@component function STDPSynapse(; name, g_max=1.0, E_rev=0.0, V_th=0.0, slope=2.0,
                                τ_s=5.0, τ_plus=20.0, τ_minus=20.0, 
                                A_plus=0.1, A_minus=0.1, w_init=0.5, w_max=1.0, w_min=0.0)
    
    @variables s(t)=0.0 w(t)=w_init x(t)=0.0 y(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ_s=τ_s τ_plus=τ_plus τ_minus=τ_minus  A_plus=A_plus A_minus=A_minus V_th=V_th E_rev=E_rev slope=slope  w_max=w_max w_min=w_min
                
    # Continuous spike detector
    σ(V) = 1.0 / (1.0 + exp(-(V - V_th) / slope))
    
    eqs = Equation[
        # Synaptic gating variable
        D(s) ~ -s / τ_s + σ(V_pre), 
        
        # Pre- and post-synaptic activity traces
        D(x) ~ -x / τ_plus + σ(V_pre),
        D(y) ~ -y / τ_minus + σ(V_post),
        
        # Continuous STDP weight update with soft bounds
        D(w) ~ A_plus * (w_max - w) * x * σ(V_post) - A_minus * (w - w_min) * y * σ(V_pre),
        
        # Synaptic current injection
        I_syn ~ w * g_max * s * (E_rev - V_post)
    ]
    
    return System(eqs, t, [s, w, x, y, I_syn, V_pre, V_post], 
                  [g_max, τ_s, τ_plus, τ_minus, A_plus, A_minus, V_th, E_rev, slope, w_max, w_min]; 
                  systems=System[], name=name)
end
```

## File: src/geometry.jl
```julia
abstract type AbstractGeometry end
struct NoGeometry <: AbstractGeometry end

Base.@kwdef struct Geometry <: AbstractGeometry
    area::Float64 = 0.0628   # default to common STG area in cm^2
    C_m::Float64  = 1.0      # default to standard specific capacitance in uF/cm^2
end

# 2. Multiple Dispatch Rules for Biophysics
# Capacitance extraction
get_capacitance(C, geom::NoGeometry) = C
get_capacitance(C, geom::Geometry) = geom.C_m * geom.area

# Conductance extraction
get_conductance(g, geom::NoGeometry) = g
get_conductance(g, geom::Geometry) = g * geom.area

# Calcium conversion factor extraction 
get_ca_conversion_factor(conv, geom::NoGeometry, tauCa) = conv
get_ca_conversion_factor(conv, geom::Geometry, tauCa) = 0.94 / (geom.C_m * geom.area * tauCa)

get_synaptic_conductance(g, geom::NoGeometry) = g
get_synaptic_conductance(g, geom::Geometry) = g * geom.area



abstract type AbstractMorphology end
struct NoMorphology <: AbstractMorphology end

 Base.@kwdef struct Morphology <: AbstractMorphology
    position::Tuple{Float64, Float64, Float64} = (0.0, 0.0, 0.0) # x, y, z in space
    shape::Symbol = :spherical                              # :spherical, :cylindrical, :point
    dimensions::NamedTuple = (radius=10.0,)                 # in microns
    color::Symbol = :blue                                   # rendering hint
end
```

## File: src/network.jl
```julia
# ==========================================
# Core Network & Compartment Builders
# ==========================================

"""
    Compartment

A struct representing a single neural compartment (e.g., a soma, axon hillock, or dendritic segment).
It wraps the generated ModelingToolkit `System` along with metadata about its physical and 
electrical properties, and exposes a tuple of `interfaces` for acausal network connections.

# Fields
- `sys::System`: The underlying MTK system representing the compartment's equations.
- `interfaces::NamedTuple`: Exposed boundary variables and pins (e.g., `V`, `p_pin`, `n_pin`, `I_ext`, `I_syn`).
- `V_init::F`: The initial membrane voltage.
- `topology::Union{Scalar, Vectorized}`: The electrical topology of the compartment.
- `geometry::G`: The physical geometry used for scaling biophysical parameters.
- `morphology::M`: The spatial morphology used for rendering or spatial simulations.
"""
struct Compartment{M<:AbstractMorphology,G<:AbstractGeometry, F<:AbstractFloat}
    sys::System
    interfaces::NamedTuple
    V_init::F
    topology::Union{Scalar, Vectorized}
    geometry::G
    morphology::M
end

"""
    Network

A struct representing the complete assembled neural network. It encapsulates the fully 
connected MTK `System` and a vector of input variables for simulation drivers.

# Fields
- `sys::System`: The final compiled MTK system representing the entire network.
- `inputs::Vector{Any}`: A collection of symbolic input variables for external stimulation.
"""
struct Network
    sys::System
    inputs::Vector{Any}
end

"""
    SynapseSpec

A specification struct used to wire a synapse between a presynaptic voltage and a postsynaptic current.
It provides the mapping needed by `wire_synapses!` to inject currents into the correct compartments.

# Fields
- `pre_V`: The symbolic voltage variable of the presynaptic compartment.
- `post_V`: The symbolic voltage variable of the postsynaptic compartment.
- `post_I_syn`: The symbolic current variable of the postsynaptic compartment where the synapse will inject.
- `synapse::System`: The MTK synapse system component (e.g., `ExpSynapse`, `CholSynapse`).
- `post_comp::Union{Compartment, Nothing}`: The postsynaptic compartment struct (used for block synapse grounding logic).
"""
struct SynapseSpec
    pre_V
    post_V
    post_I_syn
    synapse
    post_comp::Union{Compartment, Nothing} 
end

"""
Outer constructor for `SynapseSpec` that defaults `post_comp` to `nothing`.
Useful for scalar synapses where block grounding logic is not required.
"""
SynapseSpec(pre_V, post_V, post_I_syn, synapse) = SynapseSpec(pre_V, post_V, post_I_syn, synapse, nothing)

"""
    CouplingSpec

A specification struct used to wire an acausal coupling (e.g., a Gap Junction) between two compartments.

# Fields
- `comp_i::Compartment`: The first compartment to be coupled.
- `comp_j::Compartment`: The second compartment to be coupled.
- `coupling::System`: The MTK coupling system component (e.g., `GapJunction`).
"""
struct CouplingSpec{C1<:Compartment, C2<:Compartment}
    comp_i::C1
    comp_j::C2
    coupling::System
end


# ==========================================
# Ion Configuration
# ==========================================

"""
    NoCalcium

A configuration struct indicating that a compartment has no Calcium dynamics.
When passed to `build_compartment`, it bypasses the creation of a `CalciumPool`.
"""
struct NoCalcium end

"""
    CalciumTracker

A configuration struct enabling Calcium dynamics within a compartment.
When passed to `build_compartment`, it instantiates a `CalciumPool` and connects it to all 
channels that expose a `ca_port`.

# Fields
- `decay::Union{Float64, Function}`: Either a time constant for linear decay, or a function 
  that takes the current Calcium concentration and returns the decay rate.
- `Ca_init::Float64`: The initial intracellular Calcium concentration.
"""
struct CalciumTracker
    decay::Union{Float64, Function}
    Ca_init::Float64
end

"""
Keyword argument constructor for `CalciumTracker`.
"""
CalciumTracker(; decay=100.0, Ca_init=0.0) = CalciumTracker(decay, Ca_init)


# ==========================================
# Internal Wiring Helpers
# ==========================================

"""
    wire_ions!(eqs, systems, channels, config, topology, name)

Internal helper function to wire ion dynamics into a compartment's equations and systems list.
Uses multiple dispatch to handle different ion configurations.

- If `config` is `NoCalcium`, it does nothing.
- If `config` is `CalciumTracker`, it creates a `CalciumPool` and connects it to all channels in the compartment that expose a `ca_port`.
"""
wire_ions!(eqs, systems, channels, ::NoCalcium, topology, name) = nothing
function wire_ions!(eqs, systems, channels, config::CalciumTracker, topology, name)
    # Pass decay to the CalciumPool
    ca_pool = CalciumPool(topology=topology, decay=config.decay, Ca_init=config.Ca_init, name=Symbol(name, :_ca_pool))
    push!(systems, ca_pool)
    
    ca_ports = System[ca_pool.port]
    for c in channels
        if hasproperty(c, :ca_port)
            push!(ca_ports, c.ca_port)
        end
    end
    push!(eqs, connect(ca_ports...))
end

"""
    wire_synapses!(eqs, systems, specs)

Internal helper function that wires a collection of `SynapseSpec`s into the network equations.
It binds the presynaptic and postsynaptic voltage variables to the synapse, and pre-collects 
convergent synapses by their target current variable to write a single summed equation per target.

Returns a tuple of `(driven_syn_targets, block_driven_targets)` used for grounding unconnected inputs.
"""
function wire_synapses!(eqs::Vector{Equation}, systems::Vector{System},
                        specs::Vector{SynapseSpec})
    syn_by_target = Dict{SymbolicT, Vector{SymbolicT}}()
    driven_syn_targets = Set{SymbolicT}()
    block_driven_targets = Set{SymbolicT}()

    for spec in specs
        push!(systems, spec.synapse)
        
        if hasproperty(spec.synapse, :V_pre)
            push!(eqs, spec.synapse.V_pre ~ spec.pre_V)
        end
        if hasproperty(spec.synapse, :V_post)
            push!(eqs, spec.synapse.V_post ~ spec.post_V)
        end

        key = spec.post_I_syn
        haskey(syn_by_target, key) || (syn_by_target[key] = SymbolicT[])
        push!(syn_by_target[key], spec.synapse.I_syn)
        push!(driven_syn_targets, key)
        
        if spec.post_I_syn isa AbstractArray
            push!(block_driven_targets, spec.post_I_syn)
        end
    end

    for (target, currents) in syn_by_target
        if length(currents) == 1
            push!(eqs, target ~ currents[1])
        else
            push!(eqs, target ~ reduce(+, currents))
        end
    end

    return driven_syn_targets, block_driven_targets
end


# ==========================================
# Compartment & Network Builders
# ==========================================

"""
    build_compartment(capacitor, channels; name, V_init, topology, ion_config, geometry, morphology)

Builds a `Compartment` by connecting a `Capacitor`, current `injector`s, and a collection of ion `channels`.
This forms the fundamental electrical unit of a neuron. All positive terminals (p) are connected together 
to the membrane potential, and all negative terminals (n) are connected to ground. 

# Arguments
- `capacitor`: A `Capacitor` system defining the membrane capacitance.
- `channels`: A vector of ion channel systems (e.g., `GenericChannel`, `CaVChannel`).
- `name::Symbol`: The name of the compartment system.
- `V_init::Float64`: Initial membrane voltage (default -65.0 mV).
- `topology`: `Scalar()` or `Vectorized(N)` (default `Scalar()`).
- `ion_config`: `NoCalcium()` or `CalciumTracker()` to handle ion pools.
- `geometry`: Geometry struct for biophysical scaling (default `NoGeometry()`).
- `morphology`: Morphology struct for spatial data (default `NoMorphology()`).

# Returns
- A `Compartment` struct containing the assembled `System` and its exposed `interfaces`.
"""
function build_compartment(capacitor, channels; name=:compartment, V_init=-65.0, 
                           topology=Scalar(), ion_config=NoCalcium(), geometry = NoGeometry(), morphology=NoMorphology())
    
    p, n = create_pins(topology)
    injector, syn_injector = create_injectors(topology)
    init_v = init_voltage(topology, V_init)
    
    vars = SymbolicT[]
    eqs  = Equation[]

    # 1. Connect all negative terminals together
    n_pins = Any[capacitor.n, injector.n, syn_injector.n, n]
    for c in channels
        push!(n_pins, c.n)
    end
    push!(eqs, connect(n_pins...))

    # 2. Connect all positive terminals together
    p_connections = System[capacitor, injector, syn_injector]
    append!(p_connections, channels)
    push!(eqs, connect([sys.p for sys in p_connections]...))

    # 3. Expose boundary pin for acausal connections (gap junctions)
    push!(eqs, connect(p, capacitor.p))

    all_systems = System[capacitor, injector, syn_injector, p, n]
    append!(all_systems, channels)

    # 4. Wire ions (dispatches on config and topology)
    wire_ions!(eqs, all_systems, channels, ion_config, topology, name)

    sys = System(eqs, t, vars, SymbolicT[];
                 systems = all_systems,
                 initial_conditions = Dict(capacitor.v => init_v),
                 name)

    cap_name = nameof(capacitor)
    V_state  = getproperty(sys, cap_name).v

    interfaces = (
        V       = V_state,
        p_pin   = getproperty(sys, nameof(p)),
        n_pin   = getproperty(sys, nameof(n)),
        I_ext   = getproperty(sys, nameof(injector)).I.u,
        I_syn   = getproperty(sys, nameof(syn_injector)).I.u,
        cap_name = cap_name
    )
    return Compartment(sys, interfaces, V_init, topology, geometry, morphology)
end

"""
    build_acausal_network(compartments; coupling_specs, synapse_specs, drivers, name)

Assembles a collection of `Compartment`s into a complete `Network` system. 
It handles grounding, wiring driving stimuli, gap junctions (via `CouplingSpec`), 
and chemical synapses (via `SynapseSpec`).

# Arguments
- `compartments::Vector{<:Compartment}`: The compartments making up the network.
- `coupling_specs`: A vector of `CouplingSpec` structs for acausal electrical connections.
- `synapse_specs`: A vector of `SynapseSpec` structs for directed chemical synapses.
- `drivers`: A vector of `(target, stim)` tuples, where `target` is a compartment or index, and `stim` is an MTK block, vector, or number.
- `name::Symbol`: The name of the overall network system.

# Returns
- A `Network` struct containing the assembled MTK `System`.
"""
function build_acausal_network(compartments::Vector{<:Compartment};
                                coupling_specs=CouplingSpec[],
                                synapse_specs=SynapseSpec[],
                                drivers=[],
                                name=:network)

    num_compartments = length(compartments)
    eqs = Equation[]
    all_systems = System[]

    for comp in compartments
        push!(all_systems, comp.sys)
    end

    driven_compartments = Set{Int}()
    gap_junctioned = Set{Int}()

    # 1. Ground each compartment individually (Dispatches on topology)
    for (i, comp) in enumerate(compartments)
        if haskey(comp.interfaces, :n_pin)
            gnd = create_ground(comp.topology, Symbol(:gnd_, i))
            push!(all_systems, gnd)
            push!(eqs, connect(gnd.g, comp.interfaces.n_pin))
        end
    end

    # 2. Driving stimuli
    for (target, stim) in drivers
        idx = target isa Compartment ? findfirst(==(target), compartments) : target
        push!(driven_compartments, idx)
        comp = compartments[idx]

        if haskey(comp.interfaces, :I_ext)
            if stim isa System
                push!(all_systems, stim)
                push!(eqs, comp.interfaces.I_ext ~ stim.output.u)
            elseif stim isa AbstractVector
                push!(eqs, comp.interfaces.I_ext ~ stim)
            elseif stim isa Number
                push!(eqs, comp.interfaces.I_ext ~ broadcast_stim(comp.topology, stim))
            end
        end
    end

    # 3. Ground undriven I_ext
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :I_ext) && !(i in driven_compartments)
            push!(eqs, comp.interfaces.I_ext ~ ground_current(comp.topology))
        end
    end

    # 4. Wire gap junctions via p_pin
    for (i, spec) in enumerate(coupling_specs)
        push!(all_systems, spec.coupling)
        
        if haskey(spec.comp_i.interfaces, :p_pin) && hasproperty(spec.coupling, :p1)
            push!(eqs, connect(spec.comp_i.interfaces.p_pin, spec.coupling.p1))
            push!(eqs, connect(spec.coupling.n1, spec.comp_i.interfaces.n_pin))
        end
        
        if haskey(spec.comp_j.interfaces, :p_pin) && hasproperty(spec.coupling, :p2)
            push!(eqs, connect(spec.comp_j.interfaces.p_pin, spec.coupling.p2))
            push!(eqs, connect(spec.coupling.n2, spec.comp_j.interfaces.n_pin))
        end
        
        push!(gap_junctioned, findfirst(==(spec.comp_i), compartments))
        push!(gap_junctioned, findfirst(==(spec.comp_j), compartments))
    end

    # 5. Identify block-synapsed compartments by index
    block_synapsed_compartments = Set{Int}()
    for spec in synapse_specs
        if spec.post_I_syn isa AbstractArray && spec.post_comp !== nothing
            idx = findfirst(==(spec.post_comp), compartments)
            if idx !== nothing
                push!(block_synapsed_compartments, idx)
            end
        end
    end

    # 6. Wire synapses
    driven_syn_targets, block_driven_targets = wire_synapses!(eqs, all_systems, synapse_specs)

    # 7. Ground non-synapsed I_syn (Dispatches on topology)
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :I_syn)
            if comp.interfaces.I_syn in block_driven_targets
                continue
            end
            ground_undriven_syn!(eqs, comp.topology, comp.interfaces.I_syn, driven_syn_targets)
        end
    end

    # 8. Ground non-gap-junctioned p_pin.i
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :p_pin) && !(i in gap_junctioned)
            push!(eqs, comp.interfaces.p_pin.i ~ ground_current(comp.topology))
        end
    end

    net_sys = System(eqs, t, SymbolicT[], SymbolicT[];
                     systems = all_systems, name = name)
                     
    return Network(net_sys, SymbolicT[])
end

"""
    build_synapse_block(pre_comp, post_comp, W; name, synapse_type, kwargs...)

Helper function to create a `SynapseSpec` using a vectorized synapse block.
It automatically determines the pre- and postsynaptic dimensions based on the weight matrix `W`
and binds it to the provided compartments.

# Arguments
- `pre_comp::Compartment`: The presynaptic compartment.
- `post_comp::Compartment`: The postsynaptic compartment.
- `W`: The weight matrix (dimensions `N_post` x `N_pre`).
- `name::Symbol`: The name for the synapse block system.
- `synapse_type`: The vectorized synapse component to use (defaults to `VectorizedExpSynapse`).
- `kwargs...`: Additional keyword arguments passed to `synapse_type` (e.g., `g_max`, `E_rev`).

# Returns
- A `SynapseSpec` configured for the network builder.
"""
function build_synapse_block(pre_comp, post_comp, W; name, 
                             synapse_type=VectorizedExpSynapse, kwargs...)
    N_pre  = size(W, 2)
    N_post = size(W, 1)
    syn = synapse_type(N_pre=N_pre, N_post=N_post, W=W; name=name, kwargs...)
    return SynapseSpec(pre_comp.interfaces.V, post_comp.interfaces.V,
                       post_comp.interfaces.I_syn, syn, post_comp)
end
```

## File: src/topology.jl
```julia
struct Scalar end
struct Vectorized
    N::Int
end

# Topology helper functions
get_N(::Scalar) = nothing
get_N(v::Vectorized) = v.N

init_voltage(::Scalar, V_init) = V_init
init_voltage(v::Vectorized, V_init) = fill(V_init, v.N)

function create_pins(::Scalar)
    @named p = Pin(); @named n = Pin()
    return (p, n)
end
function create_pins(v::Vectorized)
    @named p = VectorizedPin(N=v.N); @named n = VectorizedPin(N=v.N)
    return (p, n)
end

function create_injectors(::Scalar)
    @named injector = CurrentSource(); @named syn_injector = CurrentSource()
    return (injector, syn_injector)
end
function create_injectors(v::Vectorized)
    @named injector = CurrentSource(topology=v)
    @named syn_injector = CurrentSource(topology=v)
    return (injector, syn_injector)
end

# Network grounding helpers
create_ground(::Scalar, name) = Ground(name=name)
create_ground(v::Vectorized, name) = Ground(topology=v, name=name)

ground_current(::Scalar) = 0.0
ground_current(v::Vectorized) = zeros(Float64, v.N)

broadcast_stim(::Scalar, stim) = stim
broadcast_stim(v::Vectorized, stim) = fill(stim, v.N)

# Synapse grounding helpers
function ground_undriven_syn!(eqs, ::Scalar, I_syn, driven_syn_targets)
    if !(I_syn in driven_syn_targets)
        push!(eqs, I_syn ~ 0.0)
    end
end
function ground_undriven_syn!(eqs, v::Vectorized, I_syn, driven_syn_targets)
    for j in 1:v.N
        i_syn_j = I_syn[j]
        if !(i_syn_j in driven_syn_targets)
            push!(eqs, i_syn_j ~ 0.0)
        end
    end
end
```

## File: src/components/electrical.jl
```julia
# ==========================================
# CORE SCALAR CONNECTORS (Custom)
# ==========================================

@connector function Pin(; name)
    vars = @variables begin
        v(t)
        i(t), [connect = Flow]
    end
    return System(Equation[], t, vars, SymbolicT[]; name=name)
end

@component function OnePort(; name)
    systems = @named begin
        p = Pin()
        n = Pin()
    end
    vars = @variables begin
        v(t)
        i(t)
    end
    equations = Equation[
        v ~ p.v - n.v,
        0 ~ p.i + n.i,
        i ~ p.i,
    ]
    return System(equations, t, vars, SymbolicT[]; name=name, systems=systems)
end

@component function TwoPort(; name)
    systems = @named begin
        p1 = Pin()
        n1 = Pin()
        p2 = Pin()
        n2 = Pin()
    end
    vars = @variables begin
        v1(t), i1(t)
        v2(t), i2(t)
    end
    equations = Equation[
        v1 ~ p1.v - n1.v,
        0 ~ p1.i + n1.i,
        i1 ~ p1.i,
        v2 ~ p2.v - n2.v,
        0 ~ p2.i + n2.i,
        i2 ~ p2.i,
    ]
    return System(equations, t, vars, SymbolicT[]; name=name, systems=systems)
end

@component function Ground(; name, topology=Scalar())
    if topology isa Scalar
        @named g = Pin()
        eqs = [g.v ~ 0]
    else
        @named g = VectorizedPin(N=topology.N)
        eqs = [g.v ~ zeros(Float64, topology.N)]
    end
    return System(eqs, t, SymbolicT[], SymbolicT[]; systems=[g], name=name)
end

@component function Capacitor(; name, C = 1.0, topology=Scalar(), geometry=NoGeometry())
    C_val = get_capacitance(C, geometry) # Dispatch handles the math
    
    if topology isa Scalar
        @named oneport = OnePort()
    else
        @named oneport = VectorizedOnePort(N=topology.N)
    end
    @unpack v, i = oneport
    @parameters C=C_val
    eqs = Equation[D(v) ~ i ./ C]
    return extend(System(eqs, t, SymbolicT[], [C]; systems=System[], name=name), oneport)
end


@component function CurrentSource(; name, topology=Scalar())
    if topology isa Scalar
        @named oneport = OnePort()
        @named I = RealInput()
    else
        @named oneport = VectorizedOnePort(N=topology.N)
        @named I = RealInputArray(nin=topology.N)
    end
    @unpack i = oneport
    
    eqs = Equation[i ~ -I.u]
    return extend(System(eqs, t, SymbolicT[], SymbolicT[]; systems=[I], name=name), oneport)
end

"""
fixed_reversal Component: A pure constant voltage source (Nernst battery).
"""
@component function FixedReversal(; name, E = 0.0)
    @named oneport = OnePort()
    @unpack v = oneport
    @parameters begin
        E = E
    end
    params = SymbolicT[]
    push!(params, E)
    vars = SymbolicT[]
    eqs = Equation[]
    push!(eqs, v ~ E)
    reversal_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        name
    )
    return extend(reversal_sys, oneport)
end

"""
SpikingCapacitor Component: Capacitor that automatically resets its voltage when a threshold is crossed 
"""
@component function SpikingCapacitor(; name, C = 10.0, V_th = -55.0, V_reset = -67.0, V_init = -65.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    
    @parameters begin
        C = C
        V_th = V_th
        V_reset = V_reset
    end
    params = SymbolicT[]
    push!(params, C, V_th, V_reset)
    
    @variables begin
        v(t) = V_init
        V(t)
    end
    vars = SymbolicT[]
    push!(vars, v, V)

    eqs = Equation[
        D(v) ~ i / C,
        V ~ v
    ]
    
    root_eqs = Equation[v ~ V_th]
    affect = Equation[v ~ V_reset]
    events = [root_eqs => affect]
    
    lif_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        continuous_events = events,
        name
    )
    
    return extend(lif_sys, oneport)
end

@component function GapJunction(; name, R = 1.0)
    @named twoport = TwoPort()
    @unpack v1, i1, v2, i2 = twoport

    @parameters R = R
    params = SymbolicT[]
    push!(params, R)

    vars = SymbolicT[]

    eqs = Equation[]
    push!(eqs, i1 ~ (v1 - v2) / R)
    push!(eqs, i2 ~ -i1)

    return extend(System(eqs, t, vars, params; systems=System[], name=name), twoport)
end

@component function EventSynapse(; name, g_max=2.0, τ=5.0, V_th=-20.0, w=0.5, E_rev=0.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ V_th=V_th w=w E_rev=E_rev
    
    eqs = [
        D(s) ~ -s / τ,
        I_syn ~ g_max * s * (E_rev - V_post)
    ]
    
    # Event: when V_pre crosses V_th upwards, add w to s
    root_eq = V_pre ~ V_th
    affect = s ~ Pre(s) + w
    events = [root_eq => affect]
    
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, τ, V_th, w, E_rev]; 
                  systems=System[], events=events, name=name)
end


# ==========================================
# VECTORIZED ELECTRICAL COMPONENTS
# ==========================================

@connector function VectorizedPin(; name, N::Int, v = nothing, i = nothing)
    vars = @variables begin
        v(t)[1:N] = v
        i(t)[1:N] = i, [connect = Flow]
    end
    return System(Equation[], t, vars, SymbolicT[]; name=name)
end

@component function VectorizedOnePort(; name, N::Int, v = nothing, i = nothing)
    pars = @parameters begin
    end
    systems = @named begin
        p = VectorizedPin(N=N)
        n = VectorizedPin(N=N)
    end
    vars = @variables begin
        v(t)[1:N] = v
        i(t)[1:N] = i
    end
    equations = Equation[
        v ~ p.v - n.v,
        collect(p.i .+ n.i .~ 0.0)...,  # splat the collected equations
        i ~ p.i,
    ]

    return System(equations, t, vars, pars; name, systems)
end
```

## File: src/library/ContinuousSpikers.jl
```julia
module ContinuousSpikers
    using ..MTKNeuralToolkit: GateSpec, GenericChannel, Scalar, OnePort
    using ModelingToolkit: t_nounits as t, D_nounits as D, @named, @variables, @parameters, @component, System, Equation, extend, @unpack
    using Symbolics: SymbolicT


    # ==========================================
    # 1. Morris-Lecar (Built via GenericChannel)
    # ==========================================
    
    # Fast Ca2+ gating (effectively instantaneous)
    const V1, V2 = -20.0, 15.0
    const ml_ca_m = v -> (
        0.5 .* (1.0 .+ tanh.((v .- V1) ./ V2)) ./ 0.1,
        0.5 .* (1.0 .- tanh.((v .- V1) ./ V2)) ./ 0.1
    )

    # Slow K+ gating (recovery variable)
    const V3, V4 = -25.0, 5.0
    const tau_n = 10.0
    const ml_k_n = v -> (
        0.5 .* (1.0 .+ tanh.((v .- V3) ./ V4)) ./ tau_n,
        0.5 .* (1.0 .- tanh.((v .- V3) ./ V4)) ./ tau_n
    )

    @component function MorrisLecar(; name, topology=Scalar(), V_init=-20.0, 
                          g_Ca=4.0, E_Ca=100.0, g_K=8.5, E_K=-70.0, g_L=0.1, E_L=-50.0)
        m0 = 0.5 * (1 + tanh((V_init - V1) / V2))
        n0 = 0.5 * (1 + tanh((V_init - V3) / V4))
        
        ca_gates = [GateSpec(:m, 1, m0, ml_ca_m)]
        k_gates  = [GateSpec(:n, 1, n0, ml_k_n)]
        
        # Note: In a real build script, you'd create the Capacitor separately, 
        # but for convenience we can just document the required channels.
        # We return a tuple of the channels to be used with build_compartment.
        @named ca_ch = GenericChannel(topology=topology, g=g_Ca, E_rev=E_Ca, gates=ca_gates)
        @named k_ch  = GenericChannel(topology=topology, g=g_K, E_rev=E_K, gates=k_gates)
        @named leak  = GenericChannel(topology=topology, g=g_L, E_rev=E_L, gates=GateSpec[])
        
        return (ca_ch, k_ch, leak)
    end

    # ==========================================
    # 2. FitzHugh-Nagumo (Custom 2D OnePort)
    # ==========================================
    
    @component function FitzHughNagumo(; name, topology=Scalar(), I_ext=0.0, a=0.7, b=0.8, c=10.0, tau=12.5)
        if topology isa Scalar
            @named oneport = OnePort()
            @unpack v, i = oneport
            
            @parameters a=a b=b c=c tau=tau
            params = SymbolicT[a, b, c, tau]
            
            @variables w(t)=0.0
            vars = SymbolicT[v, w]
            
            # The channel provides the cubic and recovery dynamics.
            # C * dV/dt = I_ext - i_channel
            # We want: dV/dt = c * (v - v^3/3 - w) + I_ext
            # So: i_channel = -c * (v - v^3/3 - w)
            eqs = Equation[
                i ~ -c * (v - (v^3)/3.0 - w),
                D(w) ~ (v + a - b * w) / tau
            ]
            
            return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
        else
            N = topology.N
            @named oneport = VectorizedOnePort(N=N)
            @unpack v, i = oneport
            
            @parameters a=a b=b c=c tau=tau
            params = SymbolicT[a, b, c, tau]
            
            @variables w(t)[1:N]=zeros(N)
            vars = SymbolicT[v, w]
            
            eqs = Equation[
                i ~ -c .* (v .- (v.^3)./ 3.0 .- w),
                D(w) ~ (v .+ a .- b .* w) ./ tau
            ]
            
            return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
        end
    end

    export MorrisLecar, FitzHughNagumo
end
```

## File: src/library/HodgkinHuxley.jl
```julia
# ==========================================
# Standard Model Library
# ==========================================
module HodgkinHuxley
    using ..MTKNeuralToolkit: GateSpec, GenericChannel, Scalar
    using ModelingToolkit: t_nounits as t


    # Standard 1952 Hodgkin-Huxley gating dynamics (Dayan & Abbott formulation)
    # where V_rest = -65 mV.
    const na_m = v -> (
        0.1 .* (v .+ 40.0) ./ (1.0 .- exp.(-(v .+ 40.0) ./ 10.0)),  # alpha_m
        4.0 .* exp.(-(v .+ 65.0) ./ 18.0)                           # beta_m
    )
    const na_h = v -> (
        0.07 .* exp.(-(v .+ 65.0) ./ 20.0),                         # alpha_h
        1.0 ./ (1.0 .+ exp.(-(v .+ 35.0) ./ 10.0))                  # beta_h
    )
    const k_n = v -> (
        0.01 .* (v .+ 55.0) ./ (1.0 .- exp.(-(v .+ 55.0) ./ 10.0)), # alpha_n
        0.125 .* exp.(-(v .+ 65.0) ./ 80.0)                         # beta_n
    )

    # Steady-state initial conditions at V = -65 mV
    const sodium_gates = [GateSpec(:m, 3, 0.052, na_m), GateSpec(:h, 1, 0.596, na_h)]
    const potassium_gates = [GateSpec(:n, 4, 0.317, k_n)]

    # Convenience constructors
    function SodiumChannel(; name, topology=Scalar(), g=120.0, E_rev=50.0)
        return GenericChannel(; name=name, g=g, E_rev=E_rev, gates=sodium_gates, topology=topology)
    end

    function PotassiumChannel(; name, topology=Scalar(), g=36.0, E_rev=-77.0)
        return GenericChannel(; name=name, g=g, E_rev=E_rev, gates=potassium_gates, topology=topology)
    end

    function LeakChannel(; name, topology=Scalar(), g=0.3, E_rev=-54.4)
        return GenericChannel(; name=name, g=g, E_rev=E_rev, gates=GateSpec[], topology=topology)
    end

    export SodiumChannel, PotassiumChannel, LeakChannel
end
```

## File: src/library/LiuCalciumNeuron.jl
```julia
module LiuCalciumNeuron
    using ..MTKNeuralToolkit: GateSpec, GenericChannel, CaVChannel, KCaChannel, CalciumTracker, Capacitor, build_compartment, Scalar
    
    import ..MTKNeuralToolkit: InfTau, InfTauCa 
    
    using ModelingToolkit: @named

    # 1. Define Inf and Tau functions
    Na_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 25.5) ./ -5.29))
    Na_tau_m(v) = 1.32 .- 1.26 ./ (1 .+ exp.((v .+ 120.0) ./ -25.0))
    
    Na_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 48.9) ./ 5.18))
    Na_tau_h(v) = (0.67 ./ (1.0 .+ exp.((v .+ 62.9) ./ -10.0))) .* (1.5 .+ 1.0 ./ (1.0 .+ exp.((v .+ 34.9) ./ 3.6)))
    
    CaS_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 33.0) ./ -8.1))
    CaS_tau_m(v) = 1.4 .+ 7.0 ./ (exp.((v .+ 27.0) ./ 10.0) .+ exp.((v .+ 70.0) ./ -13.0))
    
    CaS_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 60.0) ./ 6.2))
    CaS_tau_h(v) = 60.0 .+ 150.0 ./ (exp.((v .+ 55.0) ./ 9.0) .+ exp.((v .+ 65.0) ./ -16.0))
    
    CaT_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 27.1) ./ -7.2))
    CaT_tau_m(v) = 21.7 .- 21.3 ./ (1.0 .+ exp.((v .+ 68.1) ./ -20.5))
    
    CaT_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 32.1) ./ 5.5))
    CaT_tau_h(v) = 105.0 .- 89.8 ./ (1.0 .+ exp.((v .+ 55.0) ./ -16.9))
    
    Ih_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 70.0) ./ 6.0))
    Ih_tau_m(v) = (272.0 .+ 1499.0 ./ (1.0 .+ exp.((v .+ 42.2) ./ -8.73)))
    
    Ka_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 27.2) ./ -8.7))
    Ka_tau_m(v) = 11.6 .- 10.4 ./ (1.0 .+ exp.((v .+ 32.9) ./ -15.2))
    
    Ka_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 56.9) ./ 4.9))
    Ka_tau_h(v) = 38.6 .- 29.2 ./ (1.0 .+ exp.((v .+ 38.9) ./ -26.5))
    
    KCa_m_inf(v, ca) = (ca ./ (ca .+ 3.0)) ./ (1.0 .+ exp.((v .+ 28.3) ./ -12.6))
    KCa_tau_m(v) = 90.3 .- 75.1 ./ (1.0 .+ exp.((v .+ 46.0) ./ -22.7))
    
    Kdr_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 12.3) ./ -11.8))
    Kdr_tau_m(v) = 7.2 .- 6.4 ./ (1.0 .+ exp.((v .+ 28.3) ./ -19.2))

    # 2. Define Gates using the global convenience functions
    const na_gates  = [GateSpec(:mNa, 3, 0.0, InfTau(Na_m_inf, Na_tau_m)), 
                       GateSpec(:hNa, 1, 0.0, InfTau(Na_h_inf, Na_tau_h))]
    
    const cas_gates = [GateSpec(:mCaS, 3, 0.0, InfTau(CaS_m_inf, CaS_tau_m)), 
                       GateSpec(:hCaS, 1, 0.0, InfTau(CaS_h_inf, CaS_tau_h))]
    
    const cat_gates = [GateSpec(:mCaT, 3, 0.0, InfTau(CaT_m_inf, CaT_tau_m)), 
                       GateSpec(:hCaT, 1, 0.0, InfTau(CaT_h_inf, CaT_tau_h))]
    
    const ih_gates  = [GateSpec(:mIh, 1, 0.0, InfTau(Ih_m_inf, Ih_tau_m))]
    
    const ka_gates  = [GateSpec(:mKa, 3, 0.0, InfTau(Ka_m_inf, Ka_tau_m)), 
                       GateSpec(:hKa, 1, 0.0, InfTau(Ka_h_inf, Ka_tau_h))]
    
    const kca_gates = [GateSpec(:mKCa, 4, 0.0, InfTauCa(KCa_m_inf, KCa_tau_m))]
    
    const kdr_gates = [GateSpec(:mKdr, 4, 0.0, InfTau(Kdr_m_inf, Kdr_tau_m))]

    # 3. Build function 
    function build_liu_neuron(; name=:Liu_AB_Neuron)
        top = Scalar()
        nernst_factor = 500.0 * 8.6174e-5 * 283.15
        
        @named na_ch  = GenericChannel(topology=top, g=100.0, E_rev=50.0, gates=na_gates)
        @named cas_ch = CaVChannel(topology=top, g=3.0, conversion_factor=0.047, gates=cas_gates, Ca_out=3000.0, nernst_factor=nernst_factor)
        @named cat_ch = CaVChannel(topology=top, g=1.3, conversion_factor=0.047, gates=cat_gates, Ca_out=3000.0, nernst_factor=nernst_factor)
        @named ih_ch  = GenericChannel(topology=top, g=0.5, E_rev=-20.0, gates=ih_gates)
        @named ka_ch  = GenericChannel(topology=top, g=5.0, E_rev=-80.0, gates=ka_gates)
        @named kca_ch = KCaChannel(topology=top, g=10.0, E_rev=-80.0, gates=kca_gates)
        @named kdr_ch = GenericChannel(topology=top, g=20.0, E_rev=-80.0, gates=kdr_gates)
        @named leak   = GenericChannel(topology=top, g=0.01, E_rev=-50.0, gates=GateSpec[])
        
        @named cap = Capacitor(topology=top, C=1.0)
        channels = [na_ch, cas_ch, cat_ch, ih_ch, ka_ch, kca_ch, kdr_ch, leak]

        decay_fn = ca -> (0.05 .- ca) ./ 20.0
        ion_config = CalciumTracker(decay=decay_fn, Ca_init=0.05)

        comp = build_compartment(cap, channels; name=name, V_init=-60.0, topology=top, ion_config=ion_config)
        
        return comp
    end

    export build_liu_neuron

    nothing
end
```

## File: src/training/pem.jl
```julia
@component function PEMObservationChannel(; name, itps::AbstractVector, K_init=1.0, topology=Scalar())
    N = length(itps)
    
    if topology isa Scalar
        @named oneport = OnePort()
        @unpack v, i = oneport
        
        @parameters K = K_init
        params = SymbolicT[K]
        vars = SymbolicT[]
        
        # Explicitly use the first element of the array
        eqs = Equation[
            i ~ K * (v - itps[1](t))
        ]
        
        return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
        
    else
        @named oneport = VectorizedOnePort(N=N)
        @unpack v, i = oneport
        
        if K_init isa AbstractArray
            @parameters K[1:N] = K_init
        else
            @parameters K = K_init
        end
        params = SymbolicT[K]
        vars = SymbolicT[]
        
        # Clean, explicit unrolling to guarantee MTK shape inference
        target_vec = SymbolicT[itps[j](t) for j in 1:N]
        
        eqs = Equation[
            i ~ K .* (v .- target_vec)
        ]
        
        return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
    end
end
```

## File: src/library/PrinzCalciumNeuron.jl
```julia
module PrinzNeuron
    using ..MTKNeuralToolkit: GateSpec, GenericChannel, CaVChannel, KCaChannel, CalciumTracker, Capacitor, build_compartment, Scalar, build_acausal_network, SynapseSpec, CholSynapse, GlutSynapse
    import ..MTKNeuralToolkit: AbstractGeometry, get_capacitance, get_conductance, get_ca_conversion_factor, get_synaptic_conductance
    import ..MTKNeuralToolkit: InfTau, InfTauCa
    using ModelingToolkit: @named


    # Prinz uses a custom conversion factor to go from geometry to calcium flow so we recreate it
    Base.@kwdef struct PrinzGeometry <: AbstractGeometry
        C_m::Float64 = 10.0
        area::Float64 = 0.0628
    end

    get_capacitance(C, geom::PrinzGeometry) = geom.C_m
    get_conductance(g, geom::PrinzGeometry) = g * (geom.C_m / geom.area)
    # get_ca_conversion_factor(conv, geom::PrinzGeometry, tauCa) = (0.94 * geom.area) / (geom.C_m * tauCa) This seems more correct, but isn't what I was given on provided scripts and doesn't produce the stg bursting
    get_ca_conversion_factor(conv, geom::PrinzGeometry, tauCa) = (0.94) / (geom.C_m * tauCa)
    get_synaptic_conductance(g, geom::PrinzGeometry) = g * (1e-3 / geom.area^2)


    # 1. Define Inf and Tau functions based on Prinz equations
    # Na channel
    Na_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 25.5) ./ -5.29))
    Na_tau_m(v) = 2.64 .- 2.52 ./ (1 .+ exp.((v .+ 120.0) ./ -25.0))
    Na_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 48.9) ./ 5.18))
    Na_tau_h(v) = (1.34 ./ (1.0 .+ exp.((v .+ 62.9) ./ -10.0))) .* (1.5 .+ 1.0 ./ (1.0 .+ exp.((v .+ 34.9) ./ 3.6)))
    const na_gates = [GateSpec(:mNa, 3, 0.0, InfTau(Na_m_inf, Na_tau_m)), 
                      GateSpec(:hNa, 1, 0.0, InfTau(Na_h_inf, Na_tau_h))]

    # CaS channel
    CaS_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 33.0) ./ -8.1))
    CaS_tau_m(v) = 2.8 .+ 14.0 ./ (exp.((v .+ 27.0) ./ 10.0) .+ exp.((v .+ 70.0) ./ -13.0))
    CaS_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 60.0) ./ 6.2))
    CaS_tau_h(v) = 120.0 .+ 300.0 ./ (exp.((v .+ 55.0) ./ 9.0) .+ exp.((v .+ 65.0) ./ -16.0))
    const cas_gates = [GateSpec(:mCaS, 3, 0.0, InfTau(CaS_m_inf, CaS_tau_m)), 
                       GateSpec(:hCaS, 1, 0.0, InfTau(CaS_h_inf, CaS_tau_h))]

    # CaT channel
    CaT_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 27.1) ./ -7.2))
    CaT_tau_m(v) = 43.4 .- 42.6 ./ (1.0 .+ exp.((v .+ 68.1) ./ -20.5))
    CaT_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 32.1) ./ 5.5))
    CaT_tau_h(v) = 210.0 .- 179.6 ./ (1.0 .+ exp.((v .+ 55.0) ./ -16.9))
    const cat_gates = [GateSpec(:mCaT, 3, 0.0, InfTau(CaT_m_inf, CaT_tau_m)), 
                       GateSpec(:hCaT, 1, 0.0, InfTau(CaT_h_inf, CaT_tau_h))]

    # Ka channel
    Ka_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 27.2) ./ -8.7))
    Ka_tau_m(v) = 23.2 .- 20.8 ./ (1.0 .+ exp.((v .+ 32.9) ./ -15.2))
    Ka_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 56.9) ./ 4.9))
    Ka_tau_h(v) = 77.2 .- 58.4 ./ (1.0 .+ exp.((v .+ 38.9) ./ -26.5))
    const ka_gates = [GateSpec(:mKa, 3, 0.0, InfTau(Ka_m_inf, Ka_tau_m)), 
                     GateSpec(:hKa, 1, 0.0, InfTau(Ka_h_inf, Ka_tau_h))]

    # KCa channel
    KCa_m_inf(v, ca) = (ca ./ (ca .+ 3.0)) ./ (1.0 .+ exp.((v .+ 28.3) ./ -12.6))
    KCa_tau_m(v) = 180.6 .- 150.2 ./ (1.0 .+ exp.((v .+ 46.0) ./ -22.7))
    const kca_gates = [GateSpec(:mKCa, 4, 0.0, InfTauCa(KCa_m_inf, KCa_tau_m))]

    # Kdr channel
    Kdr_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 12.3) ./ -11.8))
    Kdr_tau_m(v) = 14.4 .- 12.8 ./ (1.0 .+ exp.((v .+ 28.3) ./ -19.2))
    const kdr_gates = [GateSpec(:mKdr, 4, 0.0, InfTau(Kdr_m_inf, Kdr_tau_m))]

    # H channel
    H_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 75.0) ./ 5.5))
    H_tau_m(v) = 2.0 ./ (exp.((v .+ 169.7) ./ -11.6) .+ exp.((v .- 26.7) ./ 14.3))
    const h_gates = [GateSpec(:mH, 1, 0.0, InfTau(H_m_inf, H_tau_m))]

    # 2. Build function
    function build_prinz_neuron(; name=:Prinz_Neuron, tauCa=200.0, Ca_inf=0.05, V_init=-50.0,
                                 gNa=100.0, gCaS=4.0, gCaT=2.0, gKa=10.0, gKCa=5.0, gKdr=10.0, gH=0.1, gleak=0.01,
                                 ENa=50.0, EK=-80.0, EH=-20.0, Eleak=-50.0, geom=PrinzGeometry())
        top = Scalar()
        nernst_factor = 500.0 * 8.6174e-5 * 283.15

        # Pass geom and tauCa to everything! The dispatch handles the math.
        @named na_ch  = GenericChannel(topology=top, g=gNa, E_rev=ENa, gates=na_gates, geometry=geom)
        @named cas_ch = CaVChannel(topology=top, g=gCaS, gates=cas_gates, Ca_out=3000.0, 
                                   nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
        @named cat_ch = CaVChannel(topology=top, g=gCaT, gates=cat_gates, Ca_out=3000.0, 
                                   nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
        @named ka_ch  = GenericChannel(topology=top, g=gKa, E_rev=EK, gates=ka_gates, geometry=geom)
        @named kca_ch = KCaChannel(topology=top, g=gKCa, E_rev=EK, gates=kca_gates, geometry=geom)
        @named kdr_ch = GenericChannel(topology=top, g=gKdr, E_rev=EK, gates=kdr_gates, geometry=geom)
        @named h_ch   = GenericChannel(topology=top, g=gH, E_rev=EH, gates=h_gates, geometry=geom)
        @named leak   = GenericChannel(topology=top, g=gleak, E_rev=Eleak, gates=GateSpec[], geometry=geom)

        @named cap = Capacitor(topology=top, geometry=geom)
        channels = [na_ch, cas_ch, cat_ch, ka_ch, kca_ch, kdr_ch, h_ch, leak]

        decay_fn = ca -> (Ca_inf .- ca) ./ tauCa
        ion_config = CalciumTracker(decay=decay_fn, Ca_init=Ca_inf)
        comp = build_compartment(cap, channels; name=name, V_init=V_init, topology=top, ion_config=ion_config)
        
        return comp
    end

    # 3. Build STG Network
    function build_stg(; name=:stg)
        geom = PrinzGeometry(area=0.0628, C_m=10.0)
        tauCa = 200.0
        Ca_inf = 0.05
        nernst_factor = 500.0 * 8.6174e-5 * 283.15
        prinz_ion_config = CalciumTracker(decay=ca -> (Ca_inf .- ca) ./ tauCa, Ca_init=Ca_inf)

        # Local Channel Builders
        NaCh(g; name)   = GenericChannel(name=name, g=g, E_rev=50.0, gates=na_gates, geometry=geom)
        CaSCh(g; name)  = CaVChannel(name=name, g=g, gates=cas_gates, Ca_out=3000.0, 
                                     nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
        CaTCh(g; name)  = CaVChannel(name=name, g=g, gates=cat_gates, Ca_out=3000.0, 
                                     nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
        HCh(g; name)    = GenericChannel(name=name, g=g, E_rev=-20.0, gates=h_gates, geometry=geom)
        KaCh(g; name)   = GenericChannel(name=name, g=g, E_rev=-80.0, gates=ka_gates, geometry=geom)
        KCaCh(g; name)  = KCaChannel(name=name, g=g, E_rev=-80.0, gates=kca_gates, geometry=geom)
        KdrCh(g; name)  = GenericChannel(name=name, g=g, E_rev=-80.0, gates=kdr_gates, geometry=geom)
        LeakCh(g; name) = GenericChannel(name=name, g=g, E_rev=-50.0, gates=GateSpec[], geometry=geom)

        # Local Neuron Builders
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

        AB = build_AB()
        PY = build_PY()
        LP = build_LP()
        neurons = [AB, PY, LP]

        # Synapses
        @named ABLP_chol = CholSynapse(g_max=30.0, geometry=geom)
        @named ABPY_chol = CholSynapse(g_max=3.0 , geometry=geom)
        @named ABLP_glut = GlutSynapse(g_max=30.0, geometry=geom)
        @named ABPY_glut = GlutSynapse(g_max=10.0, geometry=geom)
        @named LPAB_glut = GlutSynapse(g_max=30.0, geometry=geom)
        @named LPPY_glut = GlutSynapse(g_max=1.0 , geometry=geom)
        @named PYLP_glut = GlutSynapse(g_max=30.0, geometry=geom)

        synapse_specs = [
            SynapseSpec(LP.interfaces.V, AB.interfaces.V, AB.interfaces.I_syn, LPAB_glut),
            SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_chol),
            SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_glut),
            SynapseSpec(LP.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, LPPY_glut),
            SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_chol),
            SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_glut),
            SynapseSpec(PY.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, PYLP_glut)
        ]

        net = build_acausal_network(neurons; synapse_specs=synapse_specs, name=name)
        return net
    end

    export build_prinz_neuron, PrinzGeometry, na_gates, cas_gates, cat_gates, ka_gates, kca_gates, kdr_gates, h_gates, build_stg
end
```

## File: src/MTKNeuralToolkit.jl
```julia
module MTKNeuralToolkit

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D, connect, Pre
using ModelingToolkit: @component, @connector, @named, @parameters, @unpack, @variables, Equation, Flow, System, extend
using OrdinaryDiffEq
import ModelingToolkitStandardLibrary.Blocks: RealInput, RealInputArray
using Symbolics: SymbolicT



# ==========================================
# 1. Core Framework
# ==========================================
include("topology.jl")
export Scalar, Vectorized

include("geometry.jl")
export AbstractMorphology, NoMorphology, Morphology
include("components/electrical.jl")

include("components/channels.jl")
export Ground, Capacitor, CurrentSource, GenericChannel, GateSpec

include("components/calcium.jl")
include("components/synapses.jl")

include("network.jl")


export build_compartment, build_acausal_network, build_synapse_block

export Compartment, Network, SynapseSpec, CouplingSpec
export CaVChannel, KCaChannel, CalciumPool, CalciumTracker, NoCalcium, CaPort
export ExpSynapse, VectorizedExpSynapse, CholSynapse, GlutSynapse, GapJunction, STDPSynapse

export ContinuousLIFChannel
export InfTau, InfTauCa
export AbstractGeometry, NoGeometry, Geometry
export get_capacitance, get_conductance, get_ca_conversion_factor
export Pin, OnePort, TwoPort, VectorizedPin, VectorizedOnePort

include("training/pem.jl")
export PEMObservationChannel


# ==========================================
# 2. Standard Model Library (Submodules)
# ==========================================
include("library/HodgkinHuxley.jl")
export HodgkinHuxley

include("library/ContinuousSpikers.jl")
export ContinuousSpikers

include("library/LiuCalciumNeuron.jl")
export LiuCalciumNeuron

include("library/PrinzCalciumNeuron.jl") 
export PrinzCalciumNeuron                  

end
```
