# ISNs
Repo for simulating Inhibitory Stabilized Networks

#
# Toy ISN Consisting of 70% excitatory neurons and 30% inhibitory neurons. Produces raster plot and line plot.
#


# Import Dependencies
import matplotlib.pyplot as plt
import pyNN.brian2 as sim
import numpy as np
# Patch for NumPy 2.0 compatibility with PyNN
if not hasattr(np, 'in1d'):
    np.in1d = np.isin



# Simulation Setup

# Timesteps (in milliseconds)
timestep = 0.1 
sim.setup(timestep=timestep, min_delay=0.1)

# Network Size and Population parameters
num_total = 1000
num_exc = int(0.70 * num_total) 
num_inh = int(0.30 * num_total)

# Simulation Duration (in milliseconds)
sim_time = 1000.0 

# Standard Leaky Integrate and Fire Neuron Parameters
cell_params = {"cm": .12,  # nF
               "tau_m": 20.0,  # ms
               "v_rest": -70.0,  # mV
               "v_reset": -65.0,  # mV
               "v_thresh": -50.0,  # mV
               "tau_refrac": 2.0,  # ms
}

# Excitatory and Inhibitory Populations
pop_exc = sim.Population(num_exc, sim.IF_curr_exp(**cell_params), label="Excitatory")
pop_inh = sim.Population(num_inh, sim.IF_curr_exp(**cell_params), label="Inhibitory")

# Poisson Background Noise
num_source = int(0.20 * num_total)
p_drive = sim.Population(num_source, sim.SpikeSourcePoisson(rate=15.0), label="Poisson Input")



# Connect External Input to the Populations
ext_connector = sim.FixedProbabilityConnector(p_connect = 0.2)

# Connection from Noise to Excitatory Population
sim.Projection(p_drive, 
               pop_exc, 
               ext_connector, 
               sim.StaticSynapse(weight=1.5,  delay=0.1), 
               receptor_type="excitatory")

# Connection from Noise to Inhibitory Population
sim.Projection(p_drive,
               pop_inh,
               ext_connector,
               sim.StaticSynapse(weight=1.5, delay=0.1),
               receptor_type="excitatory")



# Connect Recurrent Synapses
conn_prob = 0.1
rec_connector = sim.FixedProbabilityConnector(p_connect=conn_prob)

# Excitatory to Excitatory Weight
w_ee = 2.5  # Strong excitation drives potential instability

# Excitatory to Inhibitory Weight
w_ei = 3.0  # Strong E to I recruits inhibition

#Inhibitory to Excitatory Weight
w_ie = -4.5  # Strong I to E stabilizes the excitation

# Inhibitory to Inhibitory Weight
w_ii = -2.0  # I to I prevents runaway inhibition

# Connection from Excitatory to Excitatory 
sim.Projection(pop_exc,
               pop_exc,
               rec_connector,
               sim.StaticSynapse(weight=w_ee, delay=0.1),
               receptor_type="excitatory")

# Connection from Excitatory to Inhibitory 
sim.Projection(pop_exc,
               pop_inh,
               rec_connector,
               sim.StaticSynapse(weight=w_ei, delay=0.1),
               receptor_type="excitatory")

# Connection from Inhibitory to Excitatory
sim.Projection(pop_inh,
               pop_exc,
               rec_connector,
               sim.StaticSynapse(weight=w_ie, delay=0.1),
               receptor_type="inhibitory")

# Connection from Inhibitory to Inhibitory
sim.Projection(pop_inh,
               pop_inh,
               rec_connector,
               sim.StaticSynapse(weight=w_ii, delay=0.1),
               receptor_type="inhibitory")



# Setup Recordings
pop_exc.record("spikes")
pop_inh.record("spikes")



# Run Simulation
sim.run(sim_time)

# Extract recorded spike data
spikes_e = pop_exc.get_data().segments[0].spiketrains
spikes_i = pop_inh.get_data().segments[0].spiketrains

sim.end()



# Process Spike Data
# Calculate population firing rates over time bins
bin_width = 10.0  # ms
time_bins = np.arange(0, sim_time + bin_width, bin_width)
bin_centers = time_bins[:-1] + bin_width / 2.0

# Extract spike times
e_spike_times = np.concatenate([st.magnitude for st in spikes_e if len(st) > 0]) if len(spikes_e) > 0 else []
i_spike_times = np.concatenate([st.magnitude for st in spikes_i if len(st) > 0]) if len(spikes_i) > 0 else []

# Compute Histogram Counts
counts_e, _ = np.histogram(e_spike_times, bins=time_bins)
counts_i, _ = np.histogram(i_spike_times, bins=time_bins)

# Convert to firing rate (Hz per neuron)
rate_e = (counts_e / (num_exc * (bin_width / 1000.0)))
rate_i = (counts_i / (num_inh * (bin_width / 1000.0)))



# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Raster Plot
for idx, st in enumerate(spikes_e):
    if len(st) > 0:
        ax1.plot(st, np.ones_like(st) * idx, "|", color="tab:blue", alpha=0.6, markersize=2)
for idx, st in enumerate(spikes_i):
    if len(st) > 0:
        ax1.plot(st, np.ones_like(st) * (idx + num_exc), "|", color="tab:red", alpha=0.6, markersize=2)
ax1.set_title("Inhibitory Stabilized Network (ISN) Simulation")
ax1.set_ylabel("Neuron ID")
ax1.set_ylim(0, num_total)
ax1.axhline(num_exc, color="black", linestyle="--", linewidth=1, label="E/I Boundary")
ax1.legend(["Excitatory (70%)", "Inhibitory (30%)"], loc="upper right")

# Line Plot
ax2.plot(bin_centers, rate_e, 
label="Excitatory Rate", color="tab:blue", linewidth=1.8)
ax2.plot(bin_centers, rate_i, 
label="Inhibitory Rate", color="tab:red", linewidth=1.8)
ax2.set_xlabel("Time (ms)")
ax2.set_xlim(0, sim_time)
ax2.set_ylabel("Firing Rate (Hz)")
max_rate = 200
ax2.set_ylim(0, max_rate)
ax2.grid(True, linestyle="-", alpha=0.6)
ax2.legend(loc="upper right")

# Display Plots
plt.tight_layout()
plt.show()
