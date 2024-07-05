import sireneta as sna
print( 'SiReNetA:', sna.__version__ )

import matplotlib.pyplot as plt
from plot_specs import *
net = np.loadtxt('../Data/Testnet_N8.txt', dtype=int)
# Number of nodes
N = len(net)

# Visualize the connectivity matrix
plt.figure()
plt.title( 'Connectivity matrix' )
plt.imshow(net, cmap='gray_r')
plt.clim(0,net.max())
plt.colorbar()
plt.xlabel( 'node index' )
plt.ylabel( 'node index' )

plt.tight_layout()

# Find the largest eigenvalue of the connectivity matrix A
evs = np.linalg.eigvals(net)
evmax = evs.real.max()
# Calculate the largest possible tau
taumax = 1.0 / evmax

print( f'Spectral radius:\t{evmax:2.5f}' )
print( f'Largest possible tau:\t{taumax:2.5f}' )

from sireneta.responses.leaky_cascade import LeakyCascade

# Define the simulation parameters
# Set the temporal resolution
tfinal = 10
dt = 0.01

# Set the leakage time-constants τ, proportional to taumax
tau = 0.8 * taumax

# Define the stimulation amplitude to every node
stim = 1.0

r = LeakyCascade().configure(con=net, S0=stim, tau=tau, tmax=tfinal, timestep=dt)

print(f"Connection matrix size = {r.N}")
print(f"Matrix size = {r.N}")
print(f"Graph directed = {r.directed}")

r.simulate()

print(r.responses)

