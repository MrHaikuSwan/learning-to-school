"""Simulation visualizer using matplotlib.

Notes:
I tried this first to implement a visualizer for a run of a simulation, but
matplotlib 1) looked bad, 2) had slow performance and 3) was very limiting.
I abandoned this once I learned that someone wrote a port of p5.js for Python.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from agents import AgentManager, FishDna


# Instantiate one simulation
# TODO: In the future, this will be more simulations, and it will run
# a genetic algorithm on these simulation

simulation_fish_dna = FishDna()
dims = np.array([1000, 1000])
manager = AgentManager(simulation_fish_dna, dims)
manager.populate(3, 3)

# Set up and run plotting
fig, ax = plt.subplots()
ax.set_xlim(0, dims[0])
ax.set_ylim(0, dims[1])
ax.set_aspect('equal', adjustable='box')

data, colors = manager.get_scatterplot_data()

scat = ax.scatter(data[:, 0], data[:, 1], c=colors)

def animation_update(frame):
    manager.update()
    data, _ = manager.get_scatterplot_data()
    scat.set_offsets(data)
    return [scat]

length_sec = 10
FPS = 60
interval = 1000 / FPS
nframes = FPS * length_sec
animation = FuncAnimation(fig,
                          func=animation_update,
                          frames=nframes,
                          cache_frame_data=False,
                          interval=interval,
                          blit=True)

plt.show()

