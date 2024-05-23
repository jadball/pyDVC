import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import regionprops_table

from pyDVC import tessellate

n_particles = 6000

# positions = np.random.rand(n_particles, 3) - 0.5

# create arbitrary-shaped 3D cuboid
positions_x = np.random.rand(n_particles) / 2 - 0.25
positions_y = np.random.rand(n_particles) / 2 - 0.25
positions_z = np.random.rand(n_particles) - 0.5

positions = np.column_stack((positions_x, positions_y, positions_z))

weights = np.clip(np.random.lognormal(0.005, 0.010, n_particles) - 1, 0.002, 0.2)
# lognormal distribution closer to a real polycrystal

# weights = np.random.rand(n_particles) * 0.005

bounds = np.array([
    [-0.25, 0.25],
    [-0.25, 0.25],
    [-0.5, 0.5]
])

ownership_array = tessellate(positions, bounds, 1 / 600, weights)

fig, ax = plt.subplots()

ax.imshow(ownership_array[0, :, :].T, cmap="magma")

print(len(np.unique(ownership_array)), "grains")

ax.set_xticks([])
ax.set_yticks([])

plt.show()

# the below commented block is a nice way to see the effect of increasing grid resolution

# ownership_array_1 = tessellate(positions, bounds, 1 / 6, weights)
# ownership_array_2 = tessellate(positions, bounds, 1 / 15, weights)
# ownership_array_3 = tessellate(positions, bounds, 1 / 25, weights)
# ownership_array_4 = tessellate(positions, bounds, 1 / 100, weights)
#
# fig, axs = plt.subplots(1, 4)
# axs[0].imshow(ownership_array_1[:, :, 0], cmap="magma")
# axs[1].imshow(ownership_array_2[:, :, 0], cmap="magma")
# axs[2].imshow(ownership_array_3[:, :, 0], cmap="magma")
# axs[3].imshow(ownership_array_4[:, :, 0], cmap="magma")
#
# axs[0].set_xticks([])
# axs[1].set_xticks([])
# axs[2].set_xticks([])
# axs[3].set_xticks([])
#
# axs[0].set_yticks([])
# axs[1].set_yticks([])
# axs[2].set_yticks([])
# axs[3].set_yticks([])
#
# axs[0].set_xlabel("(a): 6x6")
# axs[1].set_xlabel("(b): 15x15")
# axs[2].set_xlabel("(c): 25x25")
# axs[3].set_xlabel("(d): 100x100")
#
# plt.show()

# look at the relationship between initial weight (radius) and the sphere-equivalent diameter of the ownership array

props = regionprops_table(ownership_array, properties=['label', 'equivalent_diameter_area'])
props_df = pd.DataFrame(props)

weights_limited = [weights[i] for i in props_df["label"]]
props_df["weights"] = weights_limited

fig, ax = plt.subplots()
plt.scatter(props_df["weights"], props_df["equivalent_diameter_area"])

plt.show()
