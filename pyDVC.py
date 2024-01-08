"""Python implementation of Discrete Voronoi Chain algorithm.
   First described by Velić, Mirko & May, Dave & Moresi, Louis (2009)
   A Fast Robust Algorithm for Computing Discrete Voronoi Diagrams. J Math Model Algor. 8. 343-355. 10.1007/s10852-008-9097-6.
   http://dx.doi.org/10.1007/s10852-008-9097-6"""

import numpy as np

from numba import njit, typed, types, int64

cell_type = types.UniTuple(types.int64, 3)
cell_list_type = types.ListType(cell_type)


@njit
def tessellate(positions, bounds, cell_length, radii):
    """Performs weighted Voronoi tessellation"""
    if positions.shape[1] != 3:
        raise ValueError("Positions should be 3-dimensional!")
    if positions.shape[0] != radii.shape[0]:
        raise ValueError("Positions and radii must be the same length!")

    xmin = bounds[0][0]
    xmax = bounds[0][1]
    ymin = bounds[1][0]
    ymax = bounds[1][1]
    zmin = bounds[2][0]
    zmax = bounds[2][1]
    size_x = xmax - xmin
    size_y = ymax - ymin
    size_z = zmax - zmin

    n_cells_x = int(np.ceil(size_x / cell_length))
    n_cells_y = int(np.ceil(size_y / cell_length))
    n_cells_z = int(np.ceil(size_z / cell_length))

    # initialize 3D integer array all to -1 (un-owned)
    ownership_array = np.zeros((n_cells_x, n_cells_y, n_cells_z), dtype=int64) - 1

    n_particles = positions.shape[0]

    # initialise them as lists of lists
    L_C_i = typed.List.empty_list(cell_list_type)
    L_B_i = typed.List.empty_list(cell_list_type)

    for a_pid in range(0, n_particles):
        L_C_i.append(typed.List.empty_list(cell_type))
        L_B_i.append(typed.List.empty_list(cell_type))

    def get_cell_with_position(position):
        cells_along_x = int(np.floor((position[0] - xmin) / cell_length))
        cells_along_y = int(np.floor((position[1] - ymin) / cell_length))
        cells_along_z = int(np.floor((position[2] - zmin) / cell_length))
        return cells_along_x, cells_along_y, cells_along_z

    def get_position_of_cell(cell):
        x, y, z = cell
        x_pos = xmin + (x + 0.5) * cell_length
        y_pos = ymin + (y + 0.5) * cell_length
        z_pos = zmin + (z + 0.5) * cell_length
        return np.array([x_pos, y_pos, z_pos])

    def is_cell_valid(cell):
        cell_x, cell_y, cell_z = cell
        if cell_x < 0:
            return False
        if cell_x > n_cells_x - 1:
            return False
        if cell_y < 0:
            return False
        if cell_y > n_cells_y - 1:
            return False
        if cell_z < 0:
            return False
        if cell_z > n_cells_z - 1:
            return False
        return True

    def claim_cell(pid, cell):
        L_C_i[pid].append(cell)
        ownership_array[cell] = pid

    def battle_for_ownership(cell, current_owner_pid, potential_owner_pid):
        cell_com = get_position_of_cell(cell)

        current_owner_pos = positions[current_owner_pid, :]
        potential_owner_pos = positions[potential_owner_pid, :]

        # power diagram metric: d^2 - r^2

        current_owner_weight = radii[current_owner_pid] ** 2
        potential_owner_weight = radii[potential_owner_pid] ** 2

        current_owner_distance_squared = np.sum((current_owner_pos - cell_com) ** 2)
        potential_owner_distance_squared = np.sum((potential_owner_pos - cell_com) ** 2)

        current_owner_distance_weighted = current_owner_distance_squared - current_owner_weight
        potential_owner_distance_weighted = potential_owner_distance_squared - potential_owner_weight

        if current_owner_distance_weighted < potential_owner_distance_weighted:  # current owner is closer than new owner
            new_owner = current_owner_pid
        else:
            new_owner = potential_owner_pid

        return new_owner

    def update_boundary_chain(pid):
        # clear existing boundary chain list L_B for this grain
        L_B_i[pid].clear()
        for claimed_cell in L_C_i[pid]:
            # Get all neighbours for each newly claimed cell
            x_min_neighbour = (claimed_cell[0] - 1, claimed_cell[1], claimed_cell[2])
            x_max_neighbour = (claimed_cell[0] + 1, claimed_cell[1], claimed_cell[2])
            y_min_neighbour = (claimed_cell[0], claimed_cell[1] - 1, claimed_cell[2])
            y_max_neighbour = (claimed_cell[0], claimed_cell[1] + 1, claimed_cell[2])
            z_min_neighbour = (claimed_cell[0], claimed_cell[1], claimed_cell[2] - 1)
            z_max_neighbour = (claimed_cell[0], claimed_cell[1], claimed_cell[2] + 1)
            for neighbour in [x_min_neighbour,
                              x_max_neighbour,
                              y_min_neighbour,
                              y_max_neighbour,
                              z_min_neighbour,
                              z_max_neighbour]:
                if is_cell_valid(neighbour):
                    L_B_i[pid].append(neighbour)

    def claim_cells(pid):
        L_C_i[pid].clear()
        for boundary_cell in L_B_i[pid]:
            # if cell is free:
            boundary_cell_owner_pid = ownership_array[boundary_cell]
            if boundary_cell_owner_pid == -1:
                # boundary cell is free, so claim it
                claim_cell(pid, boundary_cell)
            elif boundary_cell_owner_pid == pid:
                # boundary cell already owned by me, so do nothing
                continue
            else:  # cell is owned by another particle, so we battle
                new_cell_owner = battle_for_ownership(boundary_cell, boundary_cell_owner_pid, pid)
                if new_cell_owner == pid:  # we won the battle, so claim it
                    claim_cell(pid, boundary_cell)

    def get_particle_cell(pid):
        # get the cell this particle is in
        return get_cell_with_position(positions[pid])

    def no_new_grains():
        # check if all L_C lists are empty
        running_length = 0
        for some_pid in range(0, n_particles):
            running_length += len(L_C_i[some_pid])

        if running_length == 0:
            return True
        else:
            return False

    # perform first loop over all particles
    # initialises the first cell for each particle
    for a_pid in range(0, n_particles):
        particle_starting_cell = get_particle_cell(a_pid)
        claim_cell(a_pid, particle_starting_cell)

    all_L_C_empty = False

    # second loop
    while not all_L_C_empty:
        # iterate over all the particles
        for a_pid in range(0, n_particles):
            # update the boundary chain for this particle
            update_boundary_chain(a_pid)
            # process the boundary chain
            claim_cells(a_pid)

        if no_new_grains():
            all_L_C_empty = True

    return ownership_array


def main():
    # Example on how to use the program:
    n_particles = 1000

    positions = np.random.rand(n_particles, 3) - 0.5
    weights = np.random.rand(n_particles) * 0.05

    bounds = np.array([
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5]
    ])

    ownership_array = tessellate(positions, bounds, 1 / 100, weights)
    print(ownership_array)


if __name__ == "__main__":
    main()
