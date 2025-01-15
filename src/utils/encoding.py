import math
import numpy as np


class IHT:
    """
    Index Hash Table Class for tile encoding from http://incompleteideas.net/tiles/tiles3.html
    Developed by Richard S. Sutton
    """

    def __init__(self, sizeval):
        self.size = sizeval
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        return (
            "Collision table: size: "
            + str(self.size)
            + " overfullCount: "
            + str(self.overfullCount)
            + " dictionary: "
            + str(len(self.dictionary))
        )

    def count(self):
        return len(self.dictionary)

    def fullp(self):
        return len(self.dictionary) >= self.size

    def getindex(self, obj, readonly=False):
        if obj in self.dictionary:
            return self.dictionary[obj]
        elif readonly:
            return None
        size = self.size
        if len(self.dictionary) >= size:
            if self.overfullCount == 0:
                print("IHT full, starting to allow collisions")
            self.overfullCount += 1
            return hash(obj) % self.size
        else:
            self.dictionary[obj] = len(self.dictionary)
            return self.dictionary[obj]


def hashcoords(coordinates, m, readonly=False):
    if isinstance(m, IHT):
        return m.getindex(tuple(coordinates), readonly)
    if isinstance(m, int):
        return hash(tuple(coordinates)) % m
    if m is None:
        return coordinates


def get_tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
    qfloats = [int(math.floor(f * numtilings)) for f in floats]
    tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // numtilings)
            b += tilingX2
        coords.extend(ints)
        tiles.append(hashcoords(coords, ihtORsize, readonly))
    return tiles


POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

IHT_SIZE = 2024
NUM_TILINGS = 8
NUM_TILES = 8

iht = IHT(IHT_SIZE)


def mc_tile_encoding(
    state,
    iht: IHT = iht,
    num_tilings: int = NUM_TILINGS,
    num_tiles: int = NUM_TILES,
):
    """
    Tile encoding function for states in the Mountain Car environment.

    This function performs tile coding to transform continuous state variables
    (position and velocity) into discrete features using tile coding. Tile coding
    is useful in reinforcement learning for handling continuous state spaces by
    discretizing them into overlapping regions (tiles), which improves generalization.

    Args:
        - state (tuple): A tuple containing the current state of the environment,
            represented as (position, velocity).
            position (float): The car's position in the range [-1.2, 0.5].
            velocity (float): The car's velocity in the range [-0.07, 0.07].
        - iht (IHT): An instance of the Index Hash Table (IHT) used for managing tile encoding.
        - iht_size (int, optional): The size of the IHT. This controls the maximum number of unique tiles that can be created.
        - num_tilings (int, optional): The number of tiling grids
        - num_tiles (int, optional): The number of tiles along each dimension per tilinggrid. This controls the resolution of the tile encoding.

    Returns:
        numpy.ndarray: A numpy array of tile indices corresponding to the given state. Each index represents an active tile for the given state in the discretized space.
    """

    # Extract the position and velocity from the gymnasium state
    position, velocity = state

    # Scale position and velocity by multiplying the inputs of each by their scale
    position_scale = num_tiles / (POSITION_MAX - POSITION_MIN)
    velocity_scale = num_tiles / (VELOCITY_MAX - VELOCITY_MIN)

    # Obtain active tiles for current position and velocity
    tiles = get_tiles(
        iht, num_tilings, [position * position_scale, velocity * velocity_scale]
    )

    return np.array(tiles)


def cw_encoding(state):
    """
    Encoding function for state in the cliff walking environment of gymnasium.
    There is no need to encode state in this environment for our TDAgent class.

    Args:
        - state (int): a state in cliff walking environment

    Returns:
        - (list): a list with only the integer representing the state
    """
    return [state]


if __name__ == "__main__":
    state = (-1.0, 0.01)
    print(mc_tile_encoding(state))
