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

iht_size = 2024
num_tilings = 8
num_tiles = 8

iht = IHT(2024)


def mc_tile_encoding(state):
    """
    Tile encoding function for states in mountain car gymnasium problem

    Args:
        - state (tuple): (position, velocity)

    Returns:
        - The list of the tiles corresponding to the given state
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

if __name__ == "__main__":
    print(mc_tile_encoding((-0.5, 0.1)))