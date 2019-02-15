# 2-D tile coding
# It is used to construct (binary) features for 2-D continuous state/input space + 1-D discrete action spacce
# for linear methods to do function approximation
import math
import numpy as np

# The following three lines are subject to change according to user need
# -- numTilings, x_start, x_range, y_start, y_range, num_actions

# Number of total tilings
numTilings = 8

# The starting point of each component of input and its range
# Please enter FLOATS (e.g use 0.0 for 0)
x_start, x_range = -1.2, 1.7
y_start, y_range = -0.07, 0.14

# Number of tiles per tiling in each dimension dimension
# (e.g. 8 means each tile covers 1/8 of the bounded distance in x-dimension)
x_num_partition = 7
y_num_partition = 7

# Number of actions available for each state
num_actions = 3

# Number of total tiles per tiling
# Notice we need one more in each dimension to ensure all tilings cover the input space
num_tiles_per_tiling = (x_num_partition + 1) * (y_num_partition + 1)




# Find the active tile indices of input (in1, in2) and store it to tileIndices
def tilecode(in1, in2, tileIndices):
    assert len(tileIndices) == numTilings

    # Change coordinate of input to be based on the first tiling, where the origin is set to
    # the left-bottom point of the first tiling
    in1 = in1 - x_start
    in2 = in2 - y_start

    # Compute the offset in each dimension per tiling
    x_offset_per_tiling = x_range / x_num_partition / numTilings
    y_offset_per_tiling = y_range / y_num_partition / numTilings

    # The height and width of each tile
    y_tile = y_range / y_num_partition
    x_tile = x_range / x_num_partition

    # Compute active tile indices for each tiling
    for idx in range(numTilings):
        # Compute the coordiate of input under the new tiling coordinate system
        x_offset, y_offset = idx * x_offset_per_tiling, idx * y_offset_per_tiling
        x, y = in1 + x_offset, in2 + y_offset

        # index = base + num_rows * 11/row + num_col
        index = idx * num_tiles_per_tiling + math.floor(y / y_tile) * (x_num_partition + 1) + math.floor(x / x_tile)

        # A sanity check: the index of tile ranges from 0 (the first tile in the first tiling)
        # to numTilings * num_tiles_per_tiling - 1 (the last tile in the last tiling)
        assert 0 <= index <= numTilings * num_tiles_per_tiling - 1

        # Write result back to tileIndices
        tileIndices[idx] = int(index)

# Get the feature vector of a single input=(in1, in2, action)
def feature(state, action):
    # Precondition check
    assert x_start <= state[0] <= (x_start + x_range) and y_start <= state[1] <= (y_start + y_range) and action in {-1, 0, 1}, 'input out of space'

    in1, in2 = state[0], state[1]
    tileIndices = [-1] * numTilings
    tilecode(in1, in2, tileIndices)
    result = np.zeros(num_tiles_per_tiling * numTilings * num_actions)
    for index in tileIndices:
            result[index + num_tiles_per_tiling * numTilings * (action+1)] += 1
    return result

# Test Code
def printTileCoderIndices(in1, in2, action):
    tileIndices = [-1] * numTilings
    tilecode(in1, in2, tileIndices)
    print('Active tile indices for input (' + str(in1) + ',' + str(in2) + ', ' + str(action) + ') are : ' + str(tileIndices))
    print('feature for this input is: ' + str(feature([in1,in2,action])))


if __name__ == '__main__':
    printTileCoderIndices(0.2, 0.01, 0)
    printTileCoderIndices(0.0, 0.07, 0)
    printTileCoderIndices(-0.5, 0.03, 1)
    printTileCoderIndices(-0.25, -0.01, 0)
