import math

numTilings = 8

    
def tilecode(in1, in2, tileIndices):
    # write your tilecoder here (5 lines or so)
    assert len(tileIndices) == numTilings
    for idx in range(numTilings):
        offset = idx * 0.6 / numTilings # A special case where x_offset is equal to y_offset
        x, y = in1 + offset, in2 + offset # Compute the coordiate of input under the new tiling coordinate system
        index = idx * 121 + math.floor(y / 0.6) * 11 + math.floor(x / 0.6) # index = base + num_rows * 11/row + num_col
        #assert 0 <= index <= 967
        tileIndices[idx] = int(index)

        
    
def printTileCoderIndices(in1, in2):
    tileIndices = [-1] * numTilings
    tilecode(in1, in2, tileIndices)
    print('Tile indices for input (', in1, ',', in2,') are : ', tileIndices)

printTileCoderIndices(0.1, 0.1)
printTileCoderIndices(4.0, 2.0)
printTileCoderIndices(5.99, 5.99)
printTileCoderIndices(4.0, 2.1)
    
