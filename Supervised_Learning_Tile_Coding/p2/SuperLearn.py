from pylab import zeros, sin, cos, normal, random
from Tilecoder import numTilings, tilecode

n =  121 * numTilings  # initialize the total number of tiles across all tilings here
theta = [0] * n    # initialize weights appropriately here
alpha = 0.1 / numTilings   # initialize step size parameter appropriately here
tileIndices =  [-1] * numTilings # initialize your list of tile indices here

def f(in1, in2):
    tilecode(in1, in2, tileIndices) # Find active tile indices of input
    result = 0
    for index in tileIndices:
        result += theta[index] # Compute approximate function value of input
    return result


    
def learn(in1, in2, target):
    tilecode(in1, in2, tileIndices) # Find active tile indices of input
    error = target - f(in1, in2) # Compute the error b/t true value and approximated value
    for index in tileIndices:
        theta[index] = theta[index] + alpha * error # Update weight vector


def test1():
   for in1, in2, target in \
         [ (0.1, 0.1, 3.0), \
           (4.0, 2.0, -1.0), \
           (5.99, 5.99, 2.0), \
           (4.0, 2.1, -1.0) ]:
        before = f(in1, in2)
        learn(in1,in2, target)
        after = f(in1, in2)
        print('Example (' + str(in1) + ', ' + str(in2) + ', ' + str(target) + ')')
        print('    f before learning: ' + str(before)) 
        print('    f after learning : ' + str(after))
    
def targetFunction(in1, in2):
    return sin(in1 - 3.0) * cos(in2) + normal(0, 0.1)

def train(numSteps):
    for i in range(numSteps):
        in1 = random() * 6.0
        in2 = random() * 6.0
        target = targetFunction(in1, in2)
        learn(in1, in2, target)
    
def writeF(filename):
    fout = open(filename, 'w')
    steps = 50
    for i in range(steps):
        for j in range(steps):
            target = f(i * 6.0 / steps, j * 6.0 / steps)
            fout.write(repr(target) + ' ')
        fout.write('\n')
    fout.close()
        
def MSE(sampleSize):
    totalSE = 0.0
    for i in range(sampleSize):
        in1 = random() * 6.0
        in2 = random() * 6.0
        error = targetFunction(in1, in2) - f(in1, in2)
        totalSE = totalSE + error * error
    print('The estimated MSE: ', (totalSE / sampleSize))
    
def test2():
    train(20)
    writeF('f20')
    MSE(10000)
    for i in range(10):
        train(1000)
        MSE(10000)
    writeF('f10000')


if __name__ == '__main__':
    test2()
