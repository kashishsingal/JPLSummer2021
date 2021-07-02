import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from GaussianProcessRegression import GP
from MogiSimulator import MogiSim
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from sklearn.metrics import mean_squared_error
import math

class LSOptimization:

    syntheticData = None
    ms = MogiSim()
    testingMatrix = None

    def __init__(self):
        pass

    def conductOptimization(self):

        # get vertical displacement synthetic data from mogi simulator
        vertDisplacementWithNoise = self.ms.createSynetheticData() #HAS NO NOISE
        self.syntheticData = vertDisplacementWithNoise
        self.testingMatrix = self.ms.testingX

        #intialParams - strength, x, y, depth
        # actual values: 64, 5, 10, 2
        p0 = np.array([100, 2, 50, 10]) #initial parameters 30, 10, 6, 6
        bnds = ((20, 80), (1, 15), (1, 15), (0.5, 10))

        # result = minimize(self.lossFunction, p0, method='Nelder-Mead', bounds = bnds, options={'disp': True, 'maxiter': 300})
        minimizer_kwargs = {"bounds": bnds}
        result = basinhopping(self.lossFunction, p0, minimizer_kwargs=minimizer_kwargs, stepsize=5, niter=100)
        #print(result.x) #prints the final parameters
        print("global minimum: strength = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f | f(x0) = %.4f" % (result.x[0], result.x[1], result.x[2], result.x[3], result.fun))

    def lossFunction(self, x):
        #surrogateMeans, surrogateStds = self.ms.createSurrogate(x)
        sampleCoordinates = np.mgrid[0:15:10j, 0:15:10j].reshape(2, -1).T
        sampleCoordinates = np.hstack((sampleCoordinates, np.full((len(sampleCoordinates), 1), x[3])))
        sampleCoordinates = np.hstack((np.full((len(sampleCoordinates), 1), x[0]), sampleCoordinates))
        surrogateMeans = self.ms.mogi(sampleCoordinates, x[1], x[2])
        rmse = math.sqrt(mean_squared_error(self.syntheticData[:, 2], surrogateMeans[:, 2]))
        print(rmse)
        print(x)
        var2 = self.testingMatrix - self.ms.testingX
        return rmse

if __name__ == '__main__':
    ls = LSOptimization()
    ls.conductOptimization()

# if(x[0] > 60 and x[0] < 65 and x[1] > 4.8 and x[1] < 5.2 and x[2] > 9.8 and x[2] < 10.2 and x[3] > 1.5 and x[3] < 2.5):
#     print(rmse)
#     print(x)

# print(np.sum(surrogateMeans[:, 2]))
# print(np.sum(self.syntheticData[:, 2]))