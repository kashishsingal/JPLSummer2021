import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from GaussianProcessRegression import GP
from MogiSimulator import MogiSim
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from sklearn.metrics import mean_squared_error
import math

class MyBounds:
    def __init__(self, xmin=[1, 1, 1, 1], xmax=[100, 15, 15, 10]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

class LSOptimization:

    syntheticData = None
    ms = MogiSim()

    def __init__(self):
        pass

    def conductOptimization(self):

        # get vertical displacement synthetic data from mogi simulator
        vertDisplacementWithNoise = self.ms.createSynetheticData() #HAS NO NOISE
        self.syntheticData = vertDisplacementWithNoise

        #intialParams - strength, x, y, depth
        # actual values: 64, 5, 10, 2
        p0 = np.array([64, 5, 10, 2]) #initial parameters 30, 10, 6, 6
        bnds = ((20, 80), (1, 15), (1, 15), (0.5, 10))

        # result = minimize(self.lossFunction, p0, method='Nelder-Mead', bounds = bnds, options={'disp': True, 'maxiter': 300})
        mybounds = MyBounds()
        minimizer_kwargs = {"bounds": bnds}
        result = basinhopping(self.lossFunction, p0, minimizer_kwargs=minimizer_kwargs, stepsize=2, niter=100, accept_test=mybounds)
        #print(result.x) #prints the final parameters
        print("global minimum: strength = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f | f(x0) = %.4f" % (result.x[0], result.x[1], result.x[2], result.x[3], result.fun))

    def lossFunction(self, x):
        print(x)
        surrogateMeans, surrogateStds = self.ms.createSurrogate(x)
        rmse = math.sqrt(mean_squared_error(self.syntheticData[:, 2], surrogateMeans[:, 2]))
        print(rmse)
        #print(np.sum(surrogateMeans[:, 2]))
        # print(np.sum(self.syntheticData[:, 2]))
        return rmse

if __name__ == '__main__':
    ls = LSOptimization()
    ls.conductOptimization()

