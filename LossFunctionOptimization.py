import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from GaussianProcessRegression import GP
from MogiSimulator import MogiSim
from scipy.optimize import minimize
from scipy.optimize import check_grad
from scipy.optimize import basinhopping
from sklearn.metrics import mean_squared_error
from GaussianProcessRegression import GP
import math

class LSOptimization:

    syntheticData = None
    ms = MogiSim()
    testingMatrix = None
    gp = GP()
    covar = None

    def __init__(self):
        pass

    def conductOptimization(self):

        # get vertical displacement synthetic data from mogi simulator
        vertDisplacementWithNoise = self.ms.createSynetheticData() #HAS NO NOISE
        self.syntheticData = vertDisplacementWithNoise
        self.testingMatrix = self.ms.testingX

        #estimate hyperparameters


        #intialParams - strength, x, y, depth
        # actual values: 64, 5, 10, 2
        p0 = np.array([100, 2, 50, 10]) #initial parameters 30, 10, 6, 6
        bnds = ((20, 80), (1, 15), (1, 15), (0.5, 10))

        # result = minimize(self.lossFunction, p0, method='Nelder-Mead', bounds = bnds, options={'disp': True, 'maxiter': 300})
        minimizer_kwargs = {"bounds": bnds}
        result = basinhopping(self.lossFunction, p0, minimizer_kwargs=minimizer_kwargs, stepsize=10, niter=60)
        #print(result.x) #prints the final parameters
        print("global minimum: strength = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f | f(x0) = %.4f" % (result.x[0], result.x[1], result.x[2], result.x[3], result.fun))


    def lossFunction(self, x):

        print(x)

        theta0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        trainingX, trainingY = self.ms.createSurrogate(x)

        vertTrainingY = trainingY[:, 2]
        meanY = np.mean(vertTrainingY)
        stdY = np.mean(vertTrainingY)
        vertTrainingY = (vertTrainingY - meanY)/stdY

        #options={'maxiter': 5},

        minimum = minimize(self.nll, x0=theta0, args=(trainingX, vertTrainingY), method='L-BFGS-B', bounds=[(1.0, 2.0), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0)], jac=False) #tau and 4 hyperparam for each param
        other = check_grad(self.nll, self.jacobianFunc, theta0, trainingX, vertTrainingY)
        hyperparameters = minimum.x
        surrogateMeans, surrogateStds = self.gp.globalGaussianProcessRegression(trainingX, vertTrainingY, self.testingMatrix, hyperparameters)

        surrogateMeans = surrogateMeans*stdY + meanY
        # muMatrix, stdVector = gp.globalGaussianProcessRegression(trainingX, trainingY, self.testingX, 1.0, 4)
        # return muMatrix, stdVector

        #Develop surrogate model output
        #surrogateMeans, surrogateStds = self.ms.createSurrogate(x)

        #replicate synthetic data x with diff parameters to see if loss function estimates actual parameters properly
        #without use of surrogate models
        # sampleCoordinates = np.mgrid[0:15:10j, 0:15:10j].reshape(2, -1).T
        # sampleCoordinates = np.hstack((sampleCoordinates, np.full((len(sampleCoordinates), 1), x[3])))
        # sampleCoordinates = np.hstack((np.full((len(sampleCoordinates), 1), x[0]), sampleCoordinates))
        # surrogateMeans = self.ms.mogi(sampleCoordinates, x[1], x[2])


        rmse = math.sqrt(mean_squared_error(self.syntheticData[:, 2], surrogateMeans))

        # print(rmse)
        # var2 = self.testingMatrix - self.ms.testingX

        return rmse

    def nll(self, theta, trainingX, trainingY):
        cov_mat = self.gp.kernel(trainingX, trainingX, theta)
        cov_mat = cov_mat + 0.00005 * np.eye(len(trainingX))
        self.covar = cov_mat
        L = np.linalg.cholesky(cov_mat)
        alpha = np.dot(np.linalg.inv(L.T), np.dot(np.linalg.inv(L), trainingY))
        secondTerm = np.linalg.slogdet(cov_mat)
        prob = - 0.5 * trainingY.T.dot(alpha) - 0.5 * secondTerm[1]
        print(prob, theta)

        # grads = np.zeros(5)
        # firstTerm = np.dot(alpha, alpha.T) - np.dot(np.linalg.inv(L.T), np.linalg.inv(L))
        # for i in range(1, 5):
        #     column = trainingX[:, i - 1][None].T
        #     squaredTerm = np.sum(column ** 2, 1).reshape(-1, 1) + np.sum(column ** 2, 1) - 2 * np.dot(column, column.T)
        #     dKdtheta = np.dot(cov_mat, -0.5 * squaredTerm)
        #     otherTerm = np.diag(np.dot(firstTerm, dKdtheta))
        #     grads[i] = 0.5 * np.trace(np.dot(firstTerm, dKdtheta))


        return -prob #, grads

    def jacobianFunc(self, hyperparameters, Xtrain, Ytrain):
        grads = np.zeros(5)
        L = np.linalg.cholesky(self.covar)
        alpha = np.matmul(np.linalg.inv(L.T), np.matmul(np.linalg.inv(L), Ytrain))
        firstTerm = np.dot(alpha, alpha.T) - np.linalg.inv(self.covar)
        for i in range(1, 5):
            column = Xtrain[:, i-1][None].T
            squaredTerm = np.sum(column ** 2, 1).reshape(-1, 1) + np.sum(column ** 2, 1) - 2 * np.dot(column, column.T)
            dKdtheta = np.multiply(self.covar, -0.5*squaredTerm)
            grads[i] = 0.5*np.trace(np.dot(firstTerm, dKdtheta))
        #return np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        return grads

if __name__ == '__main__':
    ls = LSOptimization()
    ls.conductOptimization()

# if(x[0] > 60 and x[0] < 65 and x[1] > 4.8 and x[1] < 5.2 and x[2] > 9.8 and x[2] < 10.2 and x[3] > 1.5 and x[3] < 2.5):
#     print(rmse)
#     print(x)

# print(np.sum(surrogateMeans[:, 2]))
# print(np.sum(self.syntheticData[:, 2]))