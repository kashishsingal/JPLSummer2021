"""
Performs Gaussian Process Regression with an example test case with bivariate input and univariate output

only functions kernel() and globalGaussianProcessRegression() are to be used for other test cases

author: Kashish Singal (NASA JPL)
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from GaussianProcessRegression import GP
from scipy.optimize import minimize
import copy

class MogiSim:

    poisson = 0.25
    shearModulus = 1
    sourceX = 5
    sourceY = 10
    sourceDepth = 2
    sourceStrength = 64
    testingX = None
    trainingX = None
    trainingY = None
    gp = GP()

    # the two variables below are now combined into a strength parameter
    # radius = 0.4
    # pressureChange = 1000

    def __init__(self):
        pass

    def newMogi(self, xyLocations, params):  # coordinates are strength, x, y, and magnitude of cavity depth
        displacements = np.empty((len(xyLocations), 3))  # initialize displacements array
        coordinates = np.empty((len(xyLocations), 4))
        coordinates[:, 0] = np.full((len(xyLocations)), params[0])
        coordinates[:, 1] = xyLocations[:, 0] - params[1]
        coordinates[:, 2] = xyLocations[:, 1] - params[2]
        coordinates[:, 3] = np.full((len(xyLocations)), params[3])
        Rvect = LA.norm(coordinates[:, 1:4], axis = 1).T
        magVect = coordinates[:, 0] * (1 - self.poisson) / self.shearModulus/(np.power(Rvect, 3))
        displacements = (coordinates[:, 1:4].T*magVect).T
        return displacements

    def mogi(self, matrixPoints, xCenter, yCenter):  # coordinates are strength, x, y, and magnitude of cavity depth
        displacements = np.empty((len(matrixPoints), 3))  # initialize displacements array
        coordinates = copy.deepcopy(matrixPoints)
        coordinates[:, 1] = coordinates[:, 1] - xCenter
        coordinates[:, 2] = coordinates[:, 2] - yCenter
        Rvect = LA.norm(coordinates[:, 1:4], axis = 1).T
        magVect = coordinates[:, 0] * (1 - self.poisson) / self.shearModulus/(np.power(Rvect, 3))
        displacements = (coordinates[:, 1:4].T*magVect).T
        return displacements

    def createSynetheticData(self):
        testingX = np.mgrid[0:16:9j, 0:16:9j].reshape(2, -1).T
        testingX = np.hstack((testingX, np.full((len(testingX), 1), self.sourceDepth)))
        testingX = np.hstack((np.full((len(testingX), 1), self.sourceStrength), testingX))
        self.testingX = testingX

        return self.mogi(testingX, self.sourceX, self.sourceY)

    def createSurrogate(self, params):
        strength = params[0]
        x = params[1]
        y = params[2]
        depth = params[3]

        numPoints = 4j

        trainingX = np.mgrid[20:80:5j, 0:16:7j, 0:16:7j,
                    0.1:10:5j].reshape(4, -1).T
        trainingY = self.mogi(trainingX, xCenter=x, yCenter=y)
        return trainingX, trainingY

    def createTraining(self):
        numPoints = 4j

        trainingX = np.mgrid[40:80:5j, 0:16:9j, 0:16:9j,
                    0.1:10:5j].reshape(4, -1).T
        return trainingX



    def testStrength(self):
        synthetics = np.zeros((51, 1))
        surrogates = np.zeros((51, 1))
        strengths = np.linspace(0, 401, 51)
        theta0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        for strength in strengths:
            testingX = np.array([strength, 5, 10, 2])
            testingX = np.vstack((testingX, testingX))

            # training data
            trainingX = np.linspace(strength - 2.5, strength + 2.5, 4, endpoint=True)[None].T
            trainingX = np.hstack((trainingX, np.full((len(trainingX), 1), 5)))
            trainingX = np.hstack((trainingX, np.full((len(trainingX), 1), 10)))
            trainingX = np.hstack((trainingX, np.full((len(trainingX), 1), 2)))

            syntheticDataVertical = self.mogi(testingX, xCenter = 5, yCenter = 10)[:, 2]
            trainingY = self.mogi(trainingX, xCenter = 5, yCenter = 10)[:, 2]
            minimum = minimize(self.nll, x0=copy.deepcopy(theta0), args=(trainingX, trainingY), method='L-BFGS-B',
                                      bounds=[(1.0, 2.0), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0)],
                                      jac=True)  # tau and 4 hyperparam for each param
            #print("h1 = %.4f, h2 = %.4f, h3 = %.4f, h4 = %.4f, h5 = %.4f | f(x0) = %.4f" % (minimum.x[0], minimum.x[1], minimum.x[2], minimum.x[3], minimum.x[4], minimum.fun))
            surrogateMeans, surrogateStds = self.gp.globalGaussianProcessRegression(trainingX, trainingY,
                                                                            testingX, minimum.x)
            #works for more sampling in strengths, otherwise, it will underestimate the strength and
            #even predict a lower strength than the training data itself (overfitting potentially)
            #hyperparameters [1.0, 0.1, 0.1, 0.1, 0.1] make it a perfect fit

            synthetics[int(strength/8)] = syntheticDataVertical[0]
            surrogates[int(strength/8)] = surrogateMeans[0]

        plot = plt.figure(1)
        plt.plot(strengths, surrogates, c='b', marker='.')
        plt.plot(strengths, synthetics, c='r', marker='.')
        plt.legend(['Surrogates', 'Synthetics'])
        plt.title("Surrogate vs Synthetic Output for Vertical Displacement at source for varying strengths (with hyperparameter optimization)")
        plt.xlabel("Strengths")
        plt.ylabel("Vertical Displacements")
        plt.show()

    def nll(self, theta, trainingX, trainingY):
        cov_mat = self.gp.kernel(trainingX, trainingX, theta)
        cov_mat = cov_mat + 0.00005 * np.eye(len(trainingX))
        self.covar = cov_mat
        L = np.linalg.cholesky(cov_mat)
        alpha = np.dot(np.linalg.inv(L.T), np.dot(np.linalg.inv(L), trainingY))
        secondTerm = np.linalg.slogdet(cov_mat)
        prob = (- 0.5 * trainingY.reshape(1, -1).dot(alpha.reshape(-1, 1)) - 0.5 * secondTerm[1])[0][0]

        grads = np.zeros(len(theta))

        #tau hyperparameter
        firstTerm = np.matmul(alpha.reshape(-1,1),alpha.reshape(1,-1)) - np.dot(np.linalg.inv(L.T), np.linalg.inv(L))
        grads[0] = 0.5 * np.trace(np.dot(firstTerm, cov_mat/theta[0]))

        for i in range(1, len(theta)):
            column = trainingX[:, i - 1].reshape(-1, 1)
            squaredTerm = np.sum(column ** 2, 1).reshape(-1, 1) + np.sum(column ** 2, 1) - 2 * np.dot(column, column.T)
            dKdtheta = np.multiply(cov_mat, (-0.5 * squaredTerm))
            grads[i] = 0.5 * np.trace(np.dot(firstTerm, dKdtheta))
        return -prob, -grads


    def simulateMogi(self):

        trainingX = self.createCoordinates()
        trainingY = self.mogi(self.poisson, self.shearModulus, trainingX)
        #gp = GP()

        testingX = np.mgrid[-5:5:10j, -5:5:10j].reshape(2, -1).T
        testingX = np.hstack((testingX, np.full((len(testingX), 1), self.sourceDepth)))
        testingX = np.hstack((np.full((len(testingX), 1), self.sourceStrength), testingX))
        # testing x matrix with constant strength and depth but multiple x y coordinates

        mogidisplacements = self.mogi(self.poisson, self.shearModulus, testingX)
        mogidisplacementswithnoise = mogidisplacements + np.random.normal(0, 0.01, size = mogidisplacements.shape)
        #muVector, stdVector = gp.globalGaussianProcessRegression(trainingX, trainingY, testingX, 1.0, 1.0)


        #return mogidisplacementswithnoise[:, 2]
    #
    # def plotter(self, testingX, mogidisplacementswithnoise):

        plot = plt.figure(1)
        plt.plot(testingX[:10, 2], mogidisplacements[:10, 2], c='b', marker='.')
        plt.plot(testingX[:10, 2], mogidisplacementswithnoise[:10, 2], c='r', marker='.')
        plt.legend(['Mogi', 'Mogi with Gaussian noise'])
        plt.title("Mogi Synthetic Data")

        # plot = plt.figure(2)
        # plt.plot(testingX[:10, 2], muVector[:10, 2])
        # plot = plt.figure(1)
        # ax = plot.add_subplot(projection="3d")
        # ax.set_xlim(-12, 12)
        # ax.set_ylim(-12, 12)
        # ax.set_zlim(-12, 20)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        #
        # ax.scatter3D(mogidisplacements[:, 0], mogidisplacements[:, 1], mogidisplacements[:, 2], c='r', marker='.')

        plt.show()

    def createCoordinates(self):
        #trainingX = 20 * (np.random.rand(nTraining, 4)) - 10  # nTrainingx2 random points from -10 to 10
        trainingX = np.mgrid[20:100:6j, -7:7:6j, -7:7:6j, 0.1:10:6j].reshape(4, -1).T
        #testingX = np.mgrid[-10:10:nTestingImaginary, -10:10:nTestingImaginary, -10:10:nTestingImaginary].reshape(3, -1).T
        return trainingX

        # rowCounter = 0
        # for rowCoordinates in coordinates:  # traverse each row in coordinate array
        #     R = LA.norm(rowCoordinates[1:4])  # calculate distance to point
        #     strength = rowCoordinates[0]*(1 - poisson) / shearModulus  # calculate strength multiplier
        #     displacements[rowCounter] = strength * rowCoordinates[1:4] / (R ** 3)  # calculate respective displacements
        #     rowCounter += 1
        return displacements

    def plotter(coordinates, displacements):
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        z = np.zeros((100, 1))
        dx = displacements[:, 0]
        dy = displacements[:, 1]
        dz = displacements[:, 2]

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.quiver(x, y, z, dx, dy, dz)
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_zlim(-12, 12)
        ax.view_init(elev=10, azim=45)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

if __name__ == '__main__':
    ms = MogiSim()
    #ms.simulateMogi()
    ms.testStrength()