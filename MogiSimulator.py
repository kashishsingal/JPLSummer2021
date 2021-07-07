"""
Performs Gaussian Process Regression with an example test case with bivariate input and univariate output

only functions kernel() and globalGaussianProcessRegression() are to be used for other test cases

author: Kashish Singal (NASA JPL)
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from GaussianProcessRegression import GP
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

    # the two variables below are now combined into a strength parameter
    # radius = 0.4
    # pressureChange = 1000

    def __init__(self):
        pass

    def createSynetheticData(self):
        testingX = np.mgrid[0:15:10j, 0:15:10j].reshape(2, -1).T
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

        trainingX = np.mgrid[strength-10.5:strength+10.5:numPoints, x-5.1:x+5.1:numPoints, y-5.1:y+5.1:numPoints, depth-3.1:depth+3.1:numPoints].reshape(4, -1).T
        trainingY = self.mogi(trainingX, xCenter = x, yCenter = y)

        return trainingX, trainingY

    def mogi(self, matrixPoints, xCenter, yCenter):  # coordinates are strength, x, y, and magnitude of cavity depth
        displacements = np.empty((len(matrixPoints), 3))  # initialize displacements array
        coordinates = copy.deepcopy(matrixPoints)
        coordinates[:, 1] = coordinates[:, 1] - xCenter
        coordinates[:, 2] = coordinates[:, 2] - yCenter
        Rvect = LA.norm(coordinates[:, 1:4], axis = 1).T
        magVect = coordinates[:, 0] * (1 - self.poisson) / self.shearModulus/(np.power(Rvect, 3))
        displacements = (coordinates[:, 1:4].T*magVect).T
        return displacements









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
    ms.simulateMogi()