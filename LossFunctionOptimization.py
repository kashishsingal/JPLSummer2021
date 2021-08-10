import copy
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from numpy import linalg as LA
from GaussianProcessRegression import GP
from MogiSimulator import MogiSim
from scipy.optimize import minimize
from scipy.optimize import check_grad
from scipy.optimize import basinhopping
from sklearn.metrics import mean_squared_error
from GaussianProcessRegression import GP
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import timeit
import pandas as pd
import xlrd
from smt.sampling_methods import LHS


class LSOptimization:
    syntheticData = None
    ms = MogiSim()
    testingMatrix = None
    gp = GP()
    gpr = None
    gprX = None
    gprY = None
    gprZ = None
    covar = None
    scaler = None
    groundStations = np.mgrid[1:15:20j, 1:15:20j].reshape(2, -1).T
    profile = np.linspace(1, 15, 30).reshape(-1, 1)
    constantVals = np.full(profile.shape, 3).reshape(-1, 1)
    profile1 = np.hstack((profile, constantVals))  # y = 3
    constantVals = np.full(profile.shape, 6).reshape(-1, 1)
    profile2 = np.hstack((constantVals, profile))  # x = 6
    constantVals = np.full(profile.shape, 5).reshape(-1, 1)
    profile3 = np.hstack((constantVals, profile))  # x = 5

    trainingX = None

    def __init__(self):
        pass

    def testGP(self):
        # create synthetic data
        vertDisplacementWithNoise = self.ms.createSynetheticData()[:, 2]
        mogidisplacementswithnoise = vertDisplacementWithNoise  # + np.random.normal(0, 0.0, size=vertDisplacementWithNoise.shape)
        self.syntheticData = mogidisplacementswithnoise
        self.testingMatrix = self.ms.testingX

        # create hyperparameters and training set
        x = [64, 5, 10, 2]  # initial set of parameters
        theta0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        trainingX, trainingY = self.ms.createSurrogate(x)

        # vertTrainingY = trainingY[:, 2][None].T
        # meanY = np.mean(vertTrainingY)
        # stdY = np.mean(vertTrainingY)
        # vertTrainingY = (vertTrainingY - meanY) / stdY
        #
        # minimum = minimize(self.nll, x0=copy.deepcopy(theta0), args=(trainingX, vertTrainingY), method='L-BFGS-B',
        #                           bounds=[(1.0, 2.0), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0)],
        #                           jac=True)  # tau and 4 hyperparam for each param
        # hyperparameters = minimum.x
        #
        # surrogateMeans, surrogateStds = self.gp.globalGaussianProcessRegression(trainingX, vertTrainingY,
        #                                                                         self.testingMatrix, hyperparameters)
        #
        # surrogateMeans = surrogateMeans * stdY + meanY

        gpr = GaussianProcessRegressor(normalize_y=True).fit(trainingX, trainingY[:, 2])
        surrogateMeans = gpr.predict(self.testingMatrix)

        plot = plt.figure(1)
        plt.plot(self.testingMatrix[:8, 2], surrogateMeans[:8], c='b', marker='.')
        plt.plot(self.testingMatrix[:8, 2], mogidisplacementswithnoise[:8], c='r', marker='.')
        plt.legend(['Mogi', 'Mogi with Gaussian noise'])
        plt.title("Mogi Interpolated Data vs. Synthetic Data (scikit GPR)")
        plt.show()

    def conductOptimization(self):

        # get vertical displacement synthetic data from mogi simulator
        # vertDisplacementWithNoise = self.ms.createSynetheticData() #HAS NO NOISE
        self.syntheticData = self.ms.newMogi(self.groundStations, np.array([6.4, 5, 10, 2]))[:, 2] # vertical displacements
        gaussianNoise = np.random.normal(0, 0.1*np.std(self.syntheticData), size = self.syntheticData.shape)
        print(gaussianNoise)
        self.syntheticData = self.syntheticData + gaussianNoise
        # self.testingMatrix = self.ms.testingX

        bounds = ((4.4, 8.4), (3, 7), (7, 13), (0.5, 3.5))
        numPoints = 100
        numGS = 20
        # strengthUniform = np.random.uniform(bounds[0][0], bounds[0][1], size=(numPoints, 1))
        # xUniform = np.random.uniform(bounds[1][0], bounds[1][1], size=(numPoints, 1))
        # yUniform = np.random.uniform(bounds[2][0], bounds[2][1], size=(numPoints, 1))
        # zUniform = np.random.uniform(bounds[3][0], bounds[3][1], size=(numPoints, 1))
        # self.trainingX = np.hstack((np.hstack((np.hstack((strengthUniform, xUniform)), yUniform)), zUniform))

        xlimits = np.array([[4.4, 8.4], [3, 7], [7, 13], [0.5, 3.5]])
        sampling = LHS(xlimits=xlimits)
        self.trainingX = sampling(numPoints)

        trainingY = np.empty((len(self.trainingX), len(self.groundStations)))

        count = 0
        for param in self.trainingX:
            displacements = self.ms.newMogi(self.groundStations, param)
            trainingY[count, :] = displacements[:, 2]
            count = count + 1

        self.scaler = StandardScaler().fit(self.trainingX)
        self.trainingX = self.scaler.transform(self.trainingX)

        kernel = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]), length_scale_bounds=(0.001, 100))
        self.gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=30)

        timestart = timeit.default_timer()
        self.gpr.fit(self.trainingX, trainingY)
        timeelaspsed = timeit.default_timer() - timestart
        print("Time taken to fit: %.4f" % (timeelaspsed))

        interpolatedData, stds = self.gpr.predict(self.scaler.transform(np.array([6.4, 5, 10, 2]).reshape(1, -1)), return_std=True)
        interpolatedData = interpolatedData.reshape(numGS, numGS)
        # print(stds)


        # actual values: 64, 5, 10, 2
        # p0 = np.array([100, 2, 50, 10])  # 100, 2, 50, 10
        # minimizer_kwargs = {"bounds": bounds}
        # timestart = timeit.default_timer()
        # result = basinhopping(self.lossFunction, p0, minimizer_kwargs=minimizer_kwargs, stepsize=2, niter=100)
        # timeelaspsed = timeit.default_timer() - timestart
        #
        # print("Time taken to run basinhopping: %.4f" % (timeelaspsed))
        # print()
        # print(
        #     "global minimum: strength = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f | f(x0) = %.4f" % (
        #     result.x[0], result.x[1], result.x[2], result.x[3], result.fun))
        #
        # bestData = self.gpr.predict(self.scaler.transform(result.x.reshape(1, -1))).reshape(numGS, numGS)

        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Mogi vs. Surrogate Vertical Displacement Output for Parameters 6.4 (GPa)($km^3$), 5 km, 10 km, 2 km')

        pltt = axs[0]
        pltt.set_aspect('equal')
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS), self.groundStations[:, 1].reshape(numGS, numGS), self.syntheticData.reshape(numGS, numGS),
                            150, vmin=0, vmax=1.2, cmap="magma")
        cbar = fig.colorbar(con, ax=pltt)
        cbar.ax.set_ylabel('Vertical Displacement (km)', rotation=90)
        cbar.ax.get_yaxis().labelpad = 10
        pltt.set_title("Mogi Model")
        pltt.set_xlabel("X (km)")
        pltt.set_ylabel("Y (km)")

        pltt = axs[1]
        pltt.set_aspect('equal')
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS), self.groundStations[:, 1].reshape(numGS, numGS), interpolatedData, 150,
                            cmap="magma", vmin=0, vmax=1.2)
        # con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
        #                     self.groundStations[:, 1].reshape(numGS, numGS), bestData, 10,
        #              cmap="magma", vmin=-5, vmax=15, extend='both')
        cbar = fig.colorbar(con, ax=pltt)
        cbar.ax.set_ylabel('Vertical Displacement (km)', rotation=90)
        cbar.ax.get_yaxis().labelpad = 10
        pltt.set_title("Surrogate Model")
        pltt.set_xlabel("X (km)")
        pltt.set_ylabel("Y (km)")

        # fig.savefig('test2png.png', dpi=100)

        fig, axs = plt.subplots(1, 1)
        axs.set_aspect('equal')
        con = axs.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS),
                            self.syntheticData.reshape(numGS, numGS) - interpolatedData, 150,
                            cmap="magma", vmin=-0.15, vmax=0.15)

        # plt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
        #              self.groundStations[:, 1].reshape(numGS, numGS),
        #              self.syntheticData.reshape(numGS, numGS) - bestData, 10,
        #              cmap="magma", vmin=-5, vmax=15)
        cbar = fig.colorbar(con, ax = axs)
        cbar.ax.set_ylabel('Vertical Displacement (km)', rotation=90)
        cbar.ax.get_yaxis().labelpad = 10
        plt.title("Difference Between Mogi and Surrogate Vertical Displacement Data")
        # axs.set_title("Difference Between Synthetic and Best Fit Model Vertical Displacement Data")
        axs.set_xlabel("X (km)")
        axs.set_ylabel("Y (km)")

        plt.show()
        #Synthetic Data
        #Best Fit Model



        # plot = plt.figure(1)
        # plt.plot(self.groundStations[:19, 1], self.syntheticData[:19].reshape(-1), 'r')
        # plt.plot(self.groundStations[:19, 1], self.gpr.predict(result.x.reshape(1, -1))[0, :19], 'b')
        # plt.xlabel("Y")
        # plt.ylabel("Vertical Displacement")
        # plt.title("Synthetic vs Surrogate Output along x=1")
        # plt.legend(["Synthetic", "Surrogate"])
        # plt.show()

    def lossFunction(self, x):
        surrogateMeans = self.gpr.predict(self.scaler.transform(x.reshape(1, -1)))
        rmse = math.sqrt(mean_squared_error(self.syntheticData.reshape(1, -1), surrogateMeans))
        return rmse

    def conductOptimization2(self):
        #for displacements in all 3 dimensions
        self.syntheticData = self.ms.newMogi(self.groundStations, np.array([64, 5, 10, 2])) # vertical displacements
        gaussianNoise = np.random.normal(0, 0.01, size = self.syntheticData.shape)
        self.syntheticData = self.syntheticData + gaussianNoise
        self.syntheticData = abs(self.syntheticData)
        # self.testingMatrix = self.ms.testingX

        bounds = ((44, 84), (3, 7), (7, 13), (0.5, 3.5))
        numPoints = 80
        numGS = 20

        xlimits = np.array([[44, 84], [3, 7], [7, 13], [0.5, 3.5]])
        sampling = LHS(xlimits=xlimits)
        self.trainingX = sampling(numPoints)

        trainingYx = np.empty((len(self.trainingX), len(self.groundStations)))
        trainingYy = np.empty((len(self.trainingX), len(self.groundStations)))
        trainingYz = np.empty((len(self.trainingX), len(self.groundStations)))

        count = 0
        for param in self.trainingX:
            displacements = abs(self.ms.newMogi(self.groundStations, param))
            trainingYx[count, :] = displacements[:, 0]
            trainingYy[count, :] = displacements[:, 1]
            trainingYz[count, :] = displacements[:, 2]
            count = count + 1

        self.scaler = StandardScaler().fit(self.trainingX)
        self.trainingX = self.scaler.transform(self.trainingX)

        kernelX = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]), length_scale_bounds=(0.001, 100))
        kernelY = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]),
                                            length_scale_bounds=(0.001, 100))
        kernelZ = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]),
                                            length_scale_bounds=(0.001, 100))

        self.gprX = GaussianProcessRegressor(kernel=kernelX, normalize_y=True, n_restarts_optimizer=30)
        self.gprY = GaussianProcessRegressor(kernel=kernelY, normalize_y=True, n_restarts_optimizer=30)
        self.gprZ = GaussianProcessRegressor(kernel=kernelZ, normalize_y=True, n_restarts_optimizer=30)

        timestart = timeit.default_timer()
        self.gprX.fit(self.trainingX, trainingYx)
        self.gprY.fit(self.trainingX, trainingYy)
        self.gprZ.fit(self.trainingX, trainingYz)
        timeelaspsed = timeit.default_timer() - timestart
        print("Time taken to fit: %.4f" % (timeelaspsed))

        interpolatedDataX, stdsX = self.gprX.predict(self.scaler.transform(np.array([64, 5, 10, 2]).reshape(1, -1)), return_std=True)
        interpolatedDataX = interpolatedDataX.reshape(numGS, numGS)

        interpolatedDataY, stdsY = self.gprY.predict(self.scaler.transform(np.array([64, 5, 10, 2]).reshape(1, -1)), return_std=True)
        interpolatedDataY = interpolatedDataY.reshape(numGS, numGS)

        interpolatedDataZ, stdsZ = self.gprZ.predict(self.scaler.transform(np.array([64, 5, 10, 2]).reshape(1, -1)), return_std=True)
        interpolatedDataZ = interpolatedDataZ.reshape(numGS, numGS)


        # actual values: 64, 5, 10, 2
        p0 = np.array([100, 2, 50, 10])  # 100, 2, 50, 10
        minimizer_kwargs = {"bounds": bounds}
        timestart = timeit.default_timer()
        result = basinhopping(self.lossFunction2, p0, minimizer_kwargs=minimizer_kwargs, stepsize=2, niter=100)
        timeelaspsed = timeit.default_timer() - timestart

        print("Time taken to run basinhopping: %.4f" % (timeelaspsed))
        print()
        print(
            "global minimum: strength = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f | f(x0) = %.4f" % (
            result.x[0], result.x[1], result.x[2], result.x[3], result.fun))

        bestDataX = self.gprX.predict(self.scaler.transform(result.x.reshape(1, -1))).reshape(numGS, numGS)
        bestDataY = self.gprY.predict(self.scaler.transform(result.x.reshape(1, -1))).reshape(numGS, numGS)
        bestDataZ = self.gprZ.predict(self.scaler.transform(result.x.reshape(1, -1))).reshape(numGS, numGS)

        fig1, axs = plt.subplots(1, 2)
        fig1.suptitle('Mogi vs. Surrogate Output for Parameters 64 GPa/m^3, 5m, 10m, 2m in X direction')

        pltt = axs[0]
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS), self.groundStations[:, 1].reshape(numGS, numGS), self.syntheticData[:, 0].reshape(numGS, numGS), 10, vmin=-5, vmax=15, cmap="magma", extend='both')
        fig1.colorbar(con, ax=pltt)
        pltt.set_title("Synthetic Data")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        pltt = axs[1]
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS), bestDataX, 10,
                            cmap="magma", vmin=-5, vmax=15, extend='both')
        fig1.colorbar(con, ax=pltt)
        pltt.set_title("Best Fit Model Data")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        fig2, axs = plt.subplots(1, 2)
        fig2.suptitle('Mogi vs. Surrogate Output for Parameters 64 GPa/m^3, 5m, 10m, 2m in Y direction')

        pltt = axs[0]
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS),
                            self.syntheticData[:, 1].reshape(numGS, numGS), 10, vmin=-5, vmax=15, cmap="magma",
                            extend='both')
        fig2.colorbar(con, ax=pltt)
        pltt.set_title("Synthetic Data")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        pltt = axs[1]
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS), bestDataY, 10,
                            cmap="magma", vmin=-5, vmax=15, extend='both')
        fig2.colorbar(con, ax=pltt)
        pltt.set_title("Best Fit Model Data")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        fig3, axs = plt.subplots(1, 2)
        fig3.suptitle('Mogi vs. Surrogate Output for Parameters 64 GPa/m^3, 5m, 10m, 2m in Z direction')

        pltt = axs[0]
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS),
                            self.syntheticData[:, 2].reshape(numGS, numGS), 10, vmin=-5, vmax=15, cmap="magma",
                            extend='both')
        fig3.colorbar(con, ax=pltt)
        pltt.set_title("Synthetic Data")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        pltt = axs[1]
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS), bestDataZ, 10,
                            cmap="magma", vmin=-5, vmax=15, extend='both')
        fig3.colorbar(con, ax=pltt)
        pltt.set_title("Best Fit Model Data")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")



        plt.show()

        #Synthetic Data
        #Best Fit Model


    def lossFunction2(self, x):
        surrogateMeansX = self.gprX.predict(self.scaler.transform(x.reshape(1, -1)))
        surrogateMeansY = self.gprY.predict(self.scaler.transform(x.reshape(1, -1)))
        surrogateMeansZ = self.gprZ.predict(self.scaler.transform(x.reshape(1, -1)))

        rmseX = math.sqrt(mean_squared_error(self.syntheticData[:, 0].reshape(1, -1), surrogateMeansX))
        rmseY = math.sqrt(mean_squared_error(self.syntheticData[:, 1].reshape(1, -1), surrogateMeansY))
        rmseZ = math.sqrt(mean_squared_error(self.syntheticData[:, 2].reshape(1, -1), surrogateMeansZ))

        rmse = rmseX + rmseY + rmseZ

        return rmse

    def conductOptimization3(self):
        self.syntheticData = self.ms.newMogi(self.groundStations, np.array([64, 5, 10, 2]))[:,
                             2]  # vertical displacements
        trainingParams = self.ms.createTraining()
        trainingY = np.empty((len(trainingParams), 1))
        count = 0
        for param in trainingParams:
            displacements = self.ms.newMogi(self.groundStations, param)
            trainingY[count] = math.sqrt(
                mean_squared_error(self.syntheticData.reshape(1, -1), displacements[:, 2].reshape(1, -1)))
            count = count + 1

        # intialParams - strength, x, y, depth
        # actual values: 64, 5, 10, 2
        p0 = np.array([100, 2, 50, 10])  # 100, 2, 50, 10
        bnds = ((20, 80), (1, 15), (1, 15), (0.5, 10))
        timestart = timeit.default_timer()
        self.gpr = GaussianProcessRegressor(normalize_y=True).fit(trainingParams, trainingY)
        timeelaspsed = timeit.default_timer() - timestart
        print(timeelaspsed)
        minimizer_kwargs = {"bounds": bnds}
        # con = [{"type": "ineq", "fun": self.lossFunction2}]
        result = basinhopping(self.lossFunction2, p0, minimizer_kwargs=minimizer_kwargs, stepsize=10,
                              niter=500)  # , accept_test=self.a)
        print(
            "global minimum: strength = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f | f(x0) = %.4f" % (
                result.x[0], result.x[1], result.x[2], result.x[3], result.fun))

    # ACTUAL: 64, 5, 10, 2
    # global minimum: strength = 75.5552, source x = 4.9812, source y = 9.9901, source depth = 2.3036 | f(x0) = 0.2931 (1)
    # global minimum: strength = 79.9998, source x = 10.5203, source y = 5.4762, source depth = 3.8920 | f(x0) = 0.0842 (2) w/ at
    # global minimum: strength = 79.9998, source x = 1.7368, source y = 14.2625, source depth = 1.6541 | f(x0) = -20.9072 w/o at
    def lossFunction3(self, x):
        print(x)
        return self.gpr.predict(x.reshape(1, -1))[0][0]

    def a(self, f_new, x_new, f_old, x_old):
        if f_new < 0:
            return False
        return True

    def optimizationLoop(self):
        timestart = timeit.default_timer()
        # create ground statiomns
        self.syntheticData = self.ms.newMogi(self.groundStations, np.array([64, 5, 10, 2]))[:,
                             2]  # vertical displacements

        self.trainingX = np.mgrid[40:80:5j, 0:16:5j, 0:16:5j, 0.1:10:5j].reshape(4, -1).T  # initial training data set

        # actual: 64, 5, 10, 2
        p0 = np.array([1, 1, 1, 1])  # 100, 2, 50, 10
        bnds = ((40, 80), (1, 15), (1, 15), (0.5, 10))  # assume bounds aren't changing through each iteration
        minimizer_kwargs = {"bounds": bnds}
        rangePoints = np.array([20, 5, 5, 5])
        stepCounter = rangePoints / 20
        allVals = np.array([1, 1, 1, 1, None, None, None, None, None, None])

        minRMSE = 100
        minParams = p0

        for i in range(20):
            # create trainingY data
            trainingY = np.empty((len(self.trainingX), len(self.groundStations)))
            count = 0
            for param in self.trainingX:
                displacements = self.ms.newMogi(self.groundStations, param)
                trainingY[count, :] = displacements[:, 2]
                count = count + 1

            # conduct regression
            self.gpr = GaussianProcessRegressor(normalize_y=True).fit(self.trainingX, trainingY)

            # predict parameter combination that best results in synthetic data from surrogate model
            result = basinhopping(self.lossFunction, p0, minimizer_kwargs=minimizer_kwargs, stepsize=10, niter=200,
                                  T=1000)
            print(
                "global minimum: strength = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f | f(x0) = %.4f" % (
                    result.x[0], result.x[1], result.x[2], result.x[3], result.fun))
            # if(result.fun < 0.15):
            #     self.ress = result.x
            #     break

            # if(result.fun < prevRMS):
            # prevRMS = result.fun

            if (result.fun < minRMSE):
                minRMSE = result.fun
                minDisplacements = self.gpr.predict(result.x.reshape(1, -1))

            allVallCurr = np.hstack((np.array(
                [result.x[0], result.x[1], result.x[2], result.x[3], result.fun, len(self.trainingX)]).reshape(1, -1),
                                     rangePoints.reshape(1, -1)))
            allVals = np.vstack((allVals, allVallCurr))

            numPointsToAdd = 5j
            rangePoints = rangePoints - stepCounter
            print(rangePoints)

            # generate more points focused around minimum point in parameter space
            # addedPoints = np.mgrid[result.x[0] - rangePoints:result.x[0] + rangePoints:numPointsToAdd,
            #               result.x[1] - rangePoints:result.x[1] + rangePoints:numPointsToAdd,
            #               result.x[2] - rangePoints:result.x[2] + rangePoints:numPointsToAdd,
            #               result.x[3] - rangePoints:result.x[3] + rangePoints:numPointsToAdd].reshape(4, -1).T
            #
            # self.trainingX = np.vstack((self.trainingX, addedPoints))

            # or shift the entire parameter space
            addedPoints = np.mgrid[result.x[0] - rangePoints[0]:result.x[0] + rangePoints[0]:numPointsToAdd,
                          result.x[1] - rangePoints[1]:result.x[1] + rangePoints[1]:numPointsToAdd,
                          result.x[2] - rangePoints[2]:result.x[2] + rangePoints[2]:numPointsToAdd,
                          result.x[3] - rangePoints[3]:result.x[3] + rangePoints[3]:numPointsToAdd].reshape(4, -1).T
            self.trainingX = copy.deepcopy(addedPoints)
            print(i)

        plot = plt.figure(1)
        plt.plot(self.groundStations[:19, 1], self.syntheticData[:19].reshape(-1), 'r')
        plt.plot(self.groundStations[:19, 1], minDisplacements[0, :19], 'b')
        plt.xlabel("y")
        plt.ylabel("Vertical Displacement")
        plt.title("Synthetic vs Surrogate Output")
        plt.legend(["Synthetic", "Surrogate"])
        plt.show()

        df = pd.DataFrame(allVals)
        df.to_clipboard(index=False, header=False)
        print(timeit.default_timer() - timestart)
        print()

    def optimizationLoop2(self):
        secondDisplacements = None
        lastDisplacements = None

        start = timeit.default_timer()
        # create ground stations
        self.syntheticData = self.ms.newMogi(self.profile3, np.array([64, 5, 10, 2]))[:, 2]  # vertical displacements

        self.trainingX = np.mgrid[44:84:3j, 3:7:3j, 7:13:3j, 0.5:3.5:3j].reshape(4, -1).T  # initial training data set

        # actual: 64, 5, 10, 2
        p0 = np.array([1, 1, 1, 1])  # 100, 2, 50, 10
        bnds = ((44, 84), (3, 7), (7, 13), (0.5, 3.5))  # assume bounds aren't changing through each iteration
        minimizer_kwargs = {"bounds": bnds}
        allVals = np.array([1, 1, 1, 1, None])
        # create trainingY data
        trainingY = np.empty((len(self.trainingX), len(self.profile3)))
        count = 0
        # m52 = Matern()
        # kernel = ConstantKernel() * RBF(length_scale=np.array([0.1, 0.1, 0.1, 0.1]))  # + WhiteKernel()
        kernel = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]), length_scale_bounds=(0.001, 100))
        self.gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=30)
        for param in self.trainingX:
            displacements = self.ms.newMogi(self.profile3, param)
            trainingY[count, :] = displacements[:, 2]
            # if(count == 1):
            #     self.gpr.fit(self.trainingX[0:2, :], trainingY[0:2, :])
            #     print()
            count = count + 1

        # Xtotal = np.mgrid[40:80:41j, 0:16:17j, 0:16:17j, 0.1:10:11j].reshape(4, -1).T

        for i in range(50):
            print(i)
            # conduct regression
            print("Time to fit:")
            startt = timeit.default_timer()
            self.scaler = StandardScaler().fit(self.trainingX)
            X_scaled = self.scaler.transform(self.trainingX)
            self.gpr.fit(X_scaled, trainingY)
            print(self.gpr.kernel_)
            print(timeit.default_timer() - startt)

            print("basinhopping")
            X_next = self.propose_location(self.expected_improvement, self.trainingX, trainingY, self.gpr, bnds)

            surrogateMeans = self.gpr.predict(self.scaler.transform(X_next.reshape(1, -1)))
            rmse = math.sqrt(mean_squared_error(self.syntheticData.reshape(1, -1), surrogateMeans))

            if (i == 0):
                secondDisplacements = surrogateMeans
            elif (i == 49):
                lastDisplacements = surrogateMeans

            print(
                "added point: strength = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f, RMSE = %.4f" % (
                    X_next[0], X_next[1], X_next[2], X_next[3], rmse))

            X_next = np.squeeze(X_next)
            allVallCurr = np.array([X_next[0], X_next[1], X_next[2], X_next[3], rmse])
            allVals = np.vstack((allVals, allVallCurr))

            # for adding multiple training points

            # addedPoints = np.mgrid[X_next[0] - 10:X_next[0] + 10:3j,
            #               X_next[1] - 3:X_next[1] + 3:3j,
            #               X_next[2] - 3:X_next[2] + 3:3j,
            #               X_next[3] - 2:X_next[3] + 2:3j].reshape(4, -1).T
            #
            # self.trainingX = np.vstack((self.trainingX, addedPoints))

            # for param in addedPoints:
            #     displacements = self.ms.newMogi(self.groundStations, param)
            #     trainingY = np.vstack((trainingY, displacements[:, 2]))

            # for adding only a single training point

            Y_next = self.ms.newMogi(self.profile3, X_next.reshape(-1, 1))[:, 2]
            # Add sample to previous samples
            self.trainingX = np.vstack((self.trainingX, X_next.reshape(1, -1)))
            trainingY = np.vstack((trainingY, Y_next.reshape(1, -1)))

        df = pd.DataFrame(allVals)
        df.to_clipboard(index=False, header=False)
        print(timeit.default_timer() - start)

        plot = plt.figure(1)
        plt.plot(self.profile3[:, 1], self.syntheticData[:].reshape(-1), 'k')
        plt.plot(self.profile3[:, 1], secondDisplacements[0, :], 'r')
        plt.plot(self.profile3[:, 1], lastDisplacements[0, :], 'b')
        # plt.plot(self.groundStations[:19, 1], self.syntheticData[:19].reshape(-1), 'k')
        # plt.plot(self.groundStations[:19, 1], secondDisplacements[0, :19], 'r')
        # plt.plot(self.groundStations[:19, 1], lastDisplacements[0, :19], 'b')
        plt.xlabel("Y (along X=5)")
        plt.ylabel("Vertical Displacement")
        plt.title("Synthetic vs. Surrogate for profile along x=5")
        plt.legend(["Synthetic", "First Predicted Parameters", "Last Predicted Parameters"])

        # plot = plt.figure(2)
        # ax = plot.add_subplot(projection="3d")
        # ax.set_xlim(0, 21)
        # ax.set_ylim(0, 21)
        # ax.set_zlim(-1, 15)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        #
        # ax.scatter3D(self.groundStations[:, 0], self.groundStations[:, 1], self.syntheticData, c='r', marker='.')
        # ax.scatter3D(self.groundStations[:, 0], self.groundStations[:, 1], lastDisplacements, c='b', marker='.')
        # plt.title("Synthetic vs Final Gaussian Process Estimated Displacements")
        # plt.legend(["Synthetic", "Last Predicted"])

        plt.show()

    def propose_location(self, acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=10):
        '''
        Proposes the next sampling point by optimizing the acquisition function.

        Args:
            acquisition: Acquisition function.
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.

        Returns:
            Location of the acquisition function maximum.
        '''

        min_val = 1e10
        min_x = None

        def min_obj(X, mu_sample_opt):
            # Minimization objective is the negative acquisition function
            return -acquisition(X.reshape(1, -1), mu_sample_opt, X_sample, Y_sample, gpr)

        # Find the best optimum by starting from n_restart different random points.

        strengthUniform = np.random.uniform(bounds[0][0], bounds[0][1], size=(n_restarts, 1))
        xUniform = np.random.uniform(bounds[1][0], bounds[1][1], size=(n_restarts, 1))
        yUniform = np.random.uniform(bounds[2][0], bounds[2][1], size=(n_restarts, 1))
        zUniform = np.random.uniform(bounds[3][0], bounds[3][1], size=(n_restarts, 1))

        uniformX = np.hstack((np.hstack((np.hstack((strengthUniform, xUniform)), yUniform)), zUniform))

        mu_sample = gpr.predict(self.scaler.transform(X_sample))
        mu_sample_rms = np.empty((len(mu_sample), 1))
        for i in range(len(mu_sample)):
            mu_sample_rms[i] = math.sqrt(
                mean_squared_error(self.syntheticData.reshape(1, -1), mu_sample[i].reshape(1, -1)))

        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [1]
        mu_sample_opt = np.min(mu_sample_rms)
        #bnds = ((40, 80), (1, 15), (1, 15), (0.5, 10))  # assume bounds aren't changing through each iteration
        minimizer_kwargs = {"bounds": bounds, "args": (mu_sample_opt)}

        x0 = np.array([45, 2, 2, 1])
        # basinhopping function
        start = timeit.default_timer()
        res = basinhopping(min_obj, x0, minimizer_kwargs=minimizer_kwargs, stepsize=2, niter=100, disp=False)
        print(timeit.default_timer() - start)
        min_val = res.fun
        min_x = res.x

        # for x0 in uniformX:

        # MINIMIZE (10 parameter points)
        # res = minimize(min_obj, x0=x0, args = (mu_sample_opt), bounds=bounds, method='L-BFGS-B')
        # if res.fun < min_val:
        #     min_val = res.fun
        #     min_x = res.x

        # JUST A BUNCH OF POINTS (1000 parameter points)
        # res = min_obj(x0, mu_sample_opt)
        # if res < min_val:
        #     min_val = res
        #     min_x = x0

        # print("time:")
        # print(timeit.default_timer() - start)
        # print()

        return min_x.reshape(-1, 1)

    def expected_improvement(self, X, mu_sample_opt, X_sample, Y_sample, gpr, xi=0.001):
        '''
        Computes the EI at points X based on existing samples X_sample
        and Y_sample using a Gaussian process surrogate model.

        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.

        Returns:
            Expected improvements at points X.
        '''

        mu, sigma = gpr.predict(self.scaler.transform(X), return_std=True)
        mu = math.sqrt(mean_squared_error(self.syntheticData.reshape(1, -1), mu))

        sigma = np.mean(sigma.reshape(-1, 1))

        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            if (sigma == 0.0):
                ei = 0.0

        return ei

    def optimizationLoop3(self):
        secondParam = None
        lastParam = None

        start = timeit.default_timer()
        self.syntheticData = self.ms.newMogi(self.profile3, np.array([64, 5, 10, 2]))[:, 2]  # vertical displacements
        self.trainingX = np.mgrid[44:84:3j, 3:7:3j, 7:13:3j, 0.5:3.5:3j].reshape(4, -1).T  # initial training data set

        bnds = ((44, 84), (3, 7), (7, 13), (0.5, 3.5))  # assume bounds aren't changing through each iteration
        allVals = np.array([1, 1, 1, 1, None])
        trainingY = np.empty([len(self.trainingX), 1])
        count = 0
        kernel = ConstantKernel() * Matern(nu=0.5, length_scale=0.1, length_scale_bounds=(0.001, 100))
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, normalize_y=True)
        for param in self.trainingX:
            displacements = self.ms.newMogi(self.profile3, param)
            trainingY[count, 0] = math.sqrt(mean_squared_error(self.syntheticData.reshape(1, -1), displacements[:, 2].reshape(1, -1)))
            count = count + 1

        for i in range(500):
            print(i)
            # conduct regression
            print("Time to fit:")
            startt = timeit.default_timer()
            self.gpr.fit(self.trainingX, trainingY)
            print(self.gpr.kernel_)
            print(timeit.default_timer() - startt)

            print("basinhopping")
            X_next = self.propose_locationRMSE(self.expected_improvementRMSE, self.trainingX, trainingY, self.gpr, bnds)

            surrogateMean = self.gpr.predict(X_next.reshape(1, -1))
            surrogateMean = surrogateMean[0][0]

            if (i == 0):
                secondParam = X_next
            elif (i == 499):
                lastParam = X_next

            print(
                "added point: strength = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f, RMSE = %.4f" % (
                    X_next[0], X_next[1], X_next[2], X_next[3], surrogateMean))

            X_next = np.squeeze(X_next)
            allVallCurr = np.array([X_next[0], X_next[1], X_next[2], X_next[3], surrogateMean])
            allVals = np.vstack((allVals, allVallCurr))

            Y_nextVerticalDisplacements = self.ms.newMogi(self.profile3, X_next.reshape(-1, 1))[:, 2]
            Y_next = math.sqrt(mean_squared_error(self.syntheticData.reshape(1, -1), Y_nextVerticalDisplacements.reshape(1, -1)))
            # Add sample to previous samples
            self.trainingX = np.vstack((self.trainingX, X_next.reshape(1, -1)))
            trainingY = np.vstack((trainingY, np.array([Y_next]).reshape(1, -1)))

        df = pd.DataFrame(allVals)
        df.to_clipboard(index=False, header=False)
        print(timeit.default_timer() - start)

        plot = plt.figure(1)
        plt.plot(self.profile3[:, 1], self.syntheticData[:].reshape(-1), 'k')
        plt.plot(self.profile3[:, 1], self.ms.newMogi(self.profile3, secondParam.reshape(-1, 1))[:, 2], 'r')
        plt.plot(self.profile3[:, 1], self.ms.newMogi(self.profile3, lastParam.reshape(-1, 1))[:, 2], 'b')
        plt.xlabel("Y (along X=5)")
        plt.ylabel("Vertical Displacement")
        plt.title("Synthetic vs. Surrogate for profile along x=5")
        plt.legend(["Synthetic", "First Predicted Parameters", "Last Predicted Parameters"])

        plt.show()

    def propose_locationRMSE(self, acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=10):
        '''
        Proposes the next sampling point by optimizing the acquisition function.

        Args:
            acquisition: Acquisition function.
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.

        Returns:
            Location of the acquisition function maximum.
        '''

        min_val = 1e10
        min_x = None

        def min_obj(X, mu_sample_opt):
            # Minimization objective is the negative acquisition function
            return -acquisition(X.reshape(1, -1), mu_sample_opt, X_sample, Y_sample, gpr)

        # Find the best optimum by starting from n_restart different random points.

        mu_sample_rms = gpr.predict(X_sample)

        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [1]
        mu_sample_opt = np.min(mu_sample_rms)
        #bnds = ((40, 80), (1, 15), (1, 15), (0.5, 10))  # assume bounds aren't changing through each iteration
        minimizer_kwargs = {"bounds": bounds, "args": (mu_sample_opt)}

        x0 = np.array([45, 2, 2, 1])
        # basinhopping function
        start = timeit.default_timer()
        res = basinhopping(min_obj, x0, minimizer_kwargs=minimizer_kwargs, stepsize=2, niter=100, disp=False)
        print(timeit.default_timer() - start)
        min_val = res.fun
        min_x = res.x

        return min_x.reshape(-1, 1)

    def expected_improvementRMSE(self, X, mu_sample_opt, X_sample, Y_sample, gpr, xi=0.001):
        '''
        Computes the EI at points X based on existing samples X_sample
        and Y_sample using a Gaussian process surrogate model.

        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.

        Returns:
            Expected improvements at points X.
        '''

        mu, sigma = gpr.predict(X, return_std=True)
        sigma = np.mean(sigma.reshape(-1, 1))

        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            if (sigma == 0.0):
                ei = 0.0

        return ei[0][0]


    def nll(self, theta, trainingX, trainingY):
        cov_mat = self.gp.kernel(trainingX, trainingX, theta)
        cov_mat = cov_mat + 0.00005 * np.eye(len(trainingX))
        self.covar = cov_mat
        L = np.linalg.cholesky(cov_mat)
        alpha = np.dot(np.linalg.inv(L.T), np.dot(np.linalg.inv(L), trainingY))
        secondTerm = np.linalg.slogdet(cov_mat)
        prob = (- 0.5 * trainingY.reshape(1, -1).dot(alpha.reshape(-1, 1)) - 0.5 * secondTerm[1])[0][0]

        grads = np.zeros(len(theta))

        # tau hyperparameter
        firstTerm = np.matmul(alpha.reshape(-1, 1), alpha.reshape(1, -1)) - np.dot(np.linalg.inv(L.T), np.linalg.inv(L))
        grads[0] = 0.5 * np.trace(np.dot(firstTerm, cov_mat / theta[0]))

        for i in range(1, len(theta)):
            column = trainingX[:, i - 1].reshape(-1, 1)
            squaredTerm = np.sum(column ** 2, 1).reshape(-1, 1) + np.sum(column ** 2, 1) - 2 * np.dot(column, column.T)
            dKdtheta = np.multiply(cov_mat, (-0.5 * squaredTerm))
            grads[i] = 0.5 * np.trace(np.dot(firstTerm, dKdtheta))
        return -prob, -grads

    def testingMinimize(self):
        trainingX = np.linspace(-1, 1, 17).reshape(-1, 1)
        trainingY = 10 * np.sin(15 * trainingX).reshape(-1, 1) + 10 * trainingX.reshape(-1, 1)
        testingX = np.linspace(-1, 1, 100).reshape(-1, 1)

        # meanY = np.mean(trainingY, axis=0)
        # stdY = np.std(trainingY)
        # vertTrainingY = (trainingY - meanY) / stdY
        #
        # K = self.gp.kernel3(trainingX, trainingX, 1, 1) + 0.005 * np.eye(len(trainingX))
        # Ks = self.gp.kernel3(trainingX, testingX, 1, 1)
        # Kss = self.gp.kernel3(testingX, testingX, 1, 1)
        #
        # L = np.linalg.cholesky(K)
        # alpha = np.matmul(np.linalg.inv(L.T), np.matmul(np.linalg.inv(L), vertTrainingY))
        # surrogateMeans = np.matmul(Ks.T, alpha)
        # v = np.matmul(np.linalg.inv(L), Ks)
        # surrogateStds = np.squeeze(np.diag(Kss - np.matmul(v.T, v)))
        #
        # surrogateMeans = surrogateMeans * stdY + meanY
        # surrogateStds = surrogateStds * stdY #+ meanY
        # print(surrogateStds)

        meanY = np.mean(trainingY, axis=0)
        stdY = np.std(trainingY)
        vertTrainingY = (trainingY - meanY) / stdY
        # vertTrainingY = trainingY
        theta0 = np.array([1.0, 1.0])
        minimum = minimize(self.nll, x0=theta0, args=(trainingX, vertTrainingY), method='L-BFGS-B',
                           bounds=[(0.1, 2.0), (0.01, 100)],
                           jac=True)  # tau and 4 hyperparam for each param
        hyperparameters = minimum.x
        surrogateMeans, surrogateStds = self.gp.globalGaussianProcessRegression(trainingX, vertTrainingY,
                                                                                testingX, hyperparameters)
        print(hyperparameters)

        surrogateMeans = surrogateMeans * stdY + meanY
        surrogateStds = surrogateStds * stdY #+ meanY
        print(surrogateMeans)
        print(surrogateStds)
        #'fixed'
        #(1e-5,1e5)
        # kernel = ConstantKernel() * RBF(length_scale=1.0, length_scale_bounds=(1e-5,1e5)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10,1e5))
        # self.gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=100)
        # self.gpr.fit(trainingX.reshape(-1, 1), trainingY.reshape(-1, 1))
        # surrogateMeans, surrogateStds = self.gpr.predict(testingX, return_std=True)
        # print(self.gpr.kernel_)
        # print(surrogateMeans)

        plot = plt.figure(1)
        plt.plot(testingX, 10 * np.sin(15 * testingX).reshape(-1, 1) + 10 * testingX.reshape(-1, 1), 'b')
        plt.plot(testingX, surrogateMeans, 'r')
        plot.gca().fill_between(np.squeeze(testingX.reshape(-1, 1)), np.squeeze(surrogateMeans.reshape(-1, 1) - 2.0*surrogateStds.reshape(-1, 1)),
                                np.squeeze(surrogateMeans.reshape(-1, 1) + 2.0*surrogateStds.reshape(-1, 1)), color="#dddddd")
        plt.title("Training function f(x) = 10*sin(15x) + 10x vs. Interpolated Gaussian Regression With Optimized Hyperparameters")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend(["Training Function", "Gaussian Process Regression Interpolation"])
        plt.show()

    def parabola(self, x):
        print()
        print("x")
        print(x)
        print("parab:")
        print((x[0] - 3) ** 2 + (x[1] - 3) ** 2)
        return (x[0] - 3) ** 2 + (x[1] - 3) ** 2

    def gradi(self, x):
        print("grad")
        print([2 * (x[0] - 3), 2 * (x[1] - 3)])
        return [2 * (x[0] - 3), 2 * (x[1] - 3)]

    def optimizationLoop1DTest(self):
        start = timeit.default_timer()

        bounds = np.array([-1, 2])
        noise = 0

        X = np.linspace(-1, 2, 100).reshape(-1, 1)
        Y = -np.sin(3 * X) + X ** 2 + 0.7 * X + noise * np.random.randn(*X.shape)

        noise = 0.2

        minLocation = X[np.argmin(Y)]
        print(minLocation)

        xTraining = np.array([-0.9, 1.1]).reshape(-1, 1)
        yTraining = -np.sin(3 * xTraining) + xTraining ** 2 + 0.7 * xTraining + noise * np.random.randn(
            *xTraining.shape)

        p0 = 0
        bnds = (-1, 2)
        minimizer_kwargs = {"bounds": bnds}

        allVals = np.array([0, None]).reshape(1, -1)

        niter = 10
        fig1, axis = plt.subplots(nrows=niter, ncols=2)
        plt.subplots_adjust(hspace=0.4)
        # fig1.tight_layout()

        eis = np.empty((X.shape))
        kernel = ConstantKernel() * RBF(length_scale=np.array([0.1]))  # + WhiteKernel()
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30)

        for i in range(niter):
            # conduct regression
            self.gpr.fit(xTraining.reshape(-1, 1), yTraining.reshape(-1, 1))

            X_next = self.propose_location1D(acquisition=self.expected_improvement1D, X_sample=xTraining,
                                             Y_sample=yTraining, gp=self.gpr, bounds=bnds, minLocation=minLocation)
            rmse = math.sqrt(mean_squared_error([minLocation], [X_next]))
            Y_next = -np.sin(3 * X_next) + X_next ** 2 + 0.7 * X_next + noise * np.random.randn(*X_next.shape)
            allVals = np.vstack((allVals, np.array([X_next, rmse]).reshape(1, -1)))

            print("added point: x=%.4f, rmse = %.4f" % (X_next, rmse))

            plt.subplot(niter, 2, 2 * i + 1)
            plt.plot(X, Y, 'k')
            plt.plot(xTraining, yTraining, 'bo')
            plt.plot(X.reshape(-1, 1), self.gpr.predict(X.reshape(-1, 1)), 'r--')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("1-D EI Case")
            # plt.legend(["Actual Function", "Training Points", "GPR Fit"])

            mu_sample = self.gpr.predict(xTraining)
            mu_sample_opt = np.min(mu_sample)
            for j in range(len(X)):
                eis[j] = self.expected_improvement1D(X=X[j], X_sample=xTraining, Y_sample=yTraining, gp=self.gpr)
            # print("EIS")
            # print(eis)
            plt.subplot(niter, 2, 2 * i + 2)
            plt.plot(X, eis)
            plt.title("EIS")

            xTraining = np.vstack((xTraining, X_next))
            yTraining = np.vstack((yTraining, Y_next))

        plt.show()
        df = pd.DataFrame(allVals)
        df.to_clipboard(index=False, header=False)
        print(timeit.default_timer() - start)

    def propose_location1D(self, acquisition, X_sample, Y_sample, gp, bounds, minLocation, n_restarts=50):
        '''
        Proposes the next sampling point by optimizing the acquisition function.

        Args:
            acquisition: Acquisition function.
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.

        Returns:
            Location of the acquisition function maximum.
        '''

        min_val = 1e10
        min_x = None

        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -acquisition(X=X, X_sample=X_sample, Y_sample=Y_sample, gp=gp)

        # mu_sample_rms = np.empty((len(mu_sample), 1))

        # for i in range(len(mu_sample)):
        #     mu_sample_rms[i] = math.sqrt(mean_squared_error(minLocation, X_sample))

        # minimizer_kwargs = {"bounds": (bounds,), "args": (mu_sample_opt)}

        uniform = np.random.uniform(bounds[0], bounds[1], size=(n_restarts, 1))

        x0 = 0
        # basinhopping function
        # res = basinhopping(min_obj, x0, minimizer_kwargs=minimizer_kwargs, stepsize=0.1, niter=100, disp=False)
        # min_val = res.fun
        # min_x = res.x[0]

        for x0 in uniform:
            res = minimize(min_obj, x0=x0, bounds=(bounds,), method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x[0]

        return min_x

    def expected_improvement1D(self, X, X_sample, Y_sample, gp, xi=0.01):
        '''
        Computes the EI at points X based on existing samples X_sample
        and Y_sample using a Gaussian process surrogate model.

        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.

        Returns:
            Expected improvements at points X.
        '''
        mu, sigma = self.gpr.predict(np.array(X).reshape(1, -1), return_std=True)
        mu_sample = self.gpr.predict(X_sample)
        mu_sample_opt = np.max(mu_sample)

        # mu = math.sqrt(mean_squared_error(self.syntheticData.reshape(1, -1), mu))

        # mu_sample_rms = np.sqrt(np.sum(self.syntheticData.reshape(1, -1) - mu_sample))

        sigma = np.mean(sigma)

        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            if (sigma == 0.0):
                ei = 0.0

        return ei[0][0]

    def optimizationArticle(self):
        bounds = np.array([[-1.0, 2.0]])
        noise = 0.2

        def f(X, noise=noise):
            return -np.sin(3 * X) + X ** 2 + 0.7 * X + noise * np.random.randn(*X.shape)

        X_init = np.array([[-0.9], [1.1]])
        Y_init = f(X_init)

        X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)

        # Noise-free objective function values at X
        Y = f(X, 0)

        minLocation = X[np.argmin(Y)]
        # m52 = ConstantKernel(1.0)*Matern(length_scale=0.5, nu=0.5)
        # gpr = GaussianProcessRegressor(kernel=m52, alpha=noise ** 2)
        #kernel = ConstantKernel() * RBF(length_scale=np.array([0.1]))  # + WhiteKernel()
        kernel = ConstantKernel() * Matern(nu = 0.5, length_scale=0.5, length_scale_bounds=(0.001, 3))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30)

        # Initialize samples
        X_sample = X_init
        Y_sample = Y_init

        # Number of iterations
        n_iter = 100

        # plt.figure(figsize=(12, n_iter * 3))
        # plt.subplots_adjust(hspace=0.4)

        for i in range(n_iter):
            # Update Gaussian process with existing samples
            gpr.fit(X_sample, Y_sample)

            # Obtain next sampling point from the acquisition function (expected_improvement)
            X_next = self.propose_locationArticle(self.expected_improvementArticle, X_sample, Y_sample, gpr, bounds)
            rmse = math.sqrt(mean_squared_error([minLocation], [X_next[0]]))
            # print(X_next)
            print(i)
            #print(kernel.theta)
            print(gpr.kernel_)
            # print(rmse)

            # Obtain next noisy sample from the objective function
            Y_next = f(X_next, noise)

            # Plot samples, surrogate function, noise-free objective and next sampling location
            if (i == n_iter - 1):
                # plt.subplot(n_iter, 2, 2 * i + 1)
                self.plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next)
                # plt.title(f'Iteration {i + 1}')

            # plt.subplot(n_iter, 2, 2 * i + 2)
            # self.plot_acquisition(X, self.expected_improvementArticle(X, X_sample, Y_sample, gpr), X_next, show_legend=i == 0)

            # Add sample to previous samples
            X_sample = np.vstack((X_sample, X_next))
            Y_sample = np.vstack((Y_sample, Y_next))
        plt.show()

    def expected_improvementArticle(self, X, X_sample, Y_sample, gpr, xi=0.001):
        '''
        Computes the EI at points X based on existing samples X_sample
        and Y_sample using a Gaussian process surrogate model.

        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.

        Returns:
            Expected improvements at points X.
        '''
        mu, sigma = gpr.predict(X, return_std=True)
        mu_sample = gpr.predict(X_sample)

        sigma = sigma.reshape(-1, 1)

        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [1]
        mu_sample_opt = np.min(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei[0]

    def propose_locationArticle(self, acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
        '''
        Proposes the next sampling point by optimizing the acquisition function.

        Args:
            acquisition: Acquisition function.
            X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.

        Returns:
            Location of the acquisition function maximum.
        '''
        dim = X_sample.shape[1]
        min_val = 1
        min_x = None

        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

        return min_x.reshape(-1, 1)

    def plot_acquisition(self, X, Y, X_next, show_legend=False):
        plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
        plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
        if show_legend:
            plt.legend()

    def plot_approximation(self, gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
        mu, std = gpr.predict(X, return_std=True)
        plt.fill_between(X.ravel(),
                         mu.ravel() + 1.96 * std,
                         mu.ravel() - 1.96 * std,
                         alpha=0.1)
        plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
        plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
        plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
        if X_next:
            plt.axvline(x=X_next, ls='--', c='k', lw=1)
        if show_legend:
            plt.legend()

    def checkGaussian(self):
        constantVals = np.full(self.profile.shape, 5).reshape(-1, 1)
        profile3 = np.hstack((constantVals, self.profile))  # x = 6
        self.syntheticData = self.ms.newMogi(profile3, np.array([64, 5, 10, 2]))[:, 2]  # vertical displacements
        bounds = ((44, 84), (3, 7), (7, 13), (0.5, 3.5))
        numPoints = 3
        strengthUniform = np.random.uniform(bounds[0][0], bounds[0][1], size=(numPoints, 1))
        xUniform = np.random.uniform(bounds[1][0], bounds[1][1], size=(numPoints, 1))
        yUniform = np.random.uniform(bounds[2][0], bounds[2][1], size=(numPoints, 1))
        zUniform = np.random.uniform(bounds[3][0], bounds[3][1], size=(numPoints, 1))

        self.trainingX = np.hstack((np.hstack((np.hstack((strengthUniform, xUniform)), yUniform)), zUniform))
        #self.trainingX = np.mgrid[44:84:3j, 3:7:3j, 7:13:3j, 0.5:3.5:3j].reshape(4, -1).T  # initial training data set

        # actual: 64, 5, 10, 2

        # create trainingY data
        trainingY = np.empty((len(self.trainingX), len(self.profile1)))
        count = 0
        for param in self.trainingX:
            displacements = self.ms.newMogi(profile3, param)
            trainingY[count, :] = displacements[:, 2]
            count = count + 1

        # m52 = Matern()
        kernel  = ConstantKernel() * RBF(length_scale = np.array([0.1,0.1,0.1,0.1])) #+ WhiteKernel()
        self.gpr = GaussianProcessRegressor(kernel = kernel, normalize_y=True, n_restarts_optimizer=30)
        self.gpr.fit(self.trainingX, trainingY)
        print(self.gpr.kernel_)
        plot = plt.figure(1)
        plt.plot(profile3[:, 1], self.syntheticData[:].reshape(-1), 'r')
        plt.plot(profile3[:, 1], self.gpr.predict(np.array([64, 5, 10, 2]).reshape(1, -1))[0, :], 'b')
        plt.xlabel("Y (along X=5)")
        plt.ylabel("Vertical Displacement")
        plt.title("Mogi vs. Surrogate for profile along x=5")
        plt.legend(["Actual Mogi Profile", "Surrogate Profile"])
        plt.show()

    def spaceInterpolationComparison(self):
        warnings.filterwarnings("ignore")
        numGS = 1000
        profile = np.linspace(1, 15, numGS).reshape(-1, 1)
        constantVals = np.full(profile.shape, 5).reshape(-1, 1)
        profile4 = np.hstack((constantVals, profile))  # x = 5

        self.syntheticData = self.ms.newMogi(profile4, np.array([64, 5, 10, 2]))[:, 2]  # vertical displacements

        # niter = 4
        fig1, axis = plt.subplots(nrows=2, ncols=2)
        fig1.suptitle("Displacement Interpolation as Vector for GSs and as scalar for single GS along x=5")
        # plt.subplots_adjust(hspace=0.4)

        for i in range(4):
            bounds = ((44, 84), (3, 7), (7, 13), (0.5, 3.5))
            strengthUniform = np.random.uniform(bounds[0][0], bounds[0][1], size=(3, 1))
            xUniform = np.random.uniform(bounds[1][0], bounds[1][1], size=(3, 1))
            yUniform = np.random.uniform(bounds[2][0], bounds[2][1], size=(3, 1))
            zUniform = np.random.uniform(bounds[3][0], bounds[3][1], size=(3, 1))

            trainingX1 = np.hstack((np.hstack((np.hstack((strengthUniform, xUniform)), yUniform)), zUniform))

            # trainingX1 = np.mgrid[40:90:3j, 2:8:3j, 6:14:3j, 0.1:4:3j].reshape(4, -1).T  # initial training data set

            trainingY = np.empty((len(trainingX1), len(profile4)))

            count = 0
            for param in trainingX1:
                displacements = self.ms.newMogi(profile4, param)
                trainingY[count, :] = displacements[:, 2]
                count = count + 1

            kernel = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]),
                                               length_scale_bounds=(0.001, 100))
            gpr1 = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=30)

            kernel2 = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                                               length_scale_bounds=(0.001, 100))
            gpr2 = GaussianProcessRegressor(kernel=kernel2, normalize_y=True, n_restarts_optimizer=30)

            trainingX2 = np.empty((len(trainingX1)*numGS, 6))
            newTrainingY = np.empty((len(trainingX1)*numGS, 1))
            trainingX1row = 0
            trainingX2row = 0
            profilerow = 0
            for verticalDisplacements in trainingY:
                for displacement in verticalDisplacements:
                    trainingX2[trainingX2row, :] = np.hstack((profile4[profilerow, :].reshape(1, 2), trainingX1[trainingX1row, :].reshape(1, 4)))
                    newTrainingY[trainingX2row, 0] = displacement
                    profilerow = profilerow + 1
                    trainingX2row = trainingX2row + 1
                profilerow = 0
                trainingX1row = trainingX1row + 1

            testingX = np.empty((len(profile4), 6))
            for rowNum in range(len(profile4)):
                testingX[rowNum, :] = np.hstack((profile4[rowNum, :], np.array([64, 5, 10, 2])))

            print("Fitting data...")

            print("Time taken to fit vector output: ")
            startTime = timeit.default_timer()
            gpr1.fit(trainingX1, trainingY)
            gpr1Time = timeit.default_timer() - startTime
            print(gpr1Time)

            print("Time taken to fit scalar output: ")
            startTime = timeit.default_timer()
            gpr2.fit(trainingX2, newTrainingY.reshape(-1, 1))
            gpr2Time = timeit.default_timer() - startTime
            print(gpr2Time)

            gpr1Data = gpr1.predict(np.array([64, 5, 10, 2]).reshape(1, -1))  # output 1x15
            gpr2Data = gpr2.predict(testingX)  # 15x1

            plot = plt.subplot(2, 2, i+1)
            plt.plot(profile4[:, 1], self.syntheticData[:].reshape(-1), 'k')
            plt.plot(profile4[:, 1], gpr1Data.reshape(-1), 'r')
            plt.plot(profile4[:, 1], gpr2Data.reshape(-1), 'b')
            plt.xlabel("Y (along X=5)")
            plt.ylabel("Vertical Displacement")
            plt.legend(["Mogi Output", "Vector Output, Run time: %.4f secs" % (gpr1Time), "Scalar Output, Run time: %.4f secs" % (gpr2Time)], loc=2)

        plt.show()

    def InSARMogiInterpolation(self):
        ascendingSheet = pd.read_excel('noOffset_strongSNR/ascending.xlsx').to_numpy()
        # self.observedDataAscendingCovar = 5;  # read from excel file
        descendingSheet = pd.read_excel('noOffset_strongSNR/descending.xlsx').to_numpy()  # read from excel file
        # self.observedDataDescendingCoVar = 5;  # read from excel file

        self.observedDataAscending = ascendingSheet[:, 3].reshape(-1, 1)
        self.observedDataDescending = descendingSheet[:, 3].reshape(-1, 1)

        self.trueObservedAscending = ascendingSheet[:, 6].reshape(-1, 1)
        self.trueObservedDescending = descendingSheet[:, 6].reshape(-1, 1)

        self.groundStations = np.hstack((ascendingSheet[:, 1].reshape(-1, 1), ascendingSheet[:, 2].reshape(-1, 1)))
        lenGS = len(self.groundStations)
        numGS = int(math.sqrt(lenGS))

        with open('noOffset_strongSNR/cd_full_asc.txt') as f:
            lines = [float(line.rstrip()) for line in f]
        self.covarAscendingInverse = np.linalg.inv(np.asarray(lines).reshape(lenGS, lenGS))

        with open('noOffset_strongSNR/cd_full_desc.txt') as f:
            lines = [float(line.rstrip()) for line in f]
        self.covarDescendingInverse = np.linalg.inv(np.asarray(lines).reshape(lenGS, lenGS))


        theta = ascendingSheet[0, 4]
        phi = ascendingSheet[0, 5]
        Ha = [-np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]

        theta = descendingSheet[0, 4]
        phi = descendingSheet[0, 5]
        Hd = [-np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]

        # bounds = ((5e5, 5e7), (-5000, 5000), (-7000, 0), (1000, 10000))
        bounds = ((math.log10(5e5), math.log10(5e7)), (-4000, 2500), (-5300, 300), (math.log10(1000), math.log10(10000)))
        print(bounds)
        numPoints = 60

        xlimits = np.array([[math.log10(5e5), math.log10(5e7)], [-4000, 2500], [-5300, 300], [math.log10(1000), math.log10(10000)]])
        sampling = LHS(xlimits=xlimits)
        self.trainingX = sampling(numPoints)

        trainingYAscending = np.empty((len(self.trainingX), len(self.groundStations)))
        trainingYDescending = np.empty((len(self.trainingX), len(self.groundStations)))

        count = 0
        for param in self.trainingX:
            displacementsAscending = self.ms.MogiWithDv(self.groundStations, param)
            LOSval = displacementsAscending[:, 0]*Ha[0] + displacementsAscending[:, 1]*Ha[1] + displacementsAscending[:, 2]*Ha[2]
            trainingYAscending[count, :] = LOSval

            displacementsDescending = self.ms.MogiWithDv(self.groundStations, param)
            LOSval = displacementsDescending[:, 0] * Hd[0] + displacementsDescending[:, 1] * Hd[1] + displacementsDescending[:, 2] * Hd[2]
            trainingYDescending[count, :] = LOSval

            count = count + 1

        self.scaler = StandardScaler().fit(self.trainingX)
        self.trainingX = self.scaler.transform(self.trainingX)

        kernel = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]),
                                           length_scale_bounds=(0.001, 100))
        self.gprA = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=30)

        kernel = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]),
                                           length_scale_bounds=(0.001, 100))
        self.gprD = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=30)

        timestart = timeit.default_timer()
        self.gprA.fit(self.trainingX, trainingYAscending)
        timeelaspsed = timeit.default_timer() - timestart
        print("Time taken to fit Ascending Data: %.4f" % (timeelaspsed))
        timestart = timeit.default_timer()
        self.gprD.fit(self.trainingX, trainingYDescending)
        timeelaspsed = timeit.default_timer() - timestart
        print("Time taken to fit Descending Data: %.4f" % (timeelaspsed))

        # actual values: 64, 5, 10, 2
        p0 = np.array([1, 1, 1, 1])  # 100, 2, 50, 10
        minimizer_kwargs = {"bounds": bounds}
        timestart = timeit.default_timer()
        result = basinhopping(self.InSARlossFunction, p0, minimizer_kwargs=minimizer_kwargs, stepsize=50, niter=200)
        timeelaspsed = timeit.default_timer() - timestart

        print("Time taken to run basinhopping: %.4f" % (timeelaspsed))
        print()
        print(
            "global minimum: dV = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f | f(x0) = %.4f" % (
            10**result.x[0], result.x[1], result.x[2], 10**result.x[3], result.fun))

        # bestDataA = self.gprA.predict(self.scaler.transform(result.x.reshape(1, -1))).reshape(numGS, numGS)
        # bestDataD = self.gprD.predict(self.scaler.transform(result.x.reshape(1, -1))).reshape(numGS, numGS)
        #
        # fig1, axs1 = plt.subplots(2, 2)
        # plt.subplots_adjust(wspace=0.6, hspace=0.4)
        # fig1.suptitle('Ascending In-SAR Observed vs. Surrogate Displacement Measurements for No Offset Strong SNR')
        #
        # pltt = axs1[0, 0]
        # pltt.set_aspect('equal')
        # con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
        #                     self.groundStations[:, 1].reshape(numGS, numGS),
        #                     self.trueObservedAscending.reshape(numGS, numGS),
        #                     150, vmin=np.min(self.observedDataAscending), vmax=np.max(self.observedDataAscending),
        #                     cmap="jet")
        # cbar = fig1.colorbar(con, ax=pltt)
        # cbar.ax.set_ylabel('Displacement', rotation=90)
        # cbar.ax.get_yaxis().labelpad = 10
        # pltt.set_title("Truth (m)")
        # pltt.set_xlabel("X")
        # pltt.set_ylabel("Y")
        #
        # pltt = axs1[0, 1]
        # pltt.set_aspect('equal')
        # con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
        #                     self.groundStations[:, 1].reshape(numGS, numGS),
        #                     self.observedDataAscending.reshape(numGS, numGS),
        #                     150, vmin=np.min(self.observedDataAscending), vmax=np.max(self.observedDataAscending),
        #                     cmap="jet")
        # cbar = fig1.colorbar(con, ax=pltt)
        # cbar.ax.set_ylabel('Displacement', rotation=90)
        # cbar.ax.get_yaxis().labelpad = 10
        # pltt.set_title("Data (m)")
        # pltt.set_xlabel("X")
        # pltt.set_ylabel("Y")
        #
        # pltt = axs1[1, 0]
        # pltt.set_aspect('equal')
        # con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
        #                     self.groundStations[:, 1].reshape(numGS, numGS),
        #                     bestDataA,
        #                     150, vmin=np.min(self.observedDataAscending), vmax=np.max(self.observedDataAscending),
        #                     cmap="jet")
        # cbar = fig1.colorbar(con, ax=pltt)
        # cbar.ax.set_ylabel('Displacement', rotation=90)
        # cbar.ax.get_yaxis().labelpad = 10
        # pltt.set_title("Surrogate (m)")
        # pltt.set_xlabel("X")
        # pltt.set_ylabel("Y")
        #
        # pltt = axs1[1, 1]
        # pltt.set_aspect('equal')
        # con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
        #                     self.groundStations[:, 1].reshape(numGS, numGS),
        #                     self.observedDataAscending.reshape(numGS, numGS) - bestDataA,
        #                     150, vmin=np.min(self.observedDataAscending.reshape(numGS, numGS) - bestDataA), vmax=np.max(self.observedDataAscending.reshape(numGS, numGS) - bestDataA),
        #                     cmap="jet")
        # cbar = fig1.colorbar(con, ax=pltt)
        # cbar.ax.set_ylabel('Displacement', rotation=90)
        # cbar.ax.get_yaxis().labelpad = 10
        # pltt.set_title("Data Minus Surrogate (m)")
        # pltt.set_xlabel("X")
        # pltt.set_ylabel("Y")
        #
        # fig2, axs2 = plt.subplots(2, 2)
        # plt.subplots_adjust(wspace=0.6, hspace=0.4)
        # fig2.suptitle('Descending In-SAR Observed vs. Surrogate Displacement Measurements for No Offset Strong SNR')
        #
        # pltt = axs2[0, 0]
        # pltt.set_aspect('equal')
        # con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
        #                     self.groundStations[:, 1].reshape(numGS, numGS),
        #                     self.trueObservedDescending.reshape(numGS, numGS),
        #                     150, vmin=np.min(self.observedDataDescending),
        #                     vmax=np.max(self.observedDataDescending),
        #                     cmap="jet")
        # cbar = fig2.colorbar(con, ax=pltt)
        # cbar.ax.set_ylabel('Displacement', rotation=90)
        # cbar.ax.get_yaxis().labelpad = 10
        # pltt.set_title("Truth (m)")
        # pltt.set_xlabel("X")
        # pltt.set_ylabel("Y")
        #
        # pltt = axs2[0, 1]
        # pltt.set_aspect('equal')
        # con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
        #                     self.groundStations[:, 1].reshape(numGS, numGS),
        #                     self.observedDataDescending.reshape(numGS, numGS),
        #                     150, vmin=np.min(self.observedDataDescending),
        #                     vmax=np.max(self.observedDataDescending),
        #                     cmap="jet")
        # cbar = fig2.colorbar(con, ax=pltt)
        # cbar.ax.set_ylabel('Displacement', rotation=90)
        # cbar.ax.get_yaxis().labelpad = 10
        # pltt.set_title("Data (m)")
        # pltt.set_xlabel("X")
        # pltt.set_ylabel("Y")
        #
        # pltt = axs2[1, 0]
        # pltt.set_aspect('equal')
        # con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
        #                     self.groundStations[:, 1].reshape(numGS, numGS),
        #                     bestDataD,
        #                     150, vmin=np.min(self.observedDataDescending),
        #                     vmax=np.max(self.observedDataDescending),
        #                     cmap="jet")
        # cbar = fig2.colorbar(con, ax=pltt)
        # cbar.ax.set_ylabel('Displacement', rotation=90)
        # cbar.ax.get_yaxis().labelpad = 10
        # pltt.set_title("Surrogate (m)")
        # pltt.set_xlabel("X")
        # pltt.set_ylabel("Y")
        #
        # pltt = axs2[1, 1]
        # pltt.set_aspect('equal')
        # con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
        #                     self.groundStations[:, 1].reshape(numGS, numGS),
        #                     self.observedDataDescending.reshape(numGS, numGS) - bestDataD,
        #                     150, vmin=np.min(self.observedDataDescending.reshape(numGS, numGS) - bestDataD),
        #                     vmax=np.max(self.observedDataDescending.reshape(numGS, numGS) - bestDataD),
        #                     cmap="jet")
        # cbar = fig2.colorbar(con, ax=pltt)
        # cbar.ax.set_ylabel('Displacement', rotation=90)
        # cbar.ax.get_yaxis().labelpad = 10
        # pltt.set_title("Data Minus Surrogate (m)")
        # pltt.set_xlabel("X")
        # pltt.set_ylabel("Y")
        #
        # plt.show()

        # Synthetic Data
        # Best Fit Model

        # plot = plt.figure(1)
        # plt.plot(self.groundStations[:19, 1], self.syntheticData[:19].reshape(-1), 'r')
        # plt.plot(self.groundStations[:19, 1], self.gpr.predict(result.x.reshape(1, -1))[0, :19], 'b')
        # plt.xlabel("Y")
        # plt.ylabel("Vertical Displacement")
        # plt.title("Synthetic vs Surrogate Output along x=1")
        # plt.legend(["Synthetic", "Surrogate"])
        # plt.show()

    def InSARlossFunction(self, x):
        surrogateMeansA = self.gprA.predict(self.scaler.transform(x.reshape(1, -1))).reshape(-1, 1)
        aDifference = self.observedDataAscending - surrogateMeansA
        rmseA = np.dot(np.dot(aDifference.T, self.covarAscendingInverse), aDifference)

        surrogateMeansD = self.gprD.predict(self.scaler.transform(x.reshape(1, -1))).reshape(-1, 1)
        dDifference = self.observedDataDescending - surrogateMeansD
        rmseD = np.dot(np.dot(dDifference.T, self.covarDescendingInverse), dDifference)

        print(x)
        rmse = rmseA+rmseD
        rmse = rmse[0][0]

        return rmse





if __name__ == '__main__':
    ls = LSOptimization()
    # ls.conductOptimization()
    ls.InSARMogiInterpolation()
    # ls.conductOptimization2()
    # ls.testingMinimize()
    # ls.testGP()
    # ls.optimizationLoop2()
    # ls.optimizationLoop3()
    # ls.optimizationLoop1DTest()
    # ls.optimizationArticle()
    # ls.checkGaussian()
    # ls.spaceInterpolationComparison()

# if(x[0] > 60 and x[0] < 65 and x[1] > 4.8 and x[1] < 5.2 and x[2] > 9.8 and x[2] < 10.2 and x[3] > 1.5 and x[3] < 2.5):
#     print(rmse)
#     print(x)

# print(np.sum(surrogateMeans[:, 2]))
# print(np.sum(self.syntheticData[:, 2]))

# def lossFunction(self, x):
#
#     print("")
#     print(x)
#
#     theta0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
#
#     trainingX, trainingY = self.ms.createSurrogate(x)
#     #trainingY = self.mogi(trainingX, xCenter=x, yCenter=y)
#     vertTrainingY = trainingY[:, 2]
#     meanY = np.mean(vertTrainingY)
#     stdY = np.mean(vertTrainingY)
#     vertTrainingY = (vertTrainingY - meanY) / stdY
#
#     #options={'maxiter': 5},
#
#     minimum = minimize(self.nll, x0=theta0, args=(trainingX, vertTrainingY), method='L-BFGS-B', bounds=[(1.0, 2.0), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0)], jac=True) #tau and 4 hyperparam for each param
#     #other = check_grad(self.nll, self.jacobianFunc, theta0, trainingX, vertTrainingY)
#     hyperparameters = minimum.x
#     surrogateMeans, surrogateStds = self.gp.globalGaussianProcessRegression(trainingX, vertTrainingY, self.testingMatrix, hyperparameters)
#
#     surrogateMeans = surrogateMeans*stdY + meanY
#     # muMatrix, stdVector = gp.globalGaussianProcessRegression(trainingX, trainingY, self.testingX, 1.0, 4)
#     # return muMatrix, stdVector
#
#     #Develop surrogate model output
#     #surrogateMeans, surrogateStds = self.ms.createSurrogate(x)
#
#     #replicate synthetic data x with diff parameters to see if loss function estimates actual parameters properly
#     #without use of surrogate models
#     # sampleCoordinates = np.mgrid[0:15:10j, 0:15:10j].reshape(2, -1).T
#     # sampleCoordinates = np.hstack((sampleCoordinates, np.full((len(sampleCoordinates), 1), x[3])))
#     # sampleCoordinates = np.hstack((np.full((len(sampleCoordinates), 1), x[0]), sampleCoordinates))
#     # surrogateMeans = self.ms.mogi(sampleCoordinates, x[1], x[2])
#
#
#     rmse = math.sqrt(mean_squared_error(self.syntheticData[:, 2], surrogateMeans))
#
#     # print(rmse)
#     # var2 = self.testingMatrix - self.ms.testingX
#
#     return rmse
