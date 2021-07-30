import copy

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
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import timeit
import pandas as pd


class LSOptimization:
    syntheticData = None
    ms = MogiSim()
    testingMatrix = None
    gp = GP()
    gpr = None
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
        self.syntheticData = self.ms.newMogi(self.groundStations, np.array([64, 5, 10, 2]))[:,
                             2]  # vertical displacements
        # self.testingMatrix = self.ms.testingX

        # get training parameters
        #trainingParams = self.ms.createTraining()
        trainingParams = np.mgrid[40:80:5j, 0:16:5j, 0:16:5j,
                    0.1:10:5j].reshape(4, -1).T
        trainingY = np.empty((len(trainingParams), len(self.groundStations)))
        count = 0
        for param in trainingParams:
            displacements = self.ms.newMogi(self.groundStations, param)
            trainingY[count, :] = displacements[:, 2]
            count = count + 1

        # intialParams - strength, x, y, depth
        # actual values: 64, 5, 10, 2
        p0 = np.array([100, 2, 50, 10])  # 100, 2, 50, 10
        bnds = ((40, 80), (1, 15), (1, 15), (0.5, 10))
        timestart = timeit.default_timer()
        self.gpr = GaussianProcessRegressor(normalize_y=True).fit(trainingParams, trainingY)
        timeelaspsed = timeit.default_timer() - timestart
        print(timeelaspsed)

        minimizer_kwargs = {"bounds": bnds}
        result = basinhopping(self.lossFunction, p0, minimizer_kwargs=minimizer_kwargs, stepsize=10, niter=500)
        print(
            "global minimum: strength = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f | f(x0) = %.4f" % (
            result.x[0], result.x[1], result.x[2], result.x[3], result.fun))

        plot = plt.figure(1)
        plt.plot(self.groundStations[:19, 1], self.syntheticData[:19].reshape(-1), 'r')
        plt.plot(self.groundStations[:19, 1], self.gpr.predict(result.x.reshape(1, -1))[0, :19], 'b')
        plt.xlabel("Y")
        plt.ylabel("Vertical Displacement")
        plt.title("Synthetic vs Surrogate Output along x=1")
        plt.legend(["Synthetic", "Surrogate"])
        plt.show()

    def lossFunction(self, x):
        # print(x)
        surrogateMeans = self.gpr.predict(x.reshape(1, -1))

        rmse = math.sqrt(mean_squared_error(self.syntheticData.reshape(1, -1), surrogateMeans))
        # print(rmse)
        # print()
        return rmse

    def conductOptimization2(self):
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
    def lossFunction2(self, x):
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
        trainingX = np.mgrid[-1:1:100j].reshape(-1, 1)
        trainingY = 100 * np.sin(15 * trainingX).reshape(-1, 1) + 10 * trainingX.reshape(-1, 1)
        testingX = np.mgrid[-1:1:100j].reshape(-1, 1)

        meanY = np.mean(trainingY, axis=0)
        stdY = np.mean(trainingY)
        vertTrainingY = (trainingY - meanY) / stdY

        theta0 = np.array([1.0, 1.0])
        minimum = minimize(self.nll, x0=theta0, args=(trainingX, vertTrainingY), method='L-BFGS-B',
                           bounds=[(1.0, 2.0), (0.01, 1.0)],
                           jac=True)  # tau and 4 hyperparam for each param
        hyperparameters = minimum.x
        surrogateMeans, surrogateStds = self.gp.globalGaussianProcessRegression(trainingX, vertTrainingY,
                                                                                testingX, hyperparameters)

        surrogateMeans = surrogateMeans * stdY + meanY

        plot = plt.figure(1)
        plt.plot(trainingX, trainingY, 'b')
        plt.plot(testingX, surrogateMeans, 'r')
        plt.title("Synthetic vs Gaussian for f(x) = 100*sin(5x) + 10x")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend(["Synthetic", "Gaussian Interpolation"])
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

if __name__ == '__main__':
    ls = LSOptimization()
    # ls.conductOptimization()
    # ls.testingMinimize()
    # ls.testGP()
    # ls.optimizationLoop2()
    ls.optimizationLoop3()
    # ls.optimizationLoop1DTest()
    # ls.optimizationArticle()
    # ls.checkGaussian()
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
