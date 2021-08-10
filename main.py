"""
Inversion Problem for Mogi source using Surrogate Models

Function syntheticSurrogateInversion() conducts an inversion to determine the Mogi parameters used to create synthetic
data by comparing the synthetic data to interpolated displacements from a Mogi surrogate for parameters estimated in a
stochastic minimization algorithm

Function InSARSurrogateInversion() conducts an inversion to determine the Mogi parameters of InSAR data by comparing the
InSAR data to interpolated displacements from a Mogi surrogate for parameters estimated in a stochastic minimization
algorithm

Each function has a corresponding lossFunction used in the stochastic minimization algorithm

2 Mogi functions, MogiWithDv and MogiWithStrength calculate Mogi displacements using strength or dV respectively in
addition to the other three Mogi parameters: source X, source Y, and source depth

author: Kashish Singal (NASA JPL)
"""

import numpy as np
from scipy.optimize import basinhopping
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
import matplotlib.pyplot as plt
import math
import timeit
import pandas as pd
from smt.sampling_methods import LHS
from numpy import linalg as LA

class Main:

    groundStations = np.mgrid[1:15:20j, 1:15:20j].reshape(2, -1).T #ground stations for synthetic data
    scaler = None
    trainingX = None
    poisson = 0.25
    shearModulus = 1

    def __init__(self):
        pass


    """
    Conducts an inversion to determine the Mogi parameters used to create synthetic data by comparing the synthetic data
    to interpolated displacements from a Mogi surrogate for parameters estimated in a stochastic minimization algorithm
    """
    def syntheticSurrogateInversion(self):
        # for displacements in all 3 dimensions for parameter combination 64, 5, 10, and 2
        self.syntheticData = self.MogiWithStrength(self.groundStations, np.array([64, 5, 10, 2])) #Mogi displacements
        gaussianNoise = np.random.normal(0, 0.01, size=self.syntheticData.shape) #Gaussian noise
        self.syntheticData = self.syntheticData + gaussianNoise #synthetic data with noise
        self.syntheticData = abs(self.syntheticData) #absolute value of synthetic data (so that contour maps can look symmetrical across x/y axis, not needed!

        bounds = ((44, 84), (3, 7), (7, 13), (0.5, 3.5)) #bounds of parameter space
        numPoints = 80 #number of training points
        numGS = 20 #number of ground stations along one axis (this represents a grid of 20x20 ground stations)

        xlimits = np.array([[44, 84], [3, 7], [7, 13], [0.5, 3.5]])
        sampling = LHS(xlimits=xlimits) #training points created from Latin Hypercube
        self.trainingX = sampling(numPoints)

        trainingYx = np.empty((len(self.trainingX), len(self.groundStations))) #x displacements of training points
        trainingYy = np.empty((len(self.trainingX), len(self.groundStations))) #y displacements of training points
        trainingYz = np.empty((len(self.trainingX), len(self.groundStations))) #z displacements of training points

        #populate training output vector for all 3 dimensions
        count = 0
        for param in self.trainingX:
            displacements = abs(self.MogiWithStrength(self.groundStations, param))
            trainingYx[count, :] = displacements[:, 0]
            trainingYy[count, :] = displacements[:, 1]
            trainingYz[count, :] = displacements[:, 2]
            count = count + 1

        #scale training inputs and add to training input matrix
        self.scaler = StandardScaler().fit(self.trainingX)
        self.trainingX = self.scaler.transform(self.trainingX)

        #create three matérn kernels for GPRs for all three dimensions
        kernelX = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]),
                                            length_scale_bounds=(0.001, 100))
        kernelY = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]),
                                            length_scale_bounds=(0.001, 100))
        kernelZ = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]),
                                            length_scale_bounds=(0.001, 100))

        #create all GPRs for all three dimensions
        self.gprX = GaussianProcessRegressor(kernel=kernelX, normalize_y=True, n_restarts_optimizer=30)
        self.gprY = GaussianProcessRegressor(kernel=kernelY, normalize_y=True, n_restarts_optimizer=30)
        self.gprZ = GaussianProcessRegressor(kernel=kernelZ, normalize_y=True, n_restarts_optimizer=30)

        #fit GPRs with input and output of training points
        timestart = timeit.default_timer()
        self.gprX.fit(self.trainingX, trainingYx)
        self.gprY.fit(self.trainingX, trainingYy)
        self.gprZ.fit(self.trainingX, trainingYz)
        timeelaspsed = timeit.default_timer() - timestart
        print("Time taken to fit: %.4f" % (timeelaspsed))

        #calculate what the interpolated displacement is for the same parameter combination 64, 5, 10, and 2
        #after fitting (to ensure GPRs work accurately prior to inversion)
        interpolatedDataX, stdsX = self.gprX.predict(self.scaler.transform(np.array([64, 5, 10, 2]).reshape(1, -1)),
                                                     return_std=True)
        interpolatedDataX = interpolatedDataX.reshape(numGS, numGS)

        interpolatedDataY, stdsY = self.gprY.predict(self.scaler.transform(np.array([64, 5, 10, 2]).reshape(1, -1)),
                                                     return_std=True)
        interpolatedDataY = interpolatedDataY.reshape(numGS, numGS)

        interpolatedDataZ, stdsZ = self.gprZ.predict(self.scaler.transform(np.array([64, 5, 10, 2]).reshape(1, -1)),
                                                     return_std=True)
        interpolatedDataZ = interpolatedDataZ.reshape(numGS, numGS)

        #stochastic minimization algorithm to determine parameter combination used to create synthetic data using surrogate interpolated displacements
        p0 = np.array([100, 2, 50, 10])  # actual values: 65, 5, 10, 2
        minimizer_kwargs = {"bounds": bounds}
        timestart = timeit.default_timer()
        result = basinhopping(self.lossFunction, p0, minimizer_kwargs=minimizer_kwargs, stepsize=2, niter=100)
        timeelaspsed = timeit.default_timer() - timestart

        print("Time taken to run basinhopping: %.4f" % (timeelaspsed))
        print()
        print(
            "global minimum: strength = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f | f(x0) = %.4f" % (
                result.x[0], result.x[1], result.x[2], result.x[3], result.fun))

        #for parameter combination estimated from stochastic minimization algorithm, what displacements are outputted
        bestDataX = self.gprX.predict(self.scaler.transform(result.x.reshape(1, -1))).reshape(numGS, numGS)
        bestDataY = self.gprY.predict(self.scaler.transform(result.x.reshape(1, -1))).reshape(numGS, numGS)
        bestDataZ = self.gprZ.predict(self.scaler.transform(result.x.reshape(1, -1))).reshape(numGS, numGS)

        #plot displacements

        #NOTE, CURRENTLY PLOTS SYNTHETIC VS. BEST FIT MODEL. TO PLOT SYNTHETIC VS SURROGATE FOR same PARAMETERS, REPLACE
        #bestaData_ with interpolatedData_ for all three dimensions

        fig1, axs = plt.subplots(1, 2)
        fig1.suptitle('Synthetic vs. Best Fit Model for Parameters 64 GPa/m^3, 5m, 10m, 2m in X direction')

        pltt = axs[0]
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS), self.syntheticData[:, 0].reshape(numGS, numGS),
                            10, vmin=-5, vmax=15, cmap="magma", extend='both')
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
        fig2.suptitle('Synthetic vs. Best Fit Model for Parameters 64 GPa/m^3, 5m, 10m, 2m in Y direction')

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
        fig3.suptitle('Synthetic vs. Best Fit Model for Parameters 64 GPa/m^3, 5m, 10m, 2m in Z direction')

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

    """
    Loss function used in stochastic minimization algorithm from syntheticSurrogateInversion()
    """
    def lossFunction(self, x):
        #interpolate displacements in all 3 dimensions for certain parameter combination x
        surrogateMeansX = self.gprX.predict(self.scaler.transform(x.reshape(1, -1)))
        surrogateMeansY = self.gprY.predict(self.scaler.transform(x.reshape(1, -1)))
        surrogateMeansZ = self.gprZ.predict(self.scaler.transform(x.reshape(1, -1)))

        #calculate RMSE between interpolate displacements and synthetic data
        rmseX = math.sqrt(mean_squared_error(self.syntheticData[:, 0].reshape(1, -1), surrogateMeansX))
        rmseY = math.sqrt(mean_squared_error(self.syntheticData[:, 1].reshape(1, -1), surrogateMeansY))
        rmseZ = math.sqrt(mean_squared_error(self.syntheticData[:, 2].reshape(1, -1), surrogateMeansZ))

        rmse = rmseX + rmseY + rmseZ

        return rmse

    """
    Conducts an inversion to determine the Mogi parameters of InSAR data by comparing the InSAR data to interpolated
    displacements from a Mogi surrogate for parameters estimated in a stochastic minimization algorithm
    """
    def InSARSurrogateInversion(self):
        ascendingSheet = pd.read_excel('noOffset_strongSNR/ascending.xlsx').to_numpy() #read ascending data sheet
        descendingSheet = pd.read_excel('noOffset_strongSNR/descending.xlsx').to_numpy() #read descending data sheet

        self.observedDataAscending = ascendingSheet[:, 3].reshape(-1, 1) #data from column (d) for ascending
        self.observedDataDescending = descendingSheet[:, 3].reshape(-1, 1) #data from column (d) for descending

        self.trueObservedAscending = ascendingSheet[:, 6].reshape(-1, 1) #data from column (true d) for ascending
        self.trueObservedDescending = descendingSheet[:, 6].reshape(-1, 1) #data from column (true d) for descending

        self.groundStations = np.hstack((ascendingSheet[:, 1].reshape(-1, 1), ascendingSheet[:, 2].reshape(-1, 1))) #ground stations from ascending sheet
        lenGS = len(self.groundStations) #total number of ground stations
        numGS = int(math.sqrt(lenGS)) #number of ground stations along one side of grid

        with open('noOffset_strongSNR/cd_full_asc.txt') as f:
            lines = [float(line.rstrip()) for line in f]
        self.covarAscendingInverse = np.linalg.inv(np.asarray(lines).reshape(lenGS, lenGS)) #covariance data for ascending

        with open('noOffset_strongSNR/cd_full_desc.txt') as f:
            lines = [float(line.rstrip()) for line in f]
        self.covarDescendingInverse = np.linalg.inv(np.asarray(lines).reshape(lenGS, lenGS)) #covariance data for ascending

        #calculation of LOS tranformation matrix for ascending data
        theta = ascendingSheet[0, 4]
        phi = ascendingSheet[0, 5]
        Ha = [-np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]

        # calculation of LOS tranformation matrix for ascending data
        theta = descendingSheet[0, 4]
        phi = descendingSheet[0, 5]
        Hd = [-np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]

        #bounds of parameter space
        bounds = ((math.log10(5e5), math.log10(5e7)), (-4000, 2500), (-5300, 300), (math.log10(1000), math.log10(10000)))
        print(bounds)
        numPoints = 250 #number of training points

        xlimits = np.array([[math.log10(5e5), math.log10(5e7)], [-4000, 2500], [-5300, 300], [math.log10(1000), math.log10(10000)]])
        sampling = LHS(xlimits=xlimits)
        self.trainingX = sampling(numPoints) #generate training data from sampling

        #training output data for ascending and decesending
        trainingYAscending = np.empty((len(self.trainingX), len(self.groundStations)))
        trainingYDescending = np.empty((len(self.trainingX), len(self.groundStations)))

        #populate training output vectors
        count = 0
        for param in self.trainingX:
            displacementsAscending = self.MogiWithDv(self.groundStations, param)
            LOSval = displacementsAscending[:, 0 ] *Ha[0] + displacementsAscending[:, 1 ] *Ha[1] + displacementsAscending[:, 2 ] *Ha[2]
            trainingYAscending[count, :] = LOSval

            displacementsDescending = self.MogiWithDv(self.groundStations, param)
            LOSval = displacementsDescending[:, 0] * Hd[0] + displacementsDescending[:, 1] * Hd[1] + displacementsDescending[:, 2] * Hd[2]
            trainingYDescending[count, :] = LOSval

            count = count + 1

        #scale input for training points
        self.scaler = StandardScaler().fit(self.trainingX)
        self.trainingX = self.scaler.transform(self.trainingX)

        #create GPRs for ascending and descending data
        kernel = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]),
                                           length_scale_bounds=(0.001, 100))
        self.gprA = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=30)

        kernel = ConstantKernel() * Matern(nu=0.5, length_scale=np.array([0.1, 0.1, 0.1, 0.1]),
                                           length_scale_bounds=(0.001, 100))
        self.gprD = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=30)

        #fit GPRs for ascending and descending data
        timestart = timeit.default_timer()
        self.gprA.fit(self.trainingX, trainingYAscending)
        timeelaspsed = timeit.default_timer() - timestart
        print("Time taken to fit Ascending Data: %.4f" % (timeelaspsed))
        timestart = timeit.default_timer()
        self.gprD.fit(self.trainingX, trainingYDescending)
        timeelaspsed = timeit.default_timer() - timestart
        print("Time taken to fit Descending Data: %.4f" % (timeelaspsed))

        #conduct stochastic minimization algorithm
        p0 = np.array([1, 1, 1, 1])  # actual values: 64, 5, 10, 2
        minimizer_kwargs = {"bounds": bounds}
        timestart = timeit.default_timer()
        result = basinhopping(self.InSARlossFunction, p0, minimizer_kwargs=minimizer_kwargs, stepsize=50, niter=200)
        timeelaspsed = timeit.default_timer() - timestart

        print("Time taken to run basinhopping: %.4f" % (timeelaspsed))
        print()
        print(
            "global minimum: dV = %.4f, source x = %.4f, source y = %.4f, source depth = %.4f | f(x0) = %.4f" % (
                10**result.x[0], result.x[1], result.x[2], 10**result.x[3], result.fun))

        #data interpolated by results from stochastic minimization algorithm
        bestDataA = self.gprA.predict(self.scaler.transform(result.x.reshape(1, -1))).reshape(numGS, numGS)
        bestDataD = self.gprD.predict(self.scaler.transform(result.x.reshape(1, -1))).reshape(numGS, numGS)

        #plot truth, data, interpolated data, and residuals for ascending and descending data
        fig1, axs1 = plt.subplots(2, 2)
        plt.subplots_adjust(wspace=0.6, hspace=0.4)
        fig1.suptitle('Ascending In-SAR Observed vs. Surrogate Displacement Measurements for No Offset Strong SNR')

        pltt = axs1[0, 0]
        pltt.set_aspect('equal')
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS),
                            self.trueObservedAscending.reshape(numGS, numGS),
                            150, vmin=np.min(self.observedDataAscending), vmax=np.max(self.observedDataAscending),
                            cmap="jet")
        cbar = fig1.colorbar(con, ax=pltt)
        cbar.ax.set_ylabel('Displacement', rotation=90)
        cbar.ax.get_yaxis().labelpad = 10
        pltt.set_title("Truth (m)")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        pltt = axs1[0, 1]
        pltt.set_aspect('equal')
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS),
                            self.observedDataAscending.reshape(numGS, numGS),
                            150, vmin=np.min(self.observedDataAscending), vmax=np.max(self.observedDataAscending),
                            cmap="jet")
        cbar = fig1.colorbar(con, ax=pltt)
        cbar.ax.set_ylabel('Displacement', rotation=90)
        cbar.ax.get_yaxis().labelpad = 10
        pltt.set_title("Data (m)")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        pltt = axs1[1, 0]
        pltt.set_aspect('equal')
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS),
                            bestDataA,
                            150, vmin=np.min(self.observedDataAscending), vmax=np.max(self.observedDataAscending),
                            cmap="jet")
        cbar = fig1.colorbar(con, ax=pltt)
        cbar.ax.set_ylabel('Displacement', rotation=90)
        cbar.ax.get_yaxis().labelpad = 10
        pltt.set_title("Surrogate (m)")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        pltt = axs1[1, 1]
        pltt.set_aspect('equal')
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS),
                            self.observedDataAscending.reshape(numGS, numGS) - bestDataA,
                            150, vmin=np.min(self.observedDataAscending.reshape(numGS, numGS) - bestDataA), vmax=np.max(self.observedDataAscending.reshape(numGS, numGS) - bestDataA),
                            cmap="jet")
        cbar = fig1.colorbar(con, ax=pltt)
        cbar.ax.set_ylabel('Displacement', rotation=90)
        cbar.ax.get_yaxis().labelpad = 10
        pltt.set_title("Data Minus Surrogate (m)")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        fig2, axs2 = plt.subplots(2, 2)
        plt.subplots_adjust(wspace=0.6, hspace=0.4)
        fig2.suptitle('Descending In-SAR Observed vs. Surrogate Displacement Measurements for No Offset Strong SNR')

        pltt = axs2[0, 0]
        pltt.set_aspect('equal')
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS),
                            self.trueObservedDescending.reshape(numGS, numGS),
                            150, vmin=np.min(self.observedDataDescending),
                            vmax=np.max(self.observedDataDescending),
                            cmap="jet")
        cbar = fig2.colorbar(con, ax=pltt)
        cbar.ax.set_ylabel('Displacement', rotation=90)
        cbar.ax.get_yaxis().labelpad = 10
        pltt.set_title("Truth (m)")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        pltt = axs2[0, 1]
        pltt.set_aspect('equal')
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS),
                            self.observedDataDescending.reshape(numGS, numGS),
                            150, vmin=np.min(self.observedDataDescending),
                            vmax=np.max(self.observedDataDescending),
                            cmap="jet")
        cbar = fig2.colorbar(con, ax=pltt)
        cbar.ax.set_ylabel('Displacement', rotation=90)
        cbar.ax.get_yaxis().labelpad = 10
        pltt.set_title("Data (m)")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        pltt = axs2[1, 0]
        pltt.set_aspect('equal')
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS),
                            bestDataD,
                            150, vmin=np.min(self.observedDataDescending),
                            vmax=np.max(self.observedDataDescending),
                            cmap="jet")
        cbar = fig2.colorbar(con, ax=pltt)
        cbar.ax.set_ylabel('Displacement', rotation=90)
        cbar.ax.get_yaxis().labelpad = 10
        pltt.set_title("Surrogate (m)")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        pltt = axs2[1, 1]
        pltt.set_aspect('equal')
        con = pltt.contourf(self.groundStations[:, 0].reshape(numGS, numGS),
                            self.groundStations[:, 1].reshape(numGS, numGS),
                            self.observedDataDescending.reshape(numGS, numGS) - bestDataD,
                            150, vmin=np.min(self.observedDataDescending.reshape(numGS, numGS) - bestDataD),
                            vmax=np.max(self.observedDataDescending.reshape(numGS, numGS) - bestDataD),
                            cmap="jet")
        cbar = fig2.colorbar(con, ax=pltt)
        cbar.ax.set_ylabel('Displacement', rotation=90)
        cbar.ax.get_yaxis().labelpad = 10
        pltt.set_title("Data Minus Surrogate (m)")
        pltt.set_xlabel("X")
        pltt.set_ylabel("Y")

        plt.show()

        # Synthetic Data
        # Best Fit Model

        plot = plt.figure(1)
        plt.plot(self.groundStations[:19, 1], self.syntheticData[:19].reshape(-1), 'r')
        plt.plot(self.groundStations[:19, 1], self.gpr.predict(result.x.reshape(1, -1))[0, :19], 'b')
        plt.xlabel("Y")
        plt.ylabel("Vertical Displacement")
        plt.title("Synthetic vs Surrogate Output along x=1")
        plt.legend(["Synthetic", "Surrogate"])
        plt.show()

    """
    Loss function used in stochastic minimization algorithm from InSARSurrogateInversion()
    """
    def InSARlossFunction(self, x):
        #calculate interpolated LOS displacements for certain parameter combination x and calculate RMSE thereafter
        #for both ascending and descending data
        surrogateMeansA = self.gprA.predict(self.scaler.transform(x.reshape(1, -1))).reshape(-1, 1)
        aDifference = self.observedDataAscending - surrogateMeansA
        rmseA = np.dot(np.dot(aDifference.T, self.covarAscendingInverse), aDifference)

        surrogateMeansD = self.gprD.predict(self.scaler.transform(x.reshape(1, -1))).reshape(-1, 1)
        dDifference = self.observedDataDescending - surrogateMeansD
        rmseD = np.dot(np.dot(dDifference.T, self.covarDescendingInverse), dDifference)

        print(x)
        rmse = rmseA + rmseD
        rmse = rmse[0][0]

        return rmse

    """
    Calculates Mogi displacements using magnitude of dV/pi instead of alpha^3*deltaP
    
    @:param xyLocations -- n x 2 matrix containing x and y locations of n ground stations
    @:param params -- log10(dV), source x, source y, and log10(source depth) vector for which displacements at ground stations should be calculated
    
    @:return a nx3 matrix of displacements in x, y, and z directions for n ground stations
    """

    def MogiWithDv(self, xyLocations, params):  # coordinates are strength, x, y, and magnitude of cavity depth
        displacements = np.empty((len(xyLocations), 3))  # initialize displacements array
        coordinates = np.empty((len(xyLocations), 4))
        coordinates[:, 0] = np.full((len(xyLocations)), 10**params[0]) #initialize first column of coordinates with dV
        coordinates[:, 1] = xyLocations[:, 0] - params[1] #initialize second column of coordinates with x difference between GSs and source
        coordinates[:, 2] = xyLocations[:, 1] - params[2] #initialize second column of coordinates with y difference between GSs and source
        coordinates[:, 3] = np.full((len(xyLocations)), 10**params[3]) #initialize second column of coordinates with x difference between GS and source
        Rvect = LA.norm(coordinates[:, 1:4], axis = 1).T #calculate R vector from source to GS
        magVect = coordinates[:, 0] * (1 - self.poisson) / math.pi /(np.power(Rvect, 3)) #calculate magnitude which will later be multiplied by x, y, and d
        displacements = (coordinates[:, 1:4].T*magVect).T #calculate displacements by multiplying magnitude with x, y, and d
        return displacements

    """
    Calculates Mogi displacements using magnitude of alpha^3*deltaP

    @:param xyLocations -- n x 2 matrix containing x and y locations of n ground stations
    @:param params -- strength, source x, source y, and source depth vector for which displacements at ground stations should be calculated

    @:return a nx3 matrix of displacements in x, y, and z directions for n ground stations
    """
    def MogiWithStrength(self, xyLocations, params):  # coordinates are strength, x, y, and magnitude of cavity depth
        displacements = np.empty((len(xyLocations), 3))  # initialize displacements array
        coordinates = np.empty((len(xyLocations), 4))
        coordinates[:, 0] = np.full((len(xyLocations)), params[0]) #initialize first column of coordinates with strengths
        coordinates[:, 1] = xyLocations[:, 0] - params[1] #initialize second column of coordinates with x difference between GSs and source
        coordinates[:, 2] = xyLocations[:, 1] - params[2] #initialize second column of coordinates with y difference between GSs and source
        coordinates[:, 3] = np.full((len(xyLocations)), params[3]) #initialize second column of coordinates with x difference between GS and source
        Rvect = LA.norm(coordinates[:, 1:4], axis = 1).T #calculate R vector from source to GS
        magVect = coordinates[:, 0] * (1 - self.poisson) / self.shearModulus/(np.power(Rvect, 3)) #calculate magnitude which will later be multiplied by x, y, and d
        displacements = (coordinates[:, 1:4].T*magVect).T #calculate displacements by multiplying magnitude with x, y, and d
        return displacements

if __name__ == '__main__':
    mainObject = Main()
    # mainObject.syntheticSurrogateInversion()
    mainObject.InSARSurrogateInversion()