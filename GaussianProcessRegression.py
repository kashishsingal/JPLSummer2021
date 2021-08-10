"""
Performs Gaussian Process Regression with an example model of bivariate input and univariate output

**only functions kernel() and globalGaussianProcessRegression() are to be used for other models**

author: Kashish Singal (NASA JPL)
"""


import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm
from scipy.optimize import minimize
from scipy.optimize import check_grad
from scipy.optimize import basinhopping
import math
from sklearn.metrics import mean_squared_error

class GP:

    def __init__(self):
        pass

    """
    
    Exponential Kernel function for multivariate input and output
    @param x, y: two different training/testing matrices whose covariance shall be calculated
    @param tau, sigma: two of kernel's hyperparameters
    
    @return: covariance matrix
    """
    def kernel3(self, x, y, tau, sigma):
        sqdist = np.sum(x ** 2, 1).reshape(-1, 1) + np.sum(y ** 2, 1) - 2 * np.dot(x, y.T) #sum squares row wise and create (3x1) vectors, add to third term
        return tau*np.exp(-.5 * (1.0/(sigma)) * sqdist)

    def kernel(self, x, y, hyperparams):
        tau = hyperparams[0]
        thetas = np.array(hyperparams[1:5])
        Mvect = thetas
        Mmatrix = np.diag(thetas)
        firstTerm = np.sum((x ** 2) * Mvect, 1).reshape(-1, 1)
        secondTerm = np.sum((y ** 2) * Mvect, 1)
        thirdTerm = -2.0 * np.dot(np.dot(x, Mmatrix), y.T)  # sum squares row wise and create (3x1) vectors, add to third term
        sqdist = firstTerm + secondTerm + thirdTerm
        return tau*np.exp(-.5 * sqdist)

    def kernel5(self, x, y, tau, numParams):
        Mvect = np.ones(numParams)
        Mmatrix = np.eye(numParams)
        xByLengthScale = x * Mvect
        yByLengthScale = y * Mvect
        firstTerm = np.sum(xByLengthScale ** 2, 1).reshape(-1, 1)
        secondTerm = np.sum(yByLengthScale ** 2, 1)
        thirdTerm = -2.0 * np.dot(np.dot(x, Mmatrix), y.T) #sum squares row wise and create (3x1) vectors, add to third term
        sqdist = firstTerm + secondTerm + thirdTerm
        return tau*np.exp(-.5 * sqdist)

    def kernel4(self, x, y, tau, sigma):
        Mvect = np.array([1.0, 1.0])
        Mmatrix = np.array([[1.0, 0], [0, 1.0]])
        xByLengthScale = x * Mvect
        yByLengthScale = y * Mvect
        firstTerm = np.sum((x ** 2) * Mvect, 1).reshape(-1, 1)
        secondTerm = np.sum((y ** 2) * Mvect, 1)
        thirdTerm = -2.0 * np.dot(np.dot(x, Mmatrix), y.T) #sum squares row wise and create (3x1) vectors, add to third term
        sqdist = firstTerm + secondTerm + thirdTerm
        return tau*np.exp(-.5 * sqdist)

    """
    Multivariate input and output GP regression
    @param trainingX: this is the training input matrix (training points along rows, input dimensions along columns)
    @param trainingY: this is the training output matrix (training points along rows, output dimensions along columns)
    @param testingX: this is the testing input matrix (testing points along rows, input dimensions along columns)
    @param tau: first hyperparameter of kernel function
    @param sigma: second hyperparameter of kernel function
    
    @return: mean (vector/matrix) and standard deviation vector of each testing point's output
    """
    def globalGaussianProcessRegression(self, trainingX, trainingY, testingX, hyperparams):
        K = self.kernel(trainingX, trainingX, hyperparams) + 0.005 * np.eye(len(trainingX))
        Ks = self.kernel(trainingX, testingX, hyperparams)
        Kss = self.kernel(testingX, testingX, hyperparams)

        # firstMult1 = np.matmul(Ks.T, np.linalg.inv(K))
        # muVector = np.squeeze(np.matmul(firstMult1, trainingY))
        #
        # firstMult2 = np.matmul(Ks.T, np.linalg.inv(K))
        # covarMatrix = Kss - np.matmul(firstMult2, Ks)
        # stdVector = np.squeeze(np.diag(covarMatrix))

        L = np.linalg.cholesky(K)
        alpha = np.matmul(np.linalg.inv(L.T), np.matmul(np.linalg.inv(L), trainingY))
        muVector = np.matmul(Ks.T, alpha)

        v = np.matmul(np.linalg.inv(L), Ks)
        stdVector = np.squeeze(np.diag(Kss - np.matmul(v.T,v)))

        return muVector, stdVector

    def kernel2(self, xi, xj, tau, sigma):
        sqdist = (xi-xj)**2
        return tau*np.exp(-.5 * (1.0/(sigma*2)) * sqdist)

    def exponential2DTest(self):
        trainingX = 20*(np.random.rand(50, 1)) - 10
        trainingY = np.sin(trainingX) + 1
        testingX = np.linspace(-10, 10, 100)[None].T
        muVector, stdVector = self.globalGaussianProcessRegression(trainingX, trainingY, testingX, 1.0, 1.0)

        plot1 = plt.figure(1)
        plt.plot(trainingX, trainingY, 'bs', ms=8)
        plt.plot(np.squeeze(testingX.transpose()), np.squeeze(muVector.transpose()), 'r--', lw=1)
        plt.plot(testingX, np.sin(testingX) + 1, 'g--', lw=1)
        plt.gca().fill_between(np.squeeze(testingX.T), np.squeeze(muVector.T - 2.0*stdVector), np.squeeze(muVector.T + 2.0*stdVector), color="#dddddd")
        plt.axis([-10, 10, -2, 3])
        plt.title('Gaussian Regression Exponential')
        plt.show()


    def conductOptimization(self):
        nTestingLinear = 50j
        nTesting = 50
        nTraining = 10

        # trainingX = 20 * (np.random.rand(nTraining, 2)) - 10  # nTrainingx2 random points from -10 to 10
        # trainingY = 6 * np.sin(trainingX[:, 0]) + 3 * np.sin(trainingX[:, 1])

        testingX = np.mgrid[-10:10:50j, -10:10:50j].reshape(2, -1).T
        synthetic = 6 * np.sin(testingX[:, 0]) + 3 * np.sin(testingX[:, 1])

        p0 = np.array([1, 7])  # initial parameters 30, 10, 6, 6
        bnds = ((0.5, 10), (0.5, 10))

        # result = minimize(self.lossFunction, p0, method='Nelder-Mead', bounds = bnds, options={'disp': True, 'maxiter': 300})
        minimizer_kwargs = {"bounds": bnds, "args": synthetic}
        result = basinhopping(self.lossFunction, p0, minimizer_kwargs=minimizer_kwargs, stepsize=5, niter=20)
        print(
            "global minimum: x1 = %.4f, x2 x = %.4f | f(x0) = %.4f" % (
            result.x[0], result.x[1], result.fun))

    def conductRegression(self, nTraining, nTestingLinear, tau, sigma):
        trainingX = 20 * (np.random.rand(nTraining, 2)) - 10  # nTrainingx2 random points from -10 to 10
        trainingY = np.sin(trainingX[:, 0]) + 3*np.cos(trainingX[:, 1])
        testingX = np.mgrid[-10:10:nTestingLinear, -10:10:nTestingLinear].reshape(2, -1).T

        # mean = np.mean(trainingY)
        # tempTrainingY = trainingY
        # trainingY = trainingY - mean

        X, Y = np.mgrid[-10:10:nTestingLinear, -10:10:nTestingLinear]
        Z = np.sin(X) + 3*np.cos(Y)

        K = self.kernel4(trainingX, trainingX, tau, sigma) + 0.00005 * np.eye(len(trainingX))
        Ks = self.kernel4(trainingX, testingX, tau, sigma)
        Kss = self.kernel4(testingX, testingX, tau, sigma)

        # firstMult1 = np.matmul(Ks.T, np.linalg.inv(K))
        # muVector = np.squeeze(np.matmul(firstMult1, trainingY))
        #
        # firstMult2 = np.matmul(Ks.T, np.linalg.inv(K))
        # covarMatrix = Kss - np.matmul(firstMult2, Ks)
        # stdVector = np.squeeze(np.diag(covarMatrix))

        L = np.linalg.cholesky(K)
        alpha = np.matmul(np.linalg.inv(L.T), np.matmul(np.linalg.inv(L), trainingY))
        muVector = np.matmul(Ks.T, alpha)
        v = np.matmul(np.linalg.inv(L), Ks)
        stdVector = np.squeeze(np.diag(Kss - np.matmul(v.T,v)))

        # muVector = muVector + mean
        return trainingX, trainingY, testingX, muVector, stdVector, X, Y, Z

    def gaussianProcess(self):
        nTestingLinear = 50j
        nTesting = 50
        nTraining = 10

        #hyperparameters
        tau = 1.0
        sigma = 1.0

        fig1, (ax1, ax2, ax3) = plt.subplots(3)
        fig2, (ax4, ax5, ax6) = plt.subplots(3)
        fig1.suptitle('x1=-10 Cross Section')
        fig2.suptitle('x2=0.2041 Cross Section')
        fig1.tight_layout(pad=3.0)
        fig2.tight_layout(pad=3.0)

        trainingX, trainingY, testingX, muVector, stdVector, X, Y, Z = self.conductRegression(nTraining, nTestingLinear, tau, sigma)

        ax1.plot(np.squeeze(testingX[:nTesting, 1].transpose()), np.squeeze(muVector[:nTesting].transpose()), c='r', linestyle='-') #mean points
        ax1.plot(testingX[:nTesting, 1], np.sin(testingX[:nTesting, 0]) + 3*np.cos(testingX[:nTesting, 1]), 'k--', lw=1) #actual function
        ax1.fill_between(np.squeeze(testingX[:nTesting, 1].transpose()),
                              np.squeeze(muVector[:nTesting] - 2.0 * stdVector[:nTesting]),
                              np.squeeze(muVector[:nTesting] + 2.0 * stdVector[:nTesting]), color="#dddddd") #uncertainty
        ax1.set_xlabel("x2")
        ax1.set_ylabel("y")
        ax1.title.set_text('10 Training Points')
        ax1.axis([-10, 10, -4.5, 4.5])

        ax4.plot(np.squeeze(testingX[25::50, 0].transpose()), np.squeeze(muVector[25::50].transpose()), c='r',
                  linestyle='-')  # mean points
        ax4.plot(testingX[25::50, 0], np.sin(testingX[25::50, 0]) + 3*np.cos(testingX[25::50, 1]), 'k--',
                 lw=1)  # actual function
        ax4.fill_between(np.squeeze(testingX[25::50, 0].transpose()),
                         np.squeeze(muVector[25::50] - 2.0 * stdVector[25::50]),
                         np.squeeze(muVector[25::50] + 2.0 * stdVector[25::50]), color="#dddddd")  # uncertainty
        ax4.set_xlabel("x1")
        ax4.set_ylabel("y")
        ax4.title.set_text('10 Training Points')
        ax4.axis([-10, 10, -4.5, 4.5])


        nTraining = 100
        trainingX, trainingY, testingX, muVector, stdVector, X, Y, Z = self.conductRegression(nTraining, nTestingLinear, tau,
                                                                                       sigma)
        ax2.plot(np.squeeze(testingX[:nTesting, 1].transpose()), np.squeeze(muVector[:nTesting].transpose()), c='r',
                  linestyle='-')  # mean points
        ax2.plot(testingX[:nTesting, 1], np.sin(testingX[:nTesting, 0]) + 3*np.cos(testingX[:nTesting, 1]), 'k--',
                 lw=1)  # actual function
        ax2.fill_between(np.squeeze(testingX[:nTesting, 1].transpose()),
                               np.squeeze(muVector[:nTesting] - 2.0 * stdVector[:nTesting]),
                               np.squeeze(muVector[:nTesting] + 2.0 * stdVector[:nTesting]), color="#dddddd")  # uncertainty
        ax2.set_xlabel("x2")
        ax2.set_ylabel("y")
        ax2.title.set_text('100 Training Points')
        ax2.axis([-10, 10, -4.5, 4.5])

        ax5.plot(np.squeeze(testingX[25::50, 0].transpose()), np.squeeze(muVector[25::50].transpose()), c='r',
                  linestyle='-')  # mean points
        ax5.plot(testingX[25::50, 0], np.sin(testingX[25::50, 0]) + 3*np.cos(testingX[25::50, 1]), 'k--',
                 lw=1)  # actual function
        ax5.fill_between(np.squeeze(testingX[25::50, 0].transpose()),
                         np.squeeze(muVector[25::50] - 2.0 * stdVector[25::50]),
                         np.squeeze(muVector[25::50] + 2.0 * stdVector[25::50]), color="#dddddd")  # uncertainty
        ax5.set_xlabel("x1")
        ax5.set_ylabel("y")
        ax5.title.set_text('100 Training Points')
        ax5.axis([-10, 10, -4.5, 4.5])

        nTraining = 1000
        trainingX, trainingY, testingX, muVector, stdVector, X, Y, Z = self.conductRegression(nTraining, nTestingLinear, tau,
                                                                                       sigma)
        ax3.plot(np.squeeze(testingX[:nTesting, 1].transpose()), np.squeeze(muVector[:nTesting].transpose()), c='r',
                  linestyle='-')  # mean points
        ax3.plot(testingX[:nTesting, 1], np.sin(testingX[:nTesting, 0]) + 3*np.cos(testingX[:nTesting, 1]), 'k--',
                 lw=1)  # actual function
        ax3.fill_between(np.squeeze(testingX[:nTesting, 1].transpose()),
                               np.squeeze(muVector[:nTesting] - 2.0 * stdVector[:nTesting]),
                               np.squeeze(muVector[:nTesting] + 2.0 * stdVector[:nTesting]), color="#dddddd")  # uncertainty
        ax3.set_xlabel("x2")
        ax3.set_ylabel("y")
        ax3.title.set_text('1000 Training Points')
        ax3.axis([-10, 10, -4.5, 4.5])

        ax6.plot(np.squeeze(testingX[25::50, 0].transpose()), np.squeeze(muVector[25::50].transpose()), c='r',
                  linestyle='-')  # mean points
        ax6.plot(testingX[25::50, 0], np.sin(testingX[25::50, 0]) + 3*np.cos(testingX[25::50, 1]), 'k--',
                 lw=1)  # actual function
        ax6.fill_between(np.squeeze(testingX[25::50, 0].transpose()),
                         np.squeeze(muVector[25::50] - 2.0 * stdVector[25::50]),
                         np.squeeze(muVector[25::50] + 2.0 * stdVector[25::50]), color="#dddddd")  # uncertainty
        ax6.set_xlabel("x1")
        ax6.set_ylabel("y")
        ax6.title.set_text('1000 Training Points')
        ax6.axis([-10, 10, -4.5, 4.5])

        # pl.figure(1)
        # # vals = np.argwhere(trainingX[:, 0]<-9)
        # # pl.plot(trainingX[vals, 1], trainingY[vals], 'bs', ms=8) #plot the training points along x1 = -10
        # pl.plot(np.squeeze(testingX[:nTesting, 1].transpose()), np.squeeze(muVector[:nTesting].transpose()), c='r', marker='o', linestyle='--') #plot the points along x1 = -10
        # pl.plot(testingX[:nTesting, 1], np.sin(testingX[:nTesting, 0] + testingX[:nTesting, 1]), 'g--', lw=1)
        # pl.gca().fill_between(np.squeeze(testingX[:nTesting, 1].transpose()), np.squeeze(muVector[:nTesting] - 2.0*stdVector[:nTesting]), np.squeeze(muVector[:nTesting] + 2.0*stdVector[:nTesting]), color="#dddddd")
        # pl.xlabel("x2")
        # pl.ylabel("y")
        # pl.axis([-10, 10, -3, 3])
        # pl.title('Gaussian Regression with x1=-10')
        # pl.show()

        nTraining = 500
        nTestingLinear = 30j
        trainingX, trainingY, testingX, muVector, stdVector, X, Y, Z = self.conductRegression(nTraining, nTestingLinear, tau,
                                                                                       sigma)
        plot3 = plt.figure(3)
        ax = plot3.add_subplot(projection="3d")
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.set_zlim(-4.5, 4.5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        #ax.scatter3D(trainingX[:, 0], trainingX[:, 1], trainingY, c='b')
        surf = ax.plot_surface(X, Y, Z, color="black")
        ax.scatter3D(testingX[:, 0], testingX[:, 1], muVector, c='r', marker='.')
        plt.title('Gaussian Regression Multiple Input Parameters')

        plt.show()

    def conductSimpleRegression(self):
        trainingX = np.array([-9, -5, -3, 2, 3, 4, 7]).reshape(-1, 1)
        trainingY = 5 * np.sin(trainingX) + 3
        testingX = np.linspace(-10, 10, 100).reshape(-1, 1)
        tau = 1.0
        sigma = 1.0

        K = self.kernel3(trainingX, trainingX, tau, sigma) + 0.00005 * np.eye(len(trainingX))
        Ks = self.kernel3(trainingX, testingX, tau, sigma)
        Kss = self.kernel3(testingX, testingX, tau, sigma)

        L = np.linalg.cholesky(K)
        alpha = np.matmul(np.linalg.inv(L.T), np.matmul(np.linalg.inv(L), trainingY))
        muVector = np.matmul(Ks.T, alpha)
        v = np.matmul(np.linalg.inv(L), Ks)
        stdVector = np.squeeze(np.diag(Kss - np.matmul(v.T, v)))

        plot1 = plt.figure(1)
        plt.plot(trainingX, trainingY, 'bs', ms=8)
        plt.plot(np.squeeze(testingX.transpose()), np.squeeze(muVector.transpose()), 'r--', lw=1)
        plt.plot(testingX, 5*np.sin(testingX) + 3, 'k-', lw=1)
        plt.gca().fill_between(np.squeeze(testingX.T), np.squeeze(muVector.T - 2.0 * stdVector),
                               np.squeeze(muVector.T + 2.0 * stdVector), color="#dddddd")
        plt.axis([-10, 10, -4, 10])
        plt.title('Gaussian Regression for f(x) = 5*sin(x)+3')
        plt.legend(["Training Points", "Testing Points", "Training Function"])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    gp = GP()
    # gp.gaussianProcess()
    gp.conductSimpleRegression()
    #gp.exponential2DTest()
    # gp.conductOptimization()

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

    #PREVIOUS NONEFFICIENT CODE

    # mat1 = np.array([[1, 2, 7], [3, 4, 8], [5, 6, 9]])
        # mat2 = np.array([[7, 8, 11], [9, 10, 12]])
        # print(kernel2(mat1, mat2, tau, sigma))
        #
        # #develop K matrix
        # K = np.zeros([nTraining, nTraining])
        # for i in range(0, nTraining):
        #     for j in range(0, nTraining):
        #         K[i, j] = kernel(trainingX[i], trainingX[j], tau, sigma)
        #
        # #develop K** matrix
        # Kss = np.zeros([nTesting, nTesting])
        # for i in range(0, nTesting):
        #     for j in range(0, nTesting):
        #         Kss[i, j] = kernel(testingX[i], testingX[j], tau, sigma)
        #
        # #develop K* matrix
        # Ks = np.zeros([nTraining, nTesting])
        # for i in range(0, nTraining):
        #     for j in range(0, nTesting):
        #         Ks[i, j] = kernel(trainingX[i], testingX[j], tau, sigma)
        #
        # firstMult1 = np.matmul(Ks.T, np.linalg.inv(K))
        # muVector = np.squeeze(np.matmul(firstMult1, trainingY))
        #
        # firstMult2 = np.matmul(Ks.T, np.linalg.inv(K))
        # covarMatrix = Kss - np.matmul(firstMult2, Ks)
        # stdVector = np.squeeze(np.diag(covarMatrix))
        #
        #
        # plot1 = pl.figure(1)
        # pl.plot(trainingX, trainingY, 'bs', ms=8)
        # pl.plot(np.squeeze(testingX.transpose()), np.squeeze(muVector.transpose()), 'r--', lw=1)
        # pl.plot(testingX, np.sin(testingX), 'g--', lw=1)
        # pl.gca().fill_between(np.squeeze(testingX.transpose()), np.squeeze(muVector - 2.0*stdVector), np.squeeze(muVector + 2.0*stdVector), color="#dddddd")
        # pl.axis([-10, 10, -3, 3])
        # pl.title('Gaussian Regression')