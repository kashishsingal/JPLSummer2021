# JPLSummer2021

This project was intended to use surrogate models for parameter estimation of volcanic system models.

main.py() runs two different types of inversions using a surrogate (GPR) of the Mogi model:

Function syntheticSurrogateInversion() conducts an inversion to determine the Mogi parameters used to create synthetic
data by comparing the synthetic data to interpolated displacements from a Mogi surrogate for parameters estimated in a
stochastic minimization algorithm

Function InSARSurrogateInversion() conducts an inversion to determine the Mogi parameters of InSAR data by comparing the
InSAR data to interpolated displacements from a Mogi surrogate for parameters estimated in a stochastic minimization
algorithm
