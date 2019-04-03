##----------------------------------------------------------------------------------
## The University of Manchester, School of Chemical Engineering and Analytical Science
## History Matching of Reservoirs
## Reservoir Case Study : SPE-10
## Reservoir Simulator : ECLIPSE

## Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
## Supervisor:Dr Rossmary Villegas
## MEng Student: Yap Shan Wei
##-----------------------------------------------------------------------------------
## Global Module for all User Input Variables

print('History Matching of SPE-10 with Covariance Localisation ')
## Dictionary for Model Parameters Investigated
print(' Model Parameters Investigated : ')
parameters = {"  1st layer :" : "Permeability",\
              "  2nd layer :" : "Porosity"}

for layers, parameter in parameters.items():
    print(layers, parameter)
    
# Number of model parameters investigated
Np = len(parameters)

## Dimensions of reservoir
Nx = int(input(' Number of grids in the x-direction(120) : '))
Ny = int(input(' Number of grids in the y-direction(60) : '))
Nz = int(input(' Number of grids in the z-direction for one reservoir property (5) : '))
totalgrids = Nx*Ny*Nz

N = int(input(' Number of realisations(100) : '))
No = int(input(' Number of observations(17) : '))
Nt = int(input(' Number of timesteps for history period(36) : '))




