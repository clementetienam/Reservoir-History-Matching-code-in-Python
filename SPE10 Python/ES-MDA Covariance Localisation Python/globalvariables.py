# This file is created to allow all Python scripts in the History Matching
# programme to be able access the variables inputted by the user.

# Acts as global variables

# Dimensions of reservoir
Nx = int(input(' Number of grids in the x-direction(120) : '))
Ny = int(input(' Number of grids in the y-direction(60) : '))
Nz = int(input(' Number of grids in the z-direction for one reservoir property (5) : '))

N = int(input(' Number of realisations(100) : '))

No = int(input(' Number of observations(17) : '))

Nt = int(input(' Number of timesteps for history period(36) : '))

Np = int(input(' Number of parameters used for history matching(4 for ES / 2 for EnKF) : '))


truedata = input(' Filename for true data (Real.RSM) : ')
print( ' ' )
# Creating a dictionary for the parameters used
parameters = {"1st layer :" : "Permeability",\
              "2nd layer :" : "Porosity",\
              "3rd layer :" : "Water Saturation( Not used for EnKF )",\
              "4th layer :" : "Pressure( Not used for EnKF )"}

for layers, parameter in parameters.items():
    print(layers, parameter)
    
totalgrids = Nx*Ny*Nz






