##----------------------------------------------------------------------------------
## The University of Manchester, School of Chemical Engineering and Analytical Science
## History Matching of Reservoirs
## Reservoir Case Study : SPE-10
## Reservoir Simulator : ECLIPSE

## Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
## Supervisor:Dr Rossmary Villegas
## MEng Student: Yap Shan Wei
##-----------------------------------------------------------------------------------

# Plot of Permeability and Porosity for True, Initial, MATLAB and Python
## Modules Used
## --------------------------------------------------------------------------
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glbvar as glb
## --------------------------------------------------------------------------

## Font Type in Graph Plotting
##-----------------------------------------------------------------------------------
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
##------------------------------------------------------------------------------------

## Creating A Folder for the Plotted Results
## -----------------------------------------------------------------------------
dir = 'Results'
if not os.path.exists(dir):
    os.makedirs(dir)
else:
    shutil.rmtree(dir)
    os.makedirs(dir)

## Creating a colormap from a list of colors
##--------------------------------------------------------------------------------
colors = [(0, 0, 0),(.3, .15, .75),(.6, .2, .50),(1, .25, .15),(.9, .5, 0),(.9, .9, .5),(1, 1, 1)]
n_bins = 7  # Discretizes the interpolation into bins
cmap_name = 'my_list'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N = n_bins)
##--------------------------------------------------------------------------------

layers = 4
print(' Plotting the ensemble permeability field data ')
oldfolder = os.getcwd()
print('  Loading the relevant files  ')
print('  Extracting True ')
## True Permeability and Porosity  realisation
##---------------------------------------------------------------------------
rossmary = open('rossmary.GRDECL')
rossmary = np.fromiter(rossmary,float)
rossmaryporo = open('rossmaryporo.GRDECL')
rossmaryporo = np.fromiter(rossmaryporo,float)
Trueperm = np.reshape(np.log10(rossmary),(glb.Nx,glb.Ny,2*glb.Nz),'F')
Trueporo = np.reshape(rossmaryporo,(glb.Nx,glb.Ny,2*glb.Nz),'F')
##---------------------------------------------------------------------------

print('  Extracting Initial ')
## Initial Permeability and Porosity  realisation
##---------------------------------------------------------------------------
print(' Extracting the initial grid cells ')
sgsim = open('sgsim.out')
sgsim = np.fromiter(sgsim,float)
sgsimporo = open('sgsimporo.out')
sgsimporo = np.fromiter(sgsimporo,float)
sgsim = np.reshape(sgsim,(2*glb.totalgrids,glb.N),'F')
sgsimporo = np.reshape(sgsimporo,(2*glb.totalgrids,glb.N),'F')
##---------------------------------------------------------------------------

print('  Extracting MATLAB ')
## History Matched MATLAB Permeability and Porosity 
##---------------------------------------------------------------------------
print(' Extracting the active grid cells for MATLAB ')
sgsimfinalmat = open('sgsimfinal.out')
sgsimfinalmat = np.fromiter(sgsimfinalmat,float)
sgsimporofinalmat = open('sgsimporofinal.out')
sgsimporofinalmat = np.fromiter(sgsimporofinalmat,float)
sgsimfinalmat = np.reshape(sgsimfinalmat,(2*glb.totalgrids,glb.N),'F')
sgsimporofinalmat = np.reshape(sgsimporofinalmat,(2*glb.totalgrids,glb.N),'F')
##---------------------------------------------------------------------------

print('  Extracting Python ')
## History Matched Python Permeability and Porosity 
##---------------------------------------------------------------------------
print(' Extracting the active grid cells for Python ')
sgsimfinalpy = open('sgsimfinalpy.out')
sgsimfinalpy = np.fromiter(sgsimfinalpy,float)
sgsimporofinalpy = open('sgsimporofinalpy.out')
sgsimporofinalpy = np.fromiter(sgsimporofinalpy,float)
sgsimfinalpy = np.reshape(sgsimfinalpy,(2*glb.totalgrids,glb.N),'F')
sgsimporofinalpy = np.reshape(sgsimporofinalpy,(2*glb.totalgrids,glb.N),'F')
##---------------------------------------------------------------------------


sg = np.empty((glb.totalgrids,glb.N))
sgporo = np.empty((glb.totalgrids,glb.N))

sgmat = np.empty((glb.totalgrids,glb.N))
sgporomat = np.empty((glb.totalgrids,glb.N))

sgpy = np.empty((glb.totalgrids,glb.N))
sgporopy = np.empty((glb.totalgrids,glb.N))

print('  Reshaping data ')
for i in range(glb.N):
    sgsimuse = np.reshape(sgsim[:,i],(glb.Nx,glb.Ny,2*glb.Nz),'F')
    sgs = sgsimuse[:,:,2:7]             # [2:7] means 2 - 6 *1 number before the end
    ex = np.reshape(sgs,(glb.totalgrids,1),'F')
    sg[:,i:i+1] = ex
        
    sgsimporouse = np.reshape(sgsimporo[:,i],(glb.Nx,glb.Ny,2*glb.Nz),'F')
    sgsporo = sgsimporouse[:,:,2:7]
    exporo = np.reshape(sgsporo,(glb.totalgrids,1),'F')
    sgporo[:,i:i+1] = exporo

    sgsimusemat = np.reshape(sgsimfinalmat[:,i],(glb.Nx,glb.Ny,2*glb.Nz),'F')
    sgsmat = sgsimusemat[:,:,2:7]             # [2:7] means 2 - 6 *1 number before the end
    exmat = np.reshape(sgsmat,(glb.totalgrids,1),'F')
    sgmat[:,i:i+1] = exmat
        
    sgsimporousemat = np.reshape(sgsimporofinalmat[:,i],(glb.Nx,glb.Ny,2*glb.Nz),'F')
    sgsporomat = sgsimporousemat[:,:,2:7]
    exporomat = np.reshape(sgsporomat,(glb.totalgrids,1),'F')
    sgporomat[:,i:i+1] = exporomat

    sgsimusepy = np.reshape(sgsimfinalpy[:,i],(glb.Nx,glb.Ny,2*glb.Nz),'F')
    sgspy = sgsimusepy[:,:,2:7]             # [2:7] means 2 - 6 *1 number before the end
    expy = np.reshape(sgspy,(glb.totalgrids,1),'F')
    sgpy[:,i:i+1] = expy
        
    sgsimporousepy = np.reshape(sgsimporofinalpy[:,i],(glb.Nx,glb.Ny,2*glb.Nz),'F')
    sgsporopy = sgsimporousepy[:,:,2:7]
    exporopy = np.reshape(sgsporopy,(glb.totalgrids,1),'F')
    sgporopy[:,i:i+1] = exporopy

perm = sg
poro = sgporo

permat = sgmat
poromat = sgporomat

permpy = sgpy
poropy = sgporopy


os.chdir('Results')
resultfold = os.getcwd()

for i in range (glb.N):                  
    folder = 'MASTER %03d'%(i+1)
    if os.path.isdir(folder):           # value of os.path.isdir(directory) = True
        shutil.rmtree(folder)           # remove folder      
    os.mkdir(folder)

for i in range (glb.N):
    folder = 'MASTER %03d'%(i+1)
    os.chdir(folder)
    # Saturation data lines [2939-5206]
   
    A1 = np.reshape(perm[:,i],(glb.Nx,glb.Ny,glb.Nz),'F')
    A2 = np.reshape(poro[:,i],(glb.Nx,glb.Ny,glb.Nz),'F')

    B1 = np.reshape(permat[:,i],(glb.Nx,glb.Ny,glb.Nz),'F')
    B2 = np.reshape(poromat[:,i],(glb.Nx,glb.Ny,glb.Nz),'F')

    C1 = np.reshape(permpy[:,i],(glb.Nx,glb.Ny,glb.Nz),'F')
    C2 = np.reshape(poropy[:,i],(glb.Nx,glb.Ny,glb.Nz),'F')

    A3 = A1
    A31 = np.log10(A3)
    A31 = np.reshape(A31,(glb.Nx,glb.Ny,glb.Nz),'F')

    B3 = B1
    B31 = np.log10(B3)
    B31 = np.reshape(B31,(glb.Nx,glb.Ny,glb.Nz),'F')

    C3 = C1
    C31 = np.log10(C3)
    C31 = np.reshape(C31,(glb.Nx,glb.Ny,glb.Nz),'F')
  
    A4 = A2   
    A4 = np.reshape(A4,(glb.Nx,glb.Ny,glb.Nz),'F')

    B4 = B2   
    B4 = np.reshape(B4,(glb.Nx,glb.Ny,glb.Nz),'F')

    C4 = C2   
    C4 = np.reshape(C4,(glb.Nx,glb.Ny,glb.Nz),'F')
    
    XX, YY = np.meshgrid(np.arange(glb.Nx),np.arange(glb.Ny))

    ## Plotting Permeability
    ##------------------------------------------------------------------------
    fig1 = plt.figure(figsize =(13,18))

    fig1.add_subplot(5,layers,1)
    plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,2],cmap = cm)
    plt.title('True Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)


    fig1.add_subplot(5,layers,5)
    plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,3],cmap = cm)
    plt.title('True Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar2 = plt.colorbar()
    cbar2.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)
    

    fig1.add_subplot(5,layers,9)
    plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,4],cmap = cm)
    plt.title('True Layer 3', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar3 = plt.colorbar()
    cbar3.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)


    fig1.add_subplot(5,layers,13)
    plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,5],cmap = cm)
    plt.title('True Layer 4', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar4 = plt.colorbar()
    cbar4.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)
    
    fig1.add_subplot(5,layers,17)
    plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,6],cmap = cm)
    plt.title('True Layer 5', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar5 = plt.colorbar()
    cbar5.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,2)
    plt.pcolormesh(XX.T,YY.T,A31[:,:,0],cmap = cm)
    plt.title('Initial Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar6 = plt.colorbar()
    cbar6.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,6)
    plt.pcolormesh(XX.T,YY.T,A31[:,:,1],cmap = cm)
    plt.title('Initial Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar7 = plt.colorbar()
    cbar7.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,10)
    plt.pcolormesh(XX.T,YY.T,A31[:,:,2],cmap = cm)
    plt.title('Initial Layer 3', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar8 = plt.colorbar()
    cbar8.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,14)
    plt.pcolormesh(XX.T,YY.T,A31[:,:,3],cmap = cm)
    plt.title('Initial Layer 4', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar9 = plt.colorbar()
    cbar9.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,18)
    plt.pcolormesh(XX.T,YY.T,A31[:,:,4],cmap = cm)
    plt.title('Initial Layer 5', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar10 = plt.colorbar()
    cbar10.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,3)
    plt.pcolormesh(XX.T,YY.T,B31[:,:,0],cmap = cm)
    plt.title('MATLAB Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar11 = plt.colorbar()
    cbar11.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,7)
    plt.pcolormesh(XX.T,YY.T,B31[:,:,1],cmap = cm)
    plt.title('MATLAB Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar12 = plt.colorbar()
    cbar12.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,11)
    plt.pcolormesh(XX.T,YY.T,B31[:,:,2],cmap = cm)
    plt.title('MATLAB Layer 3', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar13 = plt.colorbar()
    cbar13.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,15)
    plt.pcolormesh(XX.T,YY.T,B31[:,:,3],cmap = cm)
    plt.title('MATLAB Layer 4', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar14 = plt.colorbar()
    cbar14.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,19)
    plt.pcolormesh(XX.T,YY.T,B31[:,:,4],cmap = cm)
    plt.title('MATLAB Layer 5', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar15 = plt.colorbar()
    cbar15.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)


    fig1.add_subplot(5,layers,4)
    plt.pcolormesh(XX.T,YY.T,C31[:,:,0],cmap = cm)
    plt.title('Python Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar11 = plt.colorbar()
    cbar11.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,8)
    plt.pcolormesh(XX.T,YY.T,C31[:,:,1],cmap = cm)
    plt.title('Python Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar12 = plt.colorbar()
    cbar12.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,12)
    plt.pcolormesh(XX.T,YY.T,C31[:,:,2],cmap = cm)
    plt.title('Python Layer 3', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar13 = plt.colorbar()
    cbar13.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,16)
    plt.pcolormesh(XX.T,YY.T,C31[:,:,3],cmap = cm)
    plt.title('Python Layer 4', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar14 = plt.colorbar()
    cbar14.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)

    fig1.add_subplot(5,layers,20)
    plt.pcolormesh(XX.T,YY.T,C31[:,:,4],cmap = cm)
    plt.title('Python Layer 5', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar15 = plt.colorbar()
    cbar15.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(1,5)


    plt.suptitle('Permeability Layers',fontsize = 25)
    plt.tight_layout(rect =[0,0,1,0.95])
    plt.savefig('figureperm1.png')
    #plt.savefig('figureperm1.eps')
    #plt.show()
    plt.clf()
    plt.close()

    ## Plotting Porosity
    ##------------------------------------------------------------------------
    fig2 = plt.figure(figsize =(13,18))

    fig2.add_subplot(5,layers,1)
    plt.pcolormesh(XX.T,YY.T,Trueporo[:,:,2],cmap = cm)
    plt.title('True Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
##    cbar1.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)


    fig2.add_subplot(5,layers,5)
    plt.pcolormesh(XX.T,YY.T,Trueporo[:,:,3],cmap = cm)
    plt.title('True Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar2 = plt.colorbar()
##    cbar2.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)
    

    fig2.add_subplot(5,layers,9)
    plt.pcolormesh(XX.T,YY.T,Trueporo[:,:,4],cmap = cm)
    plt.title('True Layer 3', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar3 = plt.colorbar()
##    cbar3.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)


    fig2.add_subplot(5,layers,13)
    plt.pcolormesh(XX.T,YY.T,Trueporo[:,:,5],cmap = cm)
    plt.title('True Layer 4', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar4 = plt.colorbar()
##    cbar4.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)
    
    fig2.add_subplot(5,layers,17)
    plt.pcolormesh(XX.T,YY.T,Trueporo[:,:,6],cmap = cm)
    plt.title('True Layer 5', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar5 = plt.colorbar()
##    cbar5.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,2)
    plt.pcolormesh(XX.T,YY.T,A4[:,:,0],cmap = cm)
    plt.title('Initial Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar6 = plt.colorbar()
##    cbar6.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,6)
    plt.pcolormesh(XX.T,YY.T,A4[:,:,1],cmap = cm)
    plt.title('Initial Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar7 = plt.colorbar()
##    cbar7.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,10)
    plt.pcolormesh(XX.T,YY.T,A4[:,:,2],cmap = cm)
    plt.title('Initial Layer 3', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar8 = plt.colorbar()
##    cbar8.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,14)
    plt.pcolormesh(XX.T,YY.T,A4[:,:,3],cmap = cm)
    plt.title('Initial Layer 4', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar9 = plt.colorbar()
##    cbar9.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,18)
    plt.pcolormesh(XX.T,YY.T,A4[:,:,4],cmap = cm)
    plt.title('Initial Layer 5', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar10 = plt.colorbar()
##    cbar10.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,3)
    plt.pcolormesh(XX.T,YY.T,B4[:,:,0],cmap = cm)
    plt.title('MATLAB Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar11 = plt.colorbar()
##    cbar11.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,7)
    plt.pcolormesh(XX.T,YY.T,B4[:,:,1],cmap = cm)
    plt.title('MATLAB Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar12 = plt.colorbar()
##    cbar12.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,11)
    plt.pcolormesh(XX.T,YY.T,B4[:,:,2],cmap = cm)
    plt.title('MATLAB Layer 3', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar13 = plt.colorbar()
##    cbar13.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,15)
    plt.pcolormesh(XX.T,YY.T,B4[:,:,3],cmap = cm)
    plt.title('MATLAB Layer 4', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar14 = plt.colorbar()
##    cbar14.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,19)
    plt.pcolormesh(XX.T,YY.T,B4[:,:,4],cmap = cm)
    plt.title('MATLAB Layer 5', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar15 = plt.colorbar()
##    cbar15.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)


    fig2.add_subplot(5,layers,4)
    plt.pcolormesh(XX.T,YY.T,C4[:,:,0],cmap = cm)
    plt.title('Python Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar11 = plt.colorbar()
##    cbar11.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,8)
    plt.pcolormesh(XX.T,YY.T,C4[:,:,1],cmap = cm)
    plt.title('Python Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar12 = plt.colorbar()
##    cbar12.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,12)
    plt.pcolormesh(XX.T,YY.T,C4[:,:,2],cmap = cm)
    plt.title('Python Layer 3', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar13 = plt.colorbar()
##    cbar13.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,16)
    plt.pcolormesh(XX.T,YY.T,C4[:,:,3],cmap = cm)
    plt.title('Python Layer 4', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar14 = plt.colorbar()
##    cbar14.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)

    fig2.add_subplot(5,layers,20)
    plt.pcolormesh(XX.T,YY.T,C4[:,:,4],cmap = cm)
    plt.title('Python Layer 5', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar15 = plt.colorbar()
##    cbar15.ax.set_ylabel('Porosity',fontsize = 13)
    plt.clim(0.1,0.4)


    plt.suptitle('Porosity Layers',fontsize = 23)
    plt.tight_layout(rect =[0,0,1,0.95])
    plt.savefig('figureporo.png')
    #plt.savefig('figureperm1.eps')
    #plt.show()
    plt.clf()
    plt.close()

    print(' Folder %03d has been plotted '%(i+1))

    os.chdir(resultfold)

layers = 3

## Dissimilarity for Permeability Reconstruction
os.chdir(resultsfold) # returning to original directory
  
print('   Geting the dissimilarity for permeability reconstruction')
True1 = Trueperm[:,:,2:7]
## Get MATLAB and Python Perm
B1 = np.log10(permat)
C1 = np.log10(permpy)

Jmat = np.empty((glb.totalgrids,glb.N))
testmat = np.empty((glb.N,1))
Jpy = np.empty((glb.totalgrids,glb.N))
testpy = np.empty((glb.N,1))

for i in range(glb.N):
    Jmat[:,i] = B1[:,i] - np.reshape(True1,(True1.size),'F')
    testmat[i,:] = sum(abs(Jmat[:,i]))
    Jpy[:,i] = C1[:,i] - np.reshape(True1,(True1.size),'F')
    testpy[i,:] = sum(abs(Jpy[:,i]))

reali = np.arange(0,glb.N)
reali =np.reshape(reali,(reali.size,1),'F')

jj3mat = np.amin(testmat)
index3mat = testmat
bestnorm3mat = (index3mat == np.amin(index3mat)).ravel().nonzero()
bestnorm3mat = int(bestnorm3mat[0])
print('MATLAB - Model with best norm Realization for Log(K) reconstruction : %d'%(bestnorm3mat+1) + ' with value %4.4f'%jj3mat)

jj3py = np.amin(testpy)
index3py = testpy
bestnorm3py = (index3py == np.amin(index3py)).ravel().nonzero()
bestnorm3py = int(bestnorm3py[0])
print('Python - Model with best norm Realization for Log(K) reconstruction : %d'%(bestnorm3py+1) + ' with value %4.4f'%jj3py)


xlim = np.arange(0,glb.N)
xlim =np.reshape(xlim,(xlim.size,1))

PlogKmat = np.reshape(B1[:,bestnorm3mat],(glb.Nx,glb.Ny,glb.Nz),'F')
PlogKpy = np.reshape(C1[:,bestnorm3py],(glb.Nx,glb.Ny,glb.Nz),'F')
 
XX, YY = np.meshgrid(np.arange(glb.Nx),np.arange(glb.Ny))

## Plotting
## --------------------------------------------------------------------------
fig3 = plt.figure(figsize =(13,18))

fig3.add_subplot(5,layers,1)
plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,2],cmap = cm)
plt.title('True Layer 1', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar1 = plt.colorbar()
cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)


fig3.add_subplot(5,layers,4)
plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,3],cmap = cm)
plt.title('True Layer 2', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar2 = plt.colorbar()
cbar2.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)


fig3.add_subplot(5,layers,7)
plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,4],cmap = cm)
plt.title('True Layer 3', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar3 = plt.colorbar()
cbar3.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)


fig3.add_subplot(5,layers,10)
plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,5],cmap = cm)
plt.title('True Layer 4', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar4 = plt.colorbar()
cbar4.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig3.add_subplot(5,layers,13)
plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,6],cmap = cm)
plt.title('True Layer 5', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar5 = plt.colorbar()
cbar5.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)


fig3.add_subplot(5,layers,2)
plt.pcolormesh(XX.T,YY.T,PlogKmat[:,:,0],cmap = cm)
plt.title(' MATLAB Best Matched Layer 1 ', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar11 = plt.colorbar()
cbar11.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig3.add_subplot(5,layers,5)
plt.pcolormesh(XX.T,YY.T,PlogKmat[:,:,1],cmap = cm)
plt.title('MATLAB Best Matched Layer 2', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar12 = plt.colorbar()
cbar12.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig3.add_subplot(5,layers,8)
plt.pcolormesh(XX.T,YY.T,PlogKmat[:,:,2],cmap = cm)
plt.title('MATLAB Best Matched Layer 3', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar13 = plt.colorbar()
cbar13.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig3.add_subplot(5,layers,11)
plt.pcolormesh(XX.T,YY.T,PlogKmat[:,:,3],cmap = cm)
plt.title('MATLAB Layer 4', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar14 = plt.colorbar()
cbar14.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig3.add_subplot(5,layers,14)
plt.pcolormesh(XX.T,YY.T,PlogKmat[:,:,4],cmap = cm)
plt.title('MATLAB Best Matched Layer 5', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar15 = plt.colorbar()
cbar15.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)


fig3.add_subplot(5,layers,3)
plt.pcolormesh(XX.T,YY.T,PlogKpy[:,:,0],cmap = cm)
plt.title('Python Best Matched Layer 1', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar11 = plt.colorbar()
cbar11.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig3.add_subplot(5,layers,6)
plt.pcolormesh(XX.T,YY.T,PlogKpy[:,:,1],cmap = cm)
plt.title('Python Best Matched Layer 2', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar12 = plt.colorbar()
cbar12.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig3.add_subplot(5,layers,9)
plt.pcolormesh(XX.T,YY.T,PlogKpy[:,:,2],cmap = cm)
plt.title('Python Best Matched Layer 3', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar13 = plt.colorbar()
cbar13.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig3.add_subplot(5,layers,12)
plt.pcolormesh(XX.T,YY.T,PlogKpy[:,:,3],cmap = cm)
plt.title('Python Best Matched Layer 4', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar14 = plt.colorbar()
cbar14.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig3.add_subplot(5,layers,15)
plt.pcolormesh(XX.T,YY.T,PlogKpy[:,:,4],cmap = cm)
plt.title('Python Best Matched Layer 5', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar15 = plt.colorbar()
cbar15.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)


plt.suptitle('Best Log Layers',fontsize = 25)
plt.tight_layout(rect =[0,0,1,0.95])
plt.savefig('bestlog.png')
#plt.savefig('bestlog.eps')
#plt.show()
plt.clf()
plt.close()
## --------------------------------------------------------------------------


##Plotting mean
##--------------------------------------------------------------------------
print('  Plotting the mean of the MATLAB and Python ensemble')
## MATLAB
B1 = np.reshape(B1,(glb.totalgrids,glb.N),'F')
B1mean = np.mean(B1,axis = 1)
B1mean = np.reshape(B1mean,(glb.Nx,glb.Ny,glb.Nz), 'F')

## Python
C1 = np.reshape(C1,(glb.totalgrids,glb.N),'F')
C1mean = np.mean(C1,axis = 1)
C1mean = np.reshape(C1mean,(glb.Nx,glb.Ny,glb.Nz), 'F')


#Plotting
fig4 = plt.figure(figsize =(13,18))

fig4.add_subplot(5,layers,1)
plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,2],cmap = cm)
plt.title('True Layer 1', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar1 = plt.colorbar()
cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)


fig4.add_subplot(5,layers,4)
plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,3],cmap = cm)
plt.title('True Layer 2', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar2 = plt.colorbar()
cbar2.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)


fig4.add_subplot(5,layers,7)
plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,4],cmap = cm)
plt.title('True Layer 3', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar3 = plt.colorbar()
cbar3.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)


fig4.add_subplot(5,layers,10)
plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,5],cmap = cm)
plt.title('True Layer 4', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar4 = plt.colorbar()
cbar4.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig4.add_subplot(5,layers,13)
plt.pcolormesh(XX.T,YY.T,Trueperm[:,:,6],cmap = cm)
plt.title('True Layer 5', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar5 = plt.colorbar()
cbar5.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)


fig4.add_subplot(5,layers,2)
plt.pcolormesh(XX.T,YY.T,B1mean[:,:,0],cmap = cm)
plt.title(' MATLAB Layer 1 Mean', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar11 = plt.colorbar()
cbar11.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig4.add_subplot(5,layers,5)
plt.pcolormesh(XX.T,YY.T,B1mean[:,:,1],cmap = cm)
plt.title('MATLAB Layer 2 Mean', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar12 = plt.colorbar()
cbar12.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig4.add_subplot(5,layers,8)
plt.pcolormesh(XX.T,YY.T,B1mean[:,:,2],cmap = cm)
plt.title('MATLAB Layer 3 Mean', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar13 = plt.colorbar()
cbar13.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig4.add_subplot(5,layers,11)
plt.pcolormesh(XX.T,YY.T,B1mean[:,:,3],cmap = cm)
plt.title('MATLAB Layer 4 Mean', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar14 = plt.colorbar()
cbar14.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig4.add_subplot(5,layers,14)
plt.pcolormesh(XX.T,YY.T,B1mean[:,:,4],cmap = cm)
plt.title('MATLAB Layer 5 Mean', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar15 = plt.colorbar()
cbar15.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)


fig4.add_subplot(5,layers,3)
plt.pcolormesh(XX.T,YY.T,C1mean[:,:,0],cmap = cm)
plt.title('Python Layer 1 Mean', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar11 = plt.colorbar()
cbar11.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig4.add_subplot(5,layers,6)
plt.pcolormesh(XX.T,YY.T,C1mean[:,:,1],cmap = cm)
plt.title('Python Layer 2 Mean', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar12 = plt.colorbar()
cbar12.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig4.add_subplot(5,layers,9)
plt.pcolormesh(XX.T,YY.T,C1mean[:,:,2],cmap = cm)
plt.title('Python Layer 3 Mean', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar13 = plt.colorbar()
cbar13.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig4.add_subplot(5,layers,12)
plt.pcolormesh(XX.T,YY.T,C1mean[:,:,3],cmap = cm)
plt.title('Python Layer 4 Mean', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar14 = plt.colorbar()
cbar14.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)

fig4.add_subplot(5,layers,15)
plt.pcolormesh(XX.T,YY.T,C1mean[:,:,4],cmap = cm)
plt.title('Python Layer 5 Mean', fontsize = 15)
plt.ylabel('Y',fontsize = 13)
plt.xlabel('X',fontsize = 13)
plt.axis([0,(glb.Nx - 1),0,(glb.Ny-1)])
plt.gca().set_xticks([])
plt.gca().set_yticks([])
cbar15 = plt.colorbar()
cbar15.ax.set_ylabel('Log K (mD)',fontsize = 13)
plt.clim(1,5)



plt.tight_layout(rect = [0,0,1,0.95])
plt.suptitle('Mean Permeability Layers', fontsize = 25)
plt.savefig('figuremean.png')
#plt.savefig('figuremean1.eps')
plt.clf()
plt.close()

print(' The Python programme has been completed')
