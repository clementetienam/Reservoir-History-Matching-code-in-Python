##----------------------------------------------------------------------------------
## History matching using several methods with ESMDA with Covariance Localisation
## Running Ensembles
## the code couples ECLIPSE reservoir simulator with MATLAB used to implement my ideas on history matching of
## The SPE 10 Reservoir

## Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
## Supervisor:Dr Rossmary Villegas
## Co-Supervisor: Dr Masoud Babei
## Co-Supervisor: Dr Oliver Dorn
## MEng Student: Yap Shan Wei
##-----------------------------------------------------------------------------------
import datetime
import time
import os
import shutil
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import globalvariables as glb
import scipy.ndimage.morphology as spndmo
from matpinv import pinvmat

##----------------------------------------------------------------------------------
## This section is to prevent Windows from sleeping when executing the Python script
class WindowsInhibitor:
    '''Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx'''
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        pass

    def inhibit(self):
        import ctypes
        #Preventing Windows from going to sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS | \
            WindowsInhibitor.ES_SYSTEM_REQUIRED)

    def uninhibit(self):
        import ctypes
        #Allowing Windows to go to sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS)


osSleep = None
# in Windows, prevent the OS from sleeping while we run
if os.name == 'nt':
    osSleep = WindowsInhibitor()
    osSleep.inhibit()
##------------------------------------------------------------------------------------
## Start of Programme

print( 'ES-MDA with Covariance Localisation History Matching ')

oldfolder = os.getcwd()

alpha = int(input( ' Alpha value (4-8) : '))
c = int(input( ' Constant value for covariance localisation(5) : '))
# alpha is the number of iteration and damping coefficient

cores = multiprocessing.cpu_count()
print(' ')
print(' This computer has %d cores, which will all be utilised in running ECLIPSE '%cores)
print(' The number of cores to be utilised can be changed in runeclipse.py and writefiles.py ')
print(' ')

start = datetime.datetime.now()
print(str(start))

print('  Loading the true permeability and porosity ')
rossmary = open('rossmary.GRDECL')
rossmary = np.fromiter(rossmary, float)
rossmaryporo = open("rossmaryporo.GRDECL")
rossmaryporo = np.fromiter(rossmaryporo, float)
            
oldfolder = os.getcwd()
os.chdir(oldfolder)

# Import true production data
print( '  Importing true production data')
True1 = np.genfromtxt(glb.truedata, delimiter = '\t' ,skip_header = 7, skip_footer = 3*(glb.Nt + 6), usecols =(1,2,3,4,5,6,7,8,9,10))
True2 = np.genfromtxt(glb.truedata, delimiter = '\t',skip_header = (2*7 + glb.Nt), skip_footer = 2*(glb.Nt + 6), usecols =(1,2,3,4,5,6,7,8,9,10))
True3 = np.genfromtxt(glb.truedata, delimiter = '\t',skip_header = (3*7 + 2*glb.Nt), skip_footer = (glb.Nt + 6), usecols =(1,2,3,4,5,6,7,8,9,10))
True4 = np.genfromtxt(glb.truedata, delimiter = '\t',skip_header = (4*7 + 3*glb.Nt), skip_footer = 0, usecols =(1,2,3,4,5,6,7,8))

TO1 = True1[:,2]
TO2 = True1[:,3]
TO3 = True1[:,4]
TO4 = True1[:,5]
 
TW1 = True2[:,1]
TW2 = True2[:,2]
TW3 = True2[:,3]
TW4 = True2[:,4]
 
TP1 = True3[:,4]
TP2 = True3[:,5]
TP3 = True3[:,6]
TP4 = True3[:,7]
 
TG1 = True3[:,8]
TG2 = True3[:,9]
TG3 = True4[:,1]
TG4 = True4[:,2]
 
TFOE = True4[:,7]

 
print('  Creating the true observations ')
observation = np.zeros((glb.No,glb.Nt))

for i in range(glb.Nt):
    obs = np.zeros((glb.No,1))
    obs[0,:] = TO1[i]
    obs[1,:] = TO2[i]
    obs[2,:] = TO3[i]
    obs[3,:] = TO4[i]
    obs[4,:] = TW1[i]
    obs[5,:] = TW2[i]
    obs[6,:] = TW3[i]
    obs[7,:] = TW4[i]
    obs[8,:] = TP1[i]
    obs[9,:] = TP2[i]
    obs[10,:] = TP3[i]
    obs[11,:] = TP4[i]
    obs[12,:] = TG1[i]
    obs[13,:] = TG2[i]
    obs[14,:] = TG3[i]
    obs[15,:] = TG4[i]
    obs[16,:] = TFOE[i]

    observation[:,i:i+1] = obs

oldfolder = os.getcwd()

####Creating Folders and Copying Simulation Datafiles
##print(' Creating the folders and copying the simulation files for the forward problem ')
##
##for j in range(glb.N):
##    folder = 'MASTER %03d'%(j+1)
##    if os.path.isdir(folder): # value of os.path.isdir(directory) = True
##        shutil.rmtree(folder)      
##    os.mkdir(folder)
##    shutil.copy2('ACTNUM.DAT',folder)
##    shutil.copy2('SPE10_PVTI.PVO',folder)
##    shutil.copy2('SPE10_PVTI_RSVD.PVO',folder)
##    shutil.copy2('SPE10_PVTI_WATER.PVO',folder)
##    shutil.copy2('MASTER0.DATA',folder)

decreasingnorm = np.zeros((glb.N,alpha+1))

for iyobo in range(alpha):
    print('Now running the code for assimilating Iteration %d '%(iyobo+1))   
    # Loading Porosity and Permeability ensemble files
    print(' Permeability and porosity fields')

    if iyobo == 0:
        print(' Loading the permeability and porosity fields')
        sgsim = open("sgsim.out")
        sgsim = np.fromiter(sgsim,float)
        sgsimporo = open("sgsimporo.out")
        sgsimporo = np.fromiter(sgsimporo,float)

        perm = np.reshape(sgsim,(2*glb.totalgrids,glb.N),'F')
        poro = np.reshape(sgsimporo,(2*glb.totalgrids,glb.N),'F')

    else:
        perm = np.reshape(mumyperm,(2*glb.totalgrids,glb.N),'F')
        poro = np.reshape(mumyporo,(2*glb.totalgrids,glb.N),'F')

    os.chdir(oldfolder) # setting original directory

##    # Saving POROVANCOUVER and KVANCOUVER
##
##    for i in range(glb.N):
##        folder = 'MASTER %03d'%(i+1)
##        os.chdir(folder)
##    
##        PORO2 = poro[:,i]
##        PERMY2 = perm[:,i]
##    
##        np.savetxt('PERMY2.GRDECL',PERMY2, fmt = '%1.7e')
##        np.savetxt('PORO2.GRDECL',PORO2, fmt = '%1.7e')
##    
##        os.chdir(oldfolder) # returning to original cd
##
##    # Inserting KEYWORDS PORO and PERMY 
##    os.system('writefiles.py')
##
##    # Running Simulations in Parallel
##    print( ' Solving the forward problem' )
##    os.chdir(oldfolder)
##    os.system("runeclipse.py")
##
##
    print('  Plotting production profile')

    oldfolder = os.getcwd()
    os.chdir(oldfolder)

    print('  Starting the plotting  ')

    WOPRA = np.zeros((glb.Nt,glb.N))
    WOPRB = np.zeros((glb.Nt,glb.N))
    WOPRC = np.zeros((glb.Nt,glb.N))
    WOPRD = np.zeros((glb.Nt,glb.N))
    
    WCTA = np.zeros((glb.Nt,glb.N))
    WCTB = np.zeros((glb.Nt,glb.N))
    WCTC = np.zeros((glb.Nt,glb.N))
    WCTD = np.zeros((glb.Nt,glb.N))
    
    BHPA = np.zeros((glb.Nt,glb.N))
    BHPB = np.zeros((glb.Nt,glb.N))
    BHPC = np.zeros((glb.Nt,glb.N))
    BHPD = np.zeros((glb.Nt,glb.N))
	
    GORA = np.zeros((glb.Nt,glb.N))
    GORB = np.zeros((glb.Nt,glb.N))
    GORC = np.zeros((glb.Nt,glb.N))
    GORD = np.zeros((glb.Nt,glb.N))
	
    FOEA = np.zeros((glb.Nt,glb.N))

    for i in range(glb.N):
        folder = 'MASTER %03d'%(i+1)
        os.chdir(folder)     
      
        # Saturation data lines [2939-5206]
        A1 = np.genfromtxt('MASTER0.RSM', delimiter = '\t',skip_header = 7, skip_footer = 3*(glb.Nt + 6), usecols =(1,2,3,4,5,6,7,8,9,10))
        A2 = np.genfromtxt('MASTER0.RSM', delimiter = '\t',skip_header = (2*7 + glb.Nt), skip_footer = 2*(glb.Nt + 6), usecols =(1,2,3,4,5,6,7,8,9,10))
        A3 = np.genfromtxt('MASTER0.RSM', delimiter = '\t',skip_header = (3*7 + 2*glb.Nt), skip_footer = (glb.Nt + 6), usecols =(1,2,3,4,5,6,7,8,9,10))
        A4 = np.genfromtxt('MASTER0.RSM', delimiter = '\t',skip_header = (4*7 + 3*glb.Nt), skip_footer = 0, usecols =(1,2,3,4,5,6,7,8))
    
        WOPR1 = A1[:,2]
        WOPR2 = A1[:,3]
        WOPR3 = A1[:,4]
        WOPR4 = A1[:,5]
        Time = A1[:,0]
     
        WWCT1 = A2[:,1]
        WWCT2 = A2[:,2]
        WWCT3 = A2[:,3]
        WWCT4 = A2[:,4]
     
        BHP1 = A3[:,4]
        BHP2 = A3[:,5]
        BHP3 = A3[:,6]
        BHP4 = A3[:,7]
	 
        GORP1 = A3[:,8]
        GORP2 = A3[:,9]
        GORP3 = A4[:,1]
        GORP4 = A4[:,2]
		 
        FOE = A4[:,7]
	 
        #Saturation
    
        WOPRA[:,i] = WOPR1
        WOPRB[:,i] = WOPR2
        WOPRC[:,i] = WOPR3
        WOPRD[:,i] = WOPR4
    
        WCTA[:,i] = WWCT1
        WCTB[:,i] = WWCT2
        WCTC[:,i] = WWCT3
        WCTD[:,i] = WWCT4
    
        BHPA[:,i] = BHP1
        BHPB[:,i] = BHP2
        BHPC[:,i] = BHP3
        BHPD[:,i] = BHP4
	
        GORA[:,i] = GORP1
        GORB[:,i] = GORP2
        GORC[:,i] = GORP3
        GORD[:,i] = GORP4
	
        FOEA[:,i] = FOE
	
        os.chdir(oldfolder)

    os.chdir(oldfolder)

    # Import true production data
    # WHy are we importing here again???
    print( ' Importing true production data')
    True1 = np.genfromtxt(glb.truedata, delimiter = '\t' ,skip_header = 7, skip_footer = 3*(glb.Nt + 6), usecols =(1,2,3,4,5,6,7,8,9,10))
    True2 = np.genfromtxt(glb.truedata, delimiter = '\t',skip_header = (2*7 + glb.Nt), skip_footer = 2*(glb.Nt + 6), usecols =(1,2,3,4,5,6,7,8,9,10))
    True3 = np.genfromtxt(glb.truedata, delimiter = '\t',skip_header = (3*7 + 2*glb.Nt), skip_footer = (glb.Nt + 6), usecols =(1,2,3,4,5,6,7,8,9,10))
    True4 = np.genfromtxt(glb.truedata, delimiter = '\t',skip_header = (4*7 + 3*glb.Nt), skip_footer = 0, usecols =(1,2,3,4,5,6,7,8))

    TO1 = True1[:,2]
    TO2 = True1[:,3]
    TO3 = True1[:,4]
    TO4 = True1[:,5]
 
    TW1 = True2[:,1]
    TW2 = True2[:,2]
    TW3 = True2[:,3]
    TW4 = True2[:,4]
 
    TP1 = True3[:,4]
    TP2 = True3[:,5]
    TP3 = True3[:,6]
    TP4 = True3[:,7]
 
    TG1 = True3[:,8]
    TG2 = True3[:,9]
    TG3 = True4[:,1]
    TG4 = True4[:,2]
 
    TFOE = True4[:,7]

##    # Plot for oil production rates
##    from collections import OrderedDict  # To group all the plots with realisations together for legend display purposes
##    #subplot(2,2,1)
##    plt.plot(Time,WOPRA[:,0:glb.N],color = 'blue', lw = '2', label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Q_o(Sm^{3}/day)')
##    plt.ylim((0,25000))
##    plt.title('Producer 1')
##
##    plt.plot(Time,TO1, color = 'red', lw = '2', label ='True model' )
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
##    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('PRO1_OIL_assimi%d'%(iyobo+1))          # save as png
##    #plt.savefig('PRO1_OIL_assimi%d.eps'%(iyobo+1))     # This is for matplotlib 2.1.2
##    #plt.show()                                     # preventing the figures from showing
##    plt.clf()                                       # clears the figure
##                      
##    #subplot(2,2,2)
##    plt.plot(Time, WOPRB[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Q_o(Sm^{3}/day)')
##    plt.ylim((0,25000))
##    plt.title('Producer 2')
##
##    plt.plot(Time,TO2, color = 'red', lw = '2', label ='True model' )
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('PRO2_OIL_assimi%d'%(iyobo+1))  
##    #plt.savefig('PRO2_OIL_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##  
##    # subplot(2,2,3)
##    plt.plot(Time, WOPRC[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Q_o(Sm^{3}/day)')
##    plt.ylim((0,25000))
##    plt.title('Producer 3')
##
##    plt.plot(Time,TO3, color = 'red', lw = '2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--') 
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('PRO3_OIL_assimi%d'%(iyobo+1))  
##    #plt.savefig('PRO3_OIL_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##
##    # subplot(2,2,4)
##    plt.plot(Time, WOPRD[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Q_o(Sm^{3}/day)')
##    plt.ylim((0,25000))
##    plt.title('Producer 4')
##
##    plt.plot(Time,TO4, color = 'red', lw = '2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('PRO4_OIL_assimi%d'%(iyobo+1))  
##    #plt.savefig('PRO4_OIL_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##                      
##    # Plot for water cut
##    # subplot(2,2,1)
##    plt.plot(Time, WCTA[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Water cut')
##    plt.title('Producer 1')
##
##    plt.plot(Time,TW1, color = 'red', lw = '2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('PRO1_WATER_assimi%d'%(iyobo+1)) 
##    #plt.savefig('PRO1_WATER_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##
##                      
##    # subplot(2,2,2)
##    plt.plot(Time, WCTB[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Water cut')
##    plt.title('Producer 2')
##
##    plt.plot(Time,TW2, color = 'red', lw = '2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('PRO2_WATER_assimi%d'%(iyobo+1)) 
##    #plt.savefig('PRO2_WATER_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##
##    # subplot(2,2,3)
##    plt.plot(Time, WCTC[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Water cut')
##    plt.title('Producer 3')
##
##    plt.plot(Time,TW3, color = 'red', lw = '2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('PRO3_WATER_assimi%d'%(iyobo+1)) 
##    #plt.savefig('PRO3_WATER_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##
##
##    # subplot(2,2,4)
##    plt.plot(Time, WCTD[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Water cut')
##    plt.title('Producer 4')
##
##    plt.plot(Time,TW4, color = 'red', lw = '2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('PRO4_WATER_assimi%d'%(iyobo+1)) 
##    #plt.savefig('PRO4_WATER_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##
##
##    # Plot for BHP
##    # subplot(2,2,1)
##    plt.plot(Time, BHPA[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('BHP(BARSA)')
##    plt.title('Injector 1')
##
##    plt.plot(Time,TP1, color = 'red',lw ='2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('inj1_BHP_assimi%d'%(iyobo+1))
##    #plt.savefig('inj1_BHP_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##
##
##    # subplot(2,2,2)
##    plt.plot(Time, BHPB[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('BHP(BARSA)')
##    plt.title('Injector 2')
##
##    plt.plot(Time,TP2, color = 'red',lw ='2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('inj2_BHP_assimi%d'%(iyobo+1))
##    #plt.savefig('inj2_BHP_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##
##
##    # subplot(2,2,3)
##    plt.plot(Time, BHPC[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('BHP(BARSA)')
##    plt.title('Injector 3')
##
##    plt.plot(Time,TP3, color = 'red',lw ='2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('inj3_BHP_assimi%d'%(iyobo+1))
##    #plt.savefig('inj3_BHP_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##
##
##    # subplot(2,2,4)
##    plt.plot(Time, BHPD[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('BHP(BARSA)')
##    plt.title('Injector 4')
##
##    plt.plot(Time,TP4, color = 'red',lw ='2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('inj4_BHP_assimi%d'%(iyobo+1))
##    #plt.savefig('inj4_BHP_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##
##
##    # Plot for GOR
##    # subplot(2,2,1)
##    plt.plot(Time, GORA[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Gas oil ratio(Sm^{3}/Sm^{3})')
##    plt.title('Gas oil ratio for producer 1')
##
##    plt.plot(Time,TG1, color = 'red',lw ='2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('PRO1_GOR_assimi%d'%(iyobo+1))  
##    #plt.savefig('PRO1_GOR_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##                          
##
##    # subplot(2,2,2)
##    plt.plot(Time, GORB[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Gas oil ratio(Sm^{3}/Sm^{3})')
##    plt.title('Gas oil ratio for producer 2')
##
##    plt.plot(Time,TG2, color = 'red',lw ='2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('PRO2_GOR_assimi%d'%(iyobo+1))  
##    #plt.savefig('PRO2_GOR_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##
##
##    # subplot(2,2,3)
##    plt.plot(Time, GORC[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Gas oil ratio(Sm^{3}/Sm^{3})')
##    plt.title('Gas oil ratio for producer 3')
##
##    plt.plot(Time,TG3, color = 'red',lw ='2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('PRO3_GOR_assimi%d'%(iyobo+1))  
##    #plt.savefig('PRO3_GOR_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##
##
##    # subplot(2,2,4)
##    plt.plot(Time, GORD[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Gas oil ratio(Sm^{3}/Sm^{3})')
##    plt.title('Gas oil ratio for producer 4')
##
##    plt.plot(Time,TG4, color = 'red',lw ='2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('PRO4_GOR_assimi%d'%(iyobo+1))  
##    #plt.savefig('PRO4_GOR_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
##
##
##    plt.plot(Time, FOEA[:,0:glb.N], color = 'blue', lw = '2',label = 'Realisations')
##    plt.xlabel('Time (days)')
##    plt.ylabel('Oil recovery ratio')
##    plt.title('Field oil recovery ratio')
##
##    plt.plot(Time,TFOE, color = 'red',lw ='2', label ='True model')
##    plt.axvline(x = 2500, color = 'black', linestyle = '--')
##    handles, labels = plt.gca().get_legend_handles_labels()
##    by_label = OrderedDict(zip(labels,handles))              
##    plt.legend(by_label.values(), by_label.keys())
##    plt.ylim(ymin = 0)
##    plt.xlim(xmin = 0)
##    plt.savefig('OilRecovery_assimi%d'%(iyobo+1))  
##    #plt.savefig('OilRecovery_assimi%d.eps'%(iyobo+1))  
##    #plt.show()
##    plt.clf()
                 
    EWOP1 = np.zeros((glb.N,1))
    EWOP2 = np.zeros((glb.N,1))
    EWOP3 = np.zeros((glb.N,1))
    EWOP4 = np.zeros((glb.N,1))
    EWCT1 = np.zeros((glb.N,1))
    EWCT2 = np.zeros((glb.N,1))
    EWCT3 = np.zeros((glb.N,1))
    EWCT4 = np.zeros((glb.N,1))
    EBHP1 = np.zeros((glb.N,1))
    EBHP2 = np.zeros((glb.N,1))
    EBHP3 = np.zeros((glb.N,1))
    EBHP4 = np.zeros((glb.N,1))
    EGORP1 = np.zeros((glb.N,1))
    EGORP2 = np.zeros((glb.N,1))
    EGORP3 = np.zeros((glb.N,1))
    EGORP4 = np.zeros((glb.N,1))

    for i in range(glb.N):
        EWOP1[i:i+1,:] = np.mean((WOPRA[:,i:i+1] - np.reshape(TO1,(glb.Nt,1)))**2)
        EWOP2[i:i+1,:] = np.mean((WOPRB[:,i:i+1] - np.reshape(TO2,(glb.Nt,1)))**2)
        EWOP3[i:i+1,:] = np.mean((WOPRC[:,i:i+1] - np.reshape(TO3,(glb.Nt,1)))**2)
        EWOP4[i:i+1,:] = np.mean((WOPRD[:,i:i+1] - np.reshape(TO4,(glb.Nt,1)))**2)
        EWCT1[i:i+1,:] = np.mean((WCTA[:,i:i+1] - np.reshape(TW1,(glb.Nt,1)))**2)
        EWCT2[i:i+1,:] = np.mean((WCTB[:,i:i+1] - np.reshape(TW2,(glb.Nt,1)))**2)
        EWCT3[i:i+1,:] = np.mean((WCTC[:,i:i+1] - np.reshape(TW3,(glb.Nt,1)))**2)
        EWCT4[i:i+1,:] = np.mean((WCTD[:,i:i+1] - np.reshape(TW4,(glb.Nt,1)))**2)
        EBHP1[i:i+1,:] = np.mean((BHPA[:,i:i+1] - np.reshape(TP1,(glb.Nt,1)))**2)
        EBHP3[i:i+1,:] = np.mean((BHPC[:,i:i+1] - np.reshape(TP3,(glb.Nt,1)))**2)
        EBHP4[i:i+1,:] = np.mean((BHPD[:,i:i+1] - np.reshape(TP4,(glb.Nt,1)))**2)
        EGORP1[i:i+1,:] = np.mean((GORA[:,i:i+1] - np.reshape(TG1,(glb.Nt,1)))**2)
        EGORP2[i:i+1,:] = np.mean((GORB[:,i:i+1] - np.reshape(TG2,(glb.Nt,1)))**2)
        EGORP3[i:i+1,:] = np.mean((GORC[:,i:i+1] - np.reshape(TG3,(glb.Nt,1)))**2)
        EGORP4[i:i+1,:] = np.mean((GORD[:,i:i+1] - np.reshape(TG4,(glb.Nt,1)))**2)

    TOTALERROR = np.ones((glb.N,1))

    TOTALERROR = (EWOP1/np.std(TO1,ddof = 1))+(EWOP2/np.std(TO2,ddof = 1))+(EWOP3/np.std(TO3,ddof = 1))+ \
                 (EWOP4/np.std(TO4,ddof = 1))+(EWCT1/np.std(TW1,ddof = 1))+(EWCT2/np.std(TW2,ddof = 1))+ \
                 (EWCT3/np.std(TW3,ddof = 1))+(EWCT4/np.std(TW4,ddof = 1))+(EBHP1/np.std(TP1,ddof = 1))+ \
                 (EBHP2/np.std(TP2,ddof = 1))+(EBHP3/np.std(TP3,ddof = 1))+(EBHP4/np.std(TP4,ddof = 1))+ \
                 (EGORP1/np.std(TG1,ddof = 1))+(EGORP2/np.std(TG2,ddof = 1))+(EGORP3/np.std(TG3,ddof = 1))\
                 +(EGORP4/np.std(TG4,ddof = 1))

    TOTALERROR = TOTALERROR/glb.Nt
    jj = np.amin(TOTALERROR)        #minimum of flattened array
    bestnorm = np.argmin(TOTALERROR)        #Index of minimum value in array
    # In MATLAB realisation 1 stored in column index 1, in Python stored in column 0
    print('The best norm is number %i ' %(bestnorm + 1) +'with value %4.4f'%jj)

    reali = np.arange(1,glb.N+1)
    plttotalerror = np.reshape(TOTALERROR,(glb.N))

    plt.bar(reali,plttotalerror)
    #plt.xticks(reali)
    plt.xlabel('Realizations')
    plt.ylabel('RMSE value')
    plt.ylim(ymin = 0)
    plt.title('Cost function for Realizations')

    plt.scatter(reali,plttotalerror, color ='k')
    plt.xlabel('Realizations')
    plt.ylabel('RMSE value')
    plt.xlim([1,(glb.N - 1)])
    plt.savefig('RMS %d iteration'%iyobo)
    #plt.savefig('RMS%d iteration'%iyobo,format = 'eps')
    #plt.show()
    plt.clf()


    print('  Programme almost executed  ')

    decreasingnorm[:,iyobo:iyobo + 1] = jj
 
 
    print( 'Get the simulated files for all the time step')

    oldfolder = os.getcwd()
    os.chdir(oldfolder)

    overallsim = np.zeros((glb.No,glb.Nt,glb.N))
    for ii in range(glb.N):
        folder = 'MASTER %03d'%(ii+1)
        os.chdir(folder)
                      
        A1 = np.genfromtxt('MASTER0.RSM', delimiter = '\t',skip_header = 7, skip_footer = 3*(glb.Nt + 6), usecols =(1,2,3,4,5,6,7,8,9,10))
        A2 = np.genfromtxt('MASTER0.RSM', delimiter = '\t',skip_header = (2*7 + glb.Nt), skip_footer = 2*(glb.Nt + 6), usecols =(1,2,3,4,5,6,7,8,9,10))
        A3 = np.genfromtxt('MASTER0.RSM', delimiter = '\t',skip_header = (3*7 + 2*glb.Nt), skip_footer = (glb.Nt + 6), usecols =(1,2,3,4,5,6,7,8,9,10))
        A4 = np.genfromtxt('MASTER0.RSM', delimiter = '\t',skip_header = (4*7 + 3*glb.Nt), skip_footer = 0, usecols =(1,2,3,4,5,6,7,8))
    
        TO1 = A1[:,2]
        TO2 = A1[:,3]
        TO3 = A1[:,4]
        TO4 = A1[:,5]
     
        WWCT1 = A2[:,1]
        WWCT2 = A2[:,2]
        WWCT3 = A2[:,3]
        WWCT4 = A2[:,4]
     
        BHP1 = A3[:,4]
        BHP2 = A3[:,5]
        BHP3 = A3[:,6]
        BHP4 = A3[:,7]
	 
        GORP1 = A3[:,8]
        GORP2 = A3[:,9]
        GORP3 = A4[:,1]
        GORP4 = A4[:,2]
		 
        FOE = A4[:,7]

        observationsim = np.zeros((glb.No,glb.Nt))

        for i in range(glb.Nt):
            obs = np.zeros((glb.No,1))
            obs[0,:] = TO1[i]        
            obs[1,:] = TO2[i]
            obs[2,:] = TO3[i]
            obs[3,:] = TO4[i]
            obs[4,:] = WWCT1[i]
            obs[5,:] = WWCT2[i]
            obs[6,:] = WWCT3[i]
            obs[7,:] = WWCT4[i]
            obs[8,:] = BHP1[i]
            obs[9,:] = BHP2[i]
            obs[10,:] = BHP3[i]
            obs[11,:] = BHP4[i]
            obs[12,:] = GORP1[i]
            obs[13,:] = GORP2[i]
            obs[14,:] = GORP3[i]
            obs[15,:] = GORP4[i]
            obs[16,:] = FOE[i]
                
            observationsim[:,i:i+1] = obs

        overallsim[:,:,ii] = observationsim 
        os.chdir(oldfolder)

    os.chdir(oldfolder)
  
    sgsim = np.reshape(perm,(2*glb.totalgrids,glb.N), 'F')
    sgsimporo = np.reshape(poro,(2*glb.totalgrids,glb.N),'F')

    sg = np.zeros((glb.totalgrids,glb.N))
    sgporo = np.zeros((glb.totalgrids,glb.N))
    
    for i in range(glb.N):
        sgsimuse = np.reshape(sgsim[:,i],(glb.Nx,glb.Ny,2*glb.Nz),'F')
        sgs = sgsimuse[:,:,2:7]             # [2:7] means 2 - 6 *1 number before the end
        ex = np.reshape(sgs,(glb.totalgrids,1),'F')
        sg[:,i:i+1] = ex
            
        sgsimporouse = np.reshape(sgsimporo[:,i],(glb.Nx,glb.Ny,2*glb.Nz),'F')
        sgsporo = sgsimporouse[:,:,2:7]
        exporo = np.reshape(sgsporo,(glb.totalgrids,1),'F')
        sgporo[:,i:i+1] = exporo

    Sim11 = np.reshape(overallsim,(glb.No,glb.Nt,glb.N), 'F')
    permsteps = np.zeros((glb.totalgrids*glb.N,glb.Nt))
    porosteps = np.zeros((glb.totalgrids*glb.N,glb.Nt))

    # History matching using ESMDA
    for i in range(glb.Nt):
        print(' Now assimilating timestep %d '%(i+1))

        Sim1 = Sim11[:,i,:]
        Sim1 = np.reshape(Sim1,(glb.No,glb.N))

        f = observation[:,i]
        
        print('Loading the files ')
        ## Get the localization for all the wells

        A = np.zeros((120,60,5))
        for jj in range(5):
            A[13,24,jj] = 1
            A[37,38,jj] = 1
            A[95,22,jj] = 1
            A[66,40,jj] = 1
            A[29,54,jj] = 1
            A[57,17,jj] = 1
            A[89,5,jj] = 1
            A[100,38,jj] = 1

        print( ' Calculate the Euclidean distance function to the 6 producer wells')
        lf = np.reshape(A,(glb.Nx,glb.Ny,glb.Nz),'F')
        young = np.zeros((int(glb.totalgrids/glb.Nz),5))
        for j in range(5):
            sdf = lf[:,:,j]
            (usdf,IDX) = spndmo.distance_transform_edt(np.logical_not(sdf), return_indices = True)
            usdf = np.reshape(usdf,(int(glb.totalgrids/glb.Nz)),'F')
            young[:,j] = usdf

        sdfbig = np.reshape(young,(glb.totalgrids,1),'F')
        sdfbig1 = abs(sdfbig)
        z = sdfbig1
        ## the value of the range should be computed accurately.
          
        c0OIL1 = np.zeros((glb.totalgrids,1))
        
        print( 'Computing the Gaspari-Cohn coefficent')
        for i in range(glb.totalgrids):
            if ( 0 <= z[i,:] or z[i,:] <= c ):
                c0OIL1[i,:] = -0.25*(z[i,:]/c)**5 + 0.5*(z[i,:]/c)**4 + 0.625*(z[i,:]/c)**3 - (5.0/3.0)*(z[i,:]/c)**2 + 1

            elif ( z < 2*c ):
                c0OIL1[i,:] = (1.0/12.0)*(z[i,:]/c)**5 - 0.5*(z[i,:]/c)**4 + 0.625*(z[i,:]/c)**3 + (5.0/3.0)*(z[i,:]/c)**2 - 5*(z[i,:]/c) + 4 - (2.0/3.0)*(c/z[i,:])

            elif ( c <= z[i,:] or z[i,:] <= 2*c ):
                c0OIL1[i,:] = -5*(z[i,:]/c) + 4 -0.667*(c/z[i,:])

            else:
                c0OIL1[i,:] = 0
          
        c0OIL1[c0OIL1 < 0 ] = 0
          
        print(' Getting the Gaspari Cohn for Cyd') 
         
        schur = c0OIL1
        Bsch = np.tile(schur,(1,glb.N))
          
        yoboschur = np.ones((glb.Np*glb.totalgrids + glb.No,glb.N))
         
        yoboschur[0:glb.totalgrids,0:glb.N] = Bsch
        yoboschur[glb.totalgrids:2*glb.totalgrids,0:glb.N] = Bsch

        sgsim11 = np.reshape(np.log(sg),(glb.totalgrids,glb.N),'F')
        sgsim11poro = np.reshape(sgporo,(glb.totalgrids,glb.N),'F')
        
        print('        Determining standard deviation of the data ')
        stddWOPR1 = 0.15*f[0]
        stddWOPR2 = 0.15*f[1]
        stddWOPR3 = 0.15*f[2]
        stddWOPR4 = 0.15*f[3]

        stddWWCT1 = 0.2*f[4]
        stddWWCT2 = 0.2*f[5]
        stddWWCT3 = 0.2*f[6]
        stddWWCT4 = 0.2*f[7]
     
        stddBHP1 = 0.1*f[8]
        stddBHP2 = 0.1*f[9]
        stddBHP3 = 0.1*f[10]
        stddBHP4 = 0.1*f[11]
         
        stddGORP1 = 0.15*f[12]
        stddGORP2 = 0.15*f[13]
        stddGORP3 = 0.15*f[14]
        stddGORP4 = 0.15*f[15]

        print('        Generating Gaussian noise ')
        Error1 = np.ones((glb.No,glb.N))                  
        Error1[0,:] = np.random.normal(0,stddWOPR1,(glb.N))
        Error1[1,:] = np.random.normal(0,stddWOPR2,(glb.N))
        Error1[2,:] = np.random.normal(0,stddWOPR3,(glb.N))
        Error1[3,:] = np.random.normal(0,stddWOPR4,(glb.N))
        Error1[4,:] = np.random.normal(0,stddWWCT1,(glb.N))
        Error1[5,:] = np.random.normal(0,stddWWCT2,(glb.N))
        Error1[6,:] = np.random.normal(0,stddWWCT3,(glb.N))
        Error1[7,:] = np.random.normal(0,stddWWCT4,(glb.N))
        Error1[8,:] =  np.random.normal(0,stddBHP1,(glb.N))
        Error1[9,:] =  np.random.normal(0,stddBHP2,(glb.N))
        Error1[10,:] =  np.random.normal(0,stddBHP3,(glb.N))
        Error1[11,:] =  np.random.normal(0,stddBHP4,(glb.N))
        Error1[12,:] =  np.random.normal(0,stddGORP1,(glb.N))
        Error1[13,:] =  np.random.normal(0,stddGORP2,(glb.N))
        Error1[14,:] =  np.random.normal(0,stddGORP3,(glb.N))
        Error1[15,:] =  np.random.normal(0,stddGORP4,(glb.N))
        Error1[16,:] =  np.random.normal(0,0.062265,(glb.N))

        Cd2 = (Error1.dot(Error1.T))/(glb.N - 1)

        Dj = np.zeros((glb.No, glb.N))
        for j in range(glb.N):
            Dj[:,j] = f + Error1[:,j]

        print('      Generating the ensemble state matrix with parameters and states ')
        overall = np.zeros((glb.Np*glb.totalgrids + glb.No,glb.N))


        overall[0:glb.totalgrids,0:glb.N] = sgsim11
        overall[glb.totalgrids:2*glb.totalgrids,0:glb.N] = sgsim11poro
        overall[glb.Np*glb.totalgrids:glb.Np*glb.totalgrids + glb.No,0:glb.N] = Sim1

        Y = overall

        M = np.mean(Sim1, axis = 1)
        M2 = np.mean(overall, axis = 1)

        S = np.zeros((Sim1.shape[0],glb.N))
        yprime = np.zeros(((glb.Np)*glb.totalgrids + glb.No,glb.N))
               
        for j in range(glb.N):
            S[:,j] = Sim1[:,j]- M
            yprime[:,j] = overall[:,j] - M2

        print ('    Updating the new ensemble')
        Cyd = (yprime.dot(S.T))/(glb.N - 1)
        Cdd = (S.dot(S.T))/(glb.N - 1)


    ##    print ('    Rescaling the denominator matrix')
    ##    Ri = np.array(splg.lapack.dpotrf(alpha*Cd2))
    ##    Rii = Ri[0]
    ##    Rii = np.reshape(Rii,(Rii.size,1),'F')
    ##    for i in range(Rii.size):
    ##        if Rii[i] != 0:
    ##            Rii[i] = Rii[i]**-1
    ##
    ##    Ri = np.reshape(Rii,(Ri[0].shape),'F')
    ##    Ctilde = ((Ri**-1).dot(Cdd.dot((Ri**(-1)).T))) + np.ones(Sim1.shape)
    ##
    ##    
    ##    Usigt,Sigt,Vsigt = np.linalg.svd(Ctilde, full_matrices = False)
    ##    xsmall = np.diag(Sigt)
    ##    Bsigt = np.cumsum(xsmallt, axis = 0)          # vertically addition
    ##    valuesigt = Bsigt[-1]                 # last element
    ##    valuesigt = valuesigt * 0.9999
    ##    indicest = ( Bsigt >= valuesigt ).ravel().nonzero()
    ##    toluset = xsmallt[indicest]
    ##    tol = toluset[0]

        Usig,Sig,Vsig = np.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
        Bsig = np.cumsum(Sig, axis = 0)          # vertically addition
        valuesig = Bsig[-1]                 # last element
        valuesig = valuesig * 0.9999
        indices = ( Bsig >= valuesig ).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]



        print('  Update the new ensemble  ')
        (V,X,U) = pinvmat((Cdd + (alpha*Cd2)),tol)
        
        Ynew = Y + yoboschur*((Cyd.dot(X)).dot(Dj - Sim1))

        print('   Extracting the active permeability fields ')
        value1 = Ynew[0:glb.totalgrids,0:glb.N]

        DupdateK = np.exp(value1)

        sgsim2 = Ynew[glb.totalgrids:glb.totalgrids*2,0:glb.N]

        
        # Condition the data
        from honour2Py import honour2
        (output,outputporo) = honour2(rossmary, rossmaryporo,sgsim2,DupdateK)

        permsteps[:,i:i+1] = (np.reshape(output,(glb.totalgrids*glb.N,1)))
        porosteps[:,i:i+1] = (np.reshape(outputporo,(glb.totalgrids*glb.N,1)))

        # End of honour2

        sg = np.reshape(output,(glb.totalgrids,glb.N),'F')
        sgporo = np.reshape(outputporo,(glb.totalgrids,glb.N),'F')

        print('Finished assimilating timestep %d'%(i+1))

    sgassimi = sg
    sgporoassimi = sgporo

    permanswers = np.reshape(sgassimi,(glb.totalgrids,glb.N),'F')
    poroanswers = np.reshape(sgporoassimi,(glb.totalgrids,glb.N),'F')

    sgsimmijana = np.zeros((2*glb.totalgrids, glb.N))
    sgsimporomijana = np.zeros((2*glb.totalgrids, glb.N))

    for i in range(glb.N):
        sgsim = np.zeros((glb.Nx,glb.Ny,2*glb.Nz))
        sgsimporo = np.zeros((glb.Nx,glb.Ny,2*glb.Nz))
        sgsim[:,:,2:7] = np.reshape(permanswers[:,i],(glb.Nx,glb.Ny,glb.Nz), 'F')
        sgsimporo[:,:,2:7] = np.reshape(poroanswers[:,i],(glb.Nx,glb.Ny,glb.Nz), 'F')

        sgsimmijana[:,i:i+1] = np.reshape(sgsim,(2*glb.totalgrids,1),'F')
        sgsimporomijana[:,i:i+1] = np.reshape(sgsimporo,(2*glb.totalgrids,1),'F')


    mumyperm = sgsimmijana
    mumyporo = sgsimporomijana

    print('  Programme executed  ')



    perm = np.reshape(mumyperm,(2*glb.totalgrids,glb.N),'F')
    poro = np.reshape(mumyporo,(2*glb.totalgrids,glb.N),'F')
    print('Finished Iteration %d'%iyobo)


##end = time.process_time()
##timetaken = end - start
##print(" Normal running time : " + str(timetaken) )
##----------------------------------------------------------------------------------
## End part of preventing Windows from sleeping
if osSleep:
    osSleep.uninhibit()
##--------------------------------------------------------------------------------------
