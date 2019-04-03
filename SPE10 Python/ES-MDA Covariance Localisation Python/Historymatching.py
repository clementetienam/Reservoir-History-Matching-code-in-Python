##----------------------------------------------------------------------------------
## History matching using ES-MDA with Covariance Localisation
## Running Ensembles
## the code couples ECLIPSE reservoir simulator with PYTHON used to implement reservoir history matching of
## The SPE 10 Reservoir

## Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
## Supervisor:Dr Rossmary Villegas
## Co-Supervisor: Dr Masoud Babaei
## Co-Supervisor: Dr Oliver Dorn
##-----------------------------------------------------------------------------------
import datetime
import time
import os
import shutil
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import globalvariables as glb

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

##Creating Folders and Copying Simulation Datafiles
print(' Creating the folders and copying the simulation files for the forward problem ')

for j in range(glb.N):
    folder = 'MASTER %03d'%(j+1)
    if os.path.isdir(folder): # value of os.path.isdir(directory) = True
        shutil.rmtree(folder)      
    os.mkdir(folder)
    shutil.copy2('ACTNUM.DAT',folder)
    shutil.copy2('SPE10_PVTI.PVO',folder)
    shutil.copy2('SPE10_PVTI_RSVD.PVO',folder)
    shutil.copy2('SPE10_PVTI_WATER.PVO',folder)
    shutil.copy2('MASTER0.DATA',folder)

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

    # Saving POROVANCOUVER and KVANCOUVER

    for i in range(glb.N):
        folder = 'MASTER %03d'%(i+1)
        os.chdir(folder)
    
        PORO2 = poro[:,i]
        PERMY2 = perm[:,i]
    
        np.savetxt('PERMY2.GRDECL',PERMY2, fmt = '%1.7e')
        np.savetxt('PORO2.GRDECL',PORO2, fmt = '%1.7e')
    
        os.chdir(oldfolder) # returning to original cd

    # Inserting KEYWORDS PORO and PERMY 
    os.system('writefiles.py')

    # Running Simulations in Parallel
    print( ' Solving the forward problem' )
    os.chdir(oldfolder)
    os.system("runeclipse.py")


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

    # Plot for oil production rates
    from collections import OrderedDict  # To group all the plots with realisations together for legend display purposes
    #subplot(2,2,1)
    plt.plot(Time,WOPRA[:,0:glb.N],color = 'c', lw = '2', label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Q_o(Sm^{3}/day)')
    plt.ylim((0,25000))
    plt.title('Producer 1')

    plt.plot(Time,TO1, color = 'red', lw = '2', label ='True model' )
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('PRO1_OIL_assimi%d'%(iyobo+1))          # save as png
    #plt.savefig('PRO1_OIL_assimi%d.eps'%(iyobo+1))     # This is for matplotlib 2.1.2
    #plt.show()                                     # preventing the figures from showing
    plt.clf()                                       # clears the figure
                      
    #subplot(2,2,2)
    plt.plot(Time, WOPRB[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Q_o(Sm^{3}/day)')
    plt.ylim((0,25000))
    plt.title('Producer 2')

    plt.plot(Time,TO2, color = 'red', lw = '2', label ='True model' )
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('PRO2_OIL_assimi%d'%(iyobo+1))  
    #plt.savefig('PRO2_OIL_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()
  
    # subplot(2,2,3)
    plt.plot(Time, WOPRC[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Q_o(Sm^{3}/day)')
    plt.ylim((0,25000))
    plt.title('Producer 3')

    plt.plot(Time,TO3, color = 'red', lw = '2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--') 
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('PRO3_OIL_assimi%d'%(iyobo+1))  
    #plt.savefig('PRO3_OIL_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()

    # subplot(2,2,4)
    plt.plot(Time, WOPRD[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Q_o(Sm^{3}/day)')
    plt.ylim((0,25000))
    plt.title('Producer 4')

    plt.plot(Time,TO4, color = 'red', lw = '2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('PRO4_OIL_assimi%d'%(iyobo+1))  
    #plt.savefig('PRO4_OIL_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()
                      
    # Plot for water cut
    # subplot(2,2,1)
    plt.plot(Time, WCTA[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Water cut')
    plt.title('Producer 1')

    plt.plot(Time,TW1, color = 'red', lw = '2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('PRO1_WATER_assimi%d'%(iyobo+1)) 
    #plt.savefig('PRO1_WATER_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()

                      
    # subplot(2,2,2)
    plt.plot(Time, WCTB[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Water cut')
    plt.title('Producer 2')

    plt.plot(Time,TW2, color = 'red', lw = '2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('PRO2_WATER_assimi%d'%(iyobo+1)) 
    #plt.savefig('PRO2_WATER_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()

    # subplot(2,2,3)
    plt.plot(Time, WCTC[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Water cut')
    plt.title('Producer 3')

    plt.plot(Time,TW3, color = 'red', lw = '2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('PRO3_WATER_assimi%d'%(iyobo+1)) 
    #plt.savefig('PRO3_WATER_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()


    # subplot(2,2,4)
    plt.plot(Time, WCTD[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Water cut')
    plt.title('Producer 4')

    plt.plot(Time,TW4, color = 'red', lw = '2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('PRO4_WATER_assimi%d'%(iyobo+1)) 
    #plt.savefig('PRO4_WATER_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()


    # Plot for BHP
    # subplot(2,2,1)
    plt.plot(Time, BHPA[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('BHP(BARSA)')
    plt.title('Injector 1')

    plt.plot(Time,TP1, color = 'red',lw ='2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('inj1_BHP_assimi%d'%(iyobo+1))
    #plt.savefig('inj1_BHP_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()


    # subplot(2,2,2)
    plt.plot(Time, BHPB[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('BHP(BARSA)')
    plt.title('Injector 2')

    plt.plot(Time,TP2, color = 'red',lw ='2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('inj2_BHP_assimi%d'%(iyobo+1))
    #plt.savefig('inj2_BHP_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()


    # subplot(2,2,3)
    plt.plot(Time, BHPC[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('BHP(BARSA)')
    plt.title('Injector 3')

    plt.plot(Time,TP3, color = 'red',lw ='2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('inj3_BHP_assimi%d'%(iyobo+1))
    #plt.savefig('inj3_BHP_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()


    # subplot(2,2,4)
    plt.plot(Time, BHPD[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('BHP(BARSA)')
    plt.title('Injector 4')

    plt.plot(Time,TP4, color = 'red',lw ='2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('inj4_BHP_assimi%d'%(iyobo+1))
    #plt.savefig('inj4_BHP_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()


    # Plot for GOR
    # subplot(2,2,1)
    plt.plot(Time, GORA[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Gas oil ratio(Sm^{3}/Sm^{3})')
    plt.title('Gas oil ratio for producer 1')

    plt.plot(Time,TG1, color = 'red',lw ='2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('PRO1_GOR_assimi%d'%(iyobo+1))  
    #plt.savefig('PRO1_GOR_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()
                          

    # subplot(2,2,2)
    plt.plot(Time, GORB[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Gas oil ratio(Sm^{3}/Sm^{3})')
    plt.title('Gas oil ratio for producer 2')

    plt.plot(Time,TG2, color = 'red',lw ='2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('PRO2_GOR_assimi%d'%(iyobo+1))  
    #plt.savefig('PRO2_GOR_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()


    # subplot(2,2,3)
    plt.plot(Time, GORC[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Gas oil ratio(Sm^{3}/Sm^{3})')
    plt.title('Gas oil ratio for producer 3')

    plt.plot(Time,TG3, color = 'red',lw ='2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('PRO3_GOR_assimi%d'%(iyobo+1))  
    #plt.savefig('PRO3_GOR_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()


    # subplot(2,2,4)
    plt.plot(Time, GORD[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Gas oil ratio(Sm^{3}/Sm^{3})')
    plt.title('Gas oil ratio for producer 4')

    plt.plot(Time,TG4, color = 'red',lw ='2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('PRO4_GOR_assimi%d'%(iyobo+1))  
    #plt.savefig('PRO4_GOR_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()


    plt.plot(Time, FOEA[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
    plt.xlabel('Time (days)')
    plt.ylabel('Oil recovery ratio')
    plt.title('Field oil recovery ratio')

    plt.plot(Time,TFOE, color = 'red',lw ='2', label ='True model')
    plt.axvline(x = 2500, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))              
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.savefig('OilRecovery_assimi%d'%(iyobo+1))  
    #plt.savefig('OilRecovery_assimi%d.eps'%(iyobo+1))  
    #plt.show()
    plt.clf()
                 
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

    plt.bar(reali,plttotalerror, color = 'c')
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
  
    from main_ESMDAcovariance import main_ESMDA_covariance
    (mumyperm,mumyporo) = main_ESMDA_covariance(observation,overallsim,rossmary,rossmaryporo,perm,poro,alpha,c)


    perm = np.reshape(mumyperm,(2*glb.totalgrids,glb.N),'F')
    poro = np.reshape(mumyporo,(2*glb.totalgrids,glb.N),'F')
    print('Finished Iteration %d'%iyobo)


## Check to see whether the last history matching update is good or not
print(' This is the final reservoir simulator run ')
for i in range(glb.N):
    folder = 'MASTER %03d'%(i+1)
    os.chdir(folder)

    PORO2 = poro[:,i]
    PERMY2 = perm[:,i]

    np.savetxt('PERMY2.GRDECL',PERMY2, fmt = '%1.7e')
    np.savetxt('PORO2.GRDECL',PORO2, fmt = '%1.7e')

    os.chdir(oldfolder) # returning to original cd

## Inserting KEYWORDS PORO and PERMY 
os.system('writefiles.py')

os.chdir(oldfolder)

## Running Simulations in Parallel
print( ' Solving the forward problem' )
os.system("runeclipse.py")

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

# Plot for oil production rates
from collections import OrderedDict  # To group all the plots with realisations together for legend display purposes
#subplot(2,2,1)
plt.plot(Time,WOPRA[:,0:glb.N],color = 'c', lw = '2', label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Q_o(Sm^{3}/day)')
plt.ylim((0,25000))
plt.title('Producer 1')

plt.plot(Time,TO1, color = 'red', lw = '2', label ='True model' )
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('PRO1_OIL_final')          # save as png
#plt.savefig('PRO1_OIL__final.eps')     # This is for matplotlib 2.1.2
#plt.show()                                     # preventing the figures from showing
plt.clf()                                       # clears the figure
                  
#subplot(2,2,2)
plt.plot(Time, WOPRB[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Q_o(Sm^{3}/day)')
plt.ylim((0,25000))
plt.title('Producer 2')

plt.plot(Time,TO2, color = 'red', lw = '2', label ='True model' )
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('PRO2_OIL_final')  
#plt.savefig('PRO2_OIL__final.eps')  
#plt.show()
plt.clf()

# subplot(2,2,3)
plt.plot(Time, WOPRC[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Q_o(Sm^{3}/day)')
plt.ylim((0,25000))
plt.title('Producer 3')

plt.plot(Time,TO3, color = 'red', lw = '2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--') 
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('PRO3_OIL_final')  
#plt.savefig('PRO3_OIL__final.eps')  
#plt.show()
plt.clf()

# subplot(2,2,4)
plt.plot(Time, WOPRD[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Q_o(Sm^{3}/day)')
plt.ylim((0,25000))
plt.title('Producer 4')

plt.plot(Time,TO4, color = 'red', lw = '2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('PRO4_OIL_final')  
#plt.savefig('PRO4_OIL__final.eps')  
#plt.show()
plt.clf()
                  
# Plot for water cut
# subplot(2,2,1)
plt.plot(Time, WCTA[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Water cut')
plt.title('Producer 1')

plt.plot(Time,TW1, color = 'red', lw = '2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('PRO1_WATER_final') 
#plt.savefig('PRO1_WATER__final.eps')  
#plt.show()
plt.clf()

                  
# subplot(2,2,2)
plt.plot(Time, WCTB[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Water cut')
plt.title('Producer 2')

plt.plot(Time,TW2, color = 'red', lw = '2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('PRO2_WATER_final') 
#plt.savefig('PRO2_WATER__final.eps')  
#plt.show()
plt.clf()

# subplot(2,2,3)
plt.plot(Time, WCTC[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Water cut')
plt.title('Producer 3')

plt.plot(Time,TW3, color = 'red', lw = '2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('PRO3_WATER_final') 
#plt.savefig('PRO3_WATER__final.eps')  
#plt.show()
plt.clf()


# subplot(2,2,4)
plt.plot(Time, WCTD[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Water cut')
plt.title('Producer 4')

plt.plot(Time,TW4, color = 'red', lw = '2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('PRO4_WATER_final') 
#plt.savefig('PRO4_WATER__final.eps')  
#plt.show()
plt.clf()


# Plot for BHP
# subplot(2,2,1)
plt.plot(Time, BHPA[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('BHP(BARSA)')
plt.title('Injector 1')

plt.plot(Time,TP1, color = 'red',lw ='2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('inj1_BHP_final')
#plt.savefig('inj1_BHP__final.eps')  
#plt.show()
plt.clf()


# subplot(2,2,2)
plt.plot(Time, BHPB[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('BHP(BARSA)')
plt.title('Injector 2')

plt.plot(Time,TP2, color = 'red',lw ='2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('inj2_BHP_final')
#plt.savefig('inj2_BHP__final.eps')  
#plt.show()
plt.clf()


# subplot(2,2,3)
plt.plot(Time, BHPC[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('BHP(BARSA)')
plt.title('Injector 3')

plt.plot(Time,TP3, color = 'red',lw ='2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('inj3_BHP_final')
#plt.savefig('inj3_BHP__final.eps')  
#plt.show()
plt.clf()


# subplot(2,2,4)
plt.plot(Time, BHPD[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('BHP(BARSA)')
plt.title('Injector 4')

plt.plot(Time,TP4, color = 'red',lw ='2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('inj4_BHP_final')
#plt.savefig('inj4_BHP__final.eps')  
#plt.show()
plt.clf()


# Plot for GOR
# subplot(2,2,1)
plt.plot(Time, GORA[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Gas oil ratio(Sm^{3}/Sm^{3})')
plt.title('Gas oil ratio for producer 1')

plt.plot(Time,TG1, color = 'red',lw ='2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('PRO1_GOR_final')  
#plt.savefig('PRO1_GOR__final.eps')  
#plt.show()
plt.clf()
                      

# subplot(2,2,2)
plt.plot(Time, GORB[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Gas oil ratio(Sm^{3}/Sm^{3})')
plt.title('Gas oil ratio for producer 2')

plt.plot(Time,TG2, color = 'red',lw ='2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('PRO2_GOR_final')  
#plt.savefig('PRO2_GOR__final.eps')  
#plt.show()
plt.clf()


# subplot(2,2,3)
plt.plot(Time, GORC[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Gas oil ratio(Sm^{3}/Sm^{3})')
plt.title('Gas oil ratio for producer 3')

plt.plot(Time,TG3, color = 'red',lw ='2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('PRO3_GOR_final')  
#plt.savefig('PRO3_GOR__final.eps')  
#plt.show()
plt.clf()


# subplot(2,2,4)
plt.plot(Time, GORD[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Gas oil ratio(Sm^{3}/Sm^{3})')
plt.title('Gas oil ratio for producer 4')

plt.plot(Time,TG4, color = 'red',lw ='2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('PRO4_GOR_final')  
#plt.savefig('PRO4_GOR__final.eps')  
#plt.show()
plt.clf()


plt.plot(Time, FOEA[:,0:glb.N], color = 'c', lw = '2',label = 'Realisations')
plt.xlabel('Time (days)')
plt.ylabel('Oil recovery ratio')
plt.title('Field oil recovery ratio')

plt.plot(Time,TFOE, color = 'red',lw ='2', label ='True model')
plt.axvline(x = 2500, color = 'black', linestyle = '--')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels,handles))              
plt.legend(by_label.values(), by_label.keys())
plt.ylim(ymin = 0)
plt.xlim(xmin = 0)
plt.savefig('OilRecovery_final')  
#plt.savefig('OilRecovery__final.eps')  
#plt.show()
plt.clf()

             
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

plt.bar(reali,plttotalerror, color = 'c')
#plt.xticks(reali)
plt.xlabel('Realizations')
plt.ylabel('RMSE value')
plt.ylim(ymin = 0)
plt.title('Cost function for Realizations')

plt.scatter(reali,plttotalerror, color ='k')
plt.xlabel('Realizations')
plt.ylabel('RMSE value')
plt.xlim([1,(glb.N - 1)])
plt.savefig('RMS final')
#plt.savefig('RMS_final.eps')
#plt.show()
plt.clf()


print('  Programme almost executed  ')

decreasingnorm[:,iyobo:iyobo + 1] = jj
end = datetime.datetime.now()
timetaken = end - start
print(' Time taken : '+ str(timetaken))
print( '  Creating the output of permeability and porosity history matched model for the next run')
np.savetxt('sgsimfinalpy.out', np.reshape(mumyperm,(mumyperm.size,1),'F'), fmt = '%4.6f', newline = '\n')
np.savetxt('sgsimporofinalpy.out', np.reshape(mumyporo,(mumyporo.size,1),'F'), fmt = '%4.6f', newline = '\n')     
np.savetxt('genesisNorm.out', np.reshape(decreasingnorm,(decreasingnorm.size,1),'F'), fmt = '%4.6f', newline = '\n') 



response = input('Do you want to plot the permeability map ( Y/N ) ? ');

if response == 'Y':
    import plot3D
else:
    print('Pixel map not needed')

print('  The overall programme has been executed  ')
##end = time.process_time()
##timetaken = end - start
##print(" Normal running time : " + str(timetaken) )
##----------------------------------------------------------------------------------
## End part of preventing Windows from sleeping
if osSleep:
    osSleep.uninhibit()
##--------------------------------------------------------------------------------------

##------------------------------------------------------------------------------
##'History matching data assimilation technique using ES-MDA with Covariance Localisation for SPE10 Reservoir'  
##'PhD Student: Clement Etienam'
##'Supervisor: Dr Rossmary Villegas'
##'Co-supervisor: Dr Masoud Babaei'
##'Co-supervisor: Dr Oliver Dorn'
##------------------------------------------------------------------------------

import numpy as np
import globalvariables as glb


def main_ESMDA_covariance(observation,overallsim,rossmary,rossmaryporo,perm,poro,alpha,c):
    
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
        
        from ESMDA_localisation2 import ESMDALocalisation2
        (sgsim2,DupdateK) = ESMDALocalisation2 (sg,sgporo, f,Sim1,alpha,c)

        
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

    return( mumyperm,mumyporo )
 
# This file is created to allow all Python scripts in the History Matching
# programme to be able access the variables inputted by the user.

# Acts as global variables
## Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
## Supervisor:Dr Rossmary Villegas
## Co-Supervisor: Dr Masoud Babaei
## Co-Supervisor: Dr Oliver Dorn


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


# This is a separate script for multiprocessing of MASTER0.DATA files
## Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
## Supervisor:Dr Rossmary Villegas
## Co-Supervisor: Dr Masoud Babaei
## Co-Supervisor: Dr Oliver Dorn
import multiprocessing
from multiprocessing import Pool

def eclipse(i):
    import os
    oldfolder = os.getcwd()
    folder = 'MASTER %03d'%(i+1)
    os.chdir(folder)
    os.system("@eclrun eclipse MASTER0.DATA")
    print(" Folder MASTER %03d has been processed"%(i+1))
    os.chdir(oldfolder)
     

if __name__ == "__main__":
    number_of_realisations = range(100)
    p = Pool(multiprocessing.cpu_count())
    p.map(eclipse,number_of_realisations)
    

"""

from multiprocessing import Process

def runeclipse(i):
    import os
    import subprocess
    oldfolder = os.getcwd()
    folder = 'MASTER %03d'%(i+1)
    os.chdir(folder)   
    subprocess.run("@eclrun eclipse MASTER0.DATA")
    print(" Folder MASTER %03d has been processed"%(i+1))
    os.chdir(oldfolder) 

if __name__ == "__main__":
    number_of_realisations = range(100)
    p = Process(multiprocessing.cpu_count)
    p.map(runeclipse,number_of_realisations)
"""
# This is a separate script for multiprocessing
## Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
## Supervisor:Dr Rossmary Villegas
## Co-Supervisor: Dr Masoud Babaei
## Co-Supervisor: Dr Oliver Dorn
import multiprocessing
from multiprocessing import Pool

def writefiles(i):
    import os
    oldfolder = os.getcwd()
    folder = 'MASTER %03d'%(i+1)
    os.chdir(folder)     

    permy2f1 = open("PERMY2.GRDECL",'r')             # problem with loading intial GRDECL file
    contents_permy2f1 = permy2f1.readlines()
    permy2f1.close()

    permy2f12 = open("PERMY2.GRDECL", "w+")
    permy2f12.writelines("PERMY\n")
    permy2f12.writelines(contents_permy2f1)
    permy2f12.writelines("\n/")
    permy2f12.close()

    poro2f1 = open("PORO2.GRDECL",'r')             
    contents_poro2f1 = poro2f1.readlines()
    poro2f1.close()

    poro2f12 = open("PORO2.GRDECL", "w+")
    poro2f12.writelines("PORO\n")
    poro2f12.writelines(contents_poro2f1)
    poro2f12.writelines("\n/")
    poro2f12.close()

    os.chdir(oldfolder)

if __name__ == "__main__":
    number_of_realisations = range(100)
    p = Pool(multiprocessing.cpu_count())
    p.map(writefiles,number_of_realisations)
## Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
## Supervisor:Dr Rossmary Villegas
## Co-Supervisor: Dr Masoud Babaei
## Co-Supervisor: Dr Oliver Dorn
import numpy as np
import globalvariables as glb

def honour2(rossmary, rossmaryporo,sgsim2,DupdateK):
        
    print('  Loading the relevant files  ')
    # Honour the well permeability and porosity values

    print('  Reading true permeability field ')
    uniehonour = np.reshape(rossmary,(glb.Nx,glb.Ny,2*glb.Nz), 'F')
    unieporohonour = np.reshape(rossmaryporo,(glb.Nx,glb.Ny,2*glb.Nz), 'F')

    # Read true porosity well values

    aa = np.zeros((7))
    bb = np.zeros((7))
    cc = np.zeros((7))
    dd = np.zeros((7))
    ee = np.zeros((7))
    ff = np.zeros((7))
    gg = np.zeros((7))
    hh = np.zeros((7))

    aa1 = np.zeros((7))
    bb1 = np.zeros((7))
    cc1 = np.zeros((7))
    dd1 = np.zeros((7))
    ee1 = np.zeros((7))
    ff1 = np.zeros((7))
    gg1 = np.zeros((7))
    hh1 = np.zeros((7))
    
    # Read true porosity well values
    for j in range(2,7):
        aa[j] = uniehonour[13,24,j]
        bb[j] = uniehonour[37,38,j]
        cc[j] = uniehonour[95,22,j]
        dd[j] = uniehonour[66,40,j]
        ee[j] = uniehonour[29,54,j]
        ff[j] = uniehonour[57,17,j]
        gg[j] = uniehonour[89,5,j]
        hh[j] = uniehonour[100,38,j]

        aa1[j] = unieporohonour[13,24,j]
        bb1[j] = unieporohonour[37,38,j]
        cc1[j] = unieporohonour[95,22,j]
        dd1[j] = unieporohonour[66,40,j]
        ee1[j] = unieporohonour[29,54,j]
        ff1[j] = unieporohonour[57,17,j]
        gg1[j] = unieporohonour[89,5,j]
        hh1[j] = unieporohonour[100,38,j]

    # Read permeability ensemble after EnKF update
    A = np.reshape(DupdateK,(glb.totalgrids,glb.N),'F')          # thses 2 are basically doing the same thing
    C = np.reshape(sgsim2,(glb.totalgrids,glb.N),'F')

    # Start the conditioning for permeability
    print('   Starting the conditioning  ')

    output = np.zeros((glb.totalgrids,glb.N))
    outputporo = np.zeros((glb.totalgrids,glb.N))

    for j in range(glb.N):
        B = np.reshape(A[:,j],(glb.Nx,glb.Ny,glb.Nz),'F')
        D = np.reshape(C[:,j],(glb.Nx,glb.Ny,glb.Nz),'F')
    
        for jj in range(5):
            B[13,24,jj] = aa[jj]
            B[37,38,jj] = bb[jj]
            B[95,22,jj] = cc[jj]
            B[66,40,jj] = dd[jj]
            B[29,54,jj] = ee[jj]
            B[57,17,jj] = ff[jj]
            B[89,5,jj] = gg[jj]
            B[100,38,jj] = hh[jj]

            D[13,24,jj] = aa1[jj]
            D[37,38,jj] = bb1[jj]
            D[95,22,jj] = cc1[jj]
            D[66,40,jj] = dd1[jj]
            D[29,54,jj] = ee1[jj]
            D[57,17,jj] = ff1[jj]
            D[89,5,jj] = gg1[jj]
            D[100,38,jj] = hh1[jj]
        
        output[:,j:j+1] = np.reshape(B,(glb.totalgrids,1), 'F')
        outputporo[:,j:j+1] = np.reshape(D,(glb.totalgrids,1), 'F')
    
    output[output >= 20000] = 20000         # highest value in true permeability
    output[output <= 5.0475] = 5.0475

    outputporo[outputporo >= 0.4] = 0.4
    outputporo[outputporo <= 0.05] = 0.05

    return (output,outputporo)
##---------------------------------------------------------------------------
## Python Implementation of MATLAB's pinv
## Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
## Supervisor:Dr Rossmary Villegas
## Co-Supervisor: Dr Masoud Babaei
## Co-Supervisor: Dr Oliver Dorn
##---------------------------------------------------------------------------
## The U, V nomenclature for this programme is inverted from that in MATLAB
## U in MATLAB is V in this code, V in MATLAB is U in this code
 

##Modules Used
import numpy as np

def pinvmat(A,tol = 0):
    V,S1,U = np.linalg.svd(A,full_matrices=0)

    # Calculate the default value for tolerance if no tolerance is specified
    if tol == 0:
        tol = np.amax((A.size)*np.spacing(np.linalg.norm(S1,np.inf)))  
    
    r1 = sum(S1 > tol)+1
    v = V[:,:r1-1]
    U1 = U.T
    u = U1[:,:r1-1]
    S11 = S1[:r1-1]
    s = S11[:]
    S = 1/s[:]
    X = (u*S.T).dot(v.T)

    return (V,X,U)


"""

np.amax((A.size)*np.spacing(np.linalg.norm(S1,np.inf)))

A = np.array([[5,2,400],[2,5,297],[3,9,345]])

tol = np.amax((A.size)*np.spacing(np.linalg.norm(S1,np.inf)))


#A = A.T
tol = 2
V,S1,U = np.linalg.svd(A,full_matrices=0)
r1 = sum(S1 > tol)+1
v = V[:,:r1-1]

U1 = U.T
u = U1[:,:r1-1]
S11 = S1[:r1-1]
s = S11[:]
S = 1/s[:]
X = (u*S.T).dot(v.T)
"""
## Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
## Supervisor:Dr Rossmary Villegas
## Co-Supervisor: Dr Masoud Babaei
## Co-Supervisor: Dr Oliver Dorn
import numpy as np
import scipy.linalg as splg


a = np.array([[3,2,1],[6,5,4],[9,8,7]])
alpha = 6

Ri = np.array(splg.lapack.dpotrf(alpha*a))
Rii = Ri[0]
Rii = np.reshape(Rii,(Rii.size,1),'F')
for i in range(Rii.size):
    if Rii[i] != 0:
        Rii[i] = Rii[i]**-1
Ri = np.reshape(Rii,(Ri[0].shape),'F')


Ctilde = ((Ri**-1).dot(Cdd.dot((Ri**(-1)).T))) + np.eye(m22)




