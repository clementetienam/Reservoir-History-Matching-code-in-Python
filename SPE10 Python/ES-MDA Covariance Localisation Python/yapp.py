##----------------------------------------------------------------------------------
## History matching using ES-MDA with Covariance Localisation
## Running Ensembles
## the code couples ECLIPSE reservoir simulator with PYTHON used to implement my ideas on history matching of
## The SPE 10 Reservoir

## Author: Clement Etienam ,PhD Petroleum Engineering 2015-2018
## Supervisor:Professor Kody Law
##-----------------------------------------------------------------------------------
import datetime
import time
import os
import shutil
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
##import globalvariables as glb

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

##alpha = int(input( ' Alpha value (4-8) : '))
##c = int(input( ' Constant value for covariance localisation(5) : '))
# alpha is the number of iteration and damping coefficient

cores = multiprocessing.cpu_count()
print(' ')
print(' This computer has %d cores, which will all be utilised in GP process '%cores)
##print(' The number of cores to be utilised can be changed in runeclipse.py and writefiles.py ')
print(' ')

start = datetime.datetime.now()
print(str(start))
a=np.zeros((5,5))
##print(a)
a[0:4,:]=6
##print(a)
d=np.ones((5,1))
print(d)
ee=(a).dot(d) ## Matrix multiplication
print(ee)
          
