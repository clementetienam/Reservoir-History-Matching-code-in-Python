# This is a separate script for multiprocessing of MASTER0.DATA files
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
