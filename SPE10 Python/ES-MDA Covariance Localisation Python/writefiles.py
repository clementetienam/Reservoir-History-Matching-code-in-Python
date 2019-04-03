# This is a separate script for multiprocessing
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
