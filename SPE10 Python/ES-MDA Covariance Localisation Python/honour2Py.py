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
