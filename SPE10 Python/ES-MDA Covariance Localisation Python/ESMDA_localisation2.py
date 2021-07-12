##---------------------------------------------------------------------------
##'History matching data assimilation technique using covariance localization with ESMDA for PUNQ Reservoir'  ) 
##'Supervisor: Dr Rossmary Villegas'
##'Co-supervisor: Dr Masoud Babei'
##'Co-supervisor: Dr Oliver Dorn'
##'PhD Student: Clement Etienam'
##'MEng Student: Yap Shan Wei
##------------------------------------------------------------------------------

import numpy as np
import scipy.ndimage.morphology as spndmo
import globalvariables as glb
from matpinv import pinvmat

def ESMDALocalisation2(sg,sgporo,f,Sim1,alpha,c):

    print('      Loading the files ')
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

    print( '      Calculate the Euclidean distance function to the 6 producer wells')
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
    
    print( '      Computing the Gaspari-Cohn coefficent')
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
      
    print('      Getting the Gaspari Cohn for Cyd') 
     
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

    return (sgsim2,DupdateK)
{"mode":"full","isActive":false}
