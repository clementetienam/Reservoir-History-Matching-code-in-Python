##------------------------------------------------------------------------------
##'History matching data assimilation technique using ES-MDA with Covariance Localisation for SPE10 Reservoir'  
##'PhD Student: Clement Etienam'
##'Supervisor: Dr Rossmary Villegas'
##'Co-supervisor: Dr Masoud Babei'
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
 
