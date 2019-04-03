##---------------------------------------------------------------------------
## Python Implementation of MATLAB's pinv
## MEng Student : Yap Shan Wei
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
