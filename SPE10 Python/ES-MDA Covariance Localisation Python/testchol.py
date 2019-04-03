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
