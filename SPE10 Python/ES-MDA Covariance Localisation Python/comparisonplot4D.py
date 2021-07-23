# -*- coding: utf-8 -*-
"""
Created on Sunday July 18 08:48:28 2021
@Company: NVIDIA
@author: Dr Clement Etienam
@ Line Manager: Dr Issam Said
@Customer-RidgewayKiteSoftware
@Description: Reservoir History macthing code with 6X Forward Simulator
@Model size: 40*40*3
@Extra: Model Errors (may or may not be) accounted for
@Parametrisation Methods:
    1)KSVD/OMP
    2)DCT
    3)DENOISING AUTOENCODER
    4)LEVEL SET
    5)AUTOENCODER
    6)SVD
    7)MoE/CCR
@Data Assimilation Methods:
    1)ESMDA - Ensemble Smoother with Multiple Data Asimilation
    2)IES -  Iterative Ensemble Smoother
    3)EnKF -  Ensemble Kalman Filter
    4)REKI - Regularised Ensemble Kalman Inversion
    5)MCMC - Markov Chain Monte Carlo
"""
from __future__ import print_function
print(__doc__)
print('.........................IMPORT SOME LIBRARIES.....................')
import os
import numpy as np
def is_available():

    """
        Check NVIDIA with nvidia-smi command
        Returning code 0 if no error, it means NVIDIA is installed
        Other codes mean not installed
    """
    code = os.system('nvidia-smi')
    return code 
Yet=is_available()
if Yet==0:
    print('GPU Available with CUDA')
    import cupy as cp
    from numba import cuda
else:
    print('No GPU Available')
    import numpy as cp
    
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
os.environ['KERAS_BACKEND'] = 'tensorflow'
import scipy.io as sio
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import shutil
from sklearn.preprocessing import MinMaxScaler
import pickle
import binary_parser
from sklearn.cluster import MiniBatchKMeans
import scipy
from sklearn.metrics import mean_squared_error
from glob import glob
import multiprocessing
from sklearn import linear_model
import mpslib as mps
import scipy.ndimage.morphology as spndmo
from keras.layers import Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
import numpy.matlib
from scipy.spatial.distance import cdist
from skimage.restoration import denoise_nl_means, estimate_sigma
from numpy.polynomial.hermite import hermval
from numpy.polynomial import Chebyshev as T
from numpy.polynomial import Legendre as L
import itertools
from sklearn.decomposition import dict_learning
from pyDOE import lhs
from kneed import KneeLocator
from skimage.segmentation import find_boundaries
from math import sqrt
from random import random
from matplotlib import rcParams
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica']
from collections import OrderedDict
from numpy.random import randn
import scipy.linalg as sla
from patlib.dict_tools import DotDict
import numpy;
from scipy.fftpack import dct, idct
import numpy.matlib
import datetime
import pandas as pd
import pyvista
from matplotlib import rcParams
from matplotlib import pyplot;
from matplotlib import cm;
from matplotlib.colors import LinearSegmentedColormap
from FyeldGenerator import generate_field
from imresize import *
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0" # I have just 1 GPU
from cpuinfo import get_cpu_info
print(cuda.detect())#Print the GPU information
# Prints a json string describing the cpu
s = get_cpu_info()
print("Cpu info")
for k,v in s.items():
    print(f"\t{k}: {v}")
    
cores = multiprocessing.cpu_count()
print(' ')
print(' This computer has %d cores, which will all be utilised in parallel '%cores)
print(' ')        
print('......................DEFINE SOME FUNCTIONS.....................')



class KSVD(object):
    def __init__(self, n_components, max_iter=30, tol=1e-6,
                 n_nonzero_coefs=None):
        """
        Sparse model Y = DX, Y is the sample matrix, use KSVD to dynamically \
update the dictionary matrix D and sparse matrix X
                 :param n_components: the number of atoms in the dictionary \
(the number of columns in the dictionary)
                 :param max_iter: maximum number of iterations
                 :param tol: sparse representation result tolerance
                 :param n_nonzero_coefs: sparsity
        """
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        """
                 Initialize the dictionary matrix
        """
        u, s, v = np.linalg.svd(y)
        self.dictionary = u[:, :self.n_components]

    def _update_dict(self, y, d, x):
        """
                 The process of using KSVD to update the dictionary
        """
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue

            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d, x

    def fit(self, y):
        """
                 KSVD iterative process
        """
        self._initialize(y)
        for i in range(self.max_iter):
            x = linear_model.orthogonal_mp(self.dictionary, y, \
                                n_nonzero_coefs=self.n_nonzero_coefs)
            e = np.linalg.norm(y - np.dot(self.dictionary, x))
            if e < self.tol:
                break
            self._update_dict(y, self.dictionary, x)

        self.sparsecode = linear_model.orthogonal_mp(self.dictionary, \
                                y, n_nonzero_coefs=self.n_nonzero_coefs)
        return self.dictionary, self.sparsecode

def norm(xx):
    # return nla.norm(xx/xx.size)
    return np.sqrt(np.mean(xx * xx))

def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)

    return Pk
# Draw samples from a normal distribution
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b

class RMS:
    """Compute RMS error & dev."""

    def __init__(self, truth, ensemble):
        mean = ensemble.mean(axis=0)
        err = truth - mean
        dev = ensemble - mean
        self.rmse = norm(err)
        self.rmsd = norm(dev)

    def __str__(self):
        return "%6.4f (rmse),  %6.4f (std)" % (self.rmse, self.rmsd)


def RMS_all(series, vs):
    """RMS for each item in series."""
    for k in series:
        if k != vs:
            print(f"{k:8}:", str(RMS(series[vs], series[k])))


def svd0(A):
    """Similar to Matlab's svd(A,0).

    Compute the

     - full    svd if nrows > ncols
     - reduced svd otherwise.

    As in Matlab: svd(A,0),
    except that the input and output are transposed

    .. seealso:: tsvd() for rank (and threshold) truncation.
    """
    M, N = A.shape
    if M > N:
        return sla.svd(A, full_matrices=True)
    return sla.svd(A, full_matrices=False)


def pad0(ss, N):
    """Pad ss with zeros so that len(ss)==N."""
    out = np.zeros(N)
    out[:len(ss)] = ss
    return out


def center(E, axis=0, rescale=False):
    """Center ensemble.

    Makes use of np features: keepdims and broadcasting.

    - rescale: Inflate to compensate for reduction in the expected variance.
    """
    x = np.mean(E, axis=axis, keepdims=True)
    X = E - x

    if rescale:
        N = E.shape[axis]
        X *= np.sqrt(N/(N-1))

    x = x.squeeze()

    return X, x


def mean0(E, axis=0, rescale=True):
    """Same as: center(E,rescale=True)[0]"""
    return center(E, axis=axis, rescale=rescale)[0]


def inflate_ens(E, factor):
    """Inflate the ensemble (center, inflate, re-combine)."""
    if factor == 1:
        return E
    X, x = center(E)
    return x + X*factor

numpy.random.seed(1)
        
def write_static_properties(Perm,Poro,i):
    itunu=i+1
    perm=Perm[:,i].T
    perm=np.reshape(perm,(-1,4))
    poro=Poro[:,i].T
    poro=np.reshape(poro,(-1,4))
    filename1='KVANCOUVER_' + str(itunu) +'.DAT'
    np.savetxt(filename1, perm, fmt='%.2f', delimiter=' \t', newline='\n',\
              header='PERMY',footer='/',comments='')
    
    filename2='POROVANCOUVER_' + str(itunu) +'.DAT'
    np.savetxt(filename2, poro, fmt='%.2f', delimiter=' \t', newline='\n',\
              header='PORO',footer='/',comments='') 


def run_model(model,inn,ouut,i,training_master,oldfolder):
    model.fit(inn, ouut )
    filename='Classifier_%d.bin'%i
    os.chdir(training_master)
    model.save_model(filename) 
    os.chdir(oldfolder)
    return model

def Prediction_CCR__Machine(nclusters,inputtest,numcols,deg):
    import numpy as np
    print('Starting Prediction')
    inputtest=np.reshape(inputtest,(-1,1),'F')
    filename1='Classifier.mat'
    mat = sio.loadmat(filename1)
    loaded_model=mat['theta']
    filenamex='clfx.asv'
    filenamey='clfy.asv'     
    clfx = pickle.load(open(filenamex, 'rb'))
    clfy = pickle.load(open(filenamey, 'rb'))  
    inputtest=(clfx.transform(inputtest))
    labelDA=Predict_classfication(inputtest,loaded_model,nclusters)        
    numrowstest=len(inputtest)
    clementanswer=np.zeros((numrowstest,1))
    labelDA=np.reshape(labelDA,(-1,1),'F')
    for i in range(nclusters):
        print('-- Predicting cluster: ' + str(i) + ' | ' + str(nclusters)) 
        filename2="Regressor_Machine_Cluster_" + str(i) +".mat"
        mat = sio.loadmat(filename2)
        model0=mat['model0']
        labelDA0=(np.asarray(np.where(labelDA == i))).T
#    ##----------------------##------------------------##
        a00=inputtest[labelDA0[:,0],:]
        a00=np.reshape(a00,(-1,numcols),'F')
        if a00.shape[0]!=0:
            clementanswer[labelDA0[:,0],:]=np.reshape\
                (predict_machine(a00,deg,model0),(-1,1))
            
        clementanswer=clfy.inverse_transform(clementanswer)
    return clementanswer

def KF_solver(A, y, sigma=1):
    X = A
    N, p = X.shape
    
    if len(y.shape) == 1: y = y[:,np.newaxis] # if we have scalar output
        
    Y_noisy = y# + sigma*np.random.randn(N, y.shape[1])

    L = 1e-3*np.eye(p)
    m = np.zeros((p, y.shape[1]))
    C = np.linalg.inv((L.T@L))
    for _, x in enumerate(X):
        x = x.reshape(-1,1)
        m_pred = m # prediction mean
        C_pred = C # prediction variance
        K = C_pred@x@np.linalg.inv(x.T@C_pred@x + sigma**2) # Kalman gain
        m = (np.eye(p) - K@x.T)@m_pred + K*Y_noisy[_] # update mean 
        C = (np.eye(p) - K@x.T)@C_pred # update variance
    return np.squeeze(m),C # Final result



def KF_solverbulk(A, y):
    #A=Bulk_X
    #y=Bulk_Y[:,0]
    X = A
    N, p = X.shape
    
    if len(y.shape) == 1: y = y[:,np.newaxis] # if we have scalar output
        
    Y_noisy = y# + sigma*np.random.randn(N, y.shape[1])
    sigma=np.eye(N)
    L = 1e-3*np.eye(p)
    m = np.zeros((p, y.shape[1]))
    C = np.linalg.inv((L.T@L))
    x=X
    x = x.T
    m_pred = m # prediction mean
    C_pred = C # prediction variance
    K = C_pred@x@np.linalg.inv(x.T@C_pred@x + sigma**2) # Kalman gain
    m =( (np.eye(p) - K@x.T)@m_pred) + K@Y_noisy # update mean 
    C = (np.eye(p) - K@x.T)@C_pred # update variance
    return np.squeeze(m),C # Final result

def fit_machine (a0, b0,deg):
    # deg=4
    # a0=X_test2
        dim = a0.shape[1]
        poly = poly_power
        p_pow = 1.0
        #total_deg = 3
        p = binom_sh(deg, dim)
        A = GenMat(p, a0, poly=poly, pow_p=p_pow)
        x_train=A
        theta,con1 = KF_solver(x_train, b0)
        #theta,con1 = KF_solverbulk(x_train, b0)
        return theta,con1
        
def predict_machine(a0,deg,model):
    dim = a0.shape[1]
    poly = poly_power
    p_pow = 1.0
    #total_deg = 3
    p = binom_sh(deg, dim)
    A = GenMat(p, a0, poly=poly, pow_p=p_pow)
    x_train=A
    theta=np.reshape(model[0,0],(-1,1))
    predicted= x_train@theta
    return predicted   

    

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Learn_Classification(Xin,y,nclusters):
    m=Xin.shape[0]
    X = np.hstack((np.ones((m,1)),Xin))
    Y = np.zeros((Xin.shape[0],nclusters))

    #no of classes
    k = np.unique(y)
    k.sort()
    
    #in Y matrix for each class columns put 1 wherin the rows belong to that class
    for cls in k.astype(int):
        Y[np.where(y[:,-1] == cls),cls] = 1
    #define theta with size 3,3 as there are 3 features and 3 models, lets take the initial value as 0
    theta = np.zeros((nclusters,X.shape[1]))
    
    #learning rate
    learning_rate = 0.001
    
    #no of iterations we want our model to be trained
    no_of_iterations = 1000
    
    #to visualise the cost function
    cost_arr = np.empty((0,nclusters))
    
    #counter
    i = 1    
    for i in range(no_of_iterations):
        
        #model/hypothesis function
        lineq = np.dot(X, theta.T)
        h = sigmoid(lineq)
        
        #cost function -1/m * sum (y log h + (1-y)log(1-h))for each class
        cost = -1/m * ((Y * np.log(h)) + ((1-Y) * np.log(1-h)))
        cost = cost.sum(axis = 0)
        cost_arr = np.vstack((cost_arr, cost))
        
        #applying gradient descent to calculate new theta = theta - (learning_rate/m * summation[(h-y)*x]) -> delta
        #summation can be replace by matrix multiplication
        delta = (learning_rate/m) * np.dot((h-Y).T, X)
        theta = theta - delta  
        
        i = i + 1;
    sio.savemat('classifier.mat', {'theta':theta})

def Predict_classfication(Xtest,theta,nclusters):
    #no of test samples
    test_m = Xtest.shape[0]
    test_X = np.hstack((np.ones((test_m,1)),Xtest))    
    #predict 
    pred = np.zeros((test_m,nclusters))
    model_predict = sigmoid(np.dot(test_X, theta.T))
    pred[model_predict > 0.5] = 1
    
    #converting the prediction matrix into vectors
    predict = np.zeros((test_m,1))
    for i in range(nclusters):
        if i==0:
            pass
        else:
            predict[pred[:,i] == 1] = i
            
    return predict

def Learn_CCR_Ensemble(stringfaa,oldfolder,jj,inpuutj,outputtj,degg):
    folder=stringfaa + str(jj)
    os.chdir(folder)
    CCR_Machine(inpuutj,outputtj,degg)
    os.chdir(oldfolder)
    

def Predict_CCR_Ensemble(stringfaa,oldfolder,jj,inpuutj,degg):
    folder=stringfaa + str(jj)
    os.chdir(folder)   
    clementanswer=Prediction_CCR__Machine(2,inpuutj,1,degg) 
    clementanswer=np.reshape(clementanswer,(-1,1),'F')
    os.chdir(oldfolder)
    return clementanswer

def Extract_Machine1(stringfaa,oldfolder,jj,nclusters):
    folder=stringfaa + str(jj)
    os.chdir(folder)
    Expermean=[]
    for i in range(nclusters):
        filename2="Regressor_Machine_Cluster_" + str(i) +".mat"
        mat = sio.loadmat(filename2)
        model0=mat['model0']
        meann=model0[0,0]
        meann=np.reshape(meann,(-1,1),'F')
        Expermean.append(meann)        
    Expermean=np.vstack(Expermean)

    os.chdir(oldfolder)
    return Expermean

def Extract_Machine2(stringfaa,oldfolder,jj,nclusters):
    folder=stringfaa + str(jj)
    os.chdir(folder)
    Expercovar=[]
    for i in range(nclusters):
        filename2="Regressor_Machine_Cluster_" + str(i) +".mat"
        mat = sio.loadmat(filename2)
        model0=mat['model0']
        covaa=model0[0,1]
        covaa=np.reshape(covaa,(-1,1),'F')
        Expercovar.append(covaa)
    Expercovar=np.vstack(Expercovar)
    os.chdir(oldfolder)
    return Expercovar

def Extract_Machine3(stringfaa,oldfolder,jj,nclusters):
    folder=stringfaa + str(jj)
    os.chdir(folder)        
    mat = sio.loadmat("classifier.mat")
    model0=mat['theta']    
    model0=np.reshape(model0,(-1,1),'F')
    os.chdir(oldfolder)
    return model0

def Extract_Machine4(stringfaa,oldfolder,jj,nclusters):
    folder=stringfaa + str(jj)
    os.chdir(folder)
    for i in range(nclusters):
        filename2="Regressor_Machine_Cluster_" + str(i) +".mat"
        mat = sio.loadmat(filename2)
        model0=mat['model0']
        meann=model0[0,0]
        shapemean=meann.shape
    os.chdir(oldfolder)
    return shapemean

def Extract_Machine5(stringfaa,oldfolder,jj,nclusters):
    folder=stringfaa + str(jj)
    os.chdir(folder)
    for i in range(nclusters):
        filename2="Regressor_Machine_Cluster_" + str(i) +".mat"
        mat = sio.loadmat(filename2)
        model0=mat['model0']
        covaa=model0[0,1]
        shapecova=covaa.shape
    os.chdir(oldfolder)
    return shapecova

def Extract_Machine6(stringfaa,oldfolder,jj,nclusters):
    folder=stringfaa + str(jj)
    os.chdir(folder)
        
    mat = sio.loadmat("classifier.mat")
    model0=mat['theta']
    shapeclass=model0.shape
    os.chdir(oldfolder)
    return shapeclass

def Split_Matrix (matrix, sizee):
    x_split=np.split(matrix, sizee, axis=0)
    return x_split  

def Insert_Machine(stringfaa,oldfolder,nclusters,Expermean,Expercovar,\
                   model0,shapemean,shapecova,shapeclass,jj):
    
    folder=stringfaa + str(jj)
    os.chdir(folder)
    theta=np.reshape(model0,(shapeclass),'F')
    sio.savemat('classifier.mat', {'theta':theta})
    tempmean=Split_Matrix (Expermean, nclusters)
    tempcova=Split_Matrix (Expercovar, nclusters)
    for i in range(nclusters):
        meann=tempmean[i]
        covarr=tempcova[i]
        
        meann=np.reshape(meann,(shapemean),'F')
        covarr=np.reshape(covarr,(shapecova),'F')
        
        model0=np.empty([1,2],dtype=object)
        model0[0,0]=meann
        model0[0,1]=covarr
        filename="Regressor_Machine_Cluster_" + str(i) +".mat"
        sio.savemat(filename, {'model0':model0})
    os.chdir(oldfolder)   
        
def CCR_Machine(inpuutj,outputtj,degg):
    
    inpuutj=np.reshape(inpuutj,(-1,1),'F')
    outputtj=np.reshape(outputtj,(-1,1),'F')
    
    X=inpuutj
    y=outputtj
    numruth = X.shape[1]   
    
    y_traind=y
    scaler1a = MinMaxScaler(feature_range=(0, 1))
    (scaler1a.fit(X))
    X=(scaler1a.transform(X))
    scaler2a = MinMaxScaler(feature_range=(0, 1))
    (scaler2a.fit(y))    
    y=(scaler2a.transform(y))
    yruth=y
    filenamex='clfx.asv'
    filenamey='clfy.asv'   
    pickle.dump(scaler1a, open(filenamex, 'wb'))
    pickle.dump(scaler2a, open(filenamey, 'wb'))   
    y_traind=numruth*10*y
    matrix=np.concatenate((X,y_traind), axis=1)
    k=getoptimumk(matrix)
    nclusters=k
    nclusters=2
    print ('Optimal k is: ', nclusters)
    kmeans =MiniBatchKMeans(n_clusters=nclusters,max_iter=2000).fit(matrix)
    dd=kmeans.labels_
    dd=dd.T
    dd=np.reshape(dd,(-1,1))
    #-------------------#---------------------------------#
    inputtrainclass=X
    outputtrainclass=np.reshape(dd,(-1,1))
    Learn_Classification(inputtrainclass,outputtrainclass,nclusters)
    #print('Split for classifier problem')
    X_train=X
    y_train=dd
    #-------------------Regression----------------#    
    for i in range(nclusters):
        print('-- Learning cluster: ' + str(i+1) + ' | ' + str(nclusters)) 
        label0=(np.asarray(np.where(y_train == i))).T
        model0=np.empty([1,2],dtype=object)
		
        a0=X_train[label0[:,0],:]
        a0=np.reshape(a0,(-1,numruth),'F')
        b0=yruth[label0[:,0],:]
        b0=np.reshape(b0,(-1,1),'F')
        if a0.shape[0]!=0 and b0.shape[0]!=0:
            #model0.fit(a0, b0,verbose=False)
            theta,con1=fit_machine (a0, b0,degg)
            model0[0,0]=theta
            model0[0,1]=con1
        filename="Regressor_Machine_Cluster_" + str(i) +".mat"
        sio.savemat(filename, {'model0':model0})
    return nclusters



def binom_sh(p,l):
    """
    Shifted binomial:
    (p+l\\p) = (p+l)!/p!*l!
    meaning number of monoms to approx. function, with l vars and poly. power <= p
    """
    return int(np.math.factorial(p+l)//(np.math.factorial(p)*np.math.factorial(l))  )

def OnesFixed(m, n):
    """
    m ones on n places
    """
    for i in itertools.combinations_with_replacement(range(n), m):
        uniq = np.unique(i)
        if len(uniq) == len(i):
            res = np.full(n, False)
            res[uniq] = True
            yield res

def indeces_K(l, q, p=1):
    """
    returns all vectors of length l with sum of indices in power \
    p <= q^p, starting form 0
    x^p + y^p <= q^p
    Elements can repeat!
    """
    qp = q**p
    m = int(qp) # max number of non-zero elements
    if m >= l:
        for cmb in itertools.product(range(q+1), repeat=l):
            if sum(np.array(cmb)**p) <= qp:
                yield cmb
    else:
        ones = list(OnesFixed(m, l))
        for cmb in itertools.product(range(q+1), repeat=m): # now m repeat
            if sum(np.array(cmb)**p) <= qp:
                for mask in ones:
                    res = np.zeros(l, dtype=int)
                    res[mask] = cmb
                    yield tuple(res)


def indeces_K_cut(l, maxn, p=1, q=1):
    """
    MAGIC FUNCTION
    q is determined automatically
    """
    while binom_sh(q, l) < maxn:
        q += 1    
    a = indeces_K(l, q, p)
    a = sorted(a, reverse=True)
    a = [el for el, _ in itertools.groupby(a)] # delete duplicates
    a = sorted(a, key=lambda e: max(e))
    a = sorted(a, key=lambda e: np.sum(np.array(e)**p))
    if len(a) < maxn:
        return indeces_K_cut(l, maxn, p, q+1)
    else:
        return a[:maxn]
    
def indeces_K_cut_new(l, p=1, q=1):
    """
    MAGIC FUNCTION
    q is determined automatically
    """
    maxn = binom_sh(q, l)
     
    a = indeces_K(l, q, p)
    a = sorted(a, reverse=True)
    a = [el for el, _ in itertools.groupby(a)] # delete duplicates
    a = sorted(a, key=lambda e: max(e))
    a = sorted(a, key=lambda e: np.sum(np.array(e)**p))
    if len(a) < maxn:
        return indeces_K_cut_new(l, p, q+1)
    else:
        return a[:maxn]


def num_of_indeces_K(l, q, max_p):
    a = indeces_K(l, q, max_p)
    a = [el for el, _ in itertools.groupby(a)] # delete duplicates
    return len(a)


# Some with polynomials
def herm_mult_many(x, xi, poly_func=None):
    """
    INPUT
    x - array of point where to calculate (np.array N x l)
    xi - array of powers of Hermite (or non-Hermite) poly (array of length l)
    
    OUTPUT
    [H_xi[0](x[0, 0])*H_xi[1](x[0, 1])*...,
     H_xi[0](x[1, 0])*H_xi[1](x[1, 1])*...,
                    ...
     H_xi[0](x[N-1, 0])*H_xi[1](x[N-1, 1])*...,]
    """
    N, l = x.shape
    assert(l == len(xi))

    if poly_func is None:
        poly_func = [herm] * l
    res = poly_func[0](x[:, 0], xi[0])
    for n in range(1, l):
        res *= poly_func[n](x[:, n], xi[n])

    return res
# Vandermone-like
def poly_power(x, n):
    return x**n

def poly_power_snorm(n):
    return 2.0/(2.0*n+1.0)

poly_power.snorm = poly_power_snorm

# Some orthogonal polynomials
def cheb(x, n):
    """
    returns T_n(x)
    value of not normalized Chebyshev polynomial
    $\int \frac1{\sqrt{1-x^2}}T_m(x)T_n(x) dx = \frac\pi2\delta_{nm}$
    """
    return T.basis(n)(x)


def cheb_snorm(n):
    return np.pi/2.0 if n != 0 else np.pi

cheb.snorm = cheb_snorm


def herm(x, n):
    """
    returns H_n(x)
    value of normalized Probabilistic polynomials
    $\int exp(-x^2/2)H_m(x)H_n(x) dx = \delta_{nm}$
    """
    cf = np.zeros(n+1)
    cf[n] = 1
    nc = np.sqrt(float(np.math.factorial(n))) # norm
    return (2**(-float(n)*0.5))*hermval(x/np.sqrt(2.0), cf)/nc

def herm_norm_snorm(n):
    """
    For uniform
    """
    return 1.0

herm.snorm = herm_norm_snorm


def legendre(x, n, interval=(-1.0, 1.0)):
    """
    Non-normed poly
    """
    xn = (interval[0] + interval[1] - 2.0*x)/(interval[0] - interval[1])
    return L.basis(n)(xn)


def legendre_snorm(n, interval=(-1.0, 1.0)):
    """
    RETURNS E[L_n L_n]
    """
    # return 2.0/(2.0*n + 1.0)
    return (interval[1] - interval[0])/(2.0*n + 1.0)

legendre.snorm = legendre_snorm


# Main func
def GenMat(n_size, x, poly=None, debug=False, pow_p=1, indeces=None, \
           IsTypeGood=True, poly_vals=None):
    """
    INPUT
        n_size — number of colomns (monoms), int
        x — points, num_pnts x l numpy array (num_pnts is arbitrary integer,\
        number of point, l — number of independent vars = number of derivatives  )
    OUTPUT 
        num_pnts*(l+1) x n_size matrix A, such that 
        a_{ij} = H_i(x_j) when i<l 
        or a_{ij}=H'_{i mod l}(x_j), where derivatives are taken on \
        coordinate with number i//l
    """

    num_pnts, l = x.shape

    ss = """<class 'autograd"""
    IsTypeGood = IsTypeGood and str(x.__class__)[:len(ss)] != ss


    calc_local_vals = False
    if poly is not None:
        use_func = True
        if not isinstance(poly, list):
            assert callable(poly), "poly must be either a func or a list of funcs"
            if IsTypeGood:
                calc_local_vals = True
            else:
                poly = [poly] * l
    else:
        assert poly_vals is not None, "Neither poly nor poly_vals parameter got"
        use_func = False

    if indeces is None:
        indeces = indeces_K_cut(l, n_size, p=pow_p)
    else:
        assert(len(indeces) == n_size)

    nA = num_pnts


    if IsTypeGood:
        A = np.empty((nA, n_size), dtype=x.dtype)
    else:
        A = []

        
    if calc_local_vals:
        #tot_elems = x.size
        max_degree = np.max(indeces)
        poly_vals = []
        for i in range(max_degree + 1):
            poly_vals.append( poly(x.ravel('F'), i ) )
        poly_vals = np.vstack(poly_vals).T

        use_func = False


    if debug:
        print('number of vars(num_pnts) = {}, dim of space \
(number of derivatives, l) = {},  number of monoms(n_size) = {}'.\
format(num_pnts, l, n_size))


    if use_func: # call poly
        for i, xp in enumerate(indeces):
            Acol = []
            if IsTypeGood:
                A[:num_pnts, i] = herm_mult_many(x, xp, poly)
            else:
                Acol.append(herm_mult_many(x, xp, poly))

            A.append(Acol[0])


    else: # use poly values
        for i, xp in enumerate(indeces):
            res = np.copy(poly_vals[:num_pnts, xp[0]])
            for n in range(1, l):
                res *= poly_vals[num_pnts*n : num_pnts*(n+1), xp[n]]

            if IsTypeGood:
                A[:num_pnts, i] = res
            else:
                A.append(res)

    if not IsTypeGood:
        A = np.vstack(A).T

    norm = False
    if norm:
        As = A
        A  = []
        for i, e in enumerate(As):
            A.append(e/np.linalg.norm(e, 2))
        A = np.vstack(A)

    return A


# Points generation

def test_points_gen(n_test, nder, interval=(-1.0, 1.0), distrib='random', **kwargs):
    return {'random' : lambda n_test, nder : (interval[1] - \
            interval[0])*np.random.rand(n_test, nder) + interval[0],\
            'lhs'    : lambda n_test, nder : (interval[1] -\
            interval[0])*lhs(nder, samples=n_test, **kwargs) + interval[0],\
            }[distrib.lower()](n_test, nder)
	
def getoptimumk(X):
    distortions = []
    Kss = range(1,10)
    
    for k in Kss:
        kmeanModel = MiniBatchKMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, \
                                            'euclidean'), axis=1)) / X.shape[0])
    
    myarray = np.array(distortions)
    
    knn = KneeLocator(Kss,myarray,curve='convex',direction='decreasing',\
                      interp_method='interp1d')
    kuse=knn.knee
    
    # Plot the elbow
    plt.figure(figsize=(10, 10))
    plt.plot(Kss, distortions, 'bx-')
    plt.xlabel('cluster size')
    plt.ylabel('Distortion')
    plt.title('optimal n_clusters for machine')
    plt.savefig("machine_elbow.jpg")
    plt.clf()
    return kuse

def ESMDA(sgsim,sgsimporo,modelError,CM,f, N, Sim1,alpha):
    if Yet==0:
        sgsim=cp.asarray(sgsim)
        sgsimporo=cp.asarray(sgsimporo)
        modelError=cp.asarray(modelError)
        CM=cp.asarray(CM)
        f=cp.asarray(f)
        N=cp.asarray(N)
        Sim1=cp.asarray(Sim1)
        alpha=cp.asarray(alpha)
        noise_levell=cp.asarray(noise_level)
    else:
        noise_levell=noise_level
        
    
    sizea=sgsim.shape[0]    
    cp.random.seed(1)
    stdall=cp.zeros((f.shape[0],1))
    for i in range (f.shape[0]):
        #aa=f[i,:] 
        if f[i,:] ==0:
           stdall[i,:]=1
        else:
           
            if Big_noise==2:
                stdall[i,:]=cp.sqrt(noise_levell*f[i,:])
            else:
                stdall[i,:]=cp.sqrt(cp.asarray(diffyet[i,:]))

    stdall=cp.reshape(stdall,(-1,1))
    
    nobs = f.shape[0]
    noise = cp.random.randn(max(10000,nobs),1)
    
    Error1=stdall
    sig=Error1
    for i in range (f.shape[0]):
        f[i,:] = f[i,:] + sig[i,:]*noise[-1-nobs+i,:]

    R = sig**2
    Dj =np.matlib.repmat(cp.asnumpy(f), 1, Ne)
    rndm=np.zeros((Dj.shape[0],Ne))
    for ik in range (Dj.shape[0]):
        #i=0
        kkk=rndm[ik,:]
        kkk=np.reshape(kkk,(1,-1),'F')
        rndm[ik,:] = np.random.randn(1,Ne) 
        rndm[ik,:] = rndm[ik,:] - np.mean(kkk,axis=1)
        rndm[ik,:] = rndm[ik,:] / np.std(kkk, axis=1)
        Dj[ik,:] = Dj[ik,:] + np.sqrt(cp.asnumpy(alpha))*np.sqrt(cp.asnumpy(R[ik,])) \
            * rndm[ik,:]   
            
    Dj=cp.asarray(Dj)  

    Cd2=cp.diag(cp.reshape(R,(-1,)))
    
    overall=cp.vstack([(sgsim),sgsimporo])

    Y=overall 
    M = cp.mean(Sim1,axis=1)

    M2=cp.mean(overall,axis=1)
    
    
    S = cp.zeros((Sim1.shape[0],N_ens))
    yprime = cp.zeros((Y.shape[0],Y.shape[1]))
           
    for j in range(N_ens):
        S[:,j] = Sim1[:,j]- M
        yprime[:,j] = overall[:,j] - M2
    Cyd = (yprime.dot(S.T))/(N_ens - 1)
    Cdd = (S.dot(S.T))/(N_ens- 1)     
    
    Usig,Sig,Vsig = cp.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    
    Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
    valuesig = Bsig[-1]                 # last element
    valuesig = valuesig * 0.9999
    indices = ( Bsig >= valuesig ).ravel().nonzero()
    toluse = Sig[indices]
    tol = toluse[0]

    (V,X,U) = pinvmatt((Cdd + (alpha*Cd2)),tol)
    
    update_term=((Cyd.dot(X)).dot(Dj - Sim1))
    Ynew = Y + update_term    
    DupdateK=Ynew[:sizea,:]
    poro=Ynew[sizea:,:]
    
    if modelError==1:
        Y=CM
        overall=Y
        M = cp.mean(Sim1,axis=1)
    
        M2=cp.mean(overall,axis=1)
        
        
        S = cp.zeros((Sim1.shape[0],N_ens))
        yprime = cp.zeros((Y.shape[0],Y.shape[1]))
               
        for j in range(N_ens):
            S[:,j] = Sim1[:,j]- M
            yprime[:,j] = overall[:,j] - M2
        Cyd = (yprime.dot(S.T))/(N_ens - 1)
        Cdd = (S.dot(S.T))/(N_ens- 1)     
        
        Usig,Sig,Vsig = cp.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    
        
        
        Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
        valuesig = Bsig[-1]                 # last element
        valuesig = valuesig * 0.9999
        indices = ( Bsig >= valuesig ).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]
    
        (V,X,U) = pinvmatt((Cdd + (alpha*Cd2)),tol)
        
        update_term=((Cyd.dot(X)).dot(Dj - Sim1))
        CM = Y + update_term 
    else:
        CM=cp.zeros((CM.shape[0],CM.shape[1])) 
        
    if Yet==0:
        modelError=cp.asnumpy(modelError)
        N=cp.asnumpy(N)
        alpha=cp.asnumpy(alpha)        
        return cp.asnumpy(DupdateK),cp.asnumpy(poro),cp.asnumpy(CM)
    else:
        return DupdateK,poro,CM

def ESMDA_New(sgsim,sgsimporo,modelError,CM,f, N, Sim1,alpha):
    if Yet==0:
        sgsim=cp.asarray(sgsim)
        sgsimporo=cp.asarray(sgsimporo)
        modelError=cp.asarray(modelError)
        CM=cp.asarray(CM)
        f=cp.asarray(f)
        N=cp.asarray(N)
        Sim1=cp.asarray(Sim1)
        alpha=cp.asarray(alpha)
        noise_levell=cp.asarray(noise_level)
    else:
        noise_levell=noise_level
    
    sizea=sgsim.shape[0]
    
    cp.random.seed(1)
    stdall=cp.zeros((f.shape[0],1))
    for iz in range (f.shape[0]):
        #aa=f[i,:] 
        if f[iz,:] ==0:
           stdall[iz,:]=1
        else:
           
            if Big_noise==2:
                stdall[iz,:]=cp.sqrt(noise_levell*f[iz,:])
            else:
                stdall[iz,:]=cp.sqrt(cp.asarray(diffyet[iz,:]))

    stdall=cp.reshape(stdall,(-1,1))    
    Error1 = cp.ones((f.shape[0],N_ens)) 
    for iz in range (f.shape[0]):
        Error1[iz,:]=cp.random.normal(0,stdall[iz,:],(N_ens))        
    
    nobs = f.shape[0]    
    #Error1=stdall    
    Dj = cp.zeros((nobs, N_ens))

    for j in range(N_ens):
        Dj[:,j] = cp.ravel(f) + Error1[:,j]

    Cd2=(Error1.dot(Error1.T))/(N_ens - 1)
    
    overall=cp.vstack([(sgsim),sgsimporo])

    Y=overall 
    M = cp.mean(Sim1,axis=1)

    M2=cp.mean(overall,axis=1)    
    S = cp.zeros((Sim1.shape[0],N_ens))
    yprime = cp.zeros((Y.shape[0],Y.shape[1]))
           
    for j in range(N_ens):
        S[:,j] = Sim1[:,j]- M
        yprime[:,j] = overall[:,j] - M2
    Cyd = (yprime.dot(S.T))/(N_ens - 1)
    Cdd = (S.dot(S.T))/(N_ens- 1)     
    
    Usig,Sig,Vsig = cp.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    
    Bsig = np.cumsm(Sig, axis = 0)          # vertically addition
    valuesig = Bsig[-1]                 # last element
    valuesig = valuesig * 0.9999
    indices = ( Bsig >= valuesig ).ravel().nonzero()
    toluse = Sig[indices]
    tol = toluse[0]

    (V,X,U) = pinvmatt((Cdd + (alpha*Cd2)),tol)
    
    update_term=((Cyd.dot(X)).dot(Dj - Sim1))
    Ynew = Y + update_term    
    DupdateK=Ynew[:sizea,:]
    poro=Ynew[sizea:,:]
    
    if modelError==1:
        Y=CM
        overall=Y
        M = cp.mean(Sim1,axis=1)
    
        M2=cp.mean(overall,axis=1)
        
        
        S = cp.zeros((Sim1.shape[0],N_ens))
        yprime = cp.zeros((Y.shape[0],Y.shape[1]))
               
        for j in range(N_ens):
            S[:,j] = Sim1[:,j]- M
            yprime[:,j] = overall[:,j] - M2
        Cyd = (yprime.dot(S.T))/(N_ens - 1)
        Cdd = (S.dot(S.T))/(N_ens- 1)     
        
        Usig,Sig,Vsig = cp.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    
        Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
        valuesig = Bsig[-1]                 # last element
        valuesig = valuesig * 0.9999
        indices = ( Bsig >= valuesig ).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]
    
        (V,X,U) = pinvmatt((Cdd + (alpha*Cd2)),tol)
        
        update_term=((Cyd.dot(X)).dot(Dj - Sim1))
        CM = Y + update_term 
    else:
        CM=cp.zeros((CM.shape[0],CM.shape[1])) 

        modelError=cp.asnumpy(modelError)
        N=cp.asnumpy(N)
        alpha=cp.asnumpy(alpha)
        noise_levell=cp.asarray(noise_level)        
    if Yet==0:
        return cp.asnumpy(DupdateK),cp.asnumpy(poro),cp.asnumpy(CM)
    else:
        return DupdateK,poro,CM



def ESMDA_AEE(sgsim,sgsimporo,modelError,CM,f, N, Sim1,alpha):
    if Yet==0:
        sgsim=cp.asarray(sgsim)
        sgsimporo=cp.asarray(sgsimporo)
        modelError=cp.asarray(modelError)
        CM=cp.asarray(CM)
        f=cp.asarray(f)
        N=cp.asarray(N)
        Sim1=cp.asarray(Sim1)
        alpha=cp.asarray(alpha)
        noise_levell=cp.asarray(noise_level)    
        sizea=sgsim.shape[0]
    else:
        sizea=sgsim.shape[0]
        noise_levell=noise_level
    
    cp.random.seed(1)

    stdall=cp.zeros((f.shape[0],1))
    for i in range (f.shape[0]):
        #aa=f[i,:] 
        if f[i,:] ==0:
           stdall[i,:]=1
        else:
           
            if Big_noise==2:
                stdall[i,:]=cp.sqrt(noise_levell*f[i,:])
            else:
                stdall[i,:]=cp.sqrt(cp.asarray(diffyet[i,:]))

    stdall=cp.reshape(stdall,(-1,1))
    
    nobs = f.shape[0]
    noise = cp.random.randn(max(10000,nobs),1)
    
    Error1=stdall
    sig=Error1
    for i in range (f.shape[0]):
        f[i,:] = f[i,:] + sig[i,:]*noise[-1-nobs+i,:]

    R = sig**2
    #Dj =cp.matlib.repmat(f, 1, N)
    Dj =np.matlib.repmat(cp.asnumpy(f), 1, Ne)
    rndm=np.zeros((Dj.shape[0],Ne))
    for ik in range (Dj.shape[0]):
        #i=0
        kkk=rndm[ik,:]
        kkk=np.reshape(kkk,(1,-1),'F')
        rndm[ik,:] = np.random.randn(1,Ne) 
        rndm[ik,:] = rndm[ik,:] - np.mean(kkk,axis=1)
        rndm[ik,:] = rndm[ik,:] / np.std(kkk, axis=1)
        Dj[ik,:] = Dj[ik,:] + np.sqrt(cp.asnumpy(alpha))*np.sqrt(cp.asnumpy(R[ik,])) \
            * rndm[ik,:]   
            
    Dj=cp.asarray(Dj)  

    Cd2=cp.diag(cp.reshape(R,(-1,)))
    
    overall=cp.vstack([sgsim,sgsimporo])

    Y=overall 
    M = cp.mean(Sim1,axis=1)

    M2=cp.mean(overall,axis=1)
    
    
    S = cp.zeros((Sim1.shape[0],N_ens))
    yprime = cp.zeros((Y.shape[0],Y.shape[1]))
           
    for j in range(N_ens):
        S[:,j] = Sim1[:,j]- M
        yprime[:,j] = overall[:,j] - M2
    Cyd = (yprime.dot(S.T))/(N_ens - 1)
    Cdd = (S.dot(S.T))/(N_ens- 1)     
    
    Usig,Sig,Vsig = cp.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)

    
    
    Bsig = np.cumsum(Sig, axis = 0)          # vertically addition
    valuesig = Bsig[-1]                 # last element
    valuesig = valuesig * 0.9999
    indices = ( Bsig >= valuesig ).ravel().nonzero()
    toluse = Sig[indices]
    tol = toluse[0]

    (V,X,U) = pinvmatt((Cdd + (alpha*Cd2)),tol)
    
    update_term=((Cyd.dot(X)).dot(Dj - Sim1))
    Ynew = Y + update_term    
    DupdateK=Ynew[:sizea,:]
    poro=Ynew[sizea:,:]
    
    if modelError==1:
        Y=CM
        overall=Y
        M = cp.mean(Sim1,axis=1)
    
        M2=cp.mean(overall,axis=1)
        
        
        S = cp.zeros((Sim1.shape[0],N_ens))
        yprime = cp.zeros((Y.shape[0],Y.shape[1]))
               
        for j in range(N_ens):
            S[:,j] = Sim1[:,j]- M
            yprime[:,j] = overall[:,j] - M2
        Cyd = (yprime.dot(S.T))/(N_ens - 1)
        Cdd = (S.dot(S.T))/(N_ens- 1)     
        
        Usig,Sig,Vsig = cp.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    
        
        
        Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
        valuesig = Bsig[-1]                 # last element
        valuesig = valuesig * 0.9999
        indices = ( Bsig >= valuesig ).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]
    
        (V,X,U) = pinvmatt((Cdd + (alpha*Cd2)),tol)
        
        update_term=((Cyd.dot(X)).dot(Dj - Sim1))
        CM = Y + update_term 
    else:
        CM=cp.zeros((CM.shape[0],CM.shape[1]))        
    if Yet==0:
        return cp.asnumpy(DupdateK),cp.asnumpy(poro),cp.asnumpy(CM)
    else:
        return DupdateK,poro,CM


def ESMDA_CCR(sgsim,f, N, Sim1,alpha):
    if Yet==0:
        sgsim=cp.asarray(sgsim)
        f=cp.asarray(f)
        N=cp.asarray(N)
        Sim1=cp.asarray(Sim1)
        alpha=cp.asarray(alpha)
        noise_levell=cp.asarray(noise_level)  
    else:
        noise_levell=noise_level
    
    
    
    
    cp.random.seed(1)
    sizea=sgsim.shape[0]

    stdall=cp.zeros((f.shape[0],1))
    for i in range (f.shape[0]):
        #aa=f[i,:] 
        if f[i,:] ==0:
           stdall[i,:]=1
        else:
           
            if Big_noise==2:
                stdall[i,:]=cp.sqrt(noise_levell*f[i,:])
            else:
                stdall[i,:]=cp.sqrt(cp.asarray(diffyet[i,:]))

    stdall=cp.reshape(stdall,(-1,1))
    
    nobs = f.shape[0]
    noise = cp.random.randn(max(10000,nobs),1)
    
    Error1=stdall
    sig=Error1
    for i in range (f.shape[0]):
        f[i,:] = f[i,:] + sig[i,:]*noise[-1-nobs+i,:]

    R = sig**2
    #Dj =cp.matlib.repmat(f, 1, N)
    Dj =np.matlib.repmat(cp.asnumpy(f), 1, Ne)
    rndm=np.zeros((Dj.shape[0],Ne))
    for ik in range (Dj.shape[0]):
        #i=0
        kkk=rndm[ik,:]
        kkk=np.reshape(kkk,(1,-1),'F')
        rndm[ik,:] = np.random.randn(1,Ne) 
        rndm[ik,:] = rndm[ik,:] - np.mean(kkk,axis=1)
        rndm[ik,:] = rndm[ik,:] / np.std(kkk, axis=1)
        Dj[ik,:] = Dj[ik,:] + np.sqrt(cp.asnumpy(alpha))*np.sqrt(cp.asnumpy(R[ik,])) \
            * rndm[ik,:]   
            
    Dj=cp.asarray(Dj)  

    Cd2=cp.diag(np.reshape(R,(-1,)))
    overall=sgsim
    #Sim1=Sim1+CM
    
    Y=overall 
    M = cp.mean(Sim1,axis=1)

    M2=cp.mean(overall,axis=1)
    
    
    S = cp.zeros((Sim1.shape[0],N_ens))
    yprime = cp.zeros((Y.shape[0],Y.shape[1]))
           
    for j in range(N_ens):
        S[:,j] = Sim1[:,j]- M
        yprime[:,j] = overall[:,j] - M2
    Cyd = (yprime.dot(S.T))/(N_ens - 1)
    Cdd = (S.dot(S.T))/(N_ens- 1)     
    
    Usig,Sig,Vsig = cp.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    
    Bsig = np.cumsum(Sig, axis = 0)          # vertically addition
    valuesig = Bsig[-1]                 # last element
    valuesig = valuesig * 0.9999
    indices = ( Bsig >= valuesig ).ravel().nonzero()
    toluse = Sig[indices]
    tol = toluse[0]

    (V,X,U) = pinvmatt((Cdd + (alpha*Cd2)),tol)
    
    update_term=((Cyd.dot(X)).dot(Dj - Sim1))
    Ynew = Y + update_term    
    DupdateK=Ynew[:sizea,:]
    if Yet==0:
        return cp.asnumpy(DupdateK)
    else:
        return DupdateK
    
def iES_flavours(w, T, Y, Y0, dy, Cowp, za, N, nIter, itr, MDA, flavour):
    N1 = N - 1
    Cow1 = Cowp(1.0)

    if MDA:  # View update as annealing (progressive assimilation).
        Cow1 = Cow1 @ T  # apply previous update
        dw = dy @ Y.T @ Cow1
        if 'PertObs' in flavour:   # == "ES-MDA". By Emerick/Reynolds
            D   = mean0(randn(*Y.shape)) * sqrt(nIter)
            T  -= (Y + D) @ Y.T @ Cow1
        elif 'Sqrt' in flavour:    # == "ETKF-ish". By Raanes
            T   = Cowp(0.5) * sqrt(za) @ T
        elif 'Order1' in flavour:  # == "DEnKF-ish". By Emerick
            T  -= 0.5 * Y @ Y.T @ Cow1
        Tinv = np.eye(N)  # [as initialized] coz MDA does not de-condition.

    else:  # View update as Gauss-Newton optimzt. of log-posterior.
        grad  = Y0@dy - w*za                  # Cost function gradient
        dw    = grad@Cow1                     # Gauss-Newton step
        # ETKF-ish". By Bocquet/Sakov.
        if 'Sqrt' in flavour:
            # Sqrt-transforms
            T     = Cowp(0.5) * sqrt(N1)
            Tinv  = Cowp(-.5) / sqrt(N1)
            # Tinv saves time [vs tinv(T)] when Nx<N
        # "EnRML". By Oliver/Chen/Raanes/Evensen/Stordal.
        elif 'PertObs' in flavour:
            if itr == 0:
                D = mean0(randn(*Y.shape))
                iES_flavours.D = D
            else:
                D = iES_flavours.D
            gradT = -(Y+D)@Y0.T + N1*(np.eye(N) - T)
            T     = T + gradT@Cow1
            # Tinv= tinv(T, threshold=N1)  # unstable
            Tinv  = sla.inv(T+1)           # the +1 is for stability.
        # "DEnKF-ish". By Raanes.
        elif 'Order1' in flavour:
            # Included for completeness; does not make much sense.
            gradT = -0.5*Y@Y0.T + N1*(np.eye(N) - T)
            T     = T + gradT@Cow1
            Tinv  = sla.pinv2(T)

    return dw, T, Tinv 

    

def iES(High_K,Low_K,High_P,Low_P,modelError,sizeclem,CM,maxx,ensemble, 
        observation, obs_err_cov,
        flavour="Sqrt", MDA=False, bundle=False,
        stepsize=1, nIter=10, wtol=1e-4):

    E = ensemble
    N = len(E)
    N1 = N - 1
    #obs_err_cov=CDd
    Rm12T = np.diag((1/np.diag(obs_err_cov))**0.5)
    #Rm12T = np.diag((np.diag(obs_err_cov))**0.5)
   

    stats = DotDict()
    stats.J_lklhd  = np.full(nIter, np.nan)
    stats.J_prior  = np.full(nIter, np.nan)
    stats.J_postr  = np.full(nIter, np.nan)
    stats.rmse     = np.full(nIter, np.nan)
    stats.stepsize = np.full(nIter, np.nan)
    stats.dw       = np.full(nIter, np.nan)

    if bundle:
        if isinstance(bundle, bool):
            EPS = 1e-4  # Sakov/Boc use T=EPS*eye(N), with EPS=1e-4, but I ...
        else:
            EPS = bundle
    else:
        EPS = 1.0  # ... prefer using  T=EPS*T, yielding a conditional cloud shape

    # Init ensemble decomposition.
    X0, x0 = center(E)    # Decompose ensemble.
    w      = np.zeros(N)  # Control vector for the mean state.
    T      = np.eye(N)    # Anomalies transform matrix.
    Tinv   = np.eye(N)
    # Explicit Tinv [instead of tinv(T)] allows for merging MDA code
    # with iEnKS/EnRML code, and flop savings in 'Sqrt' case.
    old = w, T, Tinv
    for itr in range(nIter):
        # Reconstruct smoothed ensemble.
        markk=itr+1
        print(str(itr+1) + ' | ' + str(nIter))
        E = x0 + (w + EPS*T)@X0
        
        #stats.rmse[itr] = RMS(TrueK, E).rmse
        Emerg1=E.T
        Emerg=(Emerg1[:sizeclem,:])
        Ep=Emerg1[sizeclem:2*sizeclem,:]

        CM=Emerg1[2*sizeclem:,:]
        if modelError!=1:
            CM=np.zeros_like(CM)

        Euse,Ep=honour2(Ep,Emerg,nx,ny,nz,N,High_K,Low_K,High_P,Low_P)       

        
        for ijesus in range(N):
            (
            write_include)(ijesus,Euse,Ep,'Realization_')      
        az=int(np.ceil(int(N/maxx)))
        a=(np.linspace(1, N, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            name='masterr.data'
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
           
            Delete_files()#Declutter folder
        Delete_files2()
        
        if markk==1:
            os.chdir('HM_RESULTS')                             
            Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Initial.jpg")
            os.chdir(oldfolder)
        
        E_obs =  Get_simulated(predMatrix,modelError,CM,N)# # 
        aa,bb,cc=funcGetDataMismatch(E_obs,observation.reshape(-1,1))
        if markk==1:
            cc_ini=cc
        else:
            pass
        
        simmean=np.reshape(np.mean(E_obs,axis=1),(-1,1),'F')
        
        
        see=((np.sum((((simmean) - observation.reshape(-1,1)) ** 2)) )**(0.5))\
            /True_data.shape[0]
        print('RMSE of the ensemble mean = : ' \
              + str(see) + '... .')           
        
        
        #E_obs=E_obs+CM #Account for Model Erros here
        E_obs=E_obs.T
        
        #N=Ne
        E_obs = E_obs.reshape((N, -1))

        # Undo the bundle scaling of ensemble.
        if EPS != 1.0:
            E     = inflate_ens(E,     1/EPS)
            E_obs = inflate_ens(E_obs, 1/EPS)
        #observation=True_data.T   
        # Prepare analysis.
        y      = observation        # Get current obs.
        Y, xo  = center(E_obs)      # Get obs {anomalies, mean}.
        dy     = (y - xo) @ Rm12T   # Transform obs space.
        Y      = Y        @ Rm12T   # Transform obs space.
        Y0     = Tinv @ Y           # "De-condition" the obs anomalies.

        # Set "cov normlzt fctr" za ("effective ensemble size")
        # => pre_infl^2 = (N-1)/za.
        za = N1
        if MDA:
            # inflation (factor: nIter) of the ObsErrCov.
            za *= nIter

        # Compute Cowp: the (approx) posterior cov. of w
        # (estiamted at this iteration), raised to some power.
        V, s, UT = svd0(Y0)
        def Cowp(expo): return (V * (pad0(s**2, N) + za)**-expo) @ V.T

        stat2 = DotDict(
            J_prior = w@w * N1,
            J_lklhd = dy@dy.T,
        )
        # J_posterior is sum of the other two
        stat2.J_postr = stat2.J_prior + stat2.J_lklhd
        # Take root, insert for [itr]:
        for name in stat2:
            stats[name][itr] = sqrt(stat2[name])

        # Accept previous increment? ...
        if (not MDA) and itr > 0 and stats.J_postr[itr] > np.nanmin(stats.J_postr):
            # ... No. Restore previous ensemble & lower the stepsize (dont compute new increment).
            stepsize   /= 10
            w, T, Tinv  = old  # noqa
        else:
            # ... Yes. Store this ensemble, boost the stepsize, and compute new increment.
            old         = w, T, Tinv
            stepsize   *= 2
            stepsize    = min(1, stepsize)
            dw, T, Tinv = iES_flavours(w, T, Y, Y0, dy, Cowp, za, N, nIter, itr, MDA, flavour)

        stats.      dw[itr] = dw@dw / N
        stats.stepsize[itr] = stepsize 

        # Step
        w = w + stepsize*dw

        if stepsize * np.sqrt(dw@dw/N) < wtol:
            break

    stats.nIter = itr + 1

    if not MDA:
        # The last step (dw, T) must be discarded,
        # because it cannot be validated without re-running the model.
        w, T, Tinv  = old

    # Reconstruct the ensemble.
    E = x0 + (w+T)@X0

    return E, stats,cc_ini

def EnKF(sgsim,sgsimporo,modelError,CM,f, N, Sim1):
    if Yet==0:
        sgsim=cp.asarray(sgsim)
        sgsimporo=cp.asarray(sgsimporo)
        modelError=cp.asarray(modelError)
        CM=cp.asarray(CM)
        f=cp.asarray(f)
        N=cp.asarray(N)
        Sim1=cp.asarray(Sim1)
        noise_levell=cp.asarray(noise_level)  
    else:
        noise_levell=noise_level
    
    
    
    sizea=sgsim.shape[0]


    cp.random.seed(1)
 
    stdall=cp.zeros((f.shape[0],1))
    for i in range (f.shape[0]):
        #aa=f[i,:]
        if f[i,:]==0:
            stdall[i,:]=1
        else:
            
            if Big_noise==2:
                stdall[i,:]=cp.sqrt(noise_levell*f[i,:])
            else:
                stdall[i,:]=cp.sqrt(cp.asarray(diffyet[i,:]))

    stdall=cp.reshape(stdall,(-1,1))
    
    nobs = f.shape[0]
    noise = cp.random.randn(max(10000,nobs),1)
    
    Error1=stdall
    sig=Error1
    for i in range (f.shape[0]):
        f[i,:] = f[i,:] + sig[i,:]*noise[-1-nobs+i,:]

    R = sig**2
    #Dj =cp.matlib.repmat(f, 1, N)
    Dj =np.matlib.repmat(cp.asnumpy(f), 1, Ne)
    rndm=np.zeros((Dj.shape[0],Ne))
    for ik in range (Dj.shape[0]):
        #i=0
        kkk=rndm[ik,:]
        kkk=np.reshape(kkk,(1,-1),'F')
        rndm[ik,:] = np.random.randn(1,Ne) 
        rndm[ik,:] = rndm[ik,:] - np.mean(kkk,axis=1)
        rndm[ik,:] = rndm[ik,:] / np.std(kkk, axis=1)
        Dj[ik,:] = Dj[ik,:] + np.sqrt(cp.asnumpy(alpha1))*np.sqrt(cp.asnumpy(R[ik,])) \
            * rndm[ik,:]   
            
    Dj=cp.asarray(Dj)  

    Cd2=cp.diag(np.reshape(R,(-1,)))
    overall=cp.vstack([(sgsim),sgsimporo])
    #Sim1=Sim1+CM
    Y=overall 
    M = cp.mean(Sim1,axis=1)

    M2=cp.mean(overall,axis=1)
    
    
    S = cp.zeros((Sim1.shape[0],N_ens))
    yprime = cp.zeros((Y.shape[0],Y.shape[1]))
           
    for j in range(N_ens):
        S[:,j] = Sim1[:,j]- M
        yprime[:,j] = overall[:,j] - M2
    Cyd = (yprime.dot(S.T))/(N_ens - 1)
    Cdd = (S.dot(S.T))/(N_ens- 1)     
    
    Usig,Sig,Vsig = cp.linalg.svd((Cdd + (Cd2)), full_matrices = False)
    Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
    valuesig = Bsig[-1]                 # last element
    valuesig = valuesig * 0.9999
    indices = ( Bsig >= valuesig ).ravel().nonzero()
    toluse = Sig[indices]
    tol = toluse[0]

    (V,X,U) = pinvmatt((Cdd + (alpha*Cd2)),tol)
    
    update_term=((Cyd.dot(X)).dot(Dj - Sim1))
    Ynew = Y + update_term    
    
    DupdateK=Ynew[:sizea,:]
    poro=Ynew[sizea:,:]
    
    if modelError==1:
        Y=CM
        overall=Y
        M = cp.mean(Sim1,axis=1)
    
        M2=cp.mean(overall,axis=1)
        
        
        S = cp.zeros((Sim1.shape[0],N_ens))
        yprime = cp.zeros((Y.shape[0],Y.shape[1]))
               
        for j in range(N_ens):
            S[:,j] = Sim1[:,j]- M
            yprime[:,j] = overall[:,j] - M2
        Cyd = (yprime.dot(S.T))/(N_ens - 1)
        Cdd = (S.dot(S.T))/(N_ens- 1)     
        
        Usig,Sig,Vsig = cp.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    
        
        
        Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
        valuesig = Bsig[-1]                 # last element
        valuesig = valuesig * 0.9999
        indices = ( Bsig >= valuesig ).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]
    
        (V,X,U) = pinvmatt((Cdd + (alpha*Cd2)),tol)
        
        update_term=((Cyd.dot(X)).dot(Dj - Sim1))
        CM = Y + update_term
    else:
        CM=cp.zeros_like(CM)
    if Yet==0:
        return cp.asnumpy(DupdateK),cp.asnumpy(poro),cp.asnumpy(CM)
    else:
        return DupdateK,poro,CM

def ESMDA_Localisation(sgsim,sgsimporo,modelError,CM,f, N, Sim1,alpha,nx,ny,nz,c):
    if Yet==0:
        sgsim=cp.asarray(sgsim)
        sgsimporo=cp.asarray(sgsimporo)
        modelError=cp.asarray(modelError)
        CM=cp.asarray(CM)
        f=cp.asarray(f)
        N=cp.asarray(N)
        Sim1=cp.asarray(Sim1)
        alpha=cp.asarray(alpha)
        noise_levell=cp.asarray(noise_level)
        nx=cp.asarray(nx)
        ny=cp.asarray(ny)
        nz=cp.asarray(nz)  
        c=cp.asarray(c)
    else:
        pass
    
    
    
    cp.random.seed(1)
    sizea=sgsim.shape[0]

    #sizec=CM.shape[0]    

    A = cp.zeros((nx,ny,nz))
    #4 injector and 4 producer wells
    for jj in range(nz):
        A[1,24,jj] = 1
        A[1,1,jj] = 1
        A[31,0,jj] = 1
        A[31,31,jj] = 1
        A[7,9,jj] = 1
        A[14,12,jj] = 1
        A[28,19,jj] = 1
        A[14,27,jj] = 1
        
        
    print( '      Calculate the Euclidean distance function to the wells')
    lf = cp.reshape(A,(nx,ny,nz),'F')
    young = cp.zeros((int(nx*ny*nz/nz),nz))
    for j in range(nz):
        sdf = lf[:,:,j]
        (usdf,IDX) = spndmo.distance_transform_edt(cp.logical_not(sdf), \
                                                   return_indices = True)
        usdf = cp.reshape(usdf,(int(nx*ny*nz/nz)),'F')
        young[:,j] = usdf

    sdfbig = cp.reshape(young,(nx*ny*nz,1),'F')
    sdfbig1 = cp.abs(sdfbig)
    z = sdfbig1
    ## the value of the range should be computed accurately.
      
    c0OIL1 = cp.zeros((nx*ny*nz,1))
    
    print( '      Computing the Gaspari-Cohn coefficent')
    for i in range(nx*ny*nz):
        if ( 0 <= z[i,:] or z[i,:] <= c ):
            c0OIL1[i,:] = -0.25*(z[i,:]/c)**5 + 0.5*(z[i,:]/c)**4 +\
                0.625*(z[i,:]/c)**3 - (5.0/3.0)*(z[i,:]/c)**2 + 1

        elif ( z < 2*c ):
            c0OIL1[i,:] = (1.0/12.0)*(z[i,:]/c)**5 - 0.5*(z[i,:]/c)**4 +\
                0.625*(z[i,:]/c)**3 + (5.0/3.0)*(z[i,:]/c)**2 - 5*(z[i,:]/c)\
                    + 4 - (2.0/3.0)*(c/z[i,:])

        elif ( c <= z[i,:] or z[i,:] <= 2*c ):
            c0OIL1[i,:] = -5*(z[i,:]/c) + 4 -0.667*(c/z[i,:])

        else:
            c0OIL1[i,:] = 0
      
    c0OIL1[c0OIL1 < 0 ] = 0
      
    print('      Getting the Gaspari Cohn for Cyd') 
     
    schur = c0OIL1
    Bsch = cp.tile(schur,(1,N))        
       

    stdall=cp.zeros((f.shape[0],1))
    for i in range (f.shape[0]):
        #aa=f[i,:]
        if f[i,:] ==0:
            stdall[i,:]=1
        else:
            
            if Big_noise==2:
                stdall[i,:]=cp.sqrt(noise_levell*f[i,:])
            else:
                stdall[i,:]=cp.sqrt(cp.asarray(diffyet[i,:]))

    stdall=cp.reshape(stdall,(-1,1))
    
    nobs = f.shape[0]
    noise = cp.random.randn(max(10000,nobs),1)
    
    Error1=stdall
    sig=Error1
    for i in range (f.shape[0]):
        f[i,:] = f[i,:] + sig[i,:]*noise[-1-nobs+i,:]

    R = sig**2
    #Dj =cp.matlib.repmat(f, 1, N)
    Dj =np.matlib.repmat(cp.asnumpy(f), 1, Ne)
    rndm=np.zeros((Dj.shape[0],Ne))
    for ik in range (Dj.shape[0]):
        #i=0
        kkk=rndm[ik,:]
        kkk=np.reshape(kkk,(1,-1),'F')
        rndm[ik,:] = np.random.randn(1,Ne) 
        rndm[ik,:] = rndm[ik,:] - np.mean(kkk,axis=1)
        rndm[ik,:] = rndm[ik,:] / np.std(kkk, axis=1)
        Dj[ik,:] = Dj[ik,:] + np.sqrt(cp.asnumpy(alpha))*np.sqrt(cp.asnumpy(R[ik,])) \
            * rndm[ik,:]   
            
    Dj=cp.asarray(Dj)  

    

    Cd2=cp.diag(cp.reshape(R,(-1,)))
    overall=cp.vstack([(sgsim),sgsimporo])
    #Sim1=Sim1+CM
    Y=overall 
    M = cp.mean(Sim1,axis=1)

    M2=cp.mean(overall,axis=1)
    
    
    S = cp.zeros((Sim1.shape[0],N_ens))
    yprime = cp.zeros((Y.shape[0],Y.shape[1]))
           
    for j in range(N_ens):
        S[:,j] = Sim1[:,j]- M
        yprime[:,j] = overall[:,j] - M2
    Cyd = (yprime.dot(S.T))/(N_ens - 1)
    Cdd = (S.dot(S.T))/(N_ens- 1)     
    
    Usig,Sig,Vsig = cp.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    Bsig = np.cumsum(Sig, axis = 0)          # vertically addition
    valuesig = Bsig[-1]                 # last element
    valuesig = valuesig * 0.9999
    indices = ( Bsig >= valuesig ).ravel().nonzero()
    toluse = Sig[indices]
    tol = toluse[0]

    (V,X,U) = pinvmatt((Cdd + (alpha*Cd2)),tol)
    Bsch=np.vstack([Bsch,Bsch])
    update_term=Bsch*((Cyd.dot(X)).dot(Dj - Sim1))
    Ynew = Y + update_term    
    
    DupdateK=Ynew[:sizea,:]
    poro=Ynew[sizea:,:]
    if modelError==1:
        Y=CM
        overall=Y
        M = cp.mean(Sim1,axis=1)
    
        M2=cp.mean(overall,axis=1)
        
        
        S = cp.zeros((Sim1.shape[0],N_ens))
        yprime = cp.zeros((Y.shape[0],Y.shape[1]))
               
        for j in range(N_ens):
            S[:,j] = Sim1[:,j]- M
            yprime[:,j] = overall[:,j] - M2
        Cyd = (yprime.dot(S.T))/(N_ens - 1)
        Cdd = (S.dot(S.T))/(N_ens- 1)     
        
        Usig,Sig,Vsig = cp.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    
        
        
        Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
        valuesig = Bsig[-1]                 # last element
        valuesig = valuesig * 0.9999
        indices = ( Bsig >= valuesig ).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]
    
        (V,X,U) = pinvmatt((Cdd + (alpha*Cd2)),tol)
        
        update_term=((Cyd.dot(X)).dot(Dj - Sim1))
        CM = Y + update_term 
    else:
        CM=cp.zeros_like(cm)
    if Yet==0:
        return cp.asnumpy(DupdateK),cp.asnumpy(poro),cp.asnumpy(CM)
    else:
        return DupdateK,poro,CM

def ESMDA_Levelset(N_ens,m_ens,prod_ens,alpha,Nop,True_signal,Nx,\
                             Ny,Nz):
    
    """
    Level set update with the ES-MDA
    Parameters
    ----------
    N_ens : Ensemble size

    m_ens : Prior ensemble of parameters

    prod_ens : Prior ensemble of simulated measurements

    alpha : Inflation factor

    Nop : Number of measurement

    True_signal : True measurement
    
    Nx : Dimesnion in X axis
    
    Ny : Dimesnion in y axis
    
    Nz : Dimesnion in z axis


    Returns
    -------
    Ynew : Posterior ensemble of Level set functions

    """
    Error1 = np.ones((Nop,N_ens))
    overall=m_ens # Signed distance function here

        
    Xspitb=Parallel(n_jobs=8,backend='loky', verbose=0)(delayed(
    Narrow_Band)(overall[:,kk],Nx,Ny,Nz)for kk in range(N_ens) )
    B = np.reshape(np.hstack(Xspitb),(Nx*Ny*Nz,Ne),'F')# 
        
    B=np.reshape(B,(-1,1),'F')
    f=np.reshape(True_signal,(-1,1))
    Sim1=prod_ens
    
    #stdall=[]
    stdall=np.zeros((f.shape[0],1))
    for i in range (f.shape[0]):
        #aa=f[i,:]
        if f[i,:]==0:
            stdall[i,]=1
        else:
            
            if Big_noise==2:
                stdall[i,:]=sqrt(noise_level*f[i,:])
            else:
                stdall[i,:]=sqrt(diffyet[i,:])
            
    stdall=np.reshape(stdall,(-1,1))
    
    nobs = f.shape[0]
    noise = np.random.randn(max(10000,nobs),1)
    
    Error1=stdall
    sig=Error1
    for i in range (f.shape[0]):
        f[i,:] = f[i,:] + sig[i,:]*noise[-1-nobs+i,:]

    R = sig**2
    #Dj =np.matlib.repmat(f, 1, N_ens)
    Dj =np.matlib.repmat(cp.asnumpy(f), 1, Ne)
    rndm=np.zeros((Dj.shape[0],Ne))
    for ik in range (Dj.shape[0]):
        #i=0
        kkk=rndm[ik,:]
        kkk=np.reshape(kkk,(1,-1),'F')
        rndm[ik,:] = np.random.randn(1,Ne) 
        rndm[ik,:] = rndm[ik,:] - np.mean(kkk,axis=1)
        rndm[ik,:] = rndm[ik,:] / np.std(kkk, axis=1)
        Dj[ik,:] = Dj[ik,:] + np.sqrt((alpha))*np.sqrt((R[ik,])) \
            * rndm[ik,:]   
            
    Dj=cp.asarray(Dj)  

    Cd2=np.diag(np.reshape(R,(-1,))) 
    Y =overall
    #Sim1=Sim1+CM
    
    M = np.mean(Sim1, axis = 1)
    M2 = np.mean(overall, axis = 1)

    S = np.zeros((Sim1.shape[0],N_ens))
    yprime = np.zeros((Y.shape[0],Y.shape[1]))
           
    for j in range(N_ens):
        S[:,j] = Sim1[:,j]- M
        yprime[:,j] = overall[:,j] - M2
    Cyd = (yprime.dot(S.T))/(N_ens - 1)
    Cdd = (S.dot(S.T))/(N_ens- 1) 
    Usig,Sig,Vsig = np.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    Bsig = np.cumsum(Sig, axis = 0)          # vertically addition
    valuesig = Bsig[-1]                 # last element
    valuesig = valuesig * 0.999
    indices = ( Bsig >= valuesig ).ravel().nonzero()
    toluse = Sig[indices]
    tol = toluse[0]

    (V,X,U) = pinvmatt((Cdd + (alpha*Cd2)),tol)
    
    update_term=((Cyd.dot(X)).dot(Dj - Sim1))
    Bb1=np.reshape(B,(-1,N_ens),'F')

    Bb=Bb1
    update_term_boundary=update_term*Bb
    Ynew = Y + update_term_boundary
    DupdateK=Ynew

  
    return DupdateK



def Forwarding_Ensemble(oldfolder,stringf,jj,nx,ny,nz,kperm,poroin):
    folder=stringf + str(jj)
    Forward_model(oldfolder,folder,kperm,poroin)

numpy.random.seed(1)
   
def Forward_model(oldfolder,folder,kperm,poroin):
#### ===================================================================== ####
#                     6X RESERVOIR SIMULATOR
#                      
#### ===================================================================== ####
    os.chdir(os.path.join(oldfolder,folder))
    poroin=np.reshape(poroin,(-1,1),'F')
    kperm=np.reshape(kperm,(-1,1),'F')
    np.savetxt("POROO.DAT",poroin, comments='',fmt='%4.4f',delimiter='\t',\
               header="PORO",footer="/")
    np.savetxt("PERM.DAT",kperm, comments='',fmt='%4.4f',delimiter='\t',\
               header="PERMY",footer="/")    
    os.system("@6X_34157_75 masterreal.data -csv -sumout 3")
    #os.system("@mpiexec -np 5 6X_34157_75 masterr.data -csv ")
    #mpiexec -np 5 6X_34157_75 masterr.data -csv   
    os.chdir(oldfolder)

def Plot_3D(knew2,nx,ny,nz):
    #from mpl_toolkits.mplot3d import Axes3D;
    pyplot.interactive(True);
    
    # Creat mesh.
    
    X, Y =np.meshgrid(cp.arange(nx),np.arange(ny))  
    # Create flat surface.
    Z = np.zeros_like(X);
    fig = plt.figure();
    ax = fig.gca(projection='3d');
    for i in range(0,nz,1):
        #print(i)
        A = knew2[:,:,i];
        A -= numpy.min(A); A /= numpy.max(A);
        ax.plot_surface(X, Y, Z+i, rstride=1, cstride=1, facecolors = cm.jet(A));
    
        #i=i+10
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.title('Permeability',fontsize=13)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_zticks([])
    plt.savefig("3D permeability.jpg")
    plt.clf()
    plt.close()
   # m = cm.ScalarMappable(cmap=cm.jet)
    ax.grid(False)    

def Add_marker(plt,XX,YY,locc):
    for i in range(locc.shape[0]):
        a=locc[i,:]
        xloc=int(a[1])
        yloc=int(a[0])
        if a[3]==2:
           plt.scatter(XX.T[xloc-1,yloc-1]+0.5,YY.T[xloc-1,yloc-1]+0.5,s=100,\
                       marker='^', color = 'red') 
        else:
            plt.scatter(XX.T[xloc-1,yloc-1]+0.5,YY.T[xloc-1,yloc-1]+\
                        0.5,s=100,marker='v', color = 'black')
                
def honour2(sgsim2,DupdateK,nx,ny,nz,N_ens,High_K,Low_K,High_P,Low_p):
    if Yet==0:
    
        sgsim2=cp.asarray(sgsim2)
        DupdateK=cp.asarray(DupdateK)
    else:
        pass
 

    #uniehonour = np.reshape(rossmary,(nx,ny,nz), 'F')
    #unieporohonour = np.reshape(rossmaryporo,(nx,ny,nz), 'F')

    # Read true porosity well values

    # aa = np.zeros((nz))
    # bb = np.zeros((nz))
    # cc = np.zeros((nz))
    # dd = np.zeros((nz))
    # ee = np.zeros((nz))
    # ff = np.zeros((nz))
    # gg = np.zeros((nz))
    # hh = np.zeros((nz))

    # aa1 = np.zeros((nz))
    # bb1 = np.zeros((nz))
    # cc1 = np.zeros((nz))
    # dd1 = np.zeros((nz))
    # ee1 = np.zeros((nz))
    # ff1 = np.zeros((nz))
    # gg1 = np.zeros((nz))
    # hh1 = np.zeros((nz))
    
    # Read true porosity well values
    """
    for j in range(nz):
        aa[j] = uniehonour[1,24,j]
        bb[j] = uniehonour[1,1,j]
        cc[j] = uniehonour[31,0,j]
        dd[j] = uniehonour[31,31,j]
        ee[j] = uniehonour[7,9,j]
        ff[j] = uniehonour[14,12,j]
        gg[j] = uniehonour[28,19,j]
        hh[j] = uniehonour[14,27,j]

        aa1[j] = unieporohonour[1,24,j]
        bb1[j] = unieporohonour[1,1,j]
        cc1[j] = unieporohonour[31,0,j]
        dd1[j] = unieporohonour[31,31,j]
        ee1[j] = unieporohonour[7,9,j]
        ff1[j] = unieporohonour[14,12,j]
        gg1[j] = unieporohonour[28,19,j]
        hh1[j] = unieporohonour[14,27,j]
    """
    # Read permeability ensemble after EnKF update
    # A = DupdateK        
    # C = sgsim2


    output = cp.zeros((nx*ny*nz,N_ens))
    outputporo = cp.zeros((nx*ny*nz,N_ens))
    """
    for j in range(N_ens):
        B = np.reshape(A[:,j],(nx,ny,nz),'F')
        D = np.reshape(C[:,j],(nx,ny,nz),'F')
    
        for jj in range(nz):
            B[1,24,jj] = aa[jj]
            B[1,1,jj] = bb[jj]
            B[31,0,jj] = cc[jj]
            B[31,31,jj] = dd[jj]
            B[7,9,jj] = ee[jj]
            B[14,12,jj] = ff[jj]
            B[28,19,jj] = gg[jj]
            B[14,27,jj] = hh[jj]

            D[1,24,jj] = aa1[jj]
            D[1,1,jj] = bb1[jj]
            D[31,0,jj] = cc1[jj]
            D[31,31,jj] = dd1[jj]
            D[7,9,jj] = ee1[jj]
            D[14,12,jj] = ff1[jj]
            D[28,19,jj] = gg1[jj]
            D[14,27,jj] = hh1[jj]
        
        output[:,j:j+1] = np.reshape(B,(nx*ny*nz,1), 'F')
        outputporo[:,j:j+1] = np.reshape(D,(nx*ny*nz,1), 'F')
    """
    output=DupdateK
    outputporo=sgsim2

                                 
                                 
    output[output >= High_K] = High_K     # highest value in true permeability
    output[output <= Low_K] = Low_K

    outputporo[outputporo >= High_P] = High_P
    outputporo[outputporo <= Low_P] = Low_P                            
    if Yet==0:
        return cp.asnumpy(output),cp.asnumpy(outputporo)
    else:
        return output,outputporo

def pinvmatt(A,tol = 0):
    """
    

    Parameters
    ----------
    A : Input Matrix to invert

    tol : Tolerance level : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    V : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    U : TYPE
        DESCRIPTION.

    """
    V,S1,U = cp.linalg.svd(A,full_matrices=0)

    # Calculate the default value for tolerance if no tolerance is specified
    if tol == 0:
        tol = cp.amax((A.size)*cp.spacing(cp.linalg.norm(S1,cp.inf)))  
    
    r1 = sum(S1 > tol)+1
    v = V[:,:r1-1]
    U1 = U.T
    u = U1[:,:r1-1]
    S11 = S1[:r1-1]
    s = S11[:]
    S = 1/s[:]
    X = (u*S.T).dot(v.T)

    return (V,X,U)

def funcGetDataMismatch(simData,measurment):
    """
    

    Parameters
    ----------
    simData : Simulated data

    measurment : True Measurement
        DESCRIPTION.

    Returns
    -------
    obj : Root mean squared error

    objStd : standard deviation

    objReal : Mean

    """

    ne=simData.shape[1]

    objReal=np.zeros((ne,1))
    for j in range(ne):
        objReal[j]=(np.sum((((simData[:,j]) - measurment) ** 2)) )  **(0.5)/\
            (measurment.shape[0])
    obj=np.mean(objReal)

    objStd=np.std(objReal)
    return obj,objStd,objReal


def Narrow_Band(M1,Nx,Ny,Nz):
    y=np.reshape(M1,(Nx,Ny,Nz),'F')

    accepted = []
    for kk in range(Nz):
        yuse=y[:,:,kk]
        yuse=np.reshape(yuse,(-1,1),'F')
        kmeans =MiniBatchKMeans(n_clusters=Experts,max_iter=2000).fit(yuse)    
        dd=kmeans.labels_
        dd=dd.T
        dd=np.reshape(dd,(-1,1))
        mudd=np.reshape(dd,(Nx,Ny),'F') 
        mudd1=mudd.astype(np.int8)  
        aa=find_boundaries((mudd1))         
        aa = aa*1
        aa=np.reshape(aa,(-1,1),'F')
        aa=aa.astype(np.float) 
        accepted.append(aa)
    return np.array(accepted)

        
def get_clus(y,Experts):
    
    """
    Parameters
    ----------
    y : Permeability field as a vector

    Experts : Number of Facies


    Returns
    -------
    sdi : Signed Distance transformation
    
    mexx : Cluster mean

    """
    y=np.reshape(y,(-1,1),'F')
    kmeans =MiniBatchKMeans(n_clusters=Experts,max_iter=2000).fit(y)
    
    dd=kmeans.labels_
    dd=dd.T
    dd=np.reshape(dd,(-1,1))
    kk=np.zeros((dd.shape[0],1))
    d=dd
    mexx=kmeans.cluster_centers_
    for i in range (d.shape[0]):
        for j in range (Experts):
             if d[i,:]==j: 
                 kk[i,:]= np.abs(np.random.normal(mexx[j,:],0.001,1))

    sdi=kk
    
    return sdi,mexx,kmeans


def pinvmat(A,tol = 0):
    """
    

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    tol : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    X : TYPE
        DESCRIPTION.

    """
    V,S1,U = cp.linalg.svd(A,full_matrices=0)

    # Calculate the default value for tolerance if no tolerance is specified
    if tol == 0:
        tol = cp.amax((A.size)*cp.spacing(cp.linalg.norm(S1,cp.inf)))  
    
    r1 = cp.sum(S1 > tol)+1
    v = V[:,:r1-1]
    U1 = U.T
    u = U1[:,:r1-1]
    S11 = S1[:r1-1]
    s = S11[:]
    S = 1/s[:]
    X = (u*S).dot(v.T)
    return X


def transistion_model(mu,theta_old,sigma,beta):
    
    mu=cp.asarray(mu)
    theta_old=cp.asarray(theta_old)
    betaa=cp.asarray(beta)
    sigmaa=cp.asarray(sigma)
    
    eta=cp.zeros((1,nx*ny*nz))
    eta=cp.random.multivariate_normal(cp.ravel(eta), sigmaa,1)
    eta=cp.reshape(eta,(1,-1))
    term_1=(1-(betaa**2))**(0.5)
    term_2=theta_old-mu
    term_3=betaa*eta
    term_4=mu
    yess=(term_1*term_2)+term_3+term_4
    theta_proposed=yess
    if Yet==0:
        return cp.asnumpy(theta_proposed)
    else:
        return theta_proposed

def prior(x):
    #x[0] = mu, x[1]=sigma (new or current)
    #returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    #returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    #It makes the new sigma infinitely unlikely.
    # if(x <=0):
    #     return 0
    return 0.0

def log_like_normal(x,data):
    #x[0]=mu, x[1]=sigma (new or current)
    #data = the observation
    xin=np.reshape(x,(-1,1),'F')
    
    xinp=machine_map.predict(xin)
    
    print('6X Reservoir Simulator Forwarding -  MCMC Model')
    Forward_model(oldfolder,'MCMC_Model',xin,xinp)
    yycheck=Get_RSM(oldfolder,'MCMC_Model')     
    usethis=yycheck
    usesim=usethis[:,1:]    
    yycheck=np.reshape(usesim,(-1,1),'F')
    yycheck=cp.asarray(yycheck)
    data=cp.asarray(data)
    
    return cp.asnumpy(cp.sum(-cp.log(1 * cp.sqrt(2* cp.pi) )-((data-\
                                                    (yycheck))**2) \
                  / (2*1**2)))


def log_prior(theta):
    # m, b, log_f = theta
    if np.sum(theta,0)<nx*ny*nz:
        return 0.0
    return 0.0 #-np.inf



def metropolis_hastings(likelihood_computer,prior, \
                        transition_model, param_init,\
                        iterations,data,acceptance_rule,meann,sigma):
    x = param_init
    accepted = []
    rejected = [] 
    beta = random()
    for i in range(iterations):
        print( str(i+1) + ' | ' + str(iterations))
        x_new =  transistion_model(meann,x,sigma,beta)#transition_model(x)    
        x_lik = likelihood_computer(x,data)
        x_new_lik = likelihood_computer(x_new,data) 
        if (acceptance_rule(x_lik + np.log(prior(x)),x_new_lik+\
                            np.log(prior(x_new)))):            
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)            
                
    return np.array(accepted), np.array(rejected)

def objective_Na(Xin,alphe,alpha1):
    
    top=1-((1/Xin)**(alphe-1))
    bottom=1-(1/Xin)    
    ypred=top/bottom
    ytruee=alpha1
    ytruee = np.reshape(ytruee, (-1, 1))
    aspit=((np.sum(((( ypred) - ytruee) ** 2)) )  **(0.5)) 
    return aspit

def intial_ensemble(Nx,Ny,Nz,N,permx):
    #for i in range(N):
    O=mps.mpslib(method='mps_snesim_tree')
    O.par['n_real']=Nz*N
    k=permx# permeability field TI
    kjenn=k
    O.ti=kjenn
    O.par['simulation_grid_size']=(Ny,Nx,1)
    O.run()
    k=np.genfromtxt("ti.dat.gslib",skip_header=((Nz*N)+2), dtype='float')
    kreali=k
        #Ini_ensemble.append(kreali)
    return kreali

def initial_ensemble_gaussian(Nx,Ny,Nz,N,minn,maxx):
    shape = (Nx, Ny)
    
    fensemble=np.zeros((Nx*Ny*Nz,N))
    for k in range(N):
        fout=[]
        for j in range(Nz):
            field = generate_field(distrib, Pkgen(3), shape)
            field = imresize(field, output_shape=shape)
            foo=np.reshape(field,(-1,1),'F')
            fout.append(foo)
        fout=np.vstack(fout)
        clfy = MinMaxScaler(feature_range=(minn, maxx))
        (clfy.fit(fout))    
        fout=(clfy.transform(fout)) 
        fensemble[:,k]=np.ravel(fout)
    return fensemble


#Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    xx=cp.asarray(x)
    x_newx=cp.asarray(x_new)
    if x_newx>xx:
        return True
    else:
        accept=cp.random.uniform(0,1)
        if Yet==0:
            return cp.asnumpy((accept < (cp.exp(x_newx-xx))))
        else:
            return (accept < (cp.exp(x_newx-xx)))

def Plot_mean(permbest,permmean,iniperm,nx,ny):
    # permbest=np.log10(permbest)
    # permmean=np.log10(permmean)
    # iniperm=np.log10(iniperm)
    
    permmean=(np.reshape(permmean,(nx,ny,nz),'F'))
    permbest=(np.reshape(permbest,(nx,ny,nz),'F'))
    iniperm=(np.reshape(iniperm,(nx,ny,nz),'F'))
    XX, YY = np.meshgrid(np.arange(nx),np.arange(ny))  
    
    plt.figure(figsize=(10, 10))


    plt.subplot(3, 3, 2)
    plt.pcolormesh(XX.T,YY.T,permmean[:,:,0],cmap = 'jet')
    plt.title('mean-Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(Low_K,High_K)
    
    plt.subplot(3, 3, 5)
    plt.pcolormesh(XX.T,YY.T,permmean[:,:,1],cmap = 'jet')
    plt.title('mean-Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(Low_K,High_K)    
    
    plt.subplot(3, 3, 8)
    plt.pcolormesh(XX.T,YY.T,permmean[:,:,2],cmap = 'jet')
    plt.title('mean-Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(Low_K,High_K)    
    

    plt.subplot(3, 3, 3)
    plt.pcolormesh(XX.T,YY.T,permbest[:,:,0],cmap = 'jet')
    plt.title('Best Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(Low_K,High_K)
    
    plt.subplot(3, 3, 6)
    plt.pcolormesh(XX.T,YY.T,permbest[:,:,1],cmap = 'jet')
    plt.title('Best Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(Low_K,High_K) 
    
    plt.subplot(3, 3, 9)
    plt.pcolormesh(XX.T,YY.T,permbest[:,:,2],cmap = 'jet')
    plt.title('Best Layer 3', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(Low_K,High_K)    
    
    
    plt.subplot(3, 3, 1)
    plt.pcolormesh(XX.T,YY.T,iniperm[:,:,0],cmap = 'jet')
    plt.title('initial -Layer 1', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(Low_K,High_K) 
    
    plt.subplot(3, 3, 4)
    plt.pcolormesh(XX.T,YY.T,iniperm[:,:,1],cmap = 'jet')
    plt.title('initial -Layer 2', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(Low_K,High_K)

    plt.subplot(3, 3, 7)
    plt.pcolormesh(XX.T,YY.T,iniperm[:,:,2],cmap = 'jet')
    plt.title('initial -Layer 3', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    plt.axis([0,(nx - 1),0,(ny-1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel('Log K (mD)',fontsize = 13)
    plt.clim(Low_K,High_K) 
        
    plt.tight_layout(rect = [0,0,1,0.95])
    plt.suptitle('Permeability comparison',\
                 fontsize = 25)
    plt.savefig("Comparison.png")
    plt.close()
    plt.clf()



class struct:
    "A structure that can have any fields defined."
    def __init__(self, **entries): self.__dict__.update(entries)

def fread(fid, nelements, dtype):

    """Equivalent to Matlab fread function"""

    if dtype is np.str:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype

    data_array = np.fromfile(fid, dt, nelements)
    data_array.shape = (nelements, 1)

    return data_array

def rectifyy(perm,ls,mexx,Experts,Nx,Ny,Nz,modelz):
    
    """    
    rectify conlifct between pixel and level set update
    Parameters
    ----------
    perm : Pixel field

    ls : Level set field

    mexx : Clsuster means for Facies

    Experts : Number of Facies

    Nx : Dimension in X axis

    Ny : Dimension in y axis

    Nz : Dimesnion in z axis


    Returns
    -------
    valuee : Corrected pixel field

    """
    perm=np.reshape(perm,(-1,1))
    mexx=np.reshape(mexx,(-1,1))
    kmeans =MiniBatchKMeans(n_clusters=Experts,max_iter=2000).fit(perm)
    #ddp=modelz.predict(perm)
    ddp=np.reshape(kmeans.labels_.T, (-1,1),'F') 
    ddp=np.reshape(ddp, (-1,1),'F') 
    ls=np.reshape(ls,(-1,1))
    kmeansp =MiniBatchKMeans(n_clusters=Experts,max_iter=2000).fit(ls)
    #dd=modelz.predict(ls)
    dd=np.reshape(kmeansp.labels_.T, (-1,1),'F')    
    dd=np.reshape(dd, (-1,1),'F') 
    valls=Nx*Ny*Nz
    valuee=np.zeros((valls,1))
    count=Nx*Ny*Nz
    for i in range (count):
        if ddp[i,:]==dd[i,:]:
            valuee[i,:]=perm[i,:]

        
        if ddp[i,:]!=dd[i,:]:
            jdx=np.ravel(mexx[dd[i,:],:])
            a=np.abs(np.random.normal(jdx,0.001,1))            
            valuee[i,:]=np.ravel(a)# np.abs(np.random.normal(mexx[dd[i,:],:],0.001,1))   #np.abs(np.random.normal(mexx[dd[i,:],:],0.01,1))

    return valuee

def Get_RSM_ensemble(oldfolder,stringf,jj):
    folder=stringf + str(jj)
    predMatrix=Get_RSM(oldfolder,folder)
    return predMatrix



def Get_RSM(oldfolder,folder):
    #folder='True_Model'
    os.chdir(folder)
    unsmry_file = "masterreal"
    parser = binary_parser.EclBinaryParser(unsmry_file)
    vectors = parser.read_vectors()
    clement5=vectors['WBHP']
    wbhp=clement5.values[1:,:] #8 + 8  producers
    df=pd.read_csv('masterreal.csv')
    Timeuse=df.values[8:,[0]].astype(np.float32)
    Timeuse=Timeuse[1:,:]
    #measurement=np.hstack([Timeuse,wbhp,wopr,wwpr,wwct,wgpr]) #Measurement
    measurement=np.hstack([Timeuse,wbhp]) #Measurement
    #measurement=np.hstack([Timeuse,wbhp,wopr,wwpr,wwct]) #Measurement
    os.chdir(oldfolder)
    return measurement


def Getporosity_ensemble(ini_ensemble,machine_map,N_ens):
    
    
    ini_ensemblep=[]
    for ja in range(N_ens):
        usek=np.reshape(ini_ensemble[:,ja],(-1,1),'F')
        porr=machine_map.predict(usek)
        ini_ensemblep.append(porr)
        
    ini_ensemblep=np.hstack(ini_ensemblep)
    return ini_ensemblep  

def Plot_RSM_percentile(pertoutt,CMens,modelError,True_mat,Basematrix,Namesz):

    timezz=True_mat[:,0].reshape(-1,1)
    
    if modelError==1:    
        P10=pertoutt[0]
        P50=pertoutt[1]
        P90=pertoutt[2]        
        sidde=True_mat.shape[1]-1  
        aa=np.atleast_2d(0)
        #CM10=np.hstack([aa,np.reshape(CMens[:,0],(-1,sidde),'F')])
        
        CM10=np.reshape(CMens[:,0],(-1,1),'F')
        a11=np.hstack([aa,np.reshape(CM10[:,0],(-1,sidde),'F')])
     
        
        CM50=np.reshape(CMens[:,1],(-1,1),'F')
        a12=np.hstack([aa,np.reshape(CM50[:,0],(-1,sidde),'F')])
       
        CM90=np.reshape(CMens[:,2],(-1,1),'F')
        a13=np.hstack([aa,np.reshape(CM90[:,0],(-1,sidde),'F')])
      
        
        result10 = np.zeros((True_mat.shape[0],True_mat.shape[1]))
        result50 = np.zeros((True_mat.shape[0],True_mat.shape[1]))
        result90 = np.zeros((True_mat.shape[0],True_mat.shape[1]))
        for ii in range(True_mat.shape[0]):
            aa1=P10[ii,:]+a11

            ress=aa1#+aa2+aa3+aa4+aa5
            ress[:,0]=P10[ii,0]
            result10[ii,:]=ress
            
            aa1=P50[ii,:]+a12

            ress=aa1#+aa2+aa3+aa4+aa5
            ress[:,0]=P50[ii,0]            
            result50[ii,:]=ress
            
            aa1=P90[ii,:]+a13

            ress=aa1#+aa2+aa3+aa4+aa5
            ress[:,0]=P90[ii,0]                        
            result90[ii,:]=ress
            
            
            
        P10=result10
        P50=result50
        P90=result90
    else:   
        P10=pertoutt[0]
        P50=pertoutt[1]
        P90=pertoutt[2]

    plt.figure(figsize=(40, 40))        

    plt.subplot(4,4,1)
    plt.plot(timezz,True_mat[:,1], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,1], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,1], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,1], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,1], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I0',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.legend()
    
    plt.subplot(4,4,2)
    plt.plot(timezz,True_mat[:,2], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,2], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,2], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,2], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,2], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I1',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.legend()

    plt.subplot(4,4,3)
    plt.plot(timezz,True_mat[:,3], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,3], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,3], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,3], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,3], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I10',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.legend()
    
    plt.subplot(4,4,4)
    plt.plot(timezz,True_mat[:,4], color = 'red', lw = '2',\
             label ='model') 
    plt.plot(timezz,P10[:,4], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,4], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,4], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,4], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I13',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    plt.legend()
    

    plt.subplot(4,4,5)
    plt.plot(timezz,True_mat[:,5], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,5], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,5], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,5], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,5], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I15',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    plt.legend()
    
    plt.subplot(4,4,6)
    plt.plot(timezz,True_mat[:,6], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,6], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,6], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,6], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,6], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I3',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    plt.legend()

    plt.subplot(4,4,7)
    plt.plot(timezz,True_mat[:,7], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,7], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,7], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,7], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,7], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I4',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)  
    plt.legend()
    
    plt.subplot(4,4,8)
    plt.plot(timezz,True_mat[:,8], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,8], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,8], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,8], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,8], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I6',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    plt.legend()
    
    plt.subplot(4,4,9)
    plt.plot(timezz,True_mat[:,9], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,9], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,9], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,9], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,9], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P11',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)    
    plt.legend()
    
    plt.subplot(4,4,10)
    plt.plot(timezz,True_mat[:,10], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,10], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,10], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,10], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,10], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P12',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)  
    plt.legend()
    
    plt.subplot(4,4,11)
    plt.plot(timezz,True_mat[:,11], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,11], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,11], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,11], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,11], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P14',fontsize = 13)

    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    plt.legend()


    plt.subplot(4,4,12)
    plt.plot(timezz,True_mat[:,12], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,12], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,12], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,12], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,12], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P2',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)     
    plt.legend()

    plt.subplot(4,4,13)
    plt.plot(timezz,True_mat[:,13], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,13], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,13], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,13], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,13], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P5',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    plt.legend()

    plt.subplot(4,4,14)
    plt.plot(timezz,True_mat[:,14], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,14], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,14], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,14], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,14], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P7',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    plt.legend()
    
    plt.subplot(4,4,15)
    plt.plot(timezz,True_mat[:,15], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,15], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,15], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,15], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,15], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P8',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    plt.legend()

    plt.subplot(4,4,16)
    plt.plot(timezz,True_mat[:,16], color = 'red', lw = '2',\
             label ='model')
    plt.plot(timezz,P10[:,16], color = 'blue', lw = '2', \
             label ='P10 Model')
    plt.plot(timezz,P50[:,16], color = 'c', lw = '2', \
             label ='P50 Model')
    plt.plot(timezz,P90[:,16], color = 'green', lw = '2', \
         label ='P90 Model')
    plt.plot(timezz,Basematrix[:,16], color = 'black', lw = '2', \
             label ='Base Model')         
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P9',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)     
    plt.legend()
    os.chdir('PERCENTILE')     
    plt.savefig(Namesz)          # save as png                                  # preventing the figures from showing
    os.chdir(oldfolder)
    plt.clf()
    plt.close()  




def Plot_RSM_single(TrueMatrix,modelEror,CM,Namesz,tag):

    timezz=TrueMatrix[:,0].reshape(-1,1)
    
    if modelError==1:
        sidde=TrueMatrix.shape[1]-1  
        aa=np.atleast_2d(0)
        CM10=np.reshape(CM,(-1,1),'F')
        a11=np.hstack([aa,np.reshape(CM10[:,0],(-1,sidde),'F')])
        
        result10 = np.zeros((TrueMatrix.shape[0],TrueMatrix.shape[1]))
       # print(result10.shape)

        for ii in range(TrueMatrix.shape[0]):
            if (tag =='True_Model') or (tag=='Base_Model'):
                ress=TrueMatrix[ii,:]
                result10[ii,:]=ress                
            else:
                aa1=TrueMatrix[ii,:]+a11
                ress=aa1#+aa2+aa3+aa4+aa5
                ress[:,0]=TrueMatrix[ii,0]            
                result10[ii,:]=ress                


        TrueMatrix=result10
    else:
        pass

    plt.figure(figsize=(40, 40))

    plt.subplot(4,4,1)
    plt.plot(timezz,TrueMatrix[:,1], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I0',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    
    plt.subplot(4,4,2)
    plt.plot(timezz,TrueMatrix[:,2], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I1',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)


    plt.subplot(4,4,3)
    plt.plot(timezz,TrueMatrix[:,3], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I10',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    
    plt.subplot(4,4,4)
    plt.plot(timezz,TrueMatrix[:,4], color = 'red', lw = '2',\
             label ='model')  
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I13',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    

    plt.subplot(4,4,5)
    plt.plot(timezz,TrueMatrix[:,5], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I15',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    
    
    plt.subplot(4,4,6)
    plt.plot(timezz,TrueMatrix[:,6], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I3',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)

    plt.subplot(4,4,7)
    plt.plot(timezz,TrueMatrix[:,7], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I4',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)  
    
    
    plt.subplot(4,4,8)
    plt.plot(timezz,TrueMatrix[:,8], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I6',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    
    
    plt.subplot(4,4,9)
    plt.plot(timezz,TrueMatrix[:,9], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P11',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)    
    
    
    plt.subplot(4,4,10)
    plt.plot(timezz,TrueMatrix[:,10], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P12',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)  
    
    
    plt.subplot(4,4,11)
    plt.plot(timezz,TrueMatrix[:,11], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P14',fontsize = 13)

    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 



    plt.subplot(4,4,12)
    plt.plot(timezz,TrueMatrix[:,12], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P2',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)     
    

    plt.subplot(4,4,13)
    plt.plot(timezz,TrueMatrix[:,13], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P5',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 


    plt.subplot(4,4,14)
    plt.plot(timezz,TrueMatrix[:,14], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P7',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    
    
    plt.subplot(4,4,15)
    plt.plot(timezz,TrueMatrix[:,15], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P8',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 


    plt.subplot(4,4,16)
    plt.plot(timezz,TrueMatrix[:,16], color = 'red', lw = '2',\
             label ='model')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P9',fontsize = 13)
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)     
    

    plt.savefig(Namesz)          # save as png                                  # preventing the figures from showing
    plt.clf()
    plt.close()  

def Plot_RSM(predMatrix,modelError,CM,N,True_mat,Namesz):
    #True_mat=TrueMatrix
    Nt=predMatrix[0].shape[0]
    timezz=predMatrix[0][:,0].reshape(-1,1)
     
    BHPA = np.zeros((Nt,N))
    BHPB = np.zeros((Nt,N))
    BHPC = np.zeros((Nt,N))
    BHPD = np.zeros((Nt,N))
    BHPE = np.zeros((Nt,N))
    BHPF = np.zeros((Nt,N))
    BHPG = np.zeros((Nt,N))
    BHPH = np.zeros((Nt,N)) 
    BHPI = np.zeros((Nt,N))
    BHPJ = np.zeros((Nt,N))
    BHPK = np.zeros((Nt,N))    
    BHPL = np.zeros((Nt,N))  
    BHPM = np.zeros((Nt,N)) 
    BHPN = np.zeros((Nt,N)) 
    BHPO = np.zeros((Nt,N))
    BHPP = np.zeros((Nt,N))    

        
    for i in range(N):
       usef=predMatrix[i]
       if modelError==1:
           sidde=usef.shape[1]-1 
           aa=np.atleast_2d(0)
           #CM10=np.hstack([aa,np.reshape(CM[:,i],(-1,sidde),'F')]) 
           CM10=np.reshape(CM[:,i],(-1,1),'F')
           a11=np.hstack([aa,np.reshape(CM10[:,0],(-1,sidde),'F')])

           
           result10 = np.zeros((usef.shape[0],usef.shape[1]))
           for ii in range(usef.shape[0]):
               
               aa1=usef[ii,:]+a11

               ress=aa1#+aa2+aa3+aa4+aa5
               ress[:,0]=usef[ii,0]                                
               result10[ii,:]=ress
    
           usef=result10 
           
       else:
           pass
        
        
       BHPA[:,i]=usef[:,1]
       BHPB[:,i]=usef[:,2]
       BHPC[:,i]=usef[:,3]
       BHPD[:,i]=usef[:,4]
       BHPE[:,i]=usef[:,5]
       BHPF[:,i]=usef[:,6]
       BHPG[:,i]=usef[:,7]
       BHPH[:,i]=usef[:,8]
       BHPI[:,i]=usef[:,9]
       BHPJ[:,i]=usef[:,10]
       BHPK[:,i]=usef[:,11]
       BHPL[:,i]=usef[:,12]
       BHPM[:,i]=usef[:,13]
       BHPN[:,i]=usef[:,14]
       BHPO[:,i]=usef[:,15]
       BHPP[:,i]=usef[:,16]


    plt.figure(figsize=(40, 40))

    plt.subplot(4,4,1)
    plt.plot(timezz,BHPA[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I0',fontsize = 13)

    plt.plot(timezz,True_mat[:,1], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    #plt.legend()
    
    plt.subplot(4,4,2)
    plt.plot(timezz,BHPB[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I1',fontsize = 13)
    

    plt.plot(timezz,True_mat[:,2], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    #plt.legend()

    plt.subplot(4,4,3)
    plt.plot(timezz,BHPC[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I10',fontsize = 13)

    plt.plot(timezz,True_mat[:,3], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    #plt.legend()
    
    plt.subplot(4,4,4)
    plt.plot(timezz,BHPD[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I13',fontsize = 13)

    plt.plot(timezz,True_mat[:,4], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    #plt.legend()

    plt.subplot(4,4,5)
    plt.plot(timezz,BHPE[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I15',fontsize = 13)

    plt.plot(timezz,True_mat[:,5], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    #plt.legend()
    
    plt.subplot(4,4,6)
    plt.plot(timezz,BHPF[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I3',fontsize = 13)

    plt.plot(timezz,True_mat[:,6], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)
    #plt.legend()
    
    plt.subplot(4,4,7)
    plt.plot(timezz,BHPG[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I4',fontsize = 13)

    plt.plot(timezz,True_mat[:,7], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)  
    #plt.legend()
    
    plt.subplot(4,4,8)
    plt.plot(timezz,BHPH[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('I6',fontsize = 13)

    plt.plot(timezz,True_mat[:,8], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    #plt.legend()
    
    plt.subplot(4,4,9)
    plt.plot(timezz,BHPI[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P11',fontsize = 13)

    plt.plot(timezz,True_mat[:,9], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)    
    #plt.legend()
    
    plt.subplot(4,4,10)
    plt.plot(timezz,BHPJ[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P12',fontsize = 13)

    plt.plot(timezz,True_mat[:,10], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)  
    #plt.legend()
    
    plt.subplot(4,4,11)
    plt.plot(timezz,BHPK[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P14',fontsize = 13)

    plt.plot(timezz,True_mat[:,11], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    #plt.legend()


    plt.subplot(4,4,12)
    plt.plot(timezz,BHPL[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P2',fontsize = 13)

    plt.plot(timezz,True_mat[:,12], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)     
    #plt.legend()

    plt.subplot(4,4,13)
    plt.plot(timezz,BHPM[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P5',fontsize = 13)

    plt.plot(timezz,True_mat[:,13], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    #plt.legend()

    plt.subplot(4,4,14)
    plt.plot(timezz,BHPN[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P7',fontsize = 13)

    plt.plot(timezz,True_mat[:,14], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    #plt.legend()
    
    plt.subplot(4,4,15)
    plt.plot(timezz,BHPO[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P8',fontsize = 13)

    plt.plot(timezz,True_mat[:,15], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0) 
    #plt.legend()

    plt.subplot(4,4,16)
    plt.plot(timezz,BHPP[:,:N],color = 'c', lw = '2',\
             label = 'Realisations')
    plt.xlabel('Time (days)',fontsize = 13)
    plt.ylabel('BHP(Psia)',fontsize = 13)
    #plt.ylim((0,25000))
    plt.title('P9',fontsize = 13)

    plt.plot(timezz,True_mat[:,16], color = 'red', lw = '2',\
             label ='True model')
    plt.axvline(x = 600, color = 'black', linestyle = '--')
    handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this 
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin = 0)
    plt.xlim(xmin = 0)     

    plt.savefig(Namesz)          # save as png                                  # preventing the figures from showing
    plt.clf()
    plt.close()    # clears the figure
    
    
def Get_Latent(ini_ensemble,N_ens,nx,ny,nz):
    X_unie=np.zeros((N_ens,nx,ny,nz))   
    for i in range(N_ens):
        X_unie[i,:,:,:]=np.reshape(ini_ensemble[:,i],(nx,ny,nz),'F')   
    ax=X_unie/High_K   
    ouut=np.zeros((20*20*4,Ne))
    decoded_imgs2=(load_model('encoder.h5').predict(ax))   
    for i in range(N_ens):        
        jj=decoded_imgs2[i].reshape(20,20,4)
        jj=np.reshape(jj,(-1,1),'F')
        ouut[:,i]=np.ravel(jj)             
    return ouut

def Recover_image(x,Ne,nx,ny,nz):
    
    X_unie=np.zeros((Ne,20,20,4))   
    for i in range(Ne):
        X_unie[i,:,:,:]=np.reshape(x[:,i],(20,20,4),'F')     
    decoded_imgs2=(load_model('decoder.h5').predict(X_unie))*High_K
    #print(decoded_imgs2.shape)
    ouut=np.zeros((nx*ny*nz,Ne))
    for i in range(Ne):
        jj=decoded_imgs2[i].reshape(nx,ny,nz)
        jj=np.reshape(jj,(-1,1),'F')
        ouut[:,i]=np.ravel(jj)   
    return ouut    

def use_denoising(ensemble,nx,ny,nz,N_ens):
    X_unie=np.zeros((N_ens,nx,ny,nz))
    for i in range(N_ens):
        X_unie[i,:,:,:]=np.reshape(ensemble[:,i],(nx,ny,nz),'F') 
    ax=X_unie/High_K
    ouut=np.zeros((nx*ny*nz,Ne))
    decoded_imgs2=(load_model('denosingautoencoder.h5').predict(ax)) *High_K  
    for i in range(N_ens):        
        jj=decoded_imgs2[i].reshape(nx,ny,nz)
        jj=np.reshape(jj,(-1,1),'F')
        ouut[:,i]=np.ravel(jj)        
    return ouut  

def DenosingAutoencoder(nx,ny,nz):
    """
    Trains  Denosing Autoencoder

    Parameters
    ----------
    nx : TYPE
        DESCRIPTION.
    ny : TYPE
        DESCRIPTION.
    nz : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    filename="Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensemble=mat['Z']
    X_unie=np.zeros((N_ens,nx,ny,nz))   
    for i in range(N_ens):
        X_unie[i,:,:,:]=np.reshape(ini_ensemble[:,i],(nx,ny,nz),'F')   
    
    #x_train=X_unie/500
    ax=X_unie/High_K
    
    x_train, x_test, y_train, y_test = train_test_split(
    ax, ax, test_size=0.1, random_state=42)
    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, \
                                            scale=1.0, size=x_train.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, \
                        scale=1.0, size=x_test.shape)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    
    
    #nx,ny,nz=40,40,3
    input_image = Input(shape=(nx,ny,nz))
    x = Conv2D(32,(3,3),activation='relu',padding='same')(input_image)
    x = MaxPooling2D((1,1),padding='same')(x)
    x = Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D((1,1),padding='same')(x)
    x = Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D((1,1),padding='same')(x)
    encoded = MaxPooling2D((2,2),padding='same')(x)#reduces it by this value
    
    encoder = Model(input_image, encoded)
    encoder.summary()
    
    decoder_input= Input(shape=(20,20,32))
    x = Conv2D(32,(3,3),activation='relu',padding='same')(decoder_input)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D((1,1))(x)
    x = Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D((1,1))(x)
    decoded = Conv2D(3,(3,3),activation='sigmoid',padding='same')(x)
    
    decoder = Model(decoder_input, decoded)
    decoder.summary()
    
    autoencoder_input = Input(shape=(nx,ny,nz))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(autoencoder_input, decoded)
    autoencoder.summary()
    
    autoencoder.compile(optimizer='adam',loss='mse')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint("denosingautoencoder.h5", monitor='val_loss', mode='min', \
                         verbose=1, save_best_only=True)    
    
    autoencoder.fit(x_train_noisy, x_train,
      epochs=1000,
      batch_size=128,
      shuffle=True,validation_data=(x_test_noisy,x_test),callbacks=[es,mc])       
 

def Autoencoder2(nx,ny,nz):
    """
    Trains  Denosing Autoencoder

    Parameters
    ----------
    nx : TYPE
        DESCRIPTION.
    ny : TYPE
        DESCRIPTION.
    nz : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    filename="Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensemble=mat['Z']
    X_unie=np.zeros((N_ens,nx,ny,nz))   
    for i in range(N_ens):
        X_unie[i,:,:,:]=np.reshape(ini_ensemble[:,i],(nx,ny,nz),'F')   

    ax=X_unie/High_K
    
    x_train, x_test, y_train, y_test = train_test_split(
    ax, ax, test_size=0.1, random_state=42)

    
    #nx,ny,nz=40,40,3
    input_image = Input(shape=(nx,ny,nz))
    x = Conv2D(4,(3,3),activation='relu',padding='same')(input_image)
    x = MaxPooling2D((1,1),padding='same')(x)
    x = Conv2D(4,(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D((1,1),padding='same')(x)
    x = Conv2D(4,(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D((1,1),padding='same')(x)
    encoded = MaxPooling2D((2,2),padding='same')(x)#reduces it by this value
    
    encoder = Model(input_image, encoded)
    encoder.summary()
    
    decoder_input= Input(shape=(20,20,4))
    x = Conv2D(4,(3,3),activation='relu',padding='same')(decoder_input)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(4,(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D((1,1))(x)
    x = Conv2D(4,(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D((1,1))(x)
    decoded = Conv2D(3,(3,3),activation='sigmoid',padding='same')(x)
    
    decoder = Model(decoder_input, decoded)
    decoder.summary()
    
    autoencoder_input = Input(shape=(nx,ny,nz))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(autoencoder_input, decoded)
    autoencoder.summary()
    
    autoencoder.compile(optimizer='adam',loss='mse')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint("autoencoder.h5", monitor='val_loss', mode='min', \
                         verbose=1, save_best_only=True)    
    
    autoencoder.fit(x_train, x_train,
      epochs=1000,
      batch_size=128,
      shuffle=True,validation_data=(x_test,x_test),callbacks=[es,mc])
    
    encoder.save("encoder.h5")
    decoder.save("decoder.h5")      
    
    

def Autoencoder(nx,ny,nz):
    encoding_dim = 500
    image_dim = nx*ny*nz
    
    #this is our input placeholder
    input_img = Input(shape=(image_dim,))
    #encoded is a encoded representation of the input
    
    encoded = Dense(5500, activation='relu')(input_img)
    encoded = Dense(2500, activation='relu')(encoded)
    encoded = Dense(1000, activation='relu')(encoded)
    encoded = Dense(800, activation='relu')(encoded)
    encoded = Dense(encoding_dim , activation='relu')(encoded)
    #decoded is the lossy reconstruction of the input
    decoded = Dense(800, activation='relu')(encoded)
    decoded = Dense(1400, activation='relu')(decoded)
    decoded = Dense(2500, activation='relu')(decoded)
    decoded = Dense(nx*ny*nz, activation='linear')(decoded)
    
    
    autoencoder = Model(input_img, decoded)
    
    encoder = Model(input_img, encoded)
    
    
    encoded_input = Input(shape=(encoding_dim,))
    
        
    decoder_layer1 = autoencoder.layers[-4]
    decoder_layer2 = autoencoder.layers[-3]
    decoder_layer3 = autoencoder.layers[-2]
    decoder_layer4 = autoencoder.layers[-1]
    
    decoder_layer = decoder_layer4(decoder_layer3(decoder_layer2\
                    (decoder_layer1(encoded_input))))
    
    #create the  decoder model
    decoder = Model(encoded_input, decoder_layer)
    
    #train the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    filename="Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensemble=mat['Z']
    
    X_unie=np.zeros((N_ens,nx,ny,nz))   
    for i in range(N_ens):
        X_unie[i,:,:,:]=np.reshape(ini_ensemble[:,i],(nx,ny,nz),'F')   
    
    #x_train=X_unie/500
    ax=X_unie/High_K
    x_train, x_test, y_train, y_test = train_test_split(
    ax, ax, test_size=0.1, random_state=42)
    
    
    #x_test=X_unie/500
    
    x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test),np.prod(x_train.shape[1:])))

    
    autoencoder.fit(x_train,x_train,epochs=2000,
    	batch_size=128,
    	shuffle=True,
    	validation_data=(x_test,x_test)) 

    
    
    encoder.save("encoder.h5")
    decoder.save("decoder.h5")    
 

def resscale(col, minn, maxx):
    rangee = col.max() - col.min()
    a = (col - col.min()) / rangee
    return a * (maxx - minn) + minn   

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho') 


def dct22(a,Ne,nx,ny,nz,size1,size2):
    ouut=np.zeros((size1*size2*nz,Ne))
    for i in range(Ne):
        origi=np.reshape(ini_ensemble[:,i],(nx,ny,nz),'F') 
        outt=[]
        for j in range(nz):
            mike=origi[:,:,j]
            dctco=dct(dct(mike.T, norm='ortho').T, norm='ortho')
            subb=dctco[:size1,:size2]
            subb=np.reshape(subb,(size1*size2,1),'F') 
            outt.append(subb)
        outt=np.vstack(outt)
        ouut[:,i]=np.ravel(outt)
    return ouut


# implement 2D IDCT
def idct22(a,Ne,nx,ny,nz,size1,size2):
    #a=updated_ensembledct
    ouut=np.zeros((nx*ny*nz,Ne))
    for ix in range(Ne):
        #i=0
        subbj=a[:,ix]
        subbj=np.reshape(subbj,(size1,size2,nz),'F')
        #subb=np.atleast_3d(subb) 
        #print(subbj.shape)
        neww=np.zeros((nx,ny))
        outt=[]
        for jg in range(nz):  
            #j=0
            usee=subbj[:,:,jg]
            neww[:size1,:size2]=usee  
            aa=idct(idct(neww.T, norm='ortho').T, norm='ortho') 
            subbout=np.reshape(aa,(-1,1),'F')
            outt.append(subbout)
        outt=np.vstack(outt)
        ouut[:,jg]=np.ravel(outt)
    return ouut

     

def Get_simulated(predMatrix,modelError,CM,N):
    simmz=[]
    for i in range(N):
        usethis=predMatrix[i]
        if modelError==1:
            usesim=usethis[:,1:]
            CC=np.reshape(CM[:,i],(-1,1),'F')
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')

            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1

                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=usethis[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')
        simmz.append(usesim)
    simmz=np.hstack(simmz)
    return simmz


def Get_simulated_EnKF(predMatrix,modelError,CM,N):
    simmz=[]
    for i in range(N):
        usethis=predMatrix[i]
        if modelError==1:
            usesim=usethis[0,1:]
            CC=np.reshape(CM[:,i],(-1,1),'F')
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')

            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1

                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=usethis[0,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')
        simmz.append(usesim)
    simmz=np.hstack(simmz)
    return simmz

def Get_simulatedTI(predMatrix,modelError,CM,N):
    simmz=[]
    for i in range(N):
        usethis=predMatrix[i]
        if modelError==1:
            usesim=usethis[:,1:]
            #CM10=np.reshape(CM[:,i],(-1,usesim.shape[1]),'F') 
            CC=np.reshape(CM[:,i],(-1,1),'F')
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')

            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
          
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:        
            usesim=usethis[:,1:]
        usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')
        simmz.append(usesim)
    simmz=np.hstack(simmz)
    return simmz

def Select_TI(oldfolder,ressimmaster,N_small,nx,ny,nz,modelError,CM):
        stringff='masterrEMTI_se-sone_' 
        valueTI=np.zeros((5,1))
        name='masterrTI.data'
        N_enss=N_small
        os.chdir(ressimmaster)
        print('')
        print('--------------------------------------------------------------')
        print('TI = 1')
        k=np.genfromtxt("iglesias2.out",skip_header=0, dtype='float')
        os.chdir(oldfolder)
        k=k.reshape(-1,1)
        clfy = MinMaxScaler(feature_range=(Low_K,High_K))
        (clfy.fit(k))    
        k=(clfy.transform(k))        
        k=np.reshape(k,(33,33),'F')
        kjenn=k
        TI1=kjenn
        see=intial_ensemble(nx,ny,nz,N_small,kjenn)
        ini_ensemblee=np.split(see, N_small, axis=1)
        ini_ensemble=[]
        for ky in range(N_small):
            aa=ini_ensemblee[ky]
            aa=np.reshape(aa,(-1,1),'F')
            ini_ensemble.append(aa)
            
        ini_ensemble1=np.hstack(ini_ensemble)
        clfy = MinMaxScaler(feature_range=(Low_P,High_P))
        (clfy.fit(ini_ensemble1))    
        ensemblep=(clfy.transform(ini_ensemble1))        
        

        
        for ijesus in range(N_enss):
            (
            write_include)(ijesus,ini_ensemble1,ensemblep,'RealizationTI_')      
        az=int(np.ceil(int(N_enss/maxx)))
        a=(np.linspace(1, N_enss, num=N_enss))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_FileTI(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEMTI.data -csv -sumout 3 ")
            os.system(string_Jesus2)
            for kk in range(maxx):
                folder=stringff + str(kk)
                namecsv=stringff + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
                   
        Delete_files()                                    

        simDatafinal = Get_simulated(predMatrix,modelError,CM,N_enss)# 
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        cc=cc.reshape(-1,1)
        clem1=np.mean(cc,axis=0) #mean best
        cle=np.argmin(cc)
        clem1a=cc[cle,:] #best
        valueTI[0,:]=(clem1+clem1a)/2
        print('Error = ' + str((clem1+clem1a)/2)) 
        yett=1
        
        print('')
        print('--------------------------------------------------------------')        
        print ('TI = 2')
        os.chdir(ressimmaster)
        k=np.genfromtxt("TI_3.out",skip_header=3, dtype='float')
        os.chdir(oldfolder)
        k=k.reshape(-1,1)
        #os.chdir(oldfolder)
        clfy = MinMaxScaler(feature_range=(Low_K,High_K))
        (clfy.fit(k))    
        k=(clfy.transform(k))
        #k=k.T
        k=np.reshape(k,(768,243),'F')
        kjenn=k
        #shape1 = (768, 243)
        TI2=kjenn
       # kti2=
        see=intial_ensemble(nx,ny,nz,N_small,kjenn)
        #kout = kjenn
        ini_ensemblee=np.split(see, N_small, axis=1)
        ini_ensemble=[]
        for ky in range(N_small):
            aa=ini_ensemblee[ky]
            aa=np.reshape(aa,(-1,1),'F')
            ini_ensemble.append(aa)
            
        ini_ensemble2=np.hstack(ini_ensemble)
       
        
        clfy = MinMaxScaler(feature_range=(Low_P,High_P))
        (clfy.fit(ini_ensemble2))    
        ensemblep=(clfy.transform(ini_ensemble2))          

        
        for ijesus in range(N_enss):
            (
            write_include)(ijesus,ini_ensemble2,ensemblep,'RealizationTI_')      
        az=int(np.ceil(int(N_enss/maxx)))
        a=(np.linspace(1, N_enss, num=N_enss))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_FileTI(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEMTI.data -csv -sumout 3 ")
            os.system(string_Jesus2)
            for kk in range(maxx):
                folder=stringff + str(kk)
                namecsv=stringff + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
                   
        Delete_files()                                    

        simDatafinal = Get_simulated(predMatrix,modelError,CM,N_enss)# 
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        cc=cc.reshape(-1,1)
        clem2=np.mean(cc,axis=0)
        cle=np.argmin(cc)
        clem2a=cc[cle,:] #best        
        valueTI[1,:]=(clem2+clem2a)/2 
        print('Error = ' + str((clem2+clem2a)/2)) 
        yett=2

        print('')
        print('--------------------------------------------------------------')
        print('TI = 3 ') 
        os.chdir(ressimmaster)
        k=np.genfromtxt("TI_2.out",skip_header=3, dtype='float')
        k=k.reshape(-1,1)
        os.chdir(oldfolder)
        clfy = MinMaxScaler(feature_range=(Low_K,High_K))
        (clfy.fit(k))    
        k=(clfy.transform(k))
        k=np.reshape(k,(250,250),'F') 
        kjenn=k.T
        TI3=kjenn
        see=intial_ensemble(nx,ny,nz,N_small,kjenn)
        #kout = kjenn
        ini_ensemblee=np.split(see, N_small, axis=1)
        ini_ensemble=[]
        for ky in range(N_small):
            aa=ini_ensemblee[ky]
            aa=np.reshape(aa,(-1,1),'F')
            ini_ensemble.append(aa)
            
        ini_ensemble3=np.hstack(ini_ensemble) 
               
        clfy = MinMaxScaler(feature_range=(Low_P,High_P))
        (clfy.fit(ini_ensemble3))    
        ensemblep=(clfy.transform(ini_ensemble3))          

        
        for ijesus in range(N_enss):
            (
            write_include)(ijesus,ini_ensemble3,ensemblep,'RealizationTI_')      
        az=int(np.ceil(int(N_enss/maxx)))
        a=(np.linspace(1, N_enss, num=N_enss))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_FileTI(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEMTI.data -csv -sumout 3 ")
            os.system(string_Jesus2)
            for kk in range(maxx):
                folder=stringff + str(kk)
                namecsv=stringff + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
                   
        Delete_files()                                    

        simDatafinal = Get_simulated(predMatrix,modelError,CM,N_enss)# 
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        cc=cc.reshape(-1,1)
        clem3=np.mean(cc,axis=0)
        cle=np.argmin(cc)
        clem3a=cc[cle,:] #best        
        valueTI[2,:]=(clem3+clem3a)/2 
        print('Error = ' +str((clem3+clem3a)/2)) 
        yett=3
        
        print('')
        print('--------------------------------------------------------------')        
        print('TI = 4')
        os.chdir(ressimmaster)  
        k=np.loadtxt("TI_4.out",skiprows=4,\
                        dtype='float')
        kuse=k[:,1]
        k=kuse.reshape(-1,1)
        os.chdir(oldfolder)
        clfy = MinMaxScaler(feature_range=(Low_K,High_K))
        (clfy.fit(k))    
        k=(clfy.transform(k))
        #k=k.T
        k=np.reshape(k,(100,100,2),'F') 
        kjenn=k[:,:,0]
        #shape1 = (100, 100) 
        TI4=kjenn
        see=intial_ensemble(nx,ny,nz,N_small,kjenn)
        #kout = kjenn
        ini_ensemblee=np.split(see, N_small, axis=1)
        ini_ensemble=[]
        for ky in range(N_small):
            aa=ini_ensemblee[ky]
            aa=np.reshape(aa,(-1,1),'F')
            ini_ensemble.append(aa)
            
        ini_ensemble4=np.hstack(ini_ensemble)
             
        clfy = MinMaxScaler(feature_range=(Low_P,High_P))
        (clfy.fit(ini_ensemble4))    
        ensemblep=(clfy.transform(ini_ensemble4))          
        
        for ijesus in range(N_enss):
            (
            write_include)(ijesus,ini_ensemble4,ensemblep,'RealizationTI_')      
        az=int(np.ceil(int(N_enss/maxx)))
        a=(np.linspace(1, N_enss, num=N_enss))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_FileTI(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEMTI.data -csv -sumout 3 ")
            os.system(string_Jesus2)
            for kk in range(maxx):
                folder=stringff + str(kk)
                namecsv=stringff + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
                   
        Delete_files()                                    

        simDatafinal = Get_simulated(predMatrix,modelError,CM,N_enss)# 
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        cc=cc.reshape(-1,1)
        clem4=np.mean(cc,axis=0)
        cle=np.argmin(cc)
        clem4a=cc[cle,:] #best        
        valueTI[3,:]=(clem4+clem4a)/2
        print('Error = ' +str((clem4+clem4a)/2)) 
        yett=4
        
        print('')
        print('--------------------------------------------------------------')        
        print('TI = 5')  
        os.chdir(ressimmaster)
        k=np.genfromtxt("TI_1.out",skip_header=3, dtype='float')
        k=k.reshape(-1,1)
        os.chdir(oldfolder)
        clfy = MinMaxScaler(feature_range=(Low_K,High_K))
        (clfy.fit(k))    
        k=(clfy.transform(k))
        #k=k.T
        k=np.reshape(k,(400,400),'F') 
        kjenn=k
        #shape1 = (260, 300)         
        TI5=kjenn    
        see=intial_ensemble(nx,ny,nz,N_small,kjenn)
        #kout = kjenn
        ini_ensemblee=np.split(see, N_small, axis=1)
        ini_ensemble=[]
        for ky in range(N_small):
            aa=ini_ensemblee[ky]
            aa=np.reshape(aa,(-1,1),'F')
            ini_ensemble.append(aa)
            
        ini_ensemble5=np.hstack(ini_ensemble)
             
        clfy = MinMaxScaler(feature_range=(Low_P,High_P))
        (clfy.fit(ini_ensemble5))    
        ensemblep=(clfy.transform(ini_ensemble5))          
        #ensemblep=Getporosity_ensemble(ini_ensemble5,machine_map,N_enss)
        
        for ijesus in range(N_enss):
            (
            write_include)(ijesus,ini_ensemble5,ensemblep,'RealizationTI_')      
        az=int(np.ceil(int(N_enss/maxx)))
        a=(np.linspace(1, N_enss, num=N_enss))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_FileTI(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEMTI.data -csv -sumout 3 ")
            os.system(string_Jesus2)
            for kk in range(maxx):
                folder=stringff + str(kk)
                namecsv=stringff + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
                   
        Delete_files()                                    

        simDatafinal = Get_simulated(predMatrix,modelError,CM,N_enss)# 
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        cc=cc.reshape(-1,1)
        clem5=np.mean(cc,axis=0)
        cle=np.argmin(cc)
        clem5a=cc[cle,:] #best        
        valueTI[4,:]=(clem5+clem5a)/2
        print('Error = ' +str((clem5+clem5a)/2)) 
        yett=5

        TIs={'TI1':TI1,'TI2':TI2,'TI3':TI3,'TI4':TI4,'TI5':TI5}             

        clem=np.argmin(valueTI)        
        valueM=valueTI[clem,:]


        print('')
        print('--------------------------------------------------------------')    
        print(' Gaussian Random Field Simulation')
        ini_ensembleG=initial_ensemble_gaussian(nx,ny,nz,N_enss,\
                            Low_K,High_K)
      
        
        clfy = MinMaxScaler(feature_range=(Low_P,High_P))
        (clfy.fit(ini_ensembleG))    
        EnsemblepG=(clfy.transform(ini_ensembleG))              
        #EnsemblepG=Getporosity_ensemble(ini_ensembleG,machine_map,N_enss)
        print('')
                
        for ijesus in range(N_enss):
            (
            write_include)(ijesus,ini_ensembleG,EnsemblepG,'RealizationTI_')      
        az=int(np.ceil(int(N_enss/maxx)))
        a=(np.linspace(1, N_enss, num=N_enss))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_FileTI(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEMTI.data -csv -sumout 3 ")
            os.system(string_Jesus2)
            for kk in range(maxx):
                folder=stringff + str(kk)
                namecsv=stringff + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
                   
        Delete_files()                                    

        simDatafinal = Get_simulated(predMatrix,modelError,CM,N_enss)# 
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        cc=cc.reshape(-1,1)
        cle=np.argmin(cc)
        clem1a=cc[cle,:] #best
        valueG=(np.mean(cc,axis=0)+clem1a)/2
        print('Error = ' + str(valueG)) 
        
        
        plt.figure(figsize=(16,16))
        plt.subplot(2,3,1)
        plt.imshow(TI1)
        #plt.gray()
        plt.title('TI1 ', fontsize = 15)
        plt.ylabel('Y',fontsize = 13)
        plt.xlabel('X',fontsize = 13)
        
        plt.subplot(2,3,2)
        plt.imshow(TI2)
        #plt.gray()
        plt.title('TI2 ', fontsize = 15)
        plt.ylabel('Y',fontsize = 13)
        plt.xlabel('X',fontsize = 13) 
        
        plt.subplot(2,3,3)
        plt.imshow(TI3)
        #plt.gray()
        plt.title('TI3 ', fontsize = 15)
        plt.ylabel('Y',fontsize = 13)
        plt.xlabel('X',fontsize = 13) 
        
        plt.subplot(2,3,4)
        plt.imshow(TI4)
        #plt.gray()
        plt.title('TI4 ', fontsize = 15)
        plt.ylabel('Y',fontsize = 13)
        plt.xlabel('X',fontsize = 13)

        plt.subplot(2,3,5)
        plt.imshow(TI5)
        #plt.gray()
        plt.title('TI5 ', fontsize = 15)
        plt.ylabel('Y',fontsize = 13)
        plt.xlabel('X',fontsize = 13)
        
        os.chdir(ressimmaster)
        plt.savefig("TISS.png")
        os.chdir(oldfolder)
        plt.close()
        plt.clf()  
        clemm=clem+1
        if (valueG < valueM):
            print('Gaussian distribution better suited')
            mum=2
            permxx=0
            yett=6
        else:
            if (valueM < valueG) and (clemm==1):
                print('MPS distribution better suited')
                mum=1
                permxx=TI1
                yett=1
            if(valueM < valueG) and  (clemm==2):
                mum=1
                permxx=TI2
                yett=2
            if (valueM < valueG) and (clemm==3):
                mum=1
                permxx=TI3
                yett=3
            if (valueM < valueG) and (clemm==4):
                mum=1
                permxx=TI4
                yett=4
            if (valueM < valueG) and (clemm==5): 
                mum=1
                permxx=TI5
                yett=5
        return permxx,mum,TIs,yett


        
def listToStringWithoutBrackets(list1):
    return str(list1).replace('[','').replace(']','')

def overwrite_Data_File(name,a):
    with open(name, 'r') as file:
        data = file.readlines()    
    anew1="     V1      REAL     DISCRETE      1       6*                                    " + a +" /\n"  
    data[20]=anew1
    filename='masterrEM.data'
    with open(filename, 'w') as file:
        file.writelines( data )  
        
def overwrite_Data_FileTI(name,a):
    with open(name, 'r') as file:
        data = file.readlines()    
    anew1="     V1      REAL     DISCRETE      1       6*                                    " + a +" /\n"  
    data[20]=anew1
    filename='masterrEMTI.data'
    with open(filename, 'w') as file:
        file.writelines( data )  
        
        
        
def write_include(jj,kperm,poro,stringfc):
    data=np.reshape(kperm[:,jj],(-1,1),'F')
    data2=np.reshape(poro[:,jj],(-1,1),'F')
    filename=stringfc + str(jj+1) +'.inc'
    with open(filename, 'w') as file:
        file.writelines( 'PERMY \n')
        for i in range (data.shape[0]):
            use=np.asscalar(data[i,:])
            dataa = str(my_formatter.format(use))
            file.write( dataa +'\n')
            
        file.writelines( '\n')
        file.writelines( '/\n')
        file.writelines( '----\n')
        file.writelines( '\n')
        file.writelines( 'PORO\n')
        for i in range(data.shape[0]):
            use=np.asscalar(data2[i,:])
            dataa = str(my_formatter.format(use))
            file.write( dataa +'\n')        
    
        file.writelines( '\n')
        file.writelines( '/\n') 
        
def Delete_files():
    
    for fa in glob("*_se-sone_*"):
        os.remove(fa) 
        
    for f3 in glob("*.grdbin"):
        os.remove(f3) 
        
    for f3 in glob("tmp*"):
         os.remove(f3)
         
    for f4 in glob("*.mrx"):
         os.remove(f4)   
         
def Delete_files2():

    for f1 in glob("*.inc"):
        os.remove(f1)      
         
def Get_RSM_6X_Ensemble(oldfolder,folder,namecsv):
    #folder='True_Model'
    os.chdir(oldfolder)
    unsmry_file = folder#"masterr"
    parser = binary_parser.EclBinaryParser(unsmry_file)
    vectors = parser.read_vectors()
    clement5=vectors['WBHP']
    wbhp=clement5.values[1:,:] #8 + 8  producers
    df=pd.read_csv(namecsv)
    Timeuse=df.values[8:,[0]].astype(np.float32)
    Timeuse=Timeuse[1:,:]
    measurement=np.hstack([Timeuse,wbhp]) #Measurement
    #measurement=np.hstack([Timeuse,wbhp,wopr,wwpr,wwct]) #Measurement
    os.chdir(oldfolder)
    return measurement 


def Normalize_data(dataa):

    Normalise=(dataa-menjesus)/sdjesus 
    return Normalise

def numpy_to_pvgrid(Data, origin=(0,0,0), spacing=(1,1,1)):
    '''
    Convert 3D numpy array to pyvista uniform grid
    
    '''    
    # Create the spatial reference
    grid = pyvista.UniformGrid()
    # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
    grid.dimensions = np.array(Data.shape) + 1
    # Edit the spatial reference
    grid.origin = origin # The bottom left corner of the data set
    grid.spacing = spacing # These are the cell sizes along each axis
    # Add the data values to the cell data
    grid.cell_arrays['values'] = Data.flatten(order='F') # Flatten the array!

    return grid

def plot_3d_pyvista(Data,filename ):
    '''
    plot 3D cube using 'pyvista' 
    '''  
    Data=np.atleast_3d(Data)
    filee=filename + '.png'


    grid = numpy_to_pvgrid(Data, origin=(0,0,0), spacing=(1,1,1))
    os.chdir('PERCENTILE') 
    grid.plot(show_edges=False,notebook=False,\
              lighting=False,colormap='jet',off_screen=True,\
                  screenshot=filee)
    os.chdir(oldfolder)
        
def parad2_TI(X_train,y_traind,namezz):
    
    namezz=namezz +'.h5'
    np.random.seed(7)
    modelDNN = Sequential()
    modelDNN.add(Dense(200, activation = 'relu', input_dim = X_train.shape[1]))
    modelDNN.add(Dense(units = 820, activation = 'relu'))
    modelDNN.add(Dense(units = 220, activation = 'relu')) 
    modelDNN.add(Dense(units = 21, activation = 'relu'))
    modelDNN.add(Dense(units = 1))
    modelDNN.compile(loss= 'mean_squared_error', optimizer='Adam', metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint(namezz, monitor='val_loss', mode='min', \
                         verbose=1, save_best_only=True)
    a0=X_train
    a0=np.reshape(a0,(-1,X_train.shape[1]),'F')

    b0=y_traind
    b0=np.reshape(b0,(-1,y_traind.shape[1]),'F')
    gff=len(a0)//100
    if gff<1:
        gff=1
    modelDNN.fit(a0, b0,validation_split=0.1, batch_size =gff, \
                 epochs = 500,callbacks=[es,mc]) 

def Select_TI_2(oldfolder,ressimmaster,nx,ny,nz,perm_high,\
              perm_low,poro_high,poro_low,i):
        os.chdir(ressimmaster)
        print('')
        print('--------------------------------------------------------------')
        if i==1:
            print('TI = 1')
            k=np.genfromtxt("iglesias2.out",skip_header=0, dtype='float')
            k=k.reshape(-1,1)
            os.chdir(oldfolder)
            # k=k.T
            clfy = MinMaxScaler(feature_range=(perm_low, perm_high))
            (clfy.fit(k))    
            k=(clfy.transform(k))            
            k=np.reshape(k,(33,33),'F')
            #k=k.T
            #k=k.reshape(ny,nx)
            kjenn=k
            TrueK=np.reshape(kjenn,(-1,1),'F')
            clfy = MinMaxScaler(feature_range=(poro_low, poro_high))
            (clfy.fit(TrueK))    
            y_train=(clfy.transform(TrueK))
            parad2_TI(TrueK,y_train,'MAP1')
 
        elif i==2:
            print('')
            print('--------------------------------------------------------------')        
            print ('TI = 2')
            os.chdir(ressimmaster)
            k=np.genfromtxt("TI_3.out",skip_header=3, dtype='float')
            os.chdir(oldfolder)
            k=k.reshape(-1,1)
            #os.chdir(oldfolder)
            clfy = MinMaxScaler(feature_range=(perm_low, perm_high))
            (clfy.fit(k))    
            k=(clfy.transform(k))
            #k=k.T
            k=np.reshape(k,(768,243),'F')
            kjenn=k
            TrueK=np.reshape(kjenn,(-1,1),'F')
            clfy = MinMaxScaler(feature_range=(poro_low, poro_high))
            (clfy.fit(TrueK))    
            y_train=(clfy.transform(TrueK))
            parad2_TI(TrueK,y_train,'MAP2')

        elif i==3:
            print('')
            print('--------------------------------------------------------------')
            print('TI = 3 ') 
            os.chdir(ressimmaster)
            k=np.genfromtxt("TI_2.out",skip_header=3, dtype='float')
            k=k.reshape(-1,1)
            os.chdir(oldfolder)
            clfy = MinMaxScaler(feature_range=(perm_low, perm_high))
            (clfy.fit(k))    
            k=(clfy.transform(k))
            #k=k.T
            k=np.reshape(k,(250,250),'F') 
            kjenn=k.T
            TrueK=np.reshape(kjenn,(-1,1),'F')
            clfy = MinMaxScaler(feature_range=(poro_low, poro_high))
            (clfy.fit(TrueK))    
            y_train=(clfy.transform(TrueK))
            parad2_TI(TrueK,y_train,'MAP3')

        elif i==4:
            print('')
            print('--------------------------------------------------------------')        
            print('TI = 4')
            os.chdir(ressimmaster)  
            k=np.loadtxt("TI_4.out",skiprows=4,\
                            dtype='float')
            kuse=k[:,1]
            k=kuse.reshape(-1,1)
            os.chdir(oldfolder)
            clfy = MinMaxScaler(feature_range=(perm_low, perm_high))
            (clfy.fit(k))    
            k=(clfy.transform(k))
            #k=k.T
            k=np.reshape(k,(100,100,2),'F') 
            kjenn=k[:,:,0]
            TrueK=np.reshape(kjenn,(-1,1),'F')
            clfy = MinMaxScaler(feature_range=(poro_low, poro_high))
            (clfy.fit(TrueK))    
            y_train=(clfy.transform(TrueK))
            parad2_TI(TrueK,y_train,'MAP4')

        elif i==5:
            print('')
            print('--------------------------------------------------------------')        
            print('TI = 5')  
            os.chdir(ressimmaster)
            k=np.genfromtxt("TI_1.out",skip_header=3, dtype='float')
            k=k.reshape(-1,1)
            os.chdir(oldfolder)
            clfy = MinMaxScaler(feature_range=(perm_low, perm_high))
            (clfy.fit(k))    
            k=(clfy.transform(k))
            #k=k.T
            k=np.reshape(k,(400,400),'F') 
            kjenn=k
            TrueK=np.reshape(kjenn,(-1,1),'F')
            clfy = MinMaxScaler(feature_range=(poro_low, poro_high))
            (clfy.fit(TrueK))    
            y_train=(clfy.transform(TrueK))
            parad2_TI(TrueK,y_train,'MAP5') 
        else:
            print(' Gaussian Random Field Simulation')
            
            fout=[]
            shape=(nx,ny)
            for j in range(nz):
                field = generate_field(distrib, Pkgen(3), shape)
                field = imresize(field, output_shape=shape)
                foo=np.reshape(field,(-1,1),'F')
                fout.append(foo)
            fout=np.vstack(fout)
            clfy = MinMaxScaler(feature_range=(perm_low, perm_high))
            (clfy.fit(fout))    
            k=(clfy.transform(fout)) 
            kjenn=k   
            TrueK=np.reshape(kjenn,(-1,1),'F')
            clfy = MinMaxScaler(feature_range=(poro_low, poro_high))
            (clfy.fit(TrueK))    
            y_train=(clfy.transform(TrueK))
            parad2_TI(TrueK,y_train,'MAP6')             
        return kjenn              


def Get_Measuremnt_CSV(namee):
    df=pd.read_csv(namee,skiprows=[0]) 
    vectorss=df
    
    clementt5=vectorss[['WBHP','WBHP.1','WBHP.10','WBHP.13','WBHP.15',\
                        'WBHP.3','WBHP.4','WBHP.6','WBHP.11','WBHP.12',\
                            'WBHP.14','WBHP.2',\
                    'WBHP.5','WBHP.7','WBHP.8','WBHP.9']]    
    wbhpt = clementt5.iloc[8:]
    wbhpt=wbhpt.values.astype(np.float32) #8 producers
    
    
    df=vectorss
    Timeuse=df.values[8:,[0]].astype(np.float32)
    #True_measurement=np.hstack([Timeuse,wbhpt,woprt,wwprt,wwctt,wgprt]) #Measurement
    True_measurement=np.hstack([Timeuse,wbhpt]) #Measurement
    
    return True_measurement

def Get_CM_ini(modelError,Base_data_available,True_yet,Nej):

    if modelError==1:
        if Base_data_available==1:    
           mE=Base_data-True_yet
           use_mean=np.abs(np.mean(mE,axis=0))
           use_mean=use_mean.reshape(1,-1)
           use_yet=np.abs(np.mean(True_yet,axis=0))
           use_yet=use_yet.reshape(1,-1)
           CM=[]
           for i in range(use_mean.shape[1]):
               mu=1*(abs(0.1*use_mean[:,i]))
               #mu=0
               sigma=0.1*mu
               s = np.random.normal(mu, sigma, Nej)
               s=s.reshape(1,-1)
               CM.append(s)
           
           CM=np.vstack(CM) 
        else:
            CM=[]
            use_mean=np.abs(np.mean(True_yet,axis=0))
            use_mean=use_mean.reshape(1,-1)            
            
            for i in range(use_mean.shape[1]):
                mu=1*(abs(0.01*use_mean[:,i]))
                #mu=0
                sigma=0.1*mu
                s = np.random.normal(mu, sigma, Nej)
                s=s.reshape(1,-1)
                CM.append(s)
        
            CM=np.vstack(CM) 
    else:
        mE=Base_data-True_yet
        use_mean=np.abs(np.mean(mE,axis=0))
        use_mean=use_mean.reshape(1,-1)
        use_yet=np.abs(np.mean(True_yet,axis=0))
        use_yet=use_yet.reshape(1,-1)
        CM=[]
        for i in range(use_mean.shape[1]):
            mu=abs(use_mean[:,i])
            #mu=0
            sigma=0.1*mu
            s = np.random.normal(mu, sigma, Nej)
            s=s.reshape(1,-1)
            CM.append(s)
        
        CM=np.vstack(CM)
        CM=np.zeros_like(CM)
        
    return CM

def De_correlate_ensemble(nx,ny,nz,Ne,High_K,Low_K):
    filename="Ganensemble.mat" #Ensemble generated offline
    mat = sio.loadmat(filename)
    ini_ensemblef=mat['Z']
    ini_ensemblef=cp.asarray(ini_ensemblef)
    
    
    beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
    
    V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
    v = V[:,:Ne]
    U1 = U.T
    u = U1[:,:Ne]
    S11 = S1[:Ne]
    s = S11[:]
    S = (1/((beta)**(0.5)))*s
    #S=s
    X = (v*S).dot(u.T)
    X=cp.asnumpy(X)
    ini_ensemblef=cp.asnumpy(ini_ensemblef)
    X[X<=Low_K]=Low_K
    X[X>=High_K]=High_K
    return X[:,:Ne] 

def Learn_Overcomplete_Dictionary(Ne):
    #Ne= ensemble size
    filename="Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensembley=mat['Z']    
    a=dict_learning(ini_ensembley.T,Ne,0,max_iter=100,method='lars',\
                    n_jobs=-1,return_n_iter=True,\
                        method_max_iter=10000,verbose=True)
    second=a[1] 
    Dicclem=second.T# Over complete dictionary
    return Dicclem

def Sparse_coding(Dictionary,inpuut,N_ens):
    sparsecode = linear_model.orthogonal_mp(Dictionary, inpuut, \
                                            n_nonzero_coefs=N_ens) 
    return sparsecode

def Recover_Dictionary_Saarse(Dicclem,sparsecode,Low_K,High_K):
    recoverr=np.dot(Dicclem,sparsecode)
    recoverr[recoverr<=Low_K]=Low_K
    recoverr[recoverr>=High_K]=High_K
    return recoverr  
  
from shutil import rmtree
def Remove_folder(N_ens,straa):
    for jj in range(N_ens):
        folderr=straa + str(jj)
        rmtree(folderr)      # remove everything
        
def whiten(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None
    
    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method =='pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / \
                                                (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)),\
                                    G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0/np.sqrt(Theta + 1e-5)), G.T),\
                      np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)        
        
def Plot_Histogram_now(N,E11,E12):
    reali = np.arange(1,N+1)
    plttotalerror = np.reshape(E11,(N)) #Initial
    plttotalerror2 = np.reshape(E12,(N)) #Final
    plt.figure(figsize=(8, 8))
    plt.subplot(1,2,1)
    plt.bar(reali,plttotalerror, color = 'c')
    plt.xlabel('Realizations')
    plt.ylabel('RMSE value')
    plt.ylim(ymin = 0)
    plt.title('Initial Cost function for Realizations')

    plt.scatter(reali,plttotalerror, color ='k')
    plt.xlabel('Realizations')
    plt.ylabel('RMSE value')
    plt.xlim([1,(N - 1)])
    
    plt.subplot(1,2,2)
    plt.bar(reali,plttotalerror2, color = 'c')
    plt.xlabel('Realizations')
    plt.ylabel('RMSE value')
    plt.ylim(ymin = 0)
    plt.title('Final Cost function for Realizations')

    plt.scatter(reali,plttotalerror2, color ='k')
    plt.xlabel('Realizations')
    plt.ylabel('RMSE value')
    plt.xlim([1,(N - 1)])    
    
    os.chdir('HM_RESULTS')
    plt.savefig("Cost_Function_Histogram.png")
    os.chdir(oldfolder)
    plt.close()
    plt.clf()    
    

def EnKF_One_Short(sgsim,sgsimporo,modelError,CM,f, N, Sim1,corellaa):
    
    if Yet==0:
        sgsim=cp.asarray(sgsim)
        sgsimporo=cp.asarray(sgsimporo)
        modelError=cp.asarray(modelError)
        CM=cp.asarray(CM)
        f=cp.asarray(f)
        N=cp.asarray(N)
        Sim1=cp.asarray(Sim1)
        noise_levell=cp.asarray(noise_level)
        corelaa=cp.asarray(corellaa)
    else:
        noise_levell=noise_level
        corelaa=corellaa
    
    
    sizea=sgsim.shape[0]
    f=f.reshape(-1,1)
    cp.random.seed(1)
    stdall=cp.zeros((f.shape[0],1))
    for iz in range (f.shape[0]):
        #aa=f[i,:] 
        if f[iz,:] ==0:
           stdall[iz,:]=1
        else:
           
            if Big_noise==2:
                stdall[iz,:]=cp.sqrt(noise_levell*f[iz,:])
            else:
                stdall[iz,:]=cp.sqrt(cp.asarray(diffyet[iz,:]))

    stdall=cp.reshape(stdall,(-1,1))
    
    
    if corelaa==1:
        R = stdall**2
        Cd2=cp.diag(cp.reshape(R,(-1,)))    
        
        f=cp.asarray(f)
        pertubations=cp.random.multivariate_normal(cp.ravel(cp.zeros\
                                                        ((f.shape[0],1))), Cd2,Ne)
        Ddraw=cp.tile(f, Ne) 
        pertubations=pertubations.T
        Dj=Ddraw + pertubations
        # Cd2=cp.asnumpy(Cd2)
        # Dj=cp.asnumpy(Dj)
        # Ddraw=cp.asnumpy(Ddraw)
        # pertubations=cp.asnumpy(pertubations)
        # f=cp.asnumpy(f)    
    
    else:
    
        Error1 = cp.ones((f.shape[0],N_ens)) 
        for iz in range (f.shape[0]):
            Error1[iz,:]=cp.random.normal(0,stdall[iz,:],(N_ens))        
        
        nobs = f.shape[0]    
        #Error1=stdall    
        Dj = cp.zeros((nobs, N_ens))
    
        for j in range(N_ens):
            Dj[:,j] = cp.ravel(f) + Error1[:,j]
    
        Cd2=(Error1.dot(Error1.T))/(N_ens - 1)
    
    
    overall=cp.vstack([(sgsim),sgsimporo,Sim1])

    Y=overall 
    M = cp.mean(Sim1,axis=1)

    M2=cp.mean(overall,axis=1)    
    S = cp.zeros((Sim1.shape[0],N_ens))
    yprime = cp.zeros((Y.shape[0],Y.shape[1]))
           
    for j in range(N_ens):
        S[:,j] = Sim1[:,j]- M
        yprime[:,j] = overall[:,j] - M2
    Cyd = (yprime.dot(S.T))/(N_ens - 1)
    Cdd = (S.dot(S.T))/(N_ens- 1)     
    
    Usig,Sig,Vsig = cp.linalg.svd((Cdd + (1*Cd2)), full_matrices = False)
    
    Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
    valuesig = Bsig[-1]                 # last element
    valuesig = valuesig * 0.9999
    indices = ( Bsig >= valuesig ).ravel().nonzero()
    toluse = Sig[indices]
    tol = toluse[0]

    (V,X,U) = pinvmatt((Cdd + (1*Cd2)),tol)
    
    update_term=((Cyd.dot(X)).dot(Dj - Sim1))
    Ynew = Y + update_term    
    DupdateK=Ynew[:sizea,:]
    poro=Ynew[sizea:2*sizea,:]
    Sim1=Ynew[2*sizea:,:]
    
    if modelError==1:
        Y=CM
        overall=Y
        M = cp.mean(Sim1,axis=1)
    
        M2=cp.mean(overall,axis=1)
        
        
        S = cp.zeros((Sim1.shape[0],N_ens))
        yprime = cp.zeros((Y.shape[0],Y.shape[1]))
               
        for j in range(N_ens):
            S[:,j] = Sim1[:,j]- M
            yprime[:,j] = overall[:,j] - M2
        Cyd = (yprime.dot(S.T))/(N_ens - 1)
        Cdd = (S.dot(S.T))/(N_ens- 1)     
        
        Usig,Sig,Vsig = np.linalg.svd((Cdd + (alpha*Cd2)), full_matrices = False)
    
        Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
        valuesig = Bsig[-1]                 # last element
        valuesig = valuesig * 0.9999
        indices = ( Bsig >= valuesig ).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]
    
        (V,X,U) = pinvmatt((Cdd + (1*Cd2)),tol)
        
        update_term=((Cyd.dot(X)).dot(Dj - Sim1))
        CM = Y + update_term 
    else:
        CM=cp.zeros((CM.shape[0],CM.shape[1]))        
    if Yet==0:
        return cp.asnumpy(DupdateK),cp.asnumpy(poro),cp.asnumpy(Sim1),cp.asnumpy(CM)
    else:
        return (DupdateK),(poro),(Sim1),(CM)
    
    
def ESMDA_CE(num_ens,m_ens,m_ensp,CM,modelError,Za,prod_ens,alpha,CD):
    if Yet==0:
        #num_ens=cp.asarray(num_ens)
        m_ens=cp.asarray(m_ens)
        m_ensp=cp.asarray(m_ensp)
        CM=cp.asarray(CM)
        modelError=cp.asarray(modelError)
        Z=cp.asarray(Za)
        prod_ens=cp.asarray(prod_ens)
        alpha=cp.asarray(alpha)
        CD=cp.asarray(CD)
    else:
        pass
    sizea=m_ens.shape[0]
    varn=0.9999#1-1/math.pow(10,numsave)
    # Initial Variavel 
    # Forecast step
    yf = cp.vstack([m_ens,m_ensp])                        # Non linear forward model 
    df = prod_ens                     # Observation Model
    ym = cp.array(yf.mean(axis=1))    # Mean of the y_f
    dm = cp.array(df.mean(axis=1))    # Mean of the d_f
    ym=ym.reshape(ym.shape[0],1)    
    dm=dm.reshape(dm.shape[0],1)    
    dmf = yf - ym
    ddf = df - dm
    
    Cmd_f = (cp.dot(dmf,ddf.T))/(num_ens-1);  # The cros-covariance matrix
    Cdd_f = (cp.dot(ddf,ddf.T))/(num_ens-1);  # The auto covariance of predicted data
    
    # Perturb the vector of observations
    R = cp.linalg.cholesky(CD) #Matriz triangular inferior
    U = R.T   #Matriz R transposta
    #p , w =cp.linalg.eig(CD)
    
    aux = cp.repeat(Z,num_ens,axis=1)
    mean = 0*(Z.T)

    noise=cp.random.multivariate_normal(mean[0], cp.eye(len(Z)), num_ens).T
    d_obs = aux+cp.sqrt(alpha)*cp.dot(U,noise)  
    
    # Analysis step
    u, s, vh = cp.linalg.svd(Cdd_f+alpha*CD); v = vh.T
    diagonal = s
    for i in range(len(diagonal)):
        if (cp.sum(diagonal[0:i+1]))/(cp.sum(diagonal)) > varn:
            diagonal = diagonal[0:i+1]
            break
    
    u=u[:,0:i+1]
    v=v[:,0:i+1]
    ss = cp.diag(diagonal**(-1))
    K=cp.dot(Cmd_f,(cp.dot(cp.dot(v,ss),(u.T))))
    # Use Kalman covariance
    # if len(corr)>0:
    #     K = corr*K
        
    ya = yf + (cp.dot(K,(d_obs-df)))
    m_ens = ya
    DupdateK=ya[:sizea,:]
    poro=ya[sizea:,:]  
    
    if modelError==1:
        yf = CM                       # Non linear forward model 
        df = prod_ens                     # Observation Model
        ym = cp.array(yf.mean(axis=1))    # Mean of the y_f
        dm = cp.array(df.mean(axis=1))    # Mean of the d_f
        ym=ym.reshape(ym.shape[0],1)    
        dm=dm.reshape(dm.shape[0],1)    
        dmf = yf - ym
        ddf = df - dm
        
        Cmd_f = (cp.dot(dmf,ddf.T))/(num_ens-1);  # The cros-covariance matrix
        Cdd_f = (cp.dot(ddf,ddf.T))/(num_ens-1);  # The auto covariance of predicted data
        
        # Perturb the vector of observations
        R = cp.linalg.cholesky(CD,lower=True) #Matriz triangular inferior
        U = R.T   #Matriz R transposta
        #p , w =cp.linalg.eig(CD)
        
        aux = cp.repeat(Z,num_ens,axis=1)
        mean = 0*(Z.T)
    
        noise=cp.random.multivariate_normal(mean[0], cp.eye(len(Z)), num_ens).T
        d_obs = aux+cp.sqrt(alpha)*cp.dot(U,noise)  
        
        # Analysis step
        u, s, vh = cp.linalg.svd(Cdd_f+alpha*CD); v = vh.T
        diagonal = s
        for i in range(len(diagonal)):
            if (cp.sum(diagonal[0:i+1]))/(cp.sum(diagonal)) > varn:
                diagonal = diagonal[0:i+1]
                break
        
        u=u[:,0:i+1]
        v=v[:,0:i+1]
        ss = cp.diag(diagonal**(-1))
        K=cp.dot(Cmd_f,(cp.dot(cp.dot(v,ss),(u.T))))
        # Use Kalman covariance
        # if len(corr)>0:
        #     K = corr*K
            
        ya = yf + (cp.dot(K,(d_obs-df)))
        CM = ya 
        
    else:
        CM=cp.zeros((CM.shape[0],CM.shape[1])) 
        
    if Yet==0:
        modelError=cp.asnumpy(modelError)
        alpha=cp.asnumpy(alpha)        
        return cp.asnumpy(DupdateK),cp.asnumpy(poro),cp.asnumpy(CM)
    else:
        return DupdateK,poro,CM    
def MPI_use(coress):
    roundedd=round(coress/5)*5
    if roundedd<=5:
        roundedd=2
    else:
        pass
    return roundedd
        
print('......................BEGIN THE MAIN CODE...........................')
start = datetime.datetime.now()
print(str(start))
oldfolder = os.getcwd()
os.chdir(oldfolder)
cur_dir = oldfolder
num_cores=5#cores on Local Machine
degg=3#Degree Polynomial
nclusters=2 # sand and shale
my_formatter = "{0:10.4f}"
name='masterr.data'
maxx=MPI_use(multiprocessing.cpu_count())
print (str(maxx) + ' Jobes Will be Ran in Batches in Reservoir Simulation Forwarding')
#maxx=10 #' Maximum number of cores or GPU available'
stringf='masterrEM_se-sone_' 
stringf2='masterrEMPI_se-sone_'   
string_Jesus='"@mpiexec -np '+str(maxx)+' 6X_34157_75 masterrEM.data -csv -sumout 3 "'
string_Jesus2='"@mpiexec -np '+str(maxx)+' 6X_34157_75 masterrEMTI.data -csv -sumout 3 "'
## Creating a colormap from a list of colors
##--------------------------------------------------------------------------------
colors = [(0, 0, 0),(.3, .15, .75),(.6, .2, .50),(1, .25, .15),(.9, .5, 0),(.9, .9, .5),(1, 1, 1)]
n_bins = 7  # Discretizes the interpolation into bins
cmap_name = 'my_list'
cmm = LinearSegmentedColormap.from_list(cmap_name, colors, N = n_bins)
print('')
print('---------------------------------------------------------------------')
porocon=np.array([0.271589, 0.184731, 0.176876,  
0.176688, 0.262434, 0.213609,  
0.229634, 0.266524, 0.12807,  
0.192296, 0.235776, 0.227984,  
0.121182, 0.152911, 0.247384,  
0.222419, 0.177298, 0.236364,  
0.119856, 0.233353, 0.134182,  
0.172742, 0.107685, 0.297675,  
0.182874, 0.148885, 0.167602,  
0.139316, 0.232505, 0.11942,  
0.275639, 0.295352, 0.190232,  
0.159228, 0.237732, 0.182853,  
0.273021, 0.20465, 0.284232,  
0.12636, 0.268277, 0.136638,  
0.213924, 0.247039, 0.186658,  
0.189425, 0.295714, 0.159487])

poro=np.reshape(porocon,(-1,1),'C')

perm=np.array([372.442, 411.782, 318.794,  
338.833, 364.447, 296.392,  
110.109, 578.578, 535.006,  
360.239, 159.137, 391.01,  
307.331, 193.166, 328.075, 
211.161, 571.874, 324.975,  
130.113, 426.57, 205.191,  
262.524, 319.301, 579.475,  
333.155, 411.755, 155.188,  
574.786, 510.497, 411.423,  
334.326, 127.857, 469.632,  
340.447, 258.992, 559.118,  
232.695, 558.361, 387.973,  
423.587, 244.703, 298.91,  
235.004, 244.238, 224.377,  
324.599, 449.74, 483.512])  
 
perm=np.reshape(perm,(-1,1),'C') 

perm_high=10000#np.asscalar(np.amax(perm,axis=0)) + 300 
perm_low=1#np.asscalar(np.amin(perm,axis=0))-50
# if (perm_low<=100):
#     perm_low=100
# if (perm_high>=1000):
#     perm_high=1000

poro_high=0.8#np.asscalar(np.amax(poro,axis=0))+0.3  
poro_low=0.05#np.asscalar(np.amin(poro,axis=0))-0.1  
# if (poro_low<=0.1):
#     poro_low=0.1
# if (poro_high>=0.5):
#     poro_high=0.5


High_K,Low_K,High_P,Low_P=perm_high,perm_low,poro_high,poro_low

if not os.path.exists('HM_RESULTS'):
    os.makedirs('HM_RESULTS')
else:
    shutil.rmtree('HM_RESULTS')
    os.makedirs('HM_RESULTS') 

print('')
print('-------------------------Simulation----------------------------------')
nx=np.int(input('Enter the size of the reservoir in x direction (40): '))
ny=np.int(input('Enter the size of the reservoir in y direction(40): '))
nz=np.int(input('Enter the size of the reservoir in z direction(3): '))
N_ens = int(input('Number of realisations(multiples of 20 (20-200)) : ')) 

#nx,ny,nz,N_ens=40,40,3,100
modelError=int(input('Account for Model Error:\n1=Yes\n2=No\n'))
#modelError=2

print('')
print('--------------------Historical data Measurement----------------------')
Ne=N_ens

print('Read Historical data')
os.chdir('True_Model')
True_measurement=Get_Measuremnt_CSV('hm0.csv')
True_mat=True_measurement

Cmtrue=np.zeros((True_measurement.shape[1]-1,1))
Plot_RSM_single(True_mat,2,Cmtrue,'Historical.png','True_Model')
os.chdir(oldfolder)
True_data=True_measurement[:,1:]
True_yet=True_data


sdjesus = np.std(True_data,axis=0)
sdjesus=np.reshape(sdjesus,(1,-1),'F')

menjesus = np.mean(True_data,axis=0)
menjesus=np.reshape(menjesus,(1,-1),'F')


True_dataTI=Normalize_data(True_data)
True_dataTI=np.reshape(True_dataTI,(-1,1),'F')
True_data=np.reshape(True_data,(-1,1),'F')
Nop=True_dataTI.shape[0]


print('')
print('--------------------------Base data Measurement-----------------------')
print('Read Base data')
os.chdir('BASE_MODEL')
Base_mat=Get_Measuremnt_CSV('hm0_b.csv')
Cmtrue=np.zeros((True_measurement.shape[1]-1,1))
Plot_RSM_single(Base_mat,2,Cmtrue,'Base.png','Base_Model')
Base_data=Base_mat[:,1:]
Basecomp=np.reshape(Base_data,(-1,1),'F')
os.chdir(oldfolder)


if modelError==1:
    Base_data_available=int(input('Is Base data available?:\n1=Yes\n2=No\n')) 
else:
    Base_data_available=2


CM=Get_CM_ini(modelError,Base_data_available,True_yet,Ne)

print('')
print('-----------------------Select Good prior-----------------------------')
#"""
N_small=20 # Size of realisations for initial selection of TI
CM_TI=Get_CM_ini(modelError,Base_data_available,True_yet,N_small)
ressimmastertest=os.path.join(oldfolder,'Training_Images')
# permx,Jesus,TIs,yett=Select_TI(oldfolder,ressimmastertest,N_small,nx,ny,nz,\
#                                2,CM_TI)
#"""

os.chdir(oldfolder)
TII=3
if TII==1:
    print('TI = 1')
    os.chdir('Training_Images')
    kq=np.genfromtxt("iglesias2.out",skip_header=0, dtype='float')
    os.chdir(oldfolder)
    kq=kq.reshape(-1,1)
    clfy = MinMaxScaler(feature_range=(Low_K,High_K))
    (clfy.fit(kq))    
    kq=np.reshape(kq,(33,33),'F')
    kjennq=kq
    permx=kjennq

else:
    os.chdir('Training_Images')
    kq=np.genfromtxt("TI_2.out",skip_header=3, dtype='float')
    kq=kq.reshape(-1,1)
    os.chdir(oldfolder)
    clfy = MinMaxScaler(feature_range=(Low_K,High_K))
    (clfy.fit(kq))    
    kq=(clfy.transform(kq))
    kq=np.reshape(kq,(250,250),'F') 
    kjennq=kq.T
    permx=kjennq
    


#permx=TIs['TI1']
yett=3
#Jesus=2 #1=MPS, 2=SGSIM
Jesus=int(input('Input Geostatistics type:\n1=MPS\n2=SGSIM\n'))



"""
Select_TI_2(oldfolder,ressimmastertest,nx,ny,nz,perm_high,\
              perm_low,poro_high,poro_low,yett)
"""


namev='MAP'+str(yett)+'.h5'
machine_map = load_model(namev)
if Jesus==1:
    print(' Multiple point statistics chosen')
    print('Plot the TI selected')
    plt.figure(figsize=(10,10))
    plt.imshow(permx)
    #plt.gray()
    plt.title('Training_Image Selected ', fontsize = 15)
    plt.ylabel('Y',fontsize = 13)
    plt.xlabel('X',fontsize = 13)
    os.chdir('HM_RESULTS')
    plt.savefig("TI.png")
    os.chdir(oldfolder)
    plt.close()
    plt.clf()    
   
else:
    print(' Gaussian distribution chosen')
    
path = os.getcwd()

Geostats=int(Jesus)
os.chdir(oldfolder)
print('-------------------------------- Pior Selected------------------------')

#"""
#if Geostats==1: #(For MPS simulation with a TI)
#     #"""
#     print('-------------------Generate Large Realisation---------------------')
#     N_tot=5000
#     see=intial_ensemble(nx,ny,nz,N_tot,permx)
#     ini_ensemblee=np.split(see, N_tot, axis=1)
#     ini_ensemble=[]
    
#     for ky in range(N_tot):
#         aa=ini_ensemblee[ky]
#         aa=np.reshape(aa,(-1,1),'F')
#         ini_ensemble.append(aa)        
#     ini_ensembleee=np.hstack(ini_ensemble)
#     sio.savemat('Ganensemble.mat', {'Z':ini_ensembleee}) 
#     #"""    
    # print('---------------------- Learn denoising Autoencoder---------------')
    # DenosingAutoencoder(nx,ny,nz)
    
    # print('-------------------------learn Autoencoder------------------------')
    # Autoencoder2(nx,ny,nz)
    # print('--------------------Section Ended--------------------------------')
#"""
print('')
print('---------------------------------------------------------------------')
Big_noise=int(input('Get noise Level from Base case compare:\n2=No\n1=Yes\n'))


if Big_noise==2:
    Estimate_noise=int(input('Estimate nosie Level:\n1=RMSE\n2=User Supplied\n'))
    if Estimate_noise==2:
        noise_level=float(input('Enter the masurement data noise level in % (1%-5%): '))
        noise_level=noise_level/100
    else:
        clfyy = MinMaxScaler(feature_range=(0,1))
        clfyy.fit(True_data)    
        bgg=(clfyy.transform(True_data)) 
        bgg1=(clfyy.transform(Basecomp)) 
        
        noise_level = (mean_squared_error(bgg1, bgg))
else:
    diffyet=np.abs(Basecomp-True_data)

print('')
print('-------------------Select Optimisation method-------------------------')
#"""
print('*******************REKI (Option 5) is Recommended*********************')
Data_Assimilation=int(input('Enter Data Assimialtion method: \n\
1 = IES\n\
2 = ESMDA\n\
3 = EnKF\n\
4 = MCMC\n\
5 = REKI\n'))



if Geostats==2:
    choice=2 
else:
    pass
 
sizeclem=nx*ny*nz  
print('')
print('-------------------Decorrelate the ensemble---------------------------')
#"""
Deccor=int(input('De-correlate the ensemble:\n1=Yes\n2=No\n'))
#"""
#Deccor=2

print('')
print('-----------------------Alpha Parameter-------------------------------')

De_alpha=int(input('Use recommended alpha:\n1=Yes\n2=No\n'))


if (Geostats==1) and (Deccor==2):
    afresh=int(input('Generate ensemble afresh  or random?:\n1=afresh\n\
2=random from Library\n'))
    
print('')
print('-----------------------SOLVE INVERSE PROBLEM-------------------------')
print('')
if Data_Assimilation==1: 
    
    print('-----------------------------------------------------------------')    
    Technique_IES=int(input('Enter optimisation with the Ensemble Smoother: \n\
1 =  ES (Ensemble Smoother)\n\
2 = IES_Version_1 (Iterative Ensemble Smoother)\n\
3 = IES_Version_2  (Iterative Ensemble Smoother 2)(slower)\n')) 
    if Technique_IES==1:
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        print('--------------------------ES----------------------------------')
        print('History Matching using the Ensemble Kalman Smoother ')
        print('Reference: (1): Geir Evensen & P.Leeuwen An ensemble Kalman \
smoother for nonlinear dynamics\n')
    
        print('')
        choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        os.chdir(oldfolder)
    
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))           
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)               
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)         
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
    
        R = ax**2
    
        CDd=np.diag(np.reshape(R,(-1,)))
        
        snn=0
        ii=0
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')        
        os.chdir(oldfolder)         

        
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                foldern=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,foldern,namecsv)
                predMatrix.append(predd)        
            Delete_files()
        Delete_files2()
        os.chdir('HM_RESULTS')
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,\
                 "Initial.jpg")
        os.chdir(oldfolder)

        
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#   
        
        CDd=cp.asarray(CDd)
        True_data=cp.asarray(True_data)
        pertubations=cp.random.multivariate_normal(cp.ravel(cp.zeros\
                                                        ((Nop,1))), CDd,Ne)
        Ddraw=cp.tile(True_data, Ne) 
        pertubations=pertubations.T
        Dd=Ddraw #+ pertubations
        if Yet==0:
            CDd=cp.asnumpy(CDd)
            Dd=cp.asnumpy(Dd)
            Ddraw=cp.asnumpy(Ddraw)
            pertubations=cp.asnumpy(pertubations)
            True_data=cp.asnumpy(True_data)
        else:
            pass
        sgsim=ensemble
        if modelError==1:
            overall=cp.vstack([cp.asarray(sgsim),cp.asarray(ensemblep),\
                              cp.asarray(CM)])
        else:
            overall=cp.vstack([cp.asarray(sgsim),cp.asarray(ensemblep)])
        
        Y=overall 
        Sim1=simDatafinal
        M = cp.mean(Sim1,axis=1)
    
        M2=cp.mean(overall,axis=1)
        
        
        S = cp.zeros((Sim1.shape[0],Ne))
        yprime = cp.zeros((Y.shape[0],Y.shape[1]))
               
        for jc in range(Ne):
            S[:,jc] = Sim1[:,jc]- M
            yprime[:,jc] = overall[:,jc] - M2
        Cyd = (yprime.dot(S.T))/(Ne - 1)
        Cdd = (S.dot(S.T))/(Ne- 1)
         
        Usig,Sig,Vsig = cp.linalg.svd((Cdd + (1*CDd)), \
                                      full_matrices = False)
        Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
        valuesig = Bsig[-1]                 # last element
        valuesig = valuesig * 0.9999
        indices = ( Bsig >= valuesig ).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]
    
        (V,X,U) = pinvmatt((Cdd + (1*CDd)),tol)
        
        update_term=((Cyd.dot(X)).dot((np.tile(cp.asarray(True_data), Ne) +\
                                       cp.asarray(pertubations))\
                                      -Sim1 ))

        
        Ynew = Y + update_term 
        sizeclem=nx*ny*nz 
        if Yet==0:                   
            updated_ensemble=cp.asnumpy(Ynew[:sizeclem,:] )
            updated_ensemblep=cp.asnumpy(Ynew[sizeclem:2*sizeclem,:])
            if modelError==1:
                CM=cp.asnumpy(Ynew[2*sizeclem:,:])
            else:
                pass
        else:
            updated_ensemble=(Ynew[:sizeclem,:] )
            updated_ensemblep=(Ynew[sizeclem:2*sizeclem,:])
            if modelError==1:
                CM=(Ynew[2*sizeclem:,:])
            else:
                pass            
        
        ensemble=updated_ensemble
        ensemblep=updated_ensemblep        
        ensemble,ensemblep=honour2(ensemblep,\
                                     ensemble,nx,ny,nz,N_ens,\
                                         High_K,Low_K,High_P,Low_P)        
         
        
        simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
        tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
            /True_data.shape[0]
        print('RMSE of the ensemble mean = : ' \
              + str(tinuke) + '... .') 
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        simmbest=simDatafinal[:,clem].reshape(-1,1)
        tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
            /True_data.shape[0]
        print('RMSE of the ensemble best = : ' \
            + str(tinukebest) + '... .')

        cc_ini=cc
       
        if (choice==1) and (Geostats==1):
            ensemble=use_denoising(ensemble,nx,ny,nz,Ne)
        else:
            pass 
        
        ensemble,ensemblep=honour2(ensemblep,\
                                     ensemble,nx,ny,nz,N_ens,\
                                         High_K,Low_K,High_P,Low_P)          
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F') 
        controljj2p= np.reshape(meannp,(-1,1),'F') 
        controlj2=controljj2
        controlj2p=controljj2p
        
        
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
    
        
            Delete_files()                                
        Delete_files2()    
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder) 
        os.chdir('HM_RESULTS')                       
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        shpw=cc[clem]
        controlbest= np.reshape(ensemble[:,clem],(-1,1),'F') 
        controlbestp= np.reshape(ensemblep[:,clem],(-1,1),'F')
        controlbest2=controlj2#controlbest
        controlbest2p=controljj2p#controlbest
    
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F') 
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')  
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        Plot_Histogram_now(Ne,cc_ini,cc)
        #os.makedirs('ESMDA')
        if not os.path.exists('ES'):
            os.makedirs('ES')
        else:
            shutil.rmtree('ES')
            os.makedirs('ES') 
            
        shutil.copy2('masterreal.data','ES')
        print('6X Reservoir Simulator Forwarding - ES Model')
        Forward_model(oldfolder,'ES',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'ES')
        os.chdir('ES')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ES')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
             
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)
        
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM}) 
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding')
        
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files() 
        Delete_files2()            
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )     
        print('--------------------ES EXECUTED---------------------------------')        
     
    elif Technique_IES==2:
        print('-------------------------I-ES_Version_1-------------------------------------')
        print('History Matching using the Iterative Ensemble Smoother')
        print('')
        print('Reference:Chen, Y. and Oliver,D., .2011 Ensemble randomized maximum\
        likelihood method as an iterative ensemble smoother, Mathematical Geosciences,\
        Online First\n')    
        choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
            
        if De_alpha==1:
            alpha=10
        else:
            alpha=10
            
        Ne=N_ens
        Nop=True_data.shape[0]
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                #ini_ensemble=De_correlate_ensemble(nx,ny,nz,N_ens,High_K,Low_K)
                
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))   
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)
         
        ini_ensemblep=Getporosity_ensemble(ini_ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
    
        diffyet=np.abs(Basecomp-True_data)
        
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                if Big_noise==2:
                    ax[iq,:]=(noise_level*True_data[iq,:])**2
                else:
                    ax[iq,:]=(diffyet[iq,:])**2
                
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
    
        
        ensemblein=np.vstack([(ini_ensemble),ini_ensemblep,CM])       
        ensembleoutt, stats_iES,cc_ini = iES(High_K=High_K,Low_K=Low_K,High_P=High_P,\
        Low_P=Low_P,modelError=modelError,sizeclem=sizeclem,CM=CM,maxx=maxx,\
        ensemble = ensemblein.T,observation = True_data.reshape(-1),\
        obs_err_cov = CDd,flavour="Sqrt", MDA=False, bundle=False, \
        stepsize=1,nIter=alpha)
        
        ensembleoutt=ensembleoutt.T
        ensembleout=(ensembleoutt[:sizeclem,:])
        ensembleoutp=ensembleoutt[sizeclem:2*sizeclem,:]
        CM=ensembleoutt[2*sizeclem:,:]
        if modelError!=1:
            CM=np.zeros_like(CM)
            
          
        
        ensembleout,ensembleoutp=honour2(ensembleoutp,\
                                         ensembleout,nx,ny,nz,N_ens,High_K,\
                                             Low_K,High_P,Low_P)
            
        controljj2=np.reshape(np.mean(ensembleout,axis=1),(-1,1),'F')
        controljj2p=np.reshape(np.mean(ensembleoutp,axis=1),(-1,1),'F')
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controlj2=controljj2
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
                           
        #Forwarding
        #predMatrix=workflow(Ne,maxx,ensembleout,ensembleoutp)
        for i in range(N_ens):
            (
            write_include)(i,ensembleout,ensembleoutp,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)         
            Delete_files()
        Delete_files2()                
        if (choice==1) and (Geostats==1):
            ensembleout=use_denoising(ensembleout,nx,ny,nz,Ne)
        else:
            pass      
                
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder)
        os.chdir('HM_RESULTS')                     
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        
        
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
    
            
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        clembad=np.argmax(cc)
        controlbest= np.reshape(ensembleout[:,clem] ,(-1,1),'F')
        controlbest2p= np.reshape(ensembleoutp[:,clem] ,(-1,1),'F')
        controlbad= np.reshape(ensembleout[:,clembad] ,(-1,1),'F')
        controlbadp= np.reshape(ensembleoutp[:,clembad] ,(-1,1),'F')
        controlbest2=controlbest
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        
        Plot_Histogram_now(Ne,cc_ini,cc)
    
        if not os.path.exists('IES_v1'):
            os.makedirs('IES_v1')
        else:
            shutil.rmtree('IES_v1')
            os.makedirs('IES_V1') 
            
        shutil.copy2('masterreal.data','IES_V1')
        print('6X Reservoir Simulator Forwarding - IES_V1 model')
        Forward_model(oldfolder,'IES_V1',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'IES_V1')
        os.chdir('IES_V1')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','IES_V1')
        os.chdir(oldfolder)
        
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim
        
        
        #yycheck=yycheck+mEbest
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
                + str(cc) ) 
            
        os.chdir('HM_RESULTS')    
        Plot_mean(controlbest,controljj2,meanini,nx,ny) 
        os.chdir(oldfolder)
        
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensembleout,\
        'PORO_Reali':ensembleoutp,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM}) 
            
        os.chdir(oldfolder)   
            
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbest2p,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding - IES model')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files()
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,\
                            True_mat,Base_mat,"P10_P50_P90.jpg")       
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )     
        print('--------------------IES_V1 EXECUTED---------------------------')
    elif Technique_IES==3:
        print('-------------------------I-ES_Version_2-------------------------------------')
        print('History Matching using the Iterative Ensemble Smoother')
        print('')
        print('Reference: Xiadong Luo: Iterative Ensemble Smoother as an Approximate \
Solution to a Regularized Minimum-Average-Cost Problem: Theory and Applications\n')    
        choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')

            
        Ne=N_ens
        Nop=True_data.shape[0]
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                #ini_ensemble=De_correlate_ensemble(nx,ny,nz,N_ens,High_K,Low_K)
                
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))   
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)
         
        ini_ensemblep=Getporosity_ensemble(ini_ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
    
        diffyet=np.abs(Basecomp-True_data)
        
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                if Big_noise==2:
                    ax[iq,:]=(noise_level*True_data[iq,:])**2
                else:
                    ax[iq,:]=(diffyet[iq,:])**2
                
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        
        ensemblein=np.vstack([(ini_ensemble),ini_ensemblep,CM])       
        maxouter=20
        maxiner=5
        beta = 0 # $beta$ determines the threshold value in one of the stopping criteria
        maxOuterIter = maxouter # maximum iteration number in the outer loop
        maxInnerIter = maxiner # maximum iteration number in the inner loop
        init_lambda = 1 # initial lambda value
        lambda_reduction_factor = 0.9 # reduction factor in case to reduce gamma
        lambda_increment_factor = 2 # increment factor in case to increase gamma 
        doTSVD = 1 # do a TSVD on the cov of simulated obs? (1 = yes)

        tsvdCut = 0.99 # discard eigenvalues/eigenvectors if they are not among the truncated leading ones

        min_RN_change = 2 # minimum residual norm (RN) change (in percentage); RN(k) - RN(k+1) > RN(k) * min_RN_change / 100
        
        iterr=0
        lambda_=init_lambda

        measurement=True_data
        nd=len(measurement)
       
        ensemble=ensemblein
        ne=ensemble.shape[1]
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        


        W=np.ones(len(measurement))
        W=np.reshape(W,(-1,1))
        
        CDd=cp.asarray(CDd)
        True_data=cp.asarray(True_data)
        pertubations=cp.random.multivariate_normal(cp.ravel(cp.zeros\
                                                        ((Nop,1))), CDd,ne)
        Ddraw=cp.tile(True_data, ne) 
        pertubations=pertubations.T
        Dd=Ddraw + pertubations
        if Yet==0:
            CDd=cp.asnumpy(CDd)
            perturbedData=cp.asnumpy(Dd)
            True_data=cp.asnumpy(True_data)  
        else:
            pertubedData=Dd
        
       
        # simulated observations of the ensemble members
        nm=len(ensemble)

        ensembleout=(ensemble[:sizeclem,:])
        ensembleoutp=ensemble[sizeclem:2*sizeclem,:]

        CM=ensemble[2*sizeclem:,:]
        if modelError!=1:
            CM=np.zeros_like(CM)                        
                        
        for i in range(N_ens):
            (
            write_include)(i,ensembleout,ensembleoutp,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)         
            Delete_files() 
        Delete_files2()            
        os.chdir('HM_RESULTS')                     
        Plot_RSM(predMatrix,modelError,CM,N_ens,True_mat,"initial.jpg")
        os.chdir(oldfolder)                       
                        
        simData =  Get_simulated(predMatrix,modelError,CM,N_ens)# #                         

        obj,objStd,objReal=funcGetDataMismatch(simData,measurement)
        objThreshold=np.dot(beta ** 2,nd)

        isTooSmallRNChange=0

        exitFlag=np.array([0,0,0])
       
        #-------------#
        #-- run iES --#
        #-------------#
        #tsvdCut=0.99
        # outer iteration loops
        while (iterr < maxOuterIter) and  (obj > objThreshold):
            #print(str(iterr+1) + '  out of ' + str(maxOuterIter))
            print('-- Outer iteration step: ' + str(iterr)) 
            deltaM=np.ones((nm,ne))
            deltaD=np.ones((nd,ne))
            mean_t=np.mean(ensemble,axis=1)
            mean_t2=np.mean(simData,axis=1)
            for i in range(ne):
            
                deltaM[:,i]=ensemble[:,i]- mean_t
                # deviation of the simulated observations
                deltaD[:,i]=simData[:,i] - mean_t2
            Ud,Wd,Vd= np.linalg.svd(deltaD, full_matrices=True)
           # Wd=np.reshape(Wd,(-1,1))
            val=np.diag(Wd)
            total=np.sum(val)
            for j in range(ne):
                svdPd=j
                if (np.sum(val[:j]) / total > tsvdCut):
                
                    break
            #svdPd=np.asscalar(svdPd)    
            Vd=Vd[:,:svdPd]
            Ud=Ud[:,:svdPd]
            Wd=Wd[:svdPd]
            del(val)
            #-- initialization of the inner loop --#
            iterLambda=1
            while iterLambda < maxInnerIter:
                print('    -- Inner iteration step: ' + str(iterLambda) + '... .')
                ensembleOld=(ensemble)
                meanOld=mean_t
                simDataOld=(simData)
                simOld=np.mean(simData,axis=1)
                if doTSVD==1:
                    alpha= np.dot(lambda_,np.sum(Wd ** 2)) / svdPd#(lambda_ @ np.sum(Wd ** 2))/svdPd #
                    yy=Wd / (Wd ** 2 + alpha)
                    yy=np.reshape(yy,(-1,1))
                    x1=np.dot(Vd,scipy.sparse.spdiags(yy.T,0,svdPd,svdPd).toarray()) #Vd @scipy.sparse.spdiags(yy.T,0,svdPd,svdPd).toarray()
                    KGain= np.dot(np.dot(deltaM,x1),Ud.T)#(deltaM@x1)@Ud.T #

                    
                else:
                    alpha=np.dot(lambda_,np.sum(np.sum(deltaD ** 2))) / nd
                    
                    first=np.dot(deltaM,deltaD.T)
                    second=(np.dot(deltaD,deltaD.T) + np.dot(alpha,np.eye(nd)))
                    
                    Usig,Sig,Vsig = np.linalg.svd(second, full_matrices = True)
                    Bsig = np.cumsum(Sig, axis = 0)          # vertically addition
                    valuesig = Bsig[-1]                 # last element
                    valuesig = valuesig * 0.99
                    indices = ( Bsig >= valuesig ).ravel().nonzero()
                    toluse = Sig[indices]
                    tol = toluse[0]
                    
                    KGain = np.dot(first,pinvmat(second,tol))
 
                ensemble=ensemble- KGain@\
                                    (simData- perturbedData)
                meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
                #ensemble=np.concatenate((iterated_ensemble, meann), axis=1)
                
                changeM=np.sqrt(np.sum((meann - np.mean(ensembleOld,axis=1).\
                                reshape(-1,1)) ** 2) / nm)
                print('        average change (in RMSE) of the ensemble mean = : ' \
                      + str(changeM) + '... .') 

                
                ensembleout=(ensemble[:sizeclem,:])
                ensembleoutp=ensemble[sizeclem:2*sizeclem,:]
                
                ensembleout,ensembleoutp=honour2(ensembleoutp,\
                                         ensembleout,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)                  
                
        
                CM=ensemble[2*sizeclem:,:]
                if modelError!=1:
                    CM=np.zeros_like(CM)                        
                                
                for i in range(N_ens):
                    (
                    write_include)(i,ensembleout,ensembleoutp,'Realization_')      
                az=int(np.ceil(int(N_ens/maxx)))
                a=(np.linspace(1, N_ens, num=Ne))
                use1=Split_Matrix (a, az)
                predMatrix=[]
                print('        6X Reservoir Simulator Forwarding')
                for xx in range(az):
                    print( '         Batch ' + str(xx+1) + ' | ' + str(az))
                    #xx=0
                    ause=use1[xx]
                    ause=ause.astype(np.int32)
                    ause=listToStringWithoutBrackets(ause)
                    overwrite_Data_File(name,ause)
                    #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                    os.system(string_Jesus)
                    for kk in range(maxx):
                        folder=stringf + str(kk)
                        namecsv=stringf + str(kk) +'.csv'
                        predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                        predMatrix.append(predd)         
                    Delete_files() 
                Delete_files2()                       
                                
                simData =  Get_simulated(predMatrix,modelError,CM,N_ens)# #                 

                    
                objNew,objStdNew,objRealNew=funcGetDataMismatch(simData,\
                                                                measurement)
                mean_tinu=np.mean(simData,axis=1)
                tinuke=(np.sum((((mean_tinu) - measurement) ** 2)) )  **(0.5)      
                print('        RMSE of the ensemble mean = : ' \
                      + str(tinuke) + '... .')                 
                  
                tmp_objReal=(objReal)
                objReal=(objRealNew)
                objReal=(tmp_objReal)
                del(tmp_objReal)
                if objNew > obj:
                    lambda_=np.dot(lambda_,lambda_increment_factor)
                    print('         increasing Lambda to: ' + str(lambda_) + '... .')
                    iterLambda=iterLambda+ 1
                    simData=(simDataOld)
                    ensemble=(ensembleOld)
                else:
                    changeStd=(objStdNew - objStd) / objStd
                    lambda_=np.dot(lambda_,lambda_reduction_factor)
                    print('         Reducing Lambda to: ' + str(lambda_) + '... .')
                    iterr=iterr+1
                    simDataOld=(simData)
                    ensembleOld=(ensemble)
                    objStd=(objStdNew)
                    obj=(objNew)
                    objReal=(objRealNew)
                    break

            # if a better update not successfully found
            if iterLambda >= maxInnerIter:
                lambda_=np.dot(lambda_,lambda_increment_factor)
                if lambda_ < init_lambda:
                    lambda_=init_lambda
                iterr=iterr+1
            if isTooSmallRNChange:

                exitFlag[3]=1
                break

        
        if iterr >= maxOuterIter:
            exitFlag[1]=1

        if obj <= objThreshold:

            exitFlag[2]=1  
        
        ensembleoutt=ensemble
        ensembleout=(ensembleoutt[:sizeclem,:])
        ensembleoutp=ensembleoutt[sizeclem:2*sizeclem,:]
        CM=ensembleoutt[2*sizeclem:,:]
        if modelError!=1:
            CM=np.zeros_like(CM)
            
          
        
        ensembleout,ensembleoutp=honour2(ensembleoutp,\
                                         ensembleout,nx,ny,nz,N_ens,High_K,\
                                             Low_K,High_P,Low_P)
            
        controljj2=np.reshape(np.mean(ensembleout,axis=1),(-1,1),'F')
        controljj2p=np.reshape(np.mean(ensembleoutp,axis=1),(-1,1),'F')
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controlj2=controljj2
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')

        for i in range(N_ens):
            (
            write_include)(i,ensembleout,ensembleoutp,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)         
            Delete_files()
        Delete_files2()
        if (choice==1) and (Geostats==1):
            ensembleout=use_denoising(ensembleout,nx,ny,nz,Ne)
        else:
            pass      
                
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder)
        
        os.chdir('HM_RESULTS')                     
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        
        
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
    
            
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        clembad=np.argmax(cc)
        controlbest= np.reshape(ensembleout[:,clem] ,(-1,1),'F')
        controlbest2p= np.reshape(ensembleoutp[:,clem] ,(-1,1),'F')
        controlbad= np.reshape(ensembleout[:,clembad] ,(-1,1),'F')
        controlbadp= np.reshape(ensembleoutp[:,clembad] ,(-1,1),'F')
        controlbest2=controlbest
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        
        Plot_Histogram_now(Ne,cc_ini,cc)
    
        if not os.path.exists('IES_V2'):
            os.makedirs('IES_V2')
        else:
            shutil.rmtree('IES_V2')
            os.makedirs('IES_V2') 
            
        shutil.copy2('masterreal.data','IES_V2')
        print('6X Reservoir Simulator Forwarding - IES_V2 model')
        Forward_model(oldfolder,'IES_V2',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'IES_V2')
        os.chdir('IES_V2')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','IES_V2')
        os.chdir(oldfolder)
        
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim
        
        
        #yycheck=yycheck+mEbest
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
                + str(cc) ) 
            
        os.chdir('HM_RESULTS')    
        Plot_mean(controlbest,controljj2,meanini,nx,ny) 
        os.chdir(oldfolder)
        
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensembleout,\
        'PORO_Reali':ensembleoutp,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM}) 
            
        os.chdir(oldfolder)   
            
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbest2p,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding - IES_Version_2 model')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files() 
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,\
                            True_mat,Base_mat,"P10_P50_P90.jpg")       
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )     
        print('--------------------IES V_2 EXECUTED--------------------------')
    print('-----------------------SECTION EXECUTED---------------------------')        
elif Data_Assimilation==2:  
    print(' Data Asimilation using the ESMDA')
    print('Reference:Emerick, A. A. and Reynolds A. C., 2013 Ensemble smoother\
with multiple data assimilation, Computers & Geosciences.\n')      
    print('')
    print('-----------------------------------------------------------------')    
    Technique_DA=int(input('Enter optimisation: \n\
    1 = ESMDA Vanilla\n\
    2 = ESMDA_Level set\n\
    3 = ESMDA Localisation\n\
    4 = ESMDA-GEO\n\
    5 = ESMDA-CCR\n\
    6 = ESMDA_autoencoder\n\
    7 = ESMDA_DNA\n\
    8 = ESMDA_DCT\n\
    9 = ESMDA_KSVD\n\
    10 = ESMDA_LBFGS\n\
    11 = ESMDA_CE\n\
    12 = ESMDA_GEO_2\n'))    

    if Technique_DA==1:
        print('--------------------------ES-MDA----------------------------------')
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')      
        print('History Matching using the Ensemble Smoother with Multiple Data \
Asiimilation\n')
        print('')
        choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        os.chdir(oldfolder)
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input(' Enter the Inflation parameter from 4-8) : '))   
    
    
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))           
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)         
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        #CDd=np.dot(ax,ax.T)
        for ii in range(alpha):
            
            print( str(ii+1) + ' | ' + str(alpha))
            
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    foldern=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,foldern,namecsv)
                    predMatrix.append(predd)        
                Delete_files()
            Delete_files2()                             
            print('Read Historical data')
            os.chdir('True_Model')
            True_measurement=Get_Measuremnt_CSV('hm0.csv')
            True_mat=True_measurement
            True_data=True_mat[:,1:]
            True_data=np.reshape(True_data,(-1,1),'F')        
            os.chdir(oldfolder)                                    
            
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,\
                         "Initial.jpg")
                os.chdir(oldfolder)
            else:
                pass
            
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#     
            updated_ensemble,updated_ensemblep,CM=ESMDA(ensemble,ensemblep,modelError,\
                                            CM,True_data, \
                                    Ne, simDatafinal,alpha)
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep
    
            
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)            
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            if ii==0:
                cc_ini=cc
            else:
                pass
            
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .') 
            
        if (choice==1) and (Geostats==1):
            ensemble=use_denoising(ensemble,nx,ny,nz,Ne)
        else:
            pass 
        
        ensemble,ensemblep=honour2(ensemblep,\
                                     ensemble,nx,ny,nz,N_ens,\
                                         High_K,Low_K,High_P,Low_P)          
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F') 
        controljj2p= np.reshape(meannp,(-1,1),'F') 
        controlj2=controljj2
        controlj2p=controljj2p
        
        
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
    
        
            Delete_files()
        Delete_files2()                                   
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder)                        
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        shpw=cc[clem]
        controlbest= np.reshape(ensemble[:,clem],(-1,1),'F') 
        controlbestp= np.reshape(ensemblep[:,clem],(-1,1),'F')
        controlbest2=controlj2#controlbest
        controlbest2p=controljj2p#controlbest
    
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F') 
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')  
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        
        Plot_Histogram_now(Ne,cc_ini,cc)
        
        #os.makedirs('ESMDA')
        if not os.path.exists('ESMDA'):
            os.makedirs('ESMDA')
        else:
            shutil.rmtree('ESMDA')
            os.makedirs('ESMDA') 
            
        shutil.copy2('masterreal.data','ESMDA')
        print('6X Reservoir Simulator Forwarding - ESMDA Model')
        Forward_model(oldfolder,'ESMDA',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'ESMDA')
        os.chdir('ESMDA')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ESMDA')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
             
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')    
        Plot_mean(controlbest,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)
        
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM})  
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding')
        
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files()
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )     
        
        
        print('--------------------VANILLA ESMDA EXECUTED--------------------')
    elif Technique_DA==2:
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        print('--------------------------ESMDA Levelset--------------------------')
        print('History Matching using the Ensemble Smoother with Multiple Data\
Assimilation with Level set parametrisation\n')
        print('Etienam, C., Villegas, R., & Dorn ,O 2018. An Improved Reservoir\
Model Calibration through Sparsity Promoting ES-MDA. ECMI conference proceeding.\
20th European Conference on Mathematics for Industry (ECMI) 18-22 June 2018,\
Budapest, Hungary\n ')
        print('')
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input(' Enter the Inflation parameter from 4-8) : '))     
        os.chdir(oldfolder)
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))              
            else:            
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)           
                
        Experts=2
        ensuse1=ini_ensemble
        ensemblep=Getporosity_ensemble(ensuse1,machine_map,N_ens)
        mexini=np.zeros((Experts,Ne))
        sdiini=np.zeros((ini_ensemble.shape[0],ini_ensemble.shape[1]))
        for j in range(Ne):
            sdi,mexx,_=get_clus(ensuse1[:,j],Experts)
            sdiini[:,j]=np.ravel(sdi)
            mexini[:,j]=np.ravel(mexx)
        sdiuse=sdiini     
                
        for i in range(alpha):
            print( str(i+1) + ' | ' + str(alpha))                            
    
            #predMatrix=workflow(Ne,maxx,ensuse1,ensemblep)
            
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensuse1,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                #xx=0
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    folder=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                    predMatrix.append(predd)
           
                Delete_files()
            Delete_files2()                
            print('Read Historical data')
            os.chdir('True_Model')
            True_measurement=Get_Measuremnt_CSV('hm0.csv')
            True_mat=True_measurement
            True_data=True_mat[:,1:]
            True_data=np.reshape(True_data,(-1,1),'F')        
            os.chdir(oldfolder)        
                                        
            if i==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,\
                         True_mat,"Initial.jpg")
                os.chdir(oldfolder)
            
            simu = Get_simulated(predMatrix,modelError,CM,Ne)#             
            
            updatedp,updatedporo,CM=ESMDA(ensuse1,ensemblep,modelError,\
                                          CM,True_data, Ne, simu,alpha)
            sdiupdated=ESMDA_Levelset (Ne,sdiuse,simu,alpha,Nop,\
                                            True_data,nx,ny,nz)
            
            meximp=np.zeros((Experts,Ne))
            models_big=[]
            for ja in range(Ne):
                _,mexx1,modelss=get_clus(sdiupdated[:,ja],Experts)
                meximp[:,ja]=np.ravel(mexx1)
                models_big.append(modelss)
        
            Xspit2=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            rectifyy)(updatedp[:,k],sdiupdated[:,k],meximp[:,k],\
                            Experts,nx,ny,nz,models_big[k])for k in range(Ne) )
            updated = np.reshape(np.hstack(Xspit2),(-1,Ne),'F')#
        
            
            ensuse1=updated
            ensemblep=updatedporo
            #ensemblep=Getporosity_ensemble(ensuse1,machine_map,N_ens)
            
            ensuse1,ensemblep=honour2(ensemblep,\
                                         ensuse1,nx,ny,nz,N_ens,\
                                            High_K,Low_K,High_P,Low_P)             
            
            
            sdiuse=sdiupdated        
            simDatafinal = simu#                           
            simmean=np.reshape(np.mean(simu,axis=1),(-1,1),'F')
            
            
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            clembad=np.argmax(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5)) \
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .') 
            if i==0:
                cc_ini=cc
            else:
                pass
        
    
        meann=np.reshape(np.mean(ensuse1,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F')  
        controlj2=controljj2
        
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        controljj2p= np.reshape(meannp,(-1,1),'F')  
        controlj2p=controljj2p
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')                            
        #predMatrix=workflow(Ne,maxx,ensuse1,ensemblep)
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensuse1,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
           # os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
      
            Delete_files() 
        Delete_files2()            
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder)
                               
        os.chdir('HM_RESULTS')
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
                            
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#        
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        controlbest= ensuse1[:,clem] 
        controlbestp= ensemblep[:,clem]
        controlbest2=controlj2#controlbest
        controlbest2p=controlj2p#controlbest
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensuse1[:,clembad] ,(-1,1),'F')
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')       
        Plot_Histogram_now(Ne,cc_ini,cc)
        #controlbest2p=machine_map.predict(controlbest2)
        if not os.path.exists('ESMDA_LS'):
            os.makedirs('ESMDA_LS')
        else:
            shutil.rmtree('ESMDA_LS')
            os.makedirs('ESMDA_LS') 
            
        shutil.copy2('masterreal.data','ESMDA_LS') 
        print('6X Reservoir Simulator Forwarding - ESMDA_LS Model')    
        Forward_model(oldfolder,'ESMDA_LS',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'ESMDA_LS') 
        os.chdir('ESMDA_LS')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ESMDA_LS')
        os.chdir(oldfolder)
    
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim
               
        
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5) )\
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F') 
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM})
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding - IES model')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files() 
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
    
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )      
        print('--------------------ESMDA LEVELSET EXECUTED-------------------')        
        
    elif Technique_DA==3:
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')
        print('----------------------ESMDA Localisation--------------------------')
        print('History Matching using the Ensemble Smoother with Multiple Data\
assimialtion with covariance localisation\n')
        print('')
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
    
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input(' Enter the Inflation parameter from 4-8) : '))     
    
        os.chdir(oldfolder)
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))              
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)         
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
            
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        #CDd=np.dot(ax,ax.T)
        for ii in range(alpha):
            
            print( str(ii+1) + ' | ' + str(alpha))            
            
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                #xx=0
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    folder=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                    predMatrix.append(predd)
        
                Delete_files()
            Delete_files2()                                        
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#              
        
            updated_ensemble,updated_ensemblep,CM=ESMDA_Localisation(ensemble,ensemblep,\
                            modelError,CM,True_data,Ne,\
                                            simDatafinal,alpha,\
                                                nx,ny,1,10)            
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep
    
            
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)             
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))  \
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5) )\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .') 
            if ii==0:
                cc_ini=cc
            else:
                pass
            
        if (choice==1) and (Geostats==1):
            ensemble=use_denoising(ensemble,nx,ny,nz,Ne)
        else:
            pass 
        
        ensemble,ensemblep=honour2(ensemblep,\
                                     ensemble,nx,ny,nz,N_ens,\
                                         High_K,Low_K,High_P,Low_P)                      
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F')  
        controlj2=controljj2
        
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        controljj2p= np.reshape(meannp,(-1,1),'F')  
        controlj2p=controljj2p
    
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
        
            Delete_files() 
        Delete_files2()            
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder) 
        os.chdir('HM_RESULTS')                       
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#          
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        
        controlbest= ensemble[:,clem] 
        controlbest2=controlj2#controlbest
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F')  
    
        controlbestp= ensemblep[:,clem] 
        controlbest2p=controlj2p#controlbest
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')
        
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')    
        Plot_Histogram_now(Ne,cc_ini,cc)
        if not os.path.exists('ESMDA_locali'):
            os.makedirs('ESMDA_locali')
        else:
            shutil.rmtree('ESMDA_locali')
            os.makedirs('ESMDA_locali') 
            
        shutil.copy2('masterreal.data','ESMDA_locali')
        print('6X Reservoir Simulator Forwarding - ESMDA_locali Model')         
        Forward_model(oldfolder,'ESMDA_locali',controlbest2,\
                      controlbest2p)
        yycheck=Get_RSM(oldfolder,'ESMDA_locali') 
        os.chdir('ESMDA_locali')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ESMDA_locali')
        os.chdir(oldfolder)  
    
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim
           
          
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5) )/\
            True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) ) 
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F') 
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny) 
        os.chdir(oldfolder)
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM})
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
    
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding - IES model')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files()
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )       
        print('-----------------ESMDA LOCALISATION EXECUTED------------------')        
        
    elif Technique_DA==4:
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')
        os.chdir(oldfolder)
        print('-------------------------ES-MDA GEO-------------------------------')
        print('History Matching using the Ensemble Smoother with Multiple data\
assimilation with Geometric inflation factors\n')
        print('') 
        choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored') 
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input('Enter the Inflation parameter from 4-8) : '))     
    
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                #ini_ensemble=De_correlate_ensemble(nx,ny,nz,N_ens,High_K,Low_K)
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))              
            else:            
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)
                
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        
        ensemble,ensemblep=honour2(ensemblep,\
                                          ensemble,nx,ny,nz,N_ens\
                                              ,High_K,Low_K,High_P,Low_P)         
        ax=cp.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=cp.sqrt(cp.asarray(noise_level)*cp.asarray(True_data[iq,:]))
                else:
                    ax[iq,:]=cp.sqrt(cp.asarray(diffyet[iq,:]))
                
        ax=cp.reshape(ax,(-1,))    
        CDd=cp.diag(ax)
        #CDd=np.dot(ax,ax.T)
        
        alphabig=[]
        for ii in range(alpha):
            
            print( str(ii+1) + ' | ' + str(alpha))
           
            #predMatrix=workflow(Ne,maxx,ensemble,ensemblep)
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                #xx=0
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    folder=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                    predMatrix.append(predd)
    
                
                Delete_files()
            Delete_files2()                
            print('Read Historical data')
            os.chdir('True_Model')
            True_measurement=Get_Measuremnt_CSV('hm0.csv')
            True_mat=True_measurement
            os.chdir(oldfolder)                            
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,\
                         True_mat,"Initial.jpg")
                os.chdir(oldfolder)
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#  
            #simDatafinal=simDatafinal+CM               
            stdall=[]
            ff=cp.asarray(True_data)
   
    
            stdall=cp.reshape(ax,(-1,1))
            
            nobs = ff.shape[0]
            noise = cp.random.randn(max(10000,nobs),1)
            
            Error1=stdall
            sig=Error1
            for im in range (ff.shape[0]):
                noisee=noise[im,:]
                ff[im,:] = ff[im,:] + sig[im,:]*noise[-1-nobs+im,:]
        
            R = sig**2
        
            Cd2=cp.diag(np.reshape(R,(-1,)))
            sgsim=ensemble
            if modelError==1:
                overall=cp.vstack([cp.asarray(sgsim),cp.asarray(ensemblep),\
                                  cp.asarray(CM)])
            else:
                overall=cp.vstack([cp.asarray(sgsim),cp.asarray(ensemblep)])
            
            Y=overall 
            Sim1=cp.asarray(simDatafinal)
            M = cp.mean(Sim1,axis=1)
        
            M2=cp.mean(overall,axis=1)
            
            
            S = cp.zeros((Sim1.shape[0],Ne))
            yprime = cp.zeros((Y.shape[0],Y.shape[1]))
                   
            for jc in range(Ne):
                S[:,jc] = Sim1[:,jc]- M
                yprime[:,jc] = overall[:,jc] - M2
            Cyd = (yprime.dot(S.T))/(Ne - 1)
            Cdd = (S.dot(S.T))/(Ne- 1)
            
            deltaM=yprime/((Ne-1)**(0.5))
            deltaD=Sim1/((Ne-1)**(0.5))
            Gd=((cp.linalg.inv(CDd))**(0.5)).dot(deltaD)
            
            Usig1,Sig1,Vsig1 = cp.linalg.svd((Gd), full_matrices = False)
            Bsig1 = cp.cumsum(Sig1, axis = 0)          # vertically addition
            valuesig1 = Bsig1[-1]                 # last element
            valuesig1 = valuesig1 * 0.9999
            indices1 = ( Bsig1 >= valuesig1 ).ravel().nonzero()
            toluse1 = Sig1[indices1]
            tol1 = toluse1[0]            
            Va,S1a,Ua = cp.linalg.svd(Gd,full_matrices=0)
        
            r1a = cp.sum(S1a > tol1)+1
            v = Va[:,:r1a-1] #needed
            U1a = Ua.T
            ua = U1a[:,:r1a-1]
            ua=ua.T#needed
            S11a = S1a[:r1a-1]#needed
            
            if ii==0:
                S11a=cp.reshape(S11a,(-1,1),'F')
                landaa=cp.mean(S11a,axis=0)
                landaa=cp.asnumpy(landaa)
                alpha=cp.asnumpy(alpha)
                alpha=np.float(np.asscalar(alpha))
                landaa=np.asscalar(landaa)
                alpha1=max(landaa, alpha)                
  
                bnds=[(0, 1)]
                # if Yet==0:
                #     alpha,alpha1=cp.asnumpy(alpha),cp.asnumpy(alpha1)
                # else:
                #     pass
                resultt=scipy.optimize.minimize(objective_Na, x0=0.06, \
                args=(alpha,alpha1), \
                    method='SLSQP', \
                bounds=bnds, options={'xtol': 1e-10,\
                                                       'ftol': 1e-10})
                beta=np.reshape(resultt.x,(1,-1),'F') 
                
    
                print('alpha = ' + str(alpha1)+ '\n')
                #beta=np.abs(beta)
            else:
                alpha1=beta*alpha1
                izz=ii+1
                print('alpha = ' + str(alpha1)+ '\n')
            alphabig.append(alpha1)
         
            
            Dj =np.matlib.repmat(cp.asnumpy(ff), 1, Ne)
            rndm=np.zeros((Dj.shape[0],Ne))
            for ik in range (Dj.shape[0]):
                #i=0
                kkk=rndm[ik,:]
                kkk=np.reshape(kkk,(1,-1),'F')
                rndm[ik,:] = np.random.randn(1,Ne) 
                rndm[ik,:] = rndm[ik,:] - np.mean(kkk,axis=1)
                rndm[ik,:] = rndm[ik,:] / np.std(kkk, axis=1)
                Dj[ik,:] = Dj[ik,:] + np.sqrt((alpha1))*\
                    np.sqrt(cp.asnumpy(R[ik,])) \
                    * rndm[ik,:]   
                    
            Dj=cp.asarray(Dj)       
            Usig,Sig,Vsig = cp.linalg.svd((Cdd + (cp.asarray(alpha1)*Cd2)), \
                                          full_matrices = False)
            Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
            valuesig = Bsig[-1]                 # last element
            valuesig = valuesig * 0.9999
            indices = ( Bsig >= valuesig ).ravel().nonzero()
            toluse = Sig[indices]
            tol = toluse[0]
        
            (V,X,U) = pinvmatt((Cdd + (cp.asarray(alpha1)*Cd2)),tol)
            
            update_term=((Cyd.dot(X)).dot(Dj - Sim1))
            Ynew = Y + update_term 
            sizeclem=nx*ny*nz 
            if Yet==0:                   
                updated_ensemble=cp.asnumpy(Ynew[:sizeclem,:]) 
                updated_ensemblep=cp.asnumpy(Ynew[sizeclem:2*sizeclem,:])
                if modelError==1:
                    CM=cp.asnumpy(Ynew[2*sizeclem:,:])
                else:
                    pass
            else:
                updated_ensemble=(Ynew[:sizeclem,:]) 
                updated_ensemblep=(Ynew[sizeclem:2*sizeclem,:])
                if modelError==1:
                    CM=(Ynew[2*sizeclem:,:])
                else:
                    pass                
            
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)  
            
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .')
            if ii==0:
                cc_ini=cc
            else:
                pass
            
        if (choice==1) and (Geostats==1):        
            ensemble=use_denoising(ensemble,nx,ny,nz,Ne)
        else:
            pass        
        alphabig =np.array(alphabig )
        alphabig =np.reshape(alphabig ,(-1,1))       
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        meannini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F')  
        controlj2=controljj2
        controljj2p= np.reshape(meannp,(-1,1),'F')  
        controlj2p=controljj2p
    
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
        
            Delete_files()
        Delete_files2()            
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        os.chdir(oldfolder) 
        
        os.chdir('HM_RESULTS')                        
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#          
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        controlbest= ensemble[:,clem].reshape(-1,1)
        controlbest2=controlj2#controlbest
        
        controlbestp= ensemblep[:,clem].reshape(-1,1) 
        controlbest2p=controlj2p#controlbest
        Plot_Histogram_now(Ne,cc_ini,cc)
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F')
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F') 
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')    
        #controlbest2p=machine_map.predict(controlbest2)
        if not os.path.exists('ESMDA_GEO'):
            os.makedirs('ESMDA_GEO')
        else:
            shutil.rmtree('ESMDA_GEO')
            os.makedirs('ESMDA_GEO') 
        shutil.copy2('masterreal.data','ESMDA_GEO') 
        print('6X Reservoir Simulator Forwarding - ESMDA_GEO Model')        
        Forward_model(oldfolder,'ESMDA_GEO',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'ESMDA_GEO')
        os.chdir('ESMDA_GEO')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ESMDA_GEO')
        os.chdir(oldfolder)
    
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim
         
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meannini,nx,ny)
        os.chdir(oldfolder)
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM})
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files() 
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )      
        
        for kka in range(2):
            sigma_est = np.mean(estimate_sigma(controljj2, multichannel=True)) 
            patch_kw = dict(patch_size=5,      # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    multichannel=True)
            denoise_fast = denoise_nl_means(controljj2, h=0.8 * sigma_est, \
                                            fast_mode=True,
                                    **patch_kw)
            controljj2=np.reshape(denoise_fast,(-1,1))
            
            controljj2[controljj2<=Low_K]=Low_K
            controljj2[controljj2>=High_K]=High_K
        
        print('--------------------ESMDA_GEO EXECUTED------------------------') 
        
    elif Technique_DA==5:
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')
        os.chdir(oldfolder)
        print('-------------------------ES-MDA CCR------------------------------')
        print('History Matching using the Ensemble Smoother with Multiple data\
assimialtion with Mixture of Experts prior\n') 
        print('References for CCR include: ')
        print(' (1): David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park,\
Kody J. H. Law, and Clement Etienam. Cluster, classify, regress:A general\
method for learning discontinuous functions.Foundations of Data Science, \
1(2639-8001-2019-4-491):491, 2019.\n')
        print('')
        print(' (2): Clement Etienam, Kody Law, Sara Wade. Ultra-fast Deep Mixtures of\
Gaussian Process Experts. arXiv preprint arXiv:2006.13309, 2020.\n')
        print('') 
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')  
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input(' Enter the Inflation parameter from 4-8) : '))     
    
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                #ini_ensemble=De_correlate_ensemble(nx,ny,nz,N_ens,High_K,Low_K)
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))              
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)        
        
        for j in range(N_ens):
            folder = 'Realizationn_%d'%(j)
            if not os.path.exists(folder):
                os.makedirs(folder)
            else:
                shutil.rmtree(folder)
                os.makedirs(folder)       
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        
        ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                            High_K,Low_K,High_P,Low_P)         
        
        Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Learn_CCR_Ensemble)('Realizationn_',oldfolder,jj,ensemble[:,jj],\
                           ensemble[:,jj],degg)\
                                 for jj in range(Ne) )   
        
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])    
        

        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        #CDd=np.dot(ax,ax.T)
        for ii in range(alpha):
            
            print( str(ii+1) + ' | ' + str(alpha))
        
            
            Expermeant=\
                Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Extract_Machine1)('Realizationn_',oldfolder,i,2)\
                                 for i in range(Ne) )
                    
                    
            Expercovart=\
                Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Extract_Machine2)('Realizationn_',oldfolder,i,2)\
                                 for i in range(Ne) )                    
                    
            model0t=\
                Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Extract_Machine3)('Realizationn_',oldfolder,i,2)\
                                 for i in range(Ne) )                    
                    
            shapemeant=\
                Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Extract_Machine4)('Realizationn_',oldfolder,i,2)\
                                 for i in range(Ne) ) 
                    
            shapecovat=\
                Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Extract_Machine5)('Realizationn_',oldfolder,i,2)\
                                 for i in range(Ne) )                    
                    
            shapeclasst=\
                Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Extract_Machine6)('Realizationn_',oldfolder,i,2)\
                                 for i in range(Ne) )                    
            
            En1=np.hstack(Expermeant)
            En2=np.hstack(Expercovart)
            En3=np.hstack(model0t)
    
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                #xx=0
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    folder=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                    predMatrix.append(predd)
         
            
                Delete_files() 
            Delete_files2()                                   
            print('Read Historical data')
            os.chdir('True_Model')
            True_measurement=Get_Measuremnt_CSV('hm0.csv')
            True_mat=True_measurement
            os.chdir(oldfolder)                                    
     
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,\
                         True_mat,"Initial.jpg")
                os.chdir(oldfolder)
            
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#     
            updated_ensemble,ensemblep,CM=ESMDA(ensemble,ensemblep,modelError,CM,\
                                                True_data,\
                                        Ne, simDatafinal,alpha)
            updated_1=ESMDA_CCR(En1,True_data,Ne, simDatafinal,alpha)
            updated_2=ESMDA_CCR(En2,True_data,Ne, simDatafinal,alpha)
            updated_3=ESMDA_CCR(En3,True_data, Ne, simDatafinal,alpha)
    
        
            
            Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Insert_Machine)('Realizationn_',oldfolder,2,updated_1[:,jj],\
                           updated_2[:,jj],\
                   updated_3[:,jj],shapemeant[0],shapecovat[0],shapeclasst[0],jj)\
                                 for jj in range(Ne) ) 
        
            
            ensembleI=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Predict_CCR_Ensemble)('Realizationn_',oldfolder,jj,\
                                 updated_ensemble[:,jj],degg)\
                                 for jj in range(Ne) )             
        
            ensemble=np.hstack(ensembleI)
            #ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
            
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                            High_K,Low_K,High_P,Low_P)             
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .') 
            if ii==0:
                cc_ini=cc
            else:
                pass
            
        Remove_folder(N_ens,'Realizationn_')    
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F')  
        controlj2=controljj2
        
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        controljj2p= np.reshape(meannp,(-1,1),'F')  
        controlj2p=controljj2p
                                    
        #predMatrix=workflow(Ne,maxx,ensemble,ensemblep)
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
     
        
            Delete_files() 
        Delete_files2()    
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        os.chdir(oldfolder) 
        os.chdir('HM_RESULTS')                                                        
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        controlbest= ensemble[:,clem] 
        controlbest2=controlj2#controlbest
        
        controlbestp= ensemblep[:,clem] 
        controlbest2p=controlj2p#controlbest
        
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F') 
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')
        
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')     
        Plot_Histogram_now(Ne,cc_ini,cc)
        #controlbest2p=machine_map.predict(controlbest2)
        if not os.path.exists('ESMDA_CCRR'):
            os.makedirs('ESMDA_CCRR')
        else:
            shutil.rmtree('ESMDA_CCRR')
            os.makedirs('ESMDA_CCRR') 
        shutil.copy2('masterreal.data','ESMDA_CCRR') 
        print('6X Reservoir Simulator Forwarding - ESMDA_CCRR Model')        
        Forward_model(oldfolder,'ESMDA_CCRR',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'ESMDA_CCRR')
        os.chdir('ESMDA_CCRR')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ESMDA_CCRR')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
        
             
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F') 
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM}) 
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding - IES model')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files()
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )       
        print('--------------------ESMDA_CCR EXECUTED------------------------')
        
    elif Technique_DA==6:
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')
        os.chdir(oldfolder)
        print('----------------------ES-MDA Autoencoder--------------------------')
        print('History Matching using the Ensemble Smoother with Multiple data\
assimialtion with convolution autoencoder\n')
        print('Novel Implementation: Author: Clement Etienam SA Energy/GPU @Nvidia')
        print('')
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input(' Enter the Inflation parameter from 4-8) : '))     
    
        if Geostats==1:  
            if Deccor==1:#Deccorelate the ensemble
                #ini_ensemble=De_correlate_ensemble(nx,ny,nz,N_ens,High_K,Low_K)
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))            
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)          
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        #CDd=np.dot(ax,ax.T)
        for ii in range(alpha):
            
            print( str(ii+1) + ' | ' + str(alpha))
    
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                #xx=0
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    folder=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                    predMatrix.append(predd)
         
                Delete_files() 
            Delete_files2()                
            print('Read Historical data')
            os.chdir('True_Model')
            True_measurement=Get_Measuremnt_CSV('hm0.csv')
            True_mat=True_measurement
            True_data=True_mat[:,1:]
            True_data=np.reshape(True_data,(-1,1),'F')        
            os.chdir(oldfolder)
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,\
                         "Initial.jpg")
                os.chdir(oldfolder)
            else:
                pass
            
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
            encoded=Get_Latent(ensemble,Ne,nx,ny,nz)
            updated_ensemble,updated_ensemblep,CM=ESMDA_AEE(encoded,ensemblep,\
                                            modelError,CM,True_data,\
                                Ne, simDatafinal,alpha)
            updated_ensemble=Recover_image(updated_ensemble,Ne,nx,ny,nz)
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep
            
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                            High_K,Low_K,High_P,Low_P)            
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .') 
            
            if ii==0:
                cc_ini=cc
            else:
                pass
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F')  
        controlj2=controljj2
    
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        controljj2p= np.reshape(meannp,(-1,1),'F')  
        controlj2p=controljj2p                                
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
       
            Delete_files()   
        Delete_files2()                 
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder) 
        os.chdir('HM_RESULTS')                         
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        controlbest= ensemble[:,clem].reshape(-1,1) 
        controlbest2=controlj2#controlbest
        controlbestp= ensemblep[:,clem].reshape(-1,1) 
        controlbest2p=controlj2p#controlbest    
        
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F')       
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')       
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F') 
        Plot_Histogram_now(Ne,cc_ini,cc)
        #os.makedirs('ESMDA_AE')
        if not os.path.exists('ESMDA_AE'):
            os.makedirs('ESMDA_AE')
        else:
            shutil.rmtree('ESMDA_AE')
            os.makedirs('ESMDA_AE') 
        shutil.copy2('masterreal.data','ESMDA_AE') 
        print('6X Reservoir Simulator Forwarding - ESMDA_AE Model')
        Forward_model(oldfolder,'ESMDA_AE',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'ESMDA_AE') 
        os.chdir('ESMDA_AE')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ESMDA_AE')
        os.chdir(oldfolder)
    
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim
       
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')  
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny) 
        os.chdir(oldfolder)
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM})
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding - IES model')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files() 
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )      
        print('-------------------ESMDA AUTOENCODER EXECUTED----------------') 
        
    elif Technique_DA==7:
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder)
        print('-----------------ES-MDA Denosing Autoencoder---------------------')
        print('History Matching using the Ensemble smoother with multiple data\
assimialtion and convolution denoising autoencoder\n')
        print('Novel Implementation : Author- Clement Etienam SA-ML/Energy @Nvidia')
        print('')
        choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        #alpha = int(input(' Inflation paramter 4-8) : '))
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input(' Enter the Inflation parameter from 4-8) : '))     
    
    
        if Geostats==1: 
            if Deccor==1:#Deccorelate the ensemble
                #ini_ensemble=De_correlate_ensemble(nx,ny,nz,N_ens,High_K,Low_K)
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))           
            else:           
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)         
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        #CDd=np.dot(ax,ax.T)
        for ii in range(alpha):
            
            print( str(ii+1) + ' | ' + str(alpha))
    
                                        
            #predMatrix=workflow(Ne,maxx,ensemble,ensemblep)
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                #xx=0
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
               # os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    folder=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                    predMatrix.append(predd)
            
                Delete_files()
            Delete_files2()                        
            print('Read Historical data')
            os.chdir('True_Model')
            True_measurement=Get_Measuremnt_CSV('hm0.csv')
            True_mat=True_measurement
            True_data=True_mat[:,1:]
            True_data=np.reshape(True_data,(-1,1),'F')        
            os.chdir(oldfolder)
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,\
                         "Initial.jpg")
                os.chdir(oldfolder)
            else:
                pass
            
            simDatafinal =  Get_simulated(predMatrix,modelError,CM,Ne)# # 
            #encoded=Get_Latent(ensemble,Ne,nx,ny,nz)
            updated_ensemble,updated_ensemblep,CM=ESMDA(ensemble,ensemblep,\
                                                        modelError,CM,\
                                            True_data, Ne, \
                                     simDatafinal,alpha)
            if (choice==1) and (Geostats==1):
                updated_ensemble=use_denoising(updated_ensemble,nx,ny,nz,Ne)
            else:
                pass
    
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep
            
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                            High_K,Low_K,High_P,Low_P)            
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .') 
            
            if ii==0:
                cc_ini=cc
            else:
                pass
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F')  
        controlj2=controljj2
    
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        controljj2p= np.reshape(meannp,(-1,1),'F')  
        controlj2p=controljj2p                                
        #predMatrix=workflow(Ne,maxx,ensemble,ensemblep)
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
     
            Delete_files() 
        Delete_files2()                   
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder) 
        os.chdir('HM_RESULTS')                        
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal =  Get_simulated(predMatrix,modelError,CM,Ne)# # 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        controlbest= ensemble[:,clem].reshape(-1,1) 
        controlbest2=controlj2#controlbest
        
        controlbestp= ensemblep[:,clem].reshape(-1,1) 
        controlbest2p=controlj2p#controlbest    
        
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F')
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F') 
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')     
        Plot_Histogram_now(Ne,cc_ini,cc)
        if not os.path.exists('ESMDA_DA'):
            os.makedirs('ESMDA_DA')
        else:
            shutil.rmtree('ESMDA_DA')
            os.makedirs('ESMDA_DA') 
        shutil.copy2('masterreal.data','ESMDA_DA') 
        print('6X Reservoir Simulator Forwarding - ESMDA_DA Model')
        Forward_model(oldfolder,'ESMDA_DA',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'ESMDA_DA') 
        os.chdir('ESMDA_DA')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ESMDA_DA')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
            
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny) 
        os.chdir(oldfolder)
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM})
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding - model')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files() 
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )      
        print('----------ESMDA DENOISING AUTOENCODER EXECUTED----------------') 
        
    elif Technique_DA==8:
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder)
        print('------------------------ES-MDA_DCT--------------------------------')
        print('History Matching using the Discrete Cosine Transform')
        print('Novel Implementation : Author- Clement Etienam SA-ML/Energy @Nvidia')        
        print('')    
        size1,size2=int(cp.ceil(int(0.7*nx))),int(cp.ceil(int(0.7*ny)))
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input(' Enter the Inflation parameter from 4-8) : '))     
    
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
    
        if Geostats==1:  
            if Deccor==1:#Deccorelate the ensemble
                #ini_ensemble=De_correlate_ensemble(nx,ny,nz,N_ens,High_K,Low_K)
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))            
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)         
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        #CDd=np.dot(ax,ax.T)
        for ii in range(alpha):
            
            print( str(ii+1) + ' | ' + str(alpha))
    
            #predMatrix=workflow(Ne,maxx,ensemble,ensemblep)
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                #xx=0
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    folder=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                    predMatrix.append(predd)
          
                Delete_files() 
            Delete_files2()                 
            print('Read Historical data')
            os.chdir('True_Model')
            True_measurement=Get_Measuremnt_CSV('hm0.csv')
            True_mat=True_measurement
            True_data=True_mat[:,1:]
            True_data=np.reshape(True_data,(-1,1),'F')        
            os.chdir(oldfolder)        
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,\
                         "Initial.jpg")
                os.chdir(oldfolder)
            else:
                pass
            
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#
            ensembledct=dct22(ensemble,Ne,nx,ny,nz,size1,size2)
            updated_ensembledct,updated_ensemblep,CM=ESMDA_AEE(ensembledct,ensemblep,\
                                                           modelError,CM,\
                                                True_data, Ne,\
                                        simDatafinal,alpha)
            updated_ensemble=idct22(updated_ensembledct,Ne,nx,ny,nz,size1,size2)
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep
                    
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)            
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .') 
            if ii==0:
                cc_ini=cc
            else:
                pass
            
        
        ensemble,ensemblep=honour2(ensemblep,\
                                     ensemble,nx,ny,nz,N_ens,\
                                         High_K,Low_K,High_P,Low_P)          
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F')  
        controlj2=controljj2
    
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        controljj2p= np.reshape(meannp,(-1,1),'F')  
        controlj2p=controljj2p                                
        #predMatrix=workflow(Ne,maxx,ensemble,ensemblep)
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
       
            Delete_files()                                
        Delete_files2()
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder) 
        os.chdir('HM_RESULTS')                       
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        controlbest= np.reshape(ensemble[:,clem] ,(-1,1),'F')
        controlbest2=controlj2#controlbest
        
        controlbestp= ensemblep[:,clem].reshape(-1,1) 
        controlbest2p=controlj2p#controlbest
        
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F')
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F') 
        
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F') 
        
        Plot_Histogram_now(Ne,cc_ini,cc)
    
        if not os.path.exists('ESMDA_DCT'):
            os.makedirs('ESMDA_DCT')
        else:
            shutil.rmtree('ESMDA_DCT')
            os.makedirs('ESMDA_DCT')
            
        shutil.copy2('masterreal.data','ESMDA_DCT')
        print('6X Reservoir Simulator Forwarding - ESMDA_DCT Model')
        Forward_model(oldfolder,'ESMDA_DCT',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'ESMDA_DCT')
        os.chdir('ESMDA_DCT')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ESMDA_DCT')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
              
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny) 
        os.chdir(oldfolder)
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM}) 
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding - ESMDA_DCT model')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files() 
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )           
        print('--------------------ESMDA DCT EXECUTED--------------------------------') 
          
    elif Technique_DA==9:
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder)
        print('------------------------ES-MDA_KSVD--------------------------------')
        print('Novel Implementation : Author- Clement Etienam SA-ML/Energy @Nvidia')        
        print('History Matching using the K-SVD/OMP Flavour')
        print('')    
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input(' Enter the Inflation parameter from 4-8) : '))     
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
            
        print('Learn KSVD Over complete dictionary')
        print('')
        Dicclem=Learn_Overcomplete_Dictionary(Ne)
    
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                #ini_ensemble=De_correlate_ensemble(nx,ny,nz,N_ens,High_K,Low_K)
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))           
            else:           
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)         
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        #CDd=np.dot(ax,ax.T)
        for ii in range(alpha):
            
            print( str(ii+1) + ' | ' + str(alpha))
    
            #predMatrix=workflow(Ne,maxx,ensemble,ensemblep)
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                #xx=0
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    folder=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                    predMatrix.append(predd)
          
                Delete_files() 
            Delete_files2()                 
            print('Read Historical data')
            os.chdir('True_Model')
            True_measurement=Get_Measuremnt_CSV('hm0.csv')
            True_mat=True_measurement
            True_data=True_mat[:,1:]
            True_data=np.reshape(True_data,(-1,1),'F')        
            os.chdir(oldfolder)        
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,\
                         "Initial.jpg")
                os.chdir(oldfolder)
            else:
                pass
            
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#
            #ensembledct=dct22(ensemble,Ne,nx,ny,nz)
            ensembledct=Sparse_coding(Dicclem,ensemble,Ne)#Get Sparse signal
            updated_ensembledct,updated_ensemblep,CM=ESMDA_AEE(ensembledct,ensemblep,\
                                                           modelError,CM,\
                                                True_data, Ne,\
                                        simDatafinal,alpha)
            updated_ensemble=Recover_Dictionary_Saarse(Dicclem,\
                                        updated_ensembledct,Low_K,High_K)
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep
            
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)            
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .') 
            if ii==0:
                cc_ini=cc
            else:
                pass
        
        ensemble,ensemblep=honour2(ensemblep,\
                                     ensemble,nx,ny,nz,N_ens,\
                                         High_K,Low_K,High_P,Low_P)          
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F')  
        controlj2=controljj2
    
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        controljj2p= np.reshape(meannp,(-1,1),'F')  
        controlj2p=controljj2p                                
        #predMatrix=workflow(Ne,maxx,ensemble,ensemblep)
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
       
            Delete_files()                                
        Delete_files2()
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder)
        os.chdir('HM_RESULTS')                        
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        controlbest= ensemble[:,clem].reshape(-1,1) 
        controlbest2=controlj2#controlbest
        
        controlbestp= ensemblep[:,clem].reshape(-1,1) 
        controlbest2p=controlj2p#controlbest
        
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F')
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F') 
        
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')     
        #controlbest2p=machine_map.predict(controlbest2)
        #os.makedirs('ESMDA_DCT')
        Plot_Histogram_now(Ne,cc_ini,cc)
        if not os.path.exists('ESMDA_KSVD'):
            os.makedirs('ESMDA_KSVD')
        else:
            shutil.rmtree('ESMDA_KSVD')
            os.makedirs('ESMDA_KSVD')
            
        shutil.copy2('masterreal.data','ESMDA_KSVD')
        print('6X Reservoir Simulator Forwarding - ESMDA_KSVD Model')
        Forward_model(oldfolder,'ESMDA_KSVD',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'ESMDA_KSVD')
        os.chdir('ESMDA_KSVD')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ESMDA_KSVD')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
            
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny) 
        os.chdir(oldfolder)
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM}) 
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding - ESMDA_KSVD model')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files()
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )           
        print('--------------------ESMDA KSVD EXECUTED-----------------------')
    elif Technique_DA==10:
        
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        print('--------------------------ES-MDA_LBFGS----------------------------------')
        print('History Matching using the Ensemble Smoother with Multiple Data\
Asiimilation LBFGS Flavour')
        print('Novel Implementation : Author- Clement Etienam SA-ML/Energy @Nvidia')
        print('')
        choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        os.chdir(oldfolder)
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input(' Enter the Inflation parameter from 4-8) : '))   
    
    
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))           
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)         
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        #CDd=np.dot(ax,ax.T)
        for ii in range(alpha):
            
            print( str(ii+1) + ' | ' + str(alpha))
            
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    foldern=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,foldern,namecsv)
                    predMatrix.append(predd)        
                Delete_files()
            Delete_files2() 
            print('Read Historical data')
            os.chdir('True_Model')
            True_measurement=Get_Measuremnt_CSV('hm0.csv')
            True_mat=True_measurement
            True_data=True_mat[:,1:]
            True_data=np.reshape(True_data,(-1,1),'F')        
            os.chdir(oldfolder)                                    
            
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,\
                         "Initial.jpg")
                os.chdir(oldfolder)
            else:
                pass
            
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#     
            updated_ensemble,updated_ensemblep,CM=ESMDA_New(ensemble,ensemblep,\
                                                            modelError,\
                                            CM,True_data, \
                                    Ne, simDatafinal,alpha)
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep        
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)            
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .') 
            if ii==0:
                cc_ini=cc
            else:
                pass
            
        if (choice==1) and (Geostats==1):
            ensemble=use_denoising(ensemble,nx,ny,nz,Ne)
        else:
            pass 
        
        ensemble,ensemblep=honour2(ensemblep,\
                                     ensemble,nx,ny,nz,N_ens,\
                                         High_K,Low_K,High_P,Low_P)          
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F') 
        controljj2p= np.reshape(meannp,(-1,1),'F') 
        controlj2=controljj2
        controlj2p=controljj2p
        
        
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
    
        
            Delete_files()                                
        Delete_files2()
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder)  
        os.chdir('HM_RESULTS')                      
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        shpw=cc[clem]
        controlbest= np.reshape(ensemble[:,clem],(-1,1),'F') 
        controlbestp= np.reshape(ensemblep[:,clem],(-1,1),'F')
        controlbest2=controlj2#controlbest
        controlbest2p=controljj2p#controlbest
    
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F') 
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')  
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        Plot_Histogram_now(Ne,cc_ini,cc)
        #os.makedirs('ESMDA')
        if not os.path.exists('ESMDA_LBFGS'):
            os.makedirs('ESMDA_LBFGS')
        else:
            shutil.rmtree('ESMDA_LBFGS')
            os.makedirs('ESMDA_LBFGS') 
            
        shutil.copy2('masterreal.data','ESMDA_LBFGS')
        print('6X Reservoir Simulator Forwarding - ESMDA_LBFGS Model')
        Forward_model(oldfolder,'ESMDA_LBFGS',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'ESMDA_LBFGS')
        os.chdir('ESMDA_LBFGS')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ESMDA_LBFGS')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
             
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)
        
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM}) 
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding')
        
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files() 
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )     
        
        
        print('--------------------ESMDA LBFGS EXECUTED----------------------')
    elif Technique_DA==11:
        
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        print('--------------------------ES-MDA_CE----------------------------------')
        print('History Matching using the Ensemble Smoother with Multiple Data \
Asiimilation CE Flavour')
        print('Novel Implementation : Author- Clement Etienam SA-ML/Energy @Nvidia')
        print('')
        choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        os.chdir(oldfolder)
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input(' Enter the Inflation parameter from 4-8) : '))   
    
    
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))           
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)         
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        #CDd=np.dot(ax,ax.T)
        for ii in range(alpha):
            
            print( str(ii+1) + ' | ' + str(alpha))
            
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    foldern=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,foldern,namecsv)
                    predMatrix.append(predd)        
                Delete_files()
            Delete_files2() 
            print('Read Historical data')
            os.chdir('True_Model')
            True_measurement=Get_Measuremnt_CSV('hm0.csv')
            True_mat=True_measurement
            True_data=True_mat[:,1:]
            True_data=np.reshape(True_data,(-1,1),'F')        
            os.chdir(oldfolder)                                    
            
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,\
                         "Initial.jpg")
                os.chdir(oldfolder)
            else:
                pass
            
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#     
            updated_ensemble,updated_ensemblep,CM=ESMDA_CE(Ne,ensemble,ensemblep,\
                                                            CM,modelError,\
                                            True_data, \
                                    simDatafinal,alpha,CDd)
                              
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep        
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)            
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .') 
            if ii==0:
                cc_ini=cc
            else:
                pass
            
        if (choice==1) and (Geostats==1):
            ensemble=use_denoising(ensemble,nx,ny,nz,Ne)
        else:
            pass 
        
        ensemble,ensemblep=honour2(ensemblep,\
                                     ensemble,nx,ny,nz,N_ens,\
                                         High_K,Low_K,High_P,Low_P)          
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F') 
        controljj2p= np.reshape(meannp,(-1,1),'F') 
        controlj2=controljj2
        controlj2p=controljj2p
        
        
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
    
        
            Delete_files()                                
        Delete_files2()
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder)  
        os.chdir('HM_RESULTS')                      
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        shpw=cc[clem]
        controlbest= np.reshape(ensemble[:,clem],(-1,1),'F') 
        controlbestp= np.reshape(ensemblep[:,clem],(-1,1),'F')
        controlbest2=controlj2#controlbest
        controlbest2p=controljj2p#controlbest
    
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F') 
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')  
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        Plot_Histogram_now(Ne,cc_ini,cc)
        #os.makedirs('ESMDA')
        if not os.path.exists('ESMDA_CE'):
            os.makedirs('ESMDA_CE')
        else:
            shutil.rmtree('ESMDA_CE')
            os.makedirs('ESMDA_CE') 
            
        shutil.copy2('masterreal.data','ESMDA_CE')
        print('6X Reservoir Simulator Forwarding - ESMDA_CE Model')
        Forward_model(oldfolder,'ESMDA_CE',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'ESMDA_CE')
        os.chdir('ESMDA_CE')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ESMDA_CE')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
             
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)
        
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM}) 
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding')
        
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files()
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )     
        
        
        print('--------------------ESMDA_CE EXECUTED----------------------')        
    elif Technique_DA==12:
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')
        os.chdir(oldfolder)
        print('-------------------------ES-MDA GEO_Versions_2-------------------------------')
        print('History Matching using the Ensemble Smoother with Multiple data\
assimilation with Geometric inflation factors\n')
        print('') 
        choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored') 
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input('Enter the Inflation parameter from 4-8) : '))     
    
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                #ini_ensemble=De_correlate_ensemble(nx,ny,nz,N_ens,High_K,Low_K)
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))              
            else:            
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)
                
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        
        ensemble,ensemblep=honour2(ensemblep,\
                                          ensemble,nx,ny,nz,N_ens\
                                              ,High_K,Low_K,High_P,Low_P)         
        ax=cp.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=cp.sqrt(cp.asarray(noise_level)*cp.asarray(True_data[iq,:]))
                else:
                    ax[iq,:]=cp.sqrt(cp.asarray(diffyet[iq,:]))
                
        ax=cp.reshape(ax,(-1,))    
        CDd=cp.diag(ax)
        #CDd=np.dot(ax,ax.T)
        
        alphabig=[]
        for ii in range(alpha):
            
            print( str(ii+1) + ' | ' + str(alpha))
           
            #predMatrix=workflow(Ne,maxx,ensemble,ensemblep)
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                #xx=0
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    folder=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                    predMatrix.append(predd)
    
                
                Delete_files()
            Delete_files2()
            print('Read Historical data')
            os.chdir('True_Model')
            True_measurement=Get_Measuremnt_CSV('hm0.csv')
            True_mat=True_measurement
            os.chdir(oldfolder)                            
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,\
                         True_mat,"Initial.jpg")
                os.chdir(oldfolder)
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#  
            #simDatafinal=simDatafinal+CM               
            stdall=[]
            ff=cp.asarray(True_data)
            sgsim=ensemble
            if modelError==1:
                overall=cp.vstack([cp.asarray(sgsim),cp.asarray(ensemblep),\
                                  cp.asarray(CM)])
            else:
                overall=cp.vstack([cp.asarray(sgsim),cp.asarray(ensemblep)])
            
            Y=overall 
            Sim1=cp.asarray(simDatafinal)
            M = cp.mean(Sim1,axis=1)
        
            M2=cp.mean(overall,axis=1)
            
            S = cp.zeros((Sim1.shape[0],Ne))
            yprime = cp.zeros((Y.shape[0],Y.shape[1]))            
            
            for jc in range(Ne):
                S[:,jc] = Sim1[:,jc]- M
                yprime[:,jc] = overall[:,jc] - M2
            Cyd = (yprime.dot(S.T))/(Ne - 1)
            Cdd = (S.dot(S.T))/(Ne- 1)
            
            deltaM=yprime/((Ne-1)**(0.5))
            deltaD=Sim1/((Ne-1)**(0.5))            

            Gd=((cp.linalg.inv(CDd))**(0.5)).dot(deltaD)
            
            Usig1,Sig1,Vsig1 = cp.linalg.svd((Gd), full_matrices = False)
            Bsig1 = cp.cumsum(Sig1, axis = 0)          # vertically addition
            valuesig1 = Bsig1[-1]                 # last element
            valuesig1 = valuesig1 * 0.9999
            indices1 = ( Bsig1 >= valuesig1 ).ravel().nonzero()
            toluse1 = Sig1[indices1]
            tol1 = toluse1[0]            
            Va,S1a,Ua = cp.linalg.svd(Gd,full_matrices=0)
        
            r1a = cp.sum(S1a > tol1)+1
            v = Va[:,:r1a-1] #needed
            U1a = Ua.T
            ua = U1a[:,:r1a-1]
            ua=ua.T#needed
            S11a = S1a[:r1a-1]#needed
            
            if ii==0:
                #alpha=np.float(alpha)
                S11a=cp.reshape(S11a,(-1,1),'F')
                landaa=(cp.mean(S11a,axis=0))
                landaa=cp.asnumpy(landaa)
                alpha=cp.asnumpy(alpha)
                alpha=np.float(np.asscalar(alpha))
                landaa=np.asscalar(landaa)
                alpha1=max(landaa, alpha)
                bnds=[(0, 1)]
                # if Yet==0:
                #     alpha,alpha1=cp.asnumpy(alpha),cp.asnumpy(alpha1)
                # else:
                #     pass
                resultt=scipy.optimize.minimize(objective_Na, x0=0.06, \
                args=(alpha,alpha1), \
                    method='SLSQP', \
                bounds=bnds, options={'xtol': 1e-10,\
                                                       'ftol': 1e-10})
                beta=np.reshape(resultt.x,(1,-1),'F') 
                
    
                print('alpha = ' + str(alpha1)+ '\n')
                #beta=np.abs(beta)
            else:
                alpha1=beta*alpha1
                izz=ii+1
                print('alpha = ' + str(alpha1)+ '\n')
            alphabig.append(alpha1)
            Z=ff
            
            varn=0.9999#1-1/math.pow(10,numsave)
            # Initial Variavel 
            # Forecast step
            yf = Y                        # Non linear forward model 
            df = Sim1                     # Observation Model
            ym = cp.array(yf.mean(axis=1))    # Mean of the y_f
            dm = cp.array(df.mean(axis=1))    # Mean of the d_f
            ym=ym.reshape(ym.shape[0],1)    
            dm=dm.reshape(dm.shape[0],1)    
            dmf = yf - ym
            ddf = df - dm
            
            Cmd_f = (cp.dot(dmf,ddf.T))/(Ne-1);  # The cros-covariance matrix
            Cdd_f = (cp.dot(ddf,ddf.T))/(Ne-1);  # The auto covariance of predicted data
            
            # Perturb the vector of observations
            R = cp.linalg.cholesky(CDd) #Matriz triangular inferior
            U = R.T   #Matriz R transposta
            #p , w =cp.linalg.eig(CDd)
            
            aux = cp.repeat(Z,Ne,axis=1)
            mean = 0*(Z.T)
        
            noise=cp.random.multivariate_normal(mean[0], cp.eye(len(Z)), Ne).T
            d_obs = aux+cp.sqrt(cp.asarray(alpha1))*cp.dot(U,noise)  
            
            # Analysis step
            u, s, vh = cp.linalg.svd(Cdd_f+cp.asarray(alpha1)*CDd); v = vh.T
            diagonal = s
            for i in range(len(diagonal)):
                if (cp.sum(diagonal[0:i+1]))/(cp.sum(diagonal)) > varn:
                    diagonal = diagonal[0:i+1]
                    break
            
            u=u[:,0:i+1]
            v=v[:,0:i+1]
            ss = cp.diag(diagonal**(-1))
            K=cp.dot(Cmd_f,(cp.dot(cp.dot(v,ss),(u.T))))
            # Use Kalman covariance
            # if len(corr)>0:
            #     K = corr*K
                
            Ynew = yf + (cp.dot(K,(d_obs-df)))

            sizeclem=nx*ny*nz 
            if Yet==0:                   
                updated_ensemble=cp.asnumpy(Ynew[:sizeclem,:]) 
                updated_ensemblep=cp.asnumpy(Ynew[sizeclem:2*sizeclem,:])
                if modelError==1:
                    CM=cp.asnumpy(Ynew[2*sizeclem:,:])
                else:
                    pass
            else:
                updated_ensemble=(Ynew[:sizeclem,:]) 
                updated_ensemblep=(Ynew[sizeclem:2*sizeclem,:])
                if modelError==1:
                    CM=(Ynew[2*sizeclem:,:])
                else:
                    pass                
            
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)  
            
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .')
            if ii==0:
                cc_ini=cc
            else:
                pass
            
        if (choice==1) and (Geostats==1):        
            ensemble=use_denoising(ensemble,nx,ny,nz,Ne)
        else:
            pass        
        alphabig =np.array(alphabig )
        alphabig =np.reshape(alphabig ,(-1,1))       
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        meannini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F')  
        controlj2=controljj2
        controljj2p= np.reshape(meannp,(-1,1),'F')  
        controlj2p=controljj2p
    
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
        
            Delete_files()
        Delete_files2()
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        os.chdir(oldfolder) 
        
        os.chdir('HM_RESULTS')                        
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#          
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        controlbest= ensemble[:,clem].reshape(-1,1)
        controlbest2=controlj2#controlbest
        
        controlbestp= ensemblep[:,clem].reshape(-1,1) 
        controlbest2p=controlj2p#controlbest
        Plot_Histogram_now(Ne,cc_ini,cc)
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F')
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F') 
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')    
        #controlbest2p=machine_map.predict(controlbest2)
        if not os.path.exists('ESMDA_GEO_2'):
            os.makedirs('ESMDA_GEO_2')
        else:
            shutil.rmtree('ESMDA_GEO_2')
            os.makedirs('ESMDA_GEO_2') 
        shutil.copy2('masterreal.data','ESMDA_GEO_2') 
        print('6X Reservoir Simulator Forwarding - ESMDA_GEO_2 Model')        
        Forward_model(oldfolder,'ESMDA_GEO_2',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'ESMDA_GEO_2')
        os.chdir('ESMDA_GEO_2')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','ESMDA_GEO_2')
        os.chdir(oldfolder)
    
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim
         
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meannini,nx,ny)
        os.chdir(oldfolder)
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM})
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files() 
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )      
        
        for kka in range(2):
            sigma_est = np.mean(estimate_sigma(controljj2, multichannel=True)) 
            patch_kw = dict(patch_size=5,      # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    multichannel=True)
            denoise_fast = denoise_nl_means(controljj2, h=0.8 * sigma_est, \
                                            fast_mode=True,
                                    **patch_kw)
            controljj2=np.reshape(denoise_fast,(-1,1))
            
            controljj2[controljj2<=Low_K]=Low_K
            controljj2[controljj2>=High_K]=High_K
        
        print('--------------------ESMDA_GEO_V2 EXECUTED---------------------')
    print('-------------------------SECTION EXECUTED-------------------------')             
elif Data_Assimilation==3:
    Technique_EnKF=int(input('Enter optimisation with the Ensemble Kalman Filter: \n\
    1 =  EnKF_One (One Short Ensemble Kalman Filter)\n\
    2 = IEnKF  (Iterative Ensemble Kalman Filter)(slower)\n'))     

    
    if Technique_EnKF==1:
        True_data_EnKF=True_mat[:,1:]
        True_ucc=np.mean(True_data_EnKF,axis=0).reshape(-1,1)
        True_data=np.reshape(True_data_EnKF,(-1,1),'F')    
        print('---------------------------EnKF-----------------------------------')
        print('History Matching using the one Short Ensemble Kalman Filter')
        print('Reference: Evensen, G., 2003. The Ensemble Kalman Filter: Theoretical formulation\
    and practical implementation. Ocean Dynamics, 53(4), pp. 343-367.\n')
        print('') 
        choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        corellaa=int(input('Use Correlated Data noise?: \n\
        1 =  Yes\n\
        2 = No\n'))      
        os.chdir(oldfolder)
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))           
            else:
            
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)
                
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((True_data_EnKF.shape[1],1))
        for iq in range(True_data_EnKF.shape[1]):
            if True_ucc[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_ucc[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        #CDd=np.dot(ax,ax.T)

    
            
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
      
            Delete_files()
        Delete_files2()
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        os.chdir(oldfolder)        

        os.chdir('HM_RESULTS')
        Plot_RSM(predMatrix,modelError,CM,Ne,\
                 True_mat,"Initial.jpg")
        os.chdir(oldfolder)
        Sim1 = Get_simulated_EnKF(predMatrix,modelError,CM,Ne)
        
        for ii in range(True_data_EnKF.shape[0]):
            
            print( str(ii+1) + ' | ' + str(True_data_EnKF.shape[0]))
            
            simmean=np.reshape(np.mean(Sim1,axis=1),(-1,1),'F')
            True_now=True_data_EnKF[ii,:].reshape(-1,1)            
            tinuke=((np.sum((((simmean) - True_now) ** 2)) )  **(0.5))\
                /True_now.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(Sim1,True_now)
            clem=np.argmin(cc)
            simmbest=Sim1[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_now) ** 2)) )  **(0.5))\
                /True_now.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .')             
            
            ensemble,ensemblep,Sim1,CM=EnKF_One_Short(ensemble,\
                        ensemblep,modelError,CM,True_data_EnKF[ii,:],\
                            Ne, Sim1,corellaa)
            
            if ii==0:
                cc_ini=cc
            else:
                pass
            
        if (choice==1) and (Geostats==1):
            ensemble=use_denoising(ensemble,nx,ny,nz,Ne)
        else:
            pass             
        
        ensemble,ensemblep=honour2(ensemblep,\
                                     ensemble,nx,ny,nz,N_ens,\
                                         High_K,Low_K,High_P,Low_P)              
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        controljj2= meann
        controlj2=controljj2
        controljj2p= meannp
        controlj2p=controljj2p
        
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
    
            
            Delete_files()
        Delete_files2()
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')        
        os.chdir(oldfolder)        
                                    
    
        os.chdir('HM_RESULTS')
        Plot_RSM(predMatrix,modelError,CM,Ne,\
                 True_mat,"Final.jpg")
        os.chdir(oldfolder)
    
        
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#             
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        controlbest= np.reshape(ensemble[:,clem] ,(-1,1),'F')
        controlbestp= np.reshape(ensemblep[:,clem] ,(-1,1),'F')
        controlbest2=controlbest
        controlbest2p=controlbestp
        simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
        simmbest=simDatafinal[:,clem].reshape(-1,1)
        tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        tinukemean=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE of the ensemble mean = : \n' \
              + str(tinukemean) + '\nRMSE of the ensemble best = : \n' \
                  + str(tinukebest) + '... .')
                
                
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F') 
    
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F') 
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        
        os.chdir('HM_RESULTS')   
        Plot_mean(controlbest2,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)
        
        Plot_Histogram_now(Ne,cc_ini,cc)
        if not os.path.exists('One_EnKF'):
            os.makedirs('One_EnKF')
        else:
            shutil.rmtree('One_EnKF')
            os.makedirs('One_EnKF') 
            
        shutil.copy2('masterreal.data','One_EnKF')
        print('6X Reservoir Simulator Forwarding - One_EnKF Model')
        Forward_model(oldfolder,'One_EnKF',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'One_EnKF')
        os.chdir('One_EnKF')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','One_EnKF')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
             
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)        
        
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,}) 
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding ')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files()
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' ) 
        print('------------EnKF ONE SHORT EXECUTED---------------------------')
    elif Technique_EnKF==2:     
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        print('---------------------------IEnKF------------------------------')
        print('History Matching using the Iterative Ensemble Kalman Filter')
        print('Reference: Evensen, G., 2003. The Ensemble Kalman Filter: Theoretical formulation\
    and practical implementation. Ocean Dynamics, 53(4), pp. 343-367.\n')
        print('') 
        choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        if De_alpha==1:
            alpha=10
        else:
            alpha = int(input(' Enter the Inflation parameter from 4-8) : '))     
    
        os.chdir(oldfolder)
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))           
            else:
            
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)
                
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        #CDd=np.dot(ax,ax.T)
        for ii in range(alpha):
            
            print( str(ii+1) + ' | ' + str(alpha))
    
            
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                #xx=0
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    folder=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                    predMatrix.append(predd)
          
            Delete_files()
            print('Read Historical data')
            os.chdir('True_Model')
            True_measurement=Get_Measuremnt_CSV('hm0.csv')
            True_mat=True_measurement
            os.chdir(oldfolder)        
                                        
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,\
                         True_mat,"Initial.jpg")
                os.chdir(oldfolder)
            else:
                pass
            
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#  
            updated_ensemble,updated_ensemblep,CM=EnKF(ensemble,ensemblep,\
                                                     modelError,CM,True_data,\
                                              Ne, simDatafinal)
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep
            if (choice==1) and (Geostats==1):
                ensemble=use_denoising(ensemble,nx,ny,nz,Ne)
            else:
                pass             
            
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)  
                
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .') 
            if ii==0:
                cc_ini=cc
            else:
                pass
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        controljj2= meann
        controlj2=controljj2
        controljj2p= meannp
        controlj2p=controljj2p
        
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
    
            
            Delete_files()
        Delete_files2()
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')        
        os.chdir(oldfolder)        
                                    
    
        os.chdir('HM_RESULTS')
        Plot_RSM(predMatrix,modelError,CM,Ne,\
                 True_mat,"Final.jpg")
        os.chdir(oldfolder)
    
        
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#             
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        controlbest= np.reshape(ensemble[:,clem] ,(-1,1),'F')
        controlbestp= np.reshape(ensemblep[:,clem] ,(-1,1),'F')
        controlbest2=controlbest
        controlbest2p=controlbestp
        simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
        simmbest=simDatafinal[:,clem].reshape(-1,1)
        tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        tinukemean=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE of the ensemble mean = : \n' \
              + str(tinukemean) + '\nRMSE of the ensemble best = : \n' \
                  + str(tinukebest) + '... .')
                
                
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F') 
    
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F') 
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        
        os.chdir('HM_RESULTS')   
        Plot_mean(controlbest2,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)
        
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,}) 
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding ')
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files() 
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )       
        print('---------------------------IENKF EXECUTED--------------------')    
    print('--------------------SECTION EnKF ENDED----------------------------')

elif Data_Assimilation==4:
    True_data=True_mat[:,1:]
    True_data=np.reshape(True_data,(-1,1),'F')
    os.chdir(oldfolder)
    print('---------------------------MCMC----------------------------------')
    print('History Matching using the Markov Chain Monte carlo method')
    print('Reference: Stuart A M ,2010 Inverse problems:  A Bayesian perspective\
Acta Numerica 19 451-559.\n')
    print('')    
    if modelError==1:
        print(' Model Error considered')
    else:
        print('Model Error ignored')    
    if Geostats==1:
        if Deccor==1:#Deccorelate the ensemble
            filename="Ganensemble.mat" #Ensemble generated offline
            mat = sio.loadmat(filename)
            ini_ensemblef=mat['Z']  
            
            avs=whiten(ini_ensemblef, method='zca_cor')
            
            index = np.random.choice(avs.shape[1], N_ens, \
                                     replace=False)
            ini_ensemblee=avs[:,index]            
            
            clfye = MinMaxScaler(feature_range=(Low_K,High_K))
            (clfye.fit(ini_ensemblee))    
            ini_ensemble=(clfye.transform(ini_ensemblee))              
        else:
            if afresh==1:
                see=intial_ensemble(nx,ny,nz,N_ens,permx)
                ini_ensemblee=np.split(see, N_ens, axis=1)
                ini_ensemble=[]
                for ky in range(N_ens):
                    aa=ini_ensemblee[ky]
                    aa=np.reshape(aa,(-1,1),'F')
                    ini_ensemble.append(aa)
                    
                ini_ensemble=np.hstack(ini_ensemble) 
            else:
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']
                index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                         replace=False)
                ini_ensemble=ini_ensemblef[:,index]
    else:
        if Deccor==1:
            ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                Low_K,High_K)            
            ini_ensemblef=cp.asarray(ini_ensemblef)
            
            
            beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
            
            V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
            v = V[:,:Ne]
            U1 = U.T
            u = U1[:,:Ne]
            S11 = S1[:Ne]
            s = S11[:]
            S = (1/((beta)**(0.5)))*s
            #S=s
            X = (v*S).dot(u.T)
            if Yet==0:
                X=cp.asnumpy(X)
                ini_ensemblef=cp.asnumpy(ini_ensemblef)
            else:
                pass
            X[X<=Low_K]=Low_K
            X[X>=High_K]=High_K 
            ini_ensemble=X[:,:Ne] 
        else:
            
            ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                Low_K,High_K)
            
    initial_guess=np.mean(ini_ensemble,axis=1)
    initial_guess=np.reshape(initial_guess,(1,-1))
    meann=initial_guess
    sigg2=(ini_ensemble-np.mean(ini_ensemble,axis=1).reshape(-1,1))
    sigz=(sigg2@sigg2.T)/(ini_ensemble.shape[1]-1)
    sigma=sigz
    sigma[sigma<=0]=0
    initial_guess=cp.random.multivariate_normal(cp.ravel(cp.asarray(meann)),\
                                                cp.asarray(sigma),\
                1,method='eigh',dtype=cp.float32)
    if Yet==0:
        initial_guess=cp.asnumpy(initial_guess)
        initial_guess=np.reshape(initial_guess,(1,-1))
    else:
        pass
    
    array_sum = np.sum(initial_guess)
    array_has_nan = np.isnan(array_sum)
    if array_has_nan==True:
       initial_guess=np.nan_to_num(initial_guess, copy=True, nan=Low_K)
    else:
        pass
 
    

    if not os.path.exists('MCMC_Model'):
        os.makedirs('MCMC_Model')
    else:
        shutil.rmtree('MCMC_Model')
        os.makedirs('MCMC_Model')     
    shutil.copy2('masterreal.data','MCMC_Model')        
    iterations=  np.int(1e3) 
    data=True_data
    x = initial_guess
    accepted = []
    rejected = [] 
    beta = random()
    #beta=cp.asarray(beta).astype(cp.float32)
    sigma=cp.asarray(sigma).astype(cp.float32)
    meann=cp.asarray(meann).astype(cp.float32)
    for i in range(iterations):
        print( str(i+1) + ' | ' + str(iterations)) 
        x=cp.asarray(x).astype(cp.float32)
        
        
        eta=cp.zeros((1,nx*ny*nz))
        eta=cp.random.multivariate_normal(cp.ravel(eta), \
            sigma,1,method='eigh')
        eta=cp.reshape(eta,(1,-1))
        #eta=eta.astype(cp.float32)
        eta=cp.asnumpy(eta)
        array_sum = np.sum(eta)
        array_has_nan = np.isnan(array_sum)
        if array_has_nan==True:
            print('Array has Nan Values')
            eta=np.nan_to_num(eta, copy=True, nan=0)
            eta=cp.asarray(eta).astype(cp.float32)

        else:
            print('Array has no Nan')
            eta=cp.asarray(eta).astype(cp.float32)
            pass        
        
        x_new=(((1-(beta**2))**(0.5))*(x.reshape(1,-1)-meann))+(beta*eta)+meann         
        x=cp.reshape(x,(-1,1),'F')
        if Yet==0:
            x=cp.asnumpy(x)
            x=np.nan_to_num(x, copy=True, nan=10)
        else:
            x=np.nan_to_num(x, copy=True, nan=10)
        x[x>=High_K]=High_K
        x[x<=Low_K]=Low_K
        xin=x
        xinp=machine_map.predict(xin)
        xinp[xinp >= High_P] = High_P
        xinp[xinp <= Low_P] = Low_P            
        
        print('6X Reservoir Simulator Forwarding for intial point -  MCMC Model')
        Forward_model(oldfolder,'MCMC_Model',xin,xinp)
        yycheck=Get_RSM(oldfolder,'MCMC_Model')     
        usethis=yycheck
        usesim=usethis[:,1:]    
        yycheck1=np.reshape(usesim,(-1,1),'F')
        
        
        x_lik= np.sum(-np.log(1 * np.sqrt(2* np.pi) )-((data-\
                                                        (yycheck1))**2) \
                      / (2*1**2))
        if Yet==0:                
            x_new=cp.asnumpy(x_new)
            x_new=np.nan_to_num(x_new, copy=True, nan=10)
        else:
            x_new=np.nan_to_num(x_new, copy=True, nan=10)
            pass
        
        x_new[x_new>=High_K]=High_K
        x_new[x_new<=Low_K]=Low_K        
        xin=np.reshape(x_new,(-1,1),'F')
        
        xinp=machine_map.predict(xin)
        xinp[xinp >= High_P] = High_P
        xinp[xinp <= Low_P] = Low_P          
        
        print('6X Reservoir Simulator Forwarding for proposal -  MCMC Model')
        Forward_model(oldfolder,'MCMC_Model',xin,xinp)
        yycheck=Get_RSM(oldfolder,'MCMC_Model')     
        usethis=yycheck
        usesim=usethis[:,1:]    
        yycheck2=np.reshape(usesim,(-1,1),'F')
        
        
        x_new_lik= np.sum(-np.log(1 * np.sqrt(2* np.pi) )-((data-\
                                                        (yycheck2))**2) \
                      / (2*1**2))            
        a=x_new_lik/x_lik
        aa=min(a,1)
        aa2=np.random.uniform(0,1,1)
        if aa2<aa:
            x=x_new
            accepted.append(x_new)
            print('proposal accepted')            
        # if (acceptance(x_lik + np.log(prior(x)),x_new_lik+\
        #                     np.log(prior(x_new)))):            
        #     x = x_new
        #     accepted.append(x_new)
        #     print('proposal accepted')
        else:
            rejected.append(x_new) 
            print('proposal rejected')
        #beta=cp.asnumpy(beta)
        #meann=cp.asnumpy(meann)
        eta=cp.asnumpy(eta)
        del eta
    beta=cp.asnumpy(beta)
    sigma=cp.asnumpy(sigma)
    meann=cp.asnumpy(meann)
    del beta
    del sigma
    del meann
    #x_new=cp.asnumpy(x_new)
    accepted=np.hstack(accepted)            
    print(accepted.shape)
    show=int(-0.75*accepted.shape[0])
    #hist_show=int(-0.75*accepted.shape[0])
    
    # fig = plt.figure(figsize=(20,10))
    # ax = fig.add_subplot(1,2,1)
    # ax.plot(accepted[show:,:])
    # ax.set_title("Figure 4: Trace ")
    # ax.set_ylabel("$\sigma$")
    # ax.set_xlabel("Iteration")
    # ax = fig.add_subplot(1,2,2)
    # ax.hist(accepted[hist_show:,:], bins=20,density=True)
    # ax.set_ylabel("Frequency (normed)")
    # ax.set_xlabel("$\sigma$")
    # ax.set_title("Figure 5: Histogram of $\sigma$")
    # fig.tight_layout()
    
    
    # ax.grid("off")
    
    jjy=accepted[show:,:]
    jjy=np.reshape(jjy,(-1,nx*ny*nz),'F')
    mu=np.mean(jjy,axis=0)
    os.chdir('HM_RESULTS')
    Plot_mean(mu,mu,mu,nx,ny)
    os.chdir(oldfolder)
    print('--------------------SECTION MCMC ENDED----------------------------') 
elif Data_Assimilation==5:
    True_data=True_mat[:,1:]
    True_data=np.reshape(True_data,(-1,1),'F')    
    print('--------------------------REKI----------------------------------')
    print('History Matching using the Regularised Ensemble Kalman Inversion')
    print('Reference: (1): M. A. Iglesias. A regularising iterative ensemble\
Kalman method for PDE-constrained inverse problems. Inverse Problems, 32, 2016.\n\
(2): M. A. Iglesias, K. J. H. Law and A. M. Stuart. Ensemble Kalman methods\
for inverse problems.Inverse Problems, 29, 2013.\n\
(3): M. A. Iglesias and Y. Yang. Adaptive regularisation for ensemble Kalman inversion with\
applications to non-destructive testing and imaging. arXiv preprint arXiv:2006.14980, 2020.\n')
    Technique_REKI=int(input('Enter optimisation with the Ensemble Kalman Inversion: \n\
1 = REKI (Vanilla Ensemble Kalman Inversion)\n\
2 = REKI_KSVD (Ensemble Kalman Inversion + KSVD)\n\
3 = REKI_DCT (Ensemble Kalman Inversion + DCT)\n\
4 = REKI_AE  (Ensemble Kalman Inversion + Convolution Autoencoder)\n')) 

    Termm=int(input('Enter the maximum iteration count: '))
    if Technique_REKI==1:
        print('')
        print('Regularised Ensemble Kalman Inversion with Denoising Autoencoder\
post-processing\n')
        choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        os.chdir(oldfolder)
    
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))           
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)         
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
    
        R = ax**2
    
        CDd=np.diag(np.reshape(R,(-1,)))
        
        snn=0
        ii=0
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')        
        os.chdir(oldfolder)         
        
        while (snn<1):        
            print( 'Iteration --' + str(ii+1))
            
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    foldern=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,foldern,namecsv)
                    predMatrix.append(predd)        
                Delete_files()
            Delete_files2() 
                                   
            
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,\
                         "Initial.jpg")
                os.chdir(oldfolder)
            else:
                pass
            
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#   
            
            CDd=cp.asarray(CDd)
            True_data=cp.asarray(True_data)
            pertubations=cp.random.multivariate_normal(cp.ravel(cp.zeros\
                                                            ((Nop,1))), CDd,Ne)
            Ddraw=cp.tile(True_data, Ne) 
            pertubations=pertubations.T
            Dd=Ddraw #+ pertubations
            if Yet==0:
                CDd=cp.asnumpy(CDd)
                Dd=cp.asnumpy(Dd)
                Ddraw=cp.asnumpy(Ddraw)
                pertubations=cp.asnumpy(pertubations)
                True_data=cp.asnumpy(True_data)
            else:
                pass
            yyy=np.mean((Dd-simDatafinal).T@((inv(CDd)))\
                        @(Dd-simDatafinal),axis=1)
            yyy=yyy**(0.5)
            yyy=yyy.reshape(-1,1)
            alpha_star=np.mean(yyy,axis=0)
            if (snn+(1/alpha_star)>=1):
                alpha=1/(1-snn)
                snn=1
            else:
               alpha=alpha_star
               snn=snn+(1/alpha)
            print('alpha = ' + str(alpha))
            print('sn = ' + str(snn))
            sgsim=ensemble
            if modelError==1:
                overall=cp.vstack([cp.asarray(sgsim),cp.asarray(ensemblep),cp.asarray(CM)])
            else:
                overall=cp.vstack([cp.asarray(sgsim),cp.asarray(ensemblep)])
            
            Y=overall 
            Sim1=cp.asarray(simDatafinal)
            M = cp.mean(Sim1,axis=1)
        
            M2=cp.mean(overall,axis=1)
            
            
            S = cp.zeros((Sim1.shape[0],Ne))
            yprime = cp.zeros((Y.shape[0],Y.shape[1]))
                   
            for jc in range(Ne):
                S[:,jc] = Sim1[:,jc]- M
                yprime[:,jc] = overall[:,jc] - M2
            Cyd = (yprime.dot(S.T))/(Ne - 1)
            Cdd = (S.dot(S.T))/(Ne- 1)
             
            Usig,Sig,Vsig = cp.linalg.svd((Cdd + (cp.asarray(alpha)*cp.asarray(CDd))), \
                                          full_matrices = False)
            Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
            valuesig = Bsig[-1]                 # last element
            valuesig = valuesig * 0.9999
            indices = ( Bsig >= valuesig ).ravel().nonzero()
            toluse = Sig[indices]
            tol = toluse[0]
        
            (V,X,U) = pinvmatt((Cdd + (cp.asarray(alpha)*cp.asarray(CDd))),tol)
            
            update_term=((Cyd.dot(X)).dot((cp.tile(cp.asarray(True_data), Ne)\
                                        + cp.asarray(pertubations))\
                                          -Sim1 ))
                
            # update_term=((Cyd.dot(X)).dot((np.tile(True_data, Ne) -Sim1)\
            #                               + pertubations ))
            
            Ynew = Y + update_term 
            sizeclem=cp.asarray(nx*ny*nz)  
            if Yet==0:                  
                updated_ensemble=cp.asnumpy(Ynew[:sizeclem,:] )
                updated_ensemblep=cp.asnumpy(Ynew[sizeclem:2*sizeclem,:])
                if modelError==1:
                    CM=cp.asnumpy(Ynew[2*sizeclem:,:])
                else:
                    pass
            else:
                updated_ensemble=(Ynew[:sizeclem,:] )
                updated_ensemblep=(Ynew[sizeclem:2*sizeclem,:])
                if modelError==1:
                    CM=(Ynew[2*sizeclem:,:])
                else:
                    pass                
            
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep        
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)        
             
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .')
            if ii==0:
                cc_ini=cc
            else:
                pass
            
            if ii==Termm:
                print('Did not converge, Max Iteration reached')            
                break
            else:
                pass
            
            ii=ii+1
    
            
        
        if (choice==1) and (Geostats==1):
            ensemble=use_denoising(ensemble,nx,ny,nz,Ne)
        else:
            pass 
        
        ensemble,ensemblep=honour2(ensemblep,\
                                     ensemble,nx,ny,nz,N_ens,\
                                         High_K,Low_K,High_P,Low_P)          
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F') 
        controljj2p= np.reshape(meannp,(-1,1),'F') 
        controlj2=controljj2
        controlj2p=controljj2p
        
        
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
    
        
            Delete_files()                                
        Delete_files2()
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder) 
        os.chdir('HM_RESULTS')                       
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        shpw=cc[clem]
        controlbest= np.reshape(ensemble[:,clem],(-1,1),'F') 
        controlbestp= np.reshape(ensemblep[:,clem],(-1,1),'F')
        controlbest2=controlj2#controlbest
        controlbest2p=controljj2p#controlbest
    
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F') 
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')  
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        Plot_Histogram_now(Ne,cc_ini,cc)
        #os.makedirs('ESMDA')
        if not os.path.exists('REKI'):
            os.makedirs('REKI')
        else:
            shutil.rmtree('REKI')
            os.makedirs('REKI') 
            
        shutil.copy2('masterreal.data','REKI')
        print('6X Reservoir Simulator Forwarding - REKI Model')
        Forward_model(oldfolder,'REKI',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'REKI')
        os.chdir('REKI')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','REKI')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
             
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)
        
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM}) 
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding')
        
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files()
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )     
        
        
        print('--------------------SECTION REKI ENDED----------------------------') 
    elif Technique_REKI==2:
        print('')
        print('Regularised Ensemble Kalman Inversion with KSVD Parametrisation')
        print('Novel Implementation: Author: Clement Etienam SA Energy/GPU @Nvidia')
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        os.chdir(oldfolder)
        print('Learn KSVD Over complete dictionary')
        print('')
        Dicclem=Learn_Overcomplete_Dictionary(Ne)    
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))           
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)         
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
    
        R = ax**2
    
        CDd=np.diag(np.reshape(R,(-1,)))
        
        snn=0
        ii=0
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')        
        os.chdir(oldfolder)         
        
        while (snn<1):        
            print( 'Iteration --' + str(ii+1))
            
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    foldern=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,foldern,namecsv)
                    predMatrix.append(predd)        
                Delete_files()
            Delete_files2() 
                                   
            
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,\
                         "Initial.jpg")
                os.chdir(oldfolder)
            else:
                pass
            
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#   
            
            CDd=cp.asarray(CDd)
            True_data=cp.asarray(True_data)
            pertubations=cp.random.multivariate_normal(cp.ravel(cp.zeros\
                                                            ((Nop,1))), CDd,Ne)
            Ddraw=cp.tile(True_data, Ne) 
            pertubations=pertubations.T
            Dd=Ddraw #+ pertubations
            if Yet==0:
                CDd=cp.asnumpy(CDd)
                Dd=cp.asnumpy(Dd)
                Ddraw=cp.asnumpy(Ddraw)
                pertubations=cp.asnumpy(pertubations)
                True_data=cp.asnumpy(True_data)
            else:
                pass
            
            yyy=np.mean((Dd-simDatafinal).T@((inv(CDd)))\
                        @(Dd-simDatafinal),axis=1)
            yyy=yyy**(0.5)
            yyy=yyy.reshape(-1,1)
            alpha_star=np.mean(yyy,axis=0)
            if (snn+(1/alpha_star)>=1):
                alpha=1/(1-snn)
                snn=1
            else:
               alpha=alpha_star
               snn=snn+(1/alpha)
            print('alpha = ' + str(alpha))
            print('sn = ' + str(snn))
            sgsim=ensemble
            ensembledct=Sparse_coding(Dicclem,sgsim,Ne)#Get Sparse signal
            shapalla=ensembledct.shape[0]
         
            if modelError==1:
                overall=cp.vstack([cp.asarray(ensembledct),\
                                   cp.asarray(ensemblep),cp.asarray(CM)])
            else:
                overall=cp.vstack([cp.asarray(ensembledct),cp.asarray(ensemblep)])
            
            Y=overall 
            Sim1=cp.asarray(simDatafinal)
            M = cp.mean(Sim1,axis=1)
        
            M2=cp.mean(overall,axis=1)
            
            
            S = cp.zeros((Sim1.shape[0],Ne))
            yprime = cp.zeros((Y.shape[0],Y.shape[1]))
                   
            for jc in range(Ne):
                S[:,jc] = Sim1[:,jc]- M
                yprime[:,jc] = overall[:,jc] - M2
            Cyd = (yprime.dot(S.T))/(Ne - 1)
            Cdd = (S.dot(S.T))/(Ne- 1)
             
            Usig,Sig,Vsig = cp.linalg.svd((Cdd + (cp.asarray(alpha)*cp.asarray(CDd))), \
                                          full_matrices = False)
            Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
            valuesig = Bsig[-1]                 # last element
            valuesig = valuesig * 0.9999
            indices = ( Bsig >= valuesig ).ravel().nonzero()
            toluse = Sig[indices]
            tol = toluse[0]
        
            (V,X,U) = pinvmatt((Cdd + (cp.asarray(alpha)*cp.asarray(CDd))),tol)
            
            update_term=((Cyd.dot(X)).dot((cp.tile(cp.asarray(True_data), Ne) \
                                + cp.asarray(pertubations))\
                                          -Sim1 ))
                
            # update_term=((Cyd.dot(X)).dot((np.tile(True_data, Ne) -Sim1)\
            #                               + pertubations ))
            
            Ynew = Y + update_term 
            sizeclem=cp.asarray(nx*ny*nz)
            if Yet==0:                    
                updated_ensembleksvd=cp.asnumpy(Ynew[:shapalla,:]) 
                updated_ensemblep=cp.asnumpy(Ynew[shapalla:shapalla+sizeclem,:])
                if modelError==1:
                    CM=cp.asnumpy(Ynew[shapalla+sizeclem:,:])
                else:
                    pass
            else:
                updated_ensembleksvd=(Ynew[:shapalla,:]) 
                updated_ensemblep=(Ynew[shapalla:shapalla+sizeclem,:])
                if modelError==1:
                    CM=(Ynew[shapalla+sizeclem:,:])
                else:
                    pass                
            updated_ensemble=Recover_Dictionary_Saarse(Dicclem,\
                                        updated_ensembleksvd,Low_K,High_K)               
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep        
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)        
             
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .')
            if ii==0:
                cc_ini=cc
            else:
                pass
            
            if ii==Termm:
                print('Did not converge, Max Iteration reached')            
                break
            else:
                pass
            
            ii=ii+1

        
        ensemble,ensemblep=honour2(ensemblep,\
                                     ensemble,nx,ny,nz,N_ens,\
                                         High_K,Low_K,High_P,Low_P)          
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F') 
        controljj2p= np.reshape(meannp,(-1,1),'F') 
        controlj2=controljj2
        controlj2p=controljj2p
        
        
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
    
        
            Delete_files()                                
        Delete_files2()
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder) 
        os.chdir('HM_RESULTS')                       
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        shpw=cc[clem]
        controlbest= np.reshape(ensemble[:,clem],(-1,1),'F') 
        controlbestp= np.reshape(ensemblep[:,clem],(-1,1),'F')
        controlbest2=controlj2#controlbest
        controlbest2p=controljj2p#controlbest
    
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F') 
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')  
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        Plot_Histogram_now(Ne,cc_ini,cc)
        #os.makedirs('ESMDA')
        if not os.path.exists('REKI_KSVD'):
            os.makedirs('REKI_KSVD')
        else:
            shutil.rmtree('REKI_KSVD')
            os.makedirs('REKI_KSVD') 
            
        shutil.copy2('masterreal.data','REKI_KSVD')
        print('6X Reservoir Simulator Forwarding - REKI_KSVD Model')
        Forward_model(oldfolder,'REKI_KSVD',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'REKI_KSVD')
        os.chdir('REKI_KSVD')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','REKI_KSVD')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
             
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)
        
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM}) 
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding')
        
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files() 
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )     
        
        
        print('--------------------SECTION REKI_KSVD ENDED-------------------')
    elif Technique_REKI==3:
        print('')
        print('Regularised Ensemble Kalman Inversion with DCT Parametrisation')
        print('Novel Implementation: Author: Clement Etienam SA Energy/GPU @Nvidia')
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        os.chdir(oldfolder)
        size1,size2=int(cp.ceil(int(0.7*nx))),int(cp.ceil(int(0.7*ny)))
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))           
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)         
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
    
        R = ax**2
    
        CDd=np.diag(np.reshape(R,(-1,)))
        
        snn=0
        ii=0
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')        
        os.chdir(oldfolder)         
        
        while (snn<1):        
            print( 'Iteration --' + str(ii+1))
            
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    foldern=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,foldern,namecsv)
                    predMatrix.append(predd)        
                Delete_files()
            Delete_files2() 
                                   
            
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,\
                         "Initial.jpg")
                os.chdir(oldfolder)
            else:
                pass
            
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#   
            
            CDd=cp.asarray(CDd)
            True_data=cp.asarray(True_data)
            pertubations=cp.random.multivariate_normal(cp.ravel(cp.zeros\
                                                            ((Nop,1))), CDd,Ne)
            Ddraw=cp.tile(True_data, Ne) 
            pertubations=pertubations.T
            Dd=Ddraw #+ pertubations
            if Yet==0:
                CDd=cp.asnumpy(CDd)
                Dd=cp.asnumpy(Dd)
                Ddraw=cp.asnumpy(Ddraw)
                pertubations=cp.asnumpy(pertubations)
                True_data=cp.asnumpy(True_data)
            else:
                pass
            
            yyy=np.mean((Dd-simDatafinal).T@((inv(CDd)))\
                        @(Dd-simDatafinal),axis=1)
            yyy=yyy**(0.5)
            yyy=yyy.reshape(-1,1)
            alpha_star=np.mean(yyy,axis=0)
            if (snn+(1/alpha_star)>=1):
                alpha=1/(1-snn)
                snn=1
            else:
               alpha=alpha_star
               snn=snn+(1/alpha)
            print('alpha = ' + str(alpha))
            print('sn = ' + str(snn))
            
            
            ensembledct=dct22(ensemble,Ne,nx,ny,nz,size1,size2)
            sgsim=ensembledct
            shapalla=sgsim.shape[0]
                       
            
            
            if modelError==1:
                overall=cp.vstack([cp.asarray(sgsim),cp.asarray(ensemblep),cp.asarray(CM)])
            else:
                overall=cp.vstack([cp.asarray(sgsim),cp.asarray(ensemblep)])
            
            Y=overall 
            Sim1=cp.asarray(simDatafinal)
            M = cp.mean(Sim1,axis=1)
        
            M2=cp.mean(overall,axis=1)
            
            
            S = cp.zeros((Sim1.shape[0],Ne))
            yprime = cp.zeros((Y.shape[0],Y.shape[1]))
                   
            for jc in range(Ne):
                S[:,jc] = Sim1[:,jc]- M
                yprime[:,jc] = overall[:,jc] - M2
            Cyd = (yprime.dot(S.T))/(Ne - 1)
            Cdd = (S.dot(S.T))/(Ne- 1)
             
            Usig,Sig,Vsig = cp.linalg.svd((Cdd + (cp.asarray(alpha)*cp.asarray(CDd))), \
                                          full_matrices = False)
            Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
            valuesig = Bsig[-1]                 # last element
            valuesig = valuesig * 0.9999
            indices = ( Bsig >= valuesig ).ravel().nonzero()
            toluse = Sig[indices]
            tol = toluse[0]
        
            (V,X,U) = pinvmatt((Cdd + (cp.asarray(alpha)*cp.asarray(CDd))),tol)
            
            update_term=((Cyd.dot(X)).dot((cp.tile(cp.asarray(True_data), Ne)\
                            + cp.asarray(pertubations))\
                                          -Sim1 ))
                
            # update_term=((Cyd.dot(X)).dot((np.tile(True_data, Ne) -Sim1)\
            #                               + pertubations ))
            
            Ynew = Y + update_term 
            sizeclem=cp.asarray(nx*ny*nz) 
            if Yet==0:                   
                updated_ensembledct=cp.asnumpy(Ynew[:shapalla,:] )
                updated_ensemble=idct22(updated_ensembledct,Ne,nx,ny,nz,size1,size2)             
                updated_ensemblep=cp.asnumpy(Ynew[shapalla:shapalla+sizeclem,:])
                if modelError==1:
                    CM=cp.asnumpy(Ynew[shapalla+sizeclem:,:])
                else:
                    pass
            else:
                updated_ensembledct=(Ynew[:shapalla,:] )
                updated_ensemble=idct22(updated_ensembledct,Ne,nx,ny,nz,size1,size2)             
                updated_ensemblep=(Ynew[shapalla:shapalla+sizeclem,:])
                if modelError==1:
                    CM=(Ynew[shapalla+sizeclem:,:])
                else:
                    pass                
            
            ensemble=updated_ensemble
            ensemblep=updated_ensemblep        
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)        
             
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .')
            if ii==0:
                cc_ini=cc
            else:
                pass
            
            if ii==Termm:
                print('Did not converge, Max Iteration reached')            
                break
            else:
                pass
            
            ii=ii+1

        
        # ensemble,ensemblep=honour2(ensemblep,\
        #                              ensemble,nx,ny,nz,N_ens,\
        #                                  High_K,Low_K,High_P,Low_P)          
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F') 
        controljj2p= np.reshape(meannp,(-1,1),'F') 
        controlj2=controljj2
        controlj2p=controljj2p
        
        
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
    
        
            Delete_files()                                
        Delete_files2()
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder) 
        os.chdir('HM_RESULTS')                       
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        shpw=cc[clem]
        controlbest= np.reshape(ensemble[:,clem],(-1,1),'F') 
        controlbestp= np.reshape(ensemblep[:,clem],(-1,1),'F')
        controlbest2=controlj2#controlbest
        controlbest2p=controljj2p#controlbest
    
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F') 
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')  
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        Plot_Histogram_now(Ne,cc_ini,cc)
        #os.makedirs('ESMDA')
        if not os.path.exists('REKI_DCT'):
            os.makedirs('REKI_DCT')
        else:
            shutil.rmtree('REKI_DCT')
            os.makedirs('REKI_DCT') 
            
        shutil.copy2('masterreal.data','REKI_DCT')
        print('6X Reservoir Simulator Forwarding - REKI_DCT Model')
        Forward_model(oldfolder,'REKI_DCT',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'REKI_DCT')
        os.chdir('REKI_DCT')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','REKI_DCT')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
             
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)
        
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM}) 
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding')
        
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files() 
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )     
        
        
        print('--------------------SECTION REKI_DCT ENDED--------------------')  
    else:
        print('')
        print('Regularised Ensemble Kalman Inversion with Convolution Autoencoder\
Parametrisation\n')
        print('Novel Implementation: Author: Clement Etienam SA Energy/GPU @Nvidia')
        if modelError==1:
            print(' Model Error considered')
        else:
            print('Model Error ignored')
        os.chdir(oldfolder)
    
        if Geostats==1:
            if Deccor==1:#Deccorelate the ensemble
                filename="Ganensemble.mat" #Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef=mat['Z']  
                
                avs=whiten(ini_ensemblef, method='zca_cor')
                
                index = np.random.choice(avs.shape[1], N_ens, \
                                         replace=False)
                ini_ensemblee=avs[:,index]            
                
                clfye = MinMaxScaler(feature_range=(Low_K,High_K))
                (clfye.fit(ini_ensemblee))    
                ini_ensemble=(clfye.transform(ini_ensemblee))           
            else:
                if afresh==1:
                    see=intial_ensemble(nx,ny,nz,N_ens,permx)
                    ini_ensemblee=np.split(see, N_ens, axis=1)
                    ini_ensemble=[]
                    for ky in range(N_ens):
                        aa=ini_ensemblee[ky]
                        aa=np.reshape(aa,(-1,1),'F')
                        ini_ensemble.append(aa)
                        
                    ini_ensemble=np.hstack(ini_ensemble) 
                else:
                    filename="Ganensemble.mat" #Ensemble generated offline
                    mat = sio.loadmat(filename)
                    ini_ensemblef=mat['Z']
                    index = np.random.choice(ini_ensemblef.shape[0], N_ens, \
                                             replace=False)
                    ini_ensemble=ini_ensemblef[:,index]
        else:
            if Deccor==1:
                ini_ensemblef=initial_ensemble_gaussian(nx,ny,nz,5000,\
                                    Low_K,High_K)            
                ini_ensemblef=cp.asarray(ini_ensemblef)
                
                
                beta=int(cp.ceil(int(ini_ensemblef.shape[0]/Ne)))
                
                V,S1,U = cp.linalg.svd(ini_ensemblef,full_matrices=1)
                v = V[:,:Ne]
                U1 = U.T
                u = U1[:,:Ne]
                S11 = S1[:Ne]
                s = S11[:]
                S = (1/((beta)**(0.5)))*s
                #S=s
                X = (v*S).dot(u.T)
                if Yet==0:
                    X=cp.asnumpy(X)
                    ini_ensemblef=cp.asnumpy(ini_ensemblef)
                else:
                    pass
                
                X[X<=Low_K]=Low_K
                X[X>=High_K]=High_K 
                ini_ensemble=X[:,:Ne] 
            else:
                
                ini_ensemble=initial_ensemble_gaussian(nx,ny,nz,N_ens,\
                                    Low_K,High_K)         
        
        ensemble=ini_ensemble
        ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
            if True_data[iq,:]==0:
                ax[iq,:]=1 
            else:
                
                if Big_noise==2:
                    ax[iq,:]=sqrt(noise_level*True_data[iq,:])
                else:
                    ax[iq,:]=sqrt(diffyet[iq,:])
                
    
        R = ax**2
    
        CDd=np.diag(np.reshape(R,(-1,)))
        
        snn=0
        ii=0
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')        
        os.chdir(oldfolder)         
        
        while (snn<1):        
            print( 'Iteration --' + str(ii+1))
            
            for ijesus in range(N_ens):
                (
                write_include)(ijesus,ensemble,ensemblep,'Realization_')      
            az=int(np.ceil(int(N_ens/maxx)))
            a=(np.linspace(1, N_ens, num=Ne))
            use1=Split_Matrix (a, az)
            predMatrix=[]
            print('...6X Reservoir Simulator Forwarding')
            for xx in range(az):
                print( '      Batch ' + str(xx+1) + ' | ' + str(az))
                ause=use1[xx]
                ause=ause.astype(np.int32)
                ause=listToStringWithoutBrackets(ause)
                overwrite_Data_File(name,ause)
                #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
                os.system(string_Jesus)
                for kk in range(maxx):
                    foldern=stringf + str(kk)
                    namecsv=stringf + str(kk) +'.csv'
                    predd=Get_RSM_6X_Ensemble(oldfolder,foldern,namecsv)
                    predMatrix.append(predd)        
                Delete_files()
            Delete_files2()
                                   
            
            if ii==0:
                os.chdir('HM_RESULTS')
                Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,\
                         "Initial.jpg")
                os.chdir(oldfolder)
            else:
                pass
            
            simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)#   
            
            CDd=cp.asarray(CDd)
            True_data=cp.asarray(True_data)
            pertubations=cp.random.multivariate_normal(cp.ravel(cp.zeros\
                                                            ((Nop,1))), CDd,Ne)
            Ddraw=cp.tile(True_data, Ne) 
            pertubations=pertubations.T
            Dd=Ddraw #+ pertubations
            if Yet==0:
                CDd=cp.asnumpy(CDd)
                Dd=cp.asnumpy(Dd)
                Ddraw=cp.asnumpy(Ddraw)
                pertubations=cp.asnumpy(pertubations)
                True_data=cp.asnumpy(True_data)
            else:
                pass
            
            yyy=np.mean((Dd-simDatafinal).T@((inv(CDd)))\
                        @(Dd-simDatafinal),axis=1)
            yyy=yyy**(0.5)
            yyy=yyy.reshape(-1,1)
            alpha_star=np.mean(yyy,axis=0)
            if (snn+(1/alpha_star)>=1):
                alpha=1/(1-snn)
                snn=1
            else:
               alpha=alpha_star
               snn=snn+(1/alpha)
            print('alpha = ' + str(alpha))
            print('sn = ' + str(snn))
            sgsim=ensemble
            encoded=Get_Latent(sgsim,Ne,nx,ny,nz)
            shapalla=encoded.shape[0]
            
            if modelError==1:
                overall=cp.vstack([cp.asarray(sgsim),cp.asarray(ensemblep),cp.asarray(CM)])
            else:
                overall=cp.vstack([cp.asarray(sgsim),cp.asarray(ensemblep)])
            
            Y=overall 
            Sim1=cp.asarray(simDatafinal)
            M = cp.mean(Sim1,axis=1)
        
            M2=cp.mean(overall,axis=1)
            
            
            S = cp.zeros((Sim1.shape[0],Ne))
            yprime = cp.zeros((Y.shape[0],Y.shape[1]))
                   
            for jc in range(Ne):
                S[:,jc] = Sim1[:,jc]- M
                yprime[:,jc] = overall[:,jc] - M2
            Cyd = (yprime.dot(S.T))/(Ne - 1)
            Cdd = (S.dot(S.T))/(Ne- 1)
             
            Usig,Sig,Vsig = cp.linalg.svd((Cdd + (cp.asarray(alpha)*cp.asarray(CDd))), \
                                          full_matrices = False)
            Bsig = cp.cumsum(Sig, axis = 0)          # vertically addition
            valuesig = Bsig[-1]                 # last element
            valuesig = valuesig * 0.9999
            indices = ( Bsig >= valuesig ).ravel().nonzero()
            toluse = Sig[indices]
            tol = toluse[0]
        
            (V,X,U) = pinvmatt((Cdd + (cp.asarray(alpha)*cp.asarray(CDd))),tol)
            
            update_term=((Cyd.dot(X)).dot((cp.tile(True_data, Ne) + \
                                cp.asarray(pertubations))\
                                          -Sim1 ))
                
            # update_term=((Cyd.dot(X)).dot((np.tile(True_data, Ne) -Sim1)\
            #                               + pertubations ))
            
            Ynew = Y + update_term 
            sizeclem=cp.asarray(nx*ny*nz)
            if Yet==0:                    
                updated_ensemble=cp.asnumpy(Ynew[:shapalla,:] )
                updated_ensemblep=cp.asnumpy(Ynew[shapalla:shapalla+sizeclem,:])
                if modelError==1:
                    CM=cp.asnumpy(Ynew[shapalla+sizeclem:,:])
                else:
                    pass
            else:
                updated_ensemble=Ynew[:shapalla,:] 
                updated_ensemblep=Ynew[shapalla:shapalla+sizeclem,:]
                if modelError==1:
                    CM=Ynew[shapalla+sizeclem:,:]
                else:
                    pass                

            ensemble=Recover_image(updated_ensemble,Ne,nx,ny,nz)             
            ensemblep=updated_ensemblep        
            ensemble,ensemblep=honour2(ensemblep,\
                                         ensemble,nx,ny,nz,N_ens,\
                                             High_K,Low_K,High_P,Low_P)        
             
            
            simmean=np.reshape(np.mean(simDatafinal,axis=1),(-1,1),'F')
            tinuke=((np.sum((((simmean) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble mean = : ' \
                  + str(tinuke) + '... .') 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
            clem=np.argmin(cc)
            simmbest=simDatafinal[:,clem].reshape(-1,1)
            tinukebest=((np.sum((((simmbest) - True_data) ** 2)) )  **(0.5))\
                /True_data.shape[0]
            print('RMSE of the ensemble best = : ' \
                + str(tinukebest) + '... .')
            if ii==0:
                cc_ini=cc
            else:
                pass
            
            if ii==Termm:
                print('Did not converge, Max Iteration reached')            
                break
            else:
                pass
            
            ii=ii+1

        ensemble,ensemblep=honour2(ensemblep,\
                                     ensemble,nx,ny,nz,N_ens,\
                                         High_K,Low_K,High_P,Low_P)          
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        meannp=np.reshape(np.mean(ensemblep,axis=1),(-1,1),'F')
        
        meanini=np.reshape(np.mean(ini_ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,1),'F') 
        controljj2p= np.reshape(meannp,(-1,1),'F') 
        controlj2=controljj2
        controlj2p=controljj2p
        
        
        for ijesus in range(N_ens):
            (
            write_include)(ijesus,ensemble,ensemblep,'Realization_')      
        az=int(np.ceil(int(N_ens/maxx)))
        a=(np.linspace(1, N_ens, num=Ne))
        use1=Split_Matrix (a, az)
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        for xx in range(az):
            print( '      Batch ' + str(xx+1) + ' | ' + str(az))
            #xx=0
            ause=use1[xx]
            ause=ause.astype(np.int32)
            ause=listToStringWithoutBrackets(ause)
            overwrite_Data_File(name,ause)
            #os.system("@mpiexec -np 10 6X_34157_75 masterrEM.data -csv -sumout 3 ")
            os.system(string_Jesus)
            for kk in range(maxx):
                folder=stringf + str(kk)
                namecsv=stringf + str(kk) +'.csv'
                predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
                predMatrix.append(predd)
    
        
            Delete_files()                                
        Delete_files2()
        print('Read Historical data')
        os.chdir('True_Model')
        True_measurement=Get_Measuremnt_CSV('hm0.csv')
        True_mat=True_measurement
        True_data=True_mat[:,1:]
        True_data=np.reshape(True_data,(-1,1),'F')    
        os.chdir(oldfolder) 
        os.chdir('HM_RESULTS')                       
        Plot_RSM(predMatrix,modelError,CM,Ne,True_mat,"Final.jpg")
        os.chdir(oldfolder)
        simDatafinal = Get_simulated(predMatrix,modelError,CM,Ne)# 
        
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_data)
        clem=np.argmin(cc)
        shpw=cc[clem]
        controlbest= np.reshape(ensemble[:,clem],(-1,1),'F') 
        controlbestp= np.reshape(ensemblep[:,clem],(-1,1),'F')
        controlbest2=controlj2#controlbest
        controlbest2p=controljj2p#controlbest
    
        clembad=np.argmax(cc)
        controlbad= np.reshape(ensemble[:,clembad] ,(-1,1),'F') 
        controlbadp= np.reshape(ensemblep[:,clembad] ,(-1,1),'F')  
        mEmean=np.reshape(np.mean(CM,axis=1),(-1,1),'F')
        mEbest=np.reshape(CM[:,clem] ,(-1,1),'F')
        mEbad=np.reshape(CM[:,clembad] ,(-1,1),'F')
        Plot_Histogram_now(Ne,cc_ini,cc)
        #os.makedirs('ESMDA')
        if not os.path.exists('REKI_AE'):
            os.makedirs('REKI_AE')
        else:
            shutil.rmtree('REKI_AE')
            os.makedirs('REKI_AE') 
            
        shutil.copy2('masterreal.data','REKI_AE')
        print('6X Reservoir Simulator Forwarding - REKI_AE Model')
        Forward_model(oldfolder,'REKI_AE',controlbest2,controlbest2p)
        yycheck=Get_RSM(oldfolder,'REKI_AE')
        os.chdir('REKI_AE')
        Plot_RSM_single(yycheck,modelError,mEbest,'Performance.jpg','REKI_AE')
        os.chdir(oldfolder) 
        
        if modelError==1:
            usesim=yycheck[:,1:]
            CC=mEbest
            a1=np.reshape(CC[:,0],(-1,usesim.shape[1]),'F')
    
            result10 = np.zeros((usesim.shape[0],usesim.shape[1]))
            for j in range(usesim.shape[0]):
                aa1=usesim[j,:]+a1
    
                result10[j,:]=aa1#+aa2+aa3+aa4+aa5
            usesim=result10
        else:
            usesim=yycheck[:,1:]
        #usesim=Normalize_data(usesim)
        usesim=np.reshape(usesim,(-1,1),'F')        
        yycheck=usesim    
             
        cc=((np.sum(((( yycheck) - True_data) ** 2)) )  **(0.5)) \
            /True_data.shape[0]
        print('RMSE  = : ' \
            + str(cc) )
        os.chdir('HM_RESULTS')
        Plot_mean(controlbest,controljj2,meanini,nx,ny)
        os.chdir(oldfolder)
        
        print(' Plot P10,P50,P90 and Base Measurment')
        os.chdir('HM_RESULTS')
        sio.savemat('Posterior_Ensembles.mat', {'PERM_Reali':ensemble,\
        'PORO_Reali':ensemblep,'P10_Perm':controlbest,'P50_Perm':controljj2,\
            'P90_Perm':controlbad,'P10_Poro':controlbest2p,'P50_Poro':controljj2p,\
            'P90_Poro':controlbadp,'modelError':modelError,'modelNoise':CM}) 
        os.chdir(oldfolder)
        ensembleout1=np.hstack([controlbest,controljj2,controlbad])
        ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])
        CMens=np.hstack([mEbest,mEmean,mEbad])
        #ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
    
        if not os.path.exists('PERCENTILE'):
            os.makedirs('PERCENTILE')
        else:
            shutil.rmtree('PERCENTILE')
            os.makedirs('PERCENTILE') 
            
        shutil.copy2('masterreal.data','PERCENTILE')
        print('6X Reservoir Simulator Forwarding')
        
        yzout=[]
        for i in range(3):       
            write_include(i,ensembleout1,ensembleoutp1,'RealizationPI_')      
        predMatrix=[]
        print('...6X Reservoir Simulator Forwarding')
        os.system("@mpiexec -np 3 6X_34157_75 masterrEMPI.data -csv -sumout 3 ")
        for kk in range(3):
            folder=stringf2 + str(kk)
            namecsv=stringf2 + str(kk) +'.csv'
            predd=Get_RSM_6X_Ensemble(oldfolder,folder,namecsv)
            yzout.append(predd) 
        Delete_files()
        Delete_files2()
        pertout = Get_simulated(yzout,modelError,CMens,3)# 
        Plot_RSM_percentile(yzout,CMens,modelError,True_mat,Base_mat,"P10_P50_P90.jpg")
            
        plot_3d_pyvista(np.reshape(controlbest,(nx,ny,nz),'F'),'P10_Perm' ) 
        plot_3d_pyvista(np.reshape(controljj2,(nx,ny,nz),'F'),'P50_Perm' )
        plot_3d_pyvista(np.reshape(controlbad,(nx,ny,nz),'F'),'P90_Perm' )     
        
        
        print('--------------------SECTION REKI_CAEE ENDED-------------------')        
print('-------------------PROGRAM EXECUTED-----------------------------------')  
