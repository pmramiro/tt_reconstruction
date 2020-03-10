from math import sqrt
import numpy as np
import random
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import h5py

start_time = time.time()


execfile("lhereader.py")

data=readLHEF('big_run_data/unweighted_events_100000_8.lhe')     # instance


## Define physical parameters
s = 10**6
mb = 4.7
#mW = 80.42
mt = 173
#w_W = 2
#w_t = 1.5
mW = 80.419
w_W = 2.0476
w_t = 1.491500

## Define counters
counter1 = 0
counter2 = 0
trap = 0
trap_good = 0
trap_bad = 0
i = 0
bad = 0
flag_branch = 0
flag_b = 0
flag_scalar = 0
flag_E = 0

## Define number of entries in arrays
n_evs = len(data.events)
n_parts = 10
#entries = 6


## Define arrays
#evs = np.zeros((n_evs, n_parts, entries), dtype=float)
p0 = np.array([sqrt(s), 0, 0, 0])
t = np.zeros(7, dtype=float)        # t -> Wbar b -> lbar nu b
tbar = np.zeros(7, dtype=float)     # tbar -> W bbar -> l nubar bbar
W = np.zeros(7, dtype=float)        # W- = W2 = W
Wbar = np.zeros(7, dtype=float)     # W+ = W1 = Wbar
b = np.zeros(7, dtype=float)
bbar = np.zeros(7, dtype=float)
l = np.zeros(7, dtype=float)
lbar = np.zeros(7, dtype=float)
nu = np.zeros(7, dtype=float)
nubar = np.zeros(7, dtype=float)

t_data_rec = np.zeros(7, dtype=float)
tbar_data_rec = np.zeros(7, dtype=float)
W_data_rec = np.zeros(7, dtype=float)        # W- = W2 = W
Wbar_data_rec = np.zeros(7, dtype=float)     # W+ = W1 = Wbar
b_data_rec = np.zeros(7, dtype=float)
bbar_data_rec = np.zeros(7, dtype=float)
l_data_rec = np.zeros(7, dtype=float)
lbar_data_rec = np.zeros(7, dtype=float)
nu_data_rec = np.zeros(7, dtype=float)
nubar_data_rec = np.zeros(7, dtype=float)

## Arrays for reconstruction
alpha = np.zeros(3, dtype=float)
beta0 = np.zeros(3, dtype=float)
beta1 = np.zeros(3, dtype=float)
beta2 = np.zeros(3, dtype=float)

## Arrays to store some data
branch = np.zeros([4, 5], dtype=float)
masses_W1 = np.zeros(10, dtype=float)

## Arrays for gradient descent
m = np.zeros(2, dtype=float)
v = np.zeros(2, dtype=float)


## Open the file to save the data
f = h5py.File('big_run_central_fit_results/results_100000_central_9.hdf5', 'w')
reconstructed = f.create_dataset('reconstructed', (n_evs, n_parts, 7), dtype='f4')
truth = f.create_dataset('truth', (n_evs, n_parts, 7), dtype='f4')
rec_good = f.create_dataset('rec_good', (1, n_parts, 7), maxshape=(n_evs, n_parts, 7), dtype='f4')
truth_good = f.create_dataset('truth_good', (1, n_parts, 7), maxshape=(n_evs, n_parts, 7), dtype='f4')
K4_good = f.create_dataset('K4_good', (1, 1), maxshape=(n_evs, 1), dtype='f4')


## List of functions that we will use, imported from functions.py

# lorentz(v1, v2)  | Input: two 4-vectors | Output: a scalar
# M2(v1)  | Input: one 4-vector | Output: a scalar
# beta(x, y)  | Input: two squared masses | Output: a 3-component vector
# alphabeta(x, y)  | Input: two squared masses | Output: a scalar
# DIS(x, y)  | Input: two squared masses | Output: a scalar
# DIS_x(x, y)  | Input: two squared masses | Output: a scalar
# DIS_y(x, y)  | Input: two squared masses | Output: a scalar
# Pnu(x, y, sign)  | Input: two squared masses and a discriminant sign | Output: a 4-vector







################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


def lorentz(v1, v2):   ## Input: two 4-vectors | Output: a scalar

    # Computes the lorent product of two 4-vectors

    if (len(v1) != 4) | (len(v2) != 4):
        print("The input of lorentz(v1, v2) is not a 4-vector")
        exit()

    total = v1[0]*v2[0] - v1[1]*v2[1] - v1[2]*v2[2] - v1[3]*v2[3]
    
    return total


def M2(v1):   ## Input: one 4-vector | Output: a scalar

    # Computes the squared mass from a 4-vector
    
    if (len(v1) != 4):
        print("The input of M2(v1) is not a 4-vector")
        exit()

    total = v1[0]**2 - v1[1]**2 - v1[2]**2 - v1[3]**2
    
    return total


def beta(x, y):   ## Input: two squared masses | Output: a 3-component vector

    vector = beta0 + beta1*x + beta2*y
    
    return vector


def alphabeta(x, y):   ## Input: two squared masses | Output: a scalar

    total = ab0 + ab1*x + ab2*y
    
    return total


def DIS(x, y):   ## Input: two squared masses | Output: a scalar

    # Discriminant

    total = d00 + d10*x + d01*y + d20*x**2 + d11*x*y  + d02*y**2
    
    return total


def DIS_x(x, y):   ## Input: two squared masses | Output: a scalar

    # Partial derivative wrt x of the discriminant

    total = d10 + 2*d20*x + d11*y
    
    return total


def DIS_y(x, y):   ## Input: two squared masses | Output: a scalar

    # Partial derivative wrt y of the discriminant

    total = d01 + d11*x + 2*d02*y
    
    return total


def DIS_var(variables):   ## Input: two squared masses | Output: a scalar

    # Discriminant

    x, y = variables

    total = d00 + d10*x + d01*y + d20*x**2 + d11*x*y  + d02*y**2
    
    return np.array([total])


def DIS_jac(variables):   ## Input: two squared masses | Output: a scalar

    # Jacobian matrix of the discriminant

    x, y = variables

    total_x = d10 + 2*d20*x + d11*y

    total_y = d01 + d11*x + 2*d02*y

    vector = np.array([total_x, total_y])
    
    return vector


def Pnu(x, y, sign):   ## Input: two squared masses and a discriminant sign | Output: a 4-vector

    # Reconstructed neutrino 4-vector

    if (DIS(x, y) > 0):
        
        Enu = np.array([ ainv*(alphabeta(x, y) + sign*sqrt(DIS(x, y))) ])
        
    elif ( (DIS(x, y) <= 0) & (DIS(x, y) > -1) ):

        Enu = np.array([ ainv*(alphabeta(x, y)) ])
    
    pnu = Enu*alpha + beta(x, y)

    vector = np.concatenate([Enu, pnu], axis=0)

    return vector


def K4(x, y, sign, signb):   ## Input: two squared masses, sign and sign of b's | Output: a scalar

    # Computes the weight function to minimize

    if (DIS(x, y) > -1):
        total = ((1./mW**2)*(1./w_W**2)*(x - mW**2)**2 + 1) * ((1./mW**2)*(1./w_W**2)*(y - mW**2)**2 + 1) * ((1./mt**2)*(1./w_t**2)*(x + mb**2 + 2*lorentz(lbar + Pnu(x, y, sign), 1./2*(1 + signb)*j1 + 1./2*(1 - signb)*j2) - mt**2)**2 + 1) * ((1./mt**2)*(1./w_t**2)*(y + mb**2 + 2*lorentz(p0 - lbar - j1 - j2 - Pnu(x, y, sign), 1./2*(1 - signb)*j1 + 1./2*(1 + signb)*j2) - mt**2)**2 + 1)

    else:
        total = 10**18

    return total


def F4(variables):   ## Input: two squared masses | Output: a scalar

    # K4 defined in a more efficient way

    x, y = variables

    if (DIS(x, y) < 0):
        total = 10**18
    else:
        
        total = (1 + invGMW*(mW**2 - x)**2)*(1 + invGMW*(mW**2 - y)**2)*(1 + invGMT*(2*b0pb - ct1 + 2*ab0*ainv*(apb - Eb) + mt**2 - x + 2*ab1*ainv*apb*x + 2*b1pb*x - 2*ab1*ainv*Eb*x + 2*ab2*ainv*apb*y + 2*b2pb*y - 2*ab2*ainv*Eb*y)**2)*(1 + invGMT*(2*b0pbbar + ct2 + 2*ab0*ainv*(apbbar - Ebbar) - mt**2 + 2*ab1*ainv*apbbar*x + 2*b1pbbar*x - 2*ab1*ainv*Ebbar*x + y + 2*ab2*ainv*apbbar*y + 2*b2pbbar*y - 2*ab2*ainv*Ebbar*y)**2) + 4*ainv**2*invGMT*(1 + invGMW*(mW**2 - x)**2)*(1 + invGMW*(mW**2 - y)**2)*(4*(apb - Eb)*(apbbar - Ebbar)*invGMT*(2*b0pb - ct1 + 2*ab0*ainv*(apb - Eb) + mt**2 - x + 2*ab1*ainv*apb*x + 2*b1pb*x - 2*ab1*ainv*Eb*x + 2*ab2*ainv*apb*y + 2*b2pb*y - 2*ab2*ainv*Eb*y)*(2*b0pbbar + ct2 + 2*ab0*ainv*(apbbar - Ebbar) - mt**2 + 2*ab1*ainv*apbbar*x + 2*b1pbbar*x - 2*ab1*ainv*Ebbar*x + y + 2*ab2*ainv*apbbar*y + 2*b2pbbar*y - 2*ab2*ainv*Ebbar*y) + (apbbar - Ebbar)**2*(1 + invGMT*(2*b0pb - ct1 + 2*ab0*ainv*(apb - Eb) + mt**2 - x + 2*ab1*ainv*apb*x + 2*b1pb*x - 2*ab1*ainv*Eb*x + 2*ab2*ainv*apb*y + 2*b2pb*y - 2*ab2*ainv*Eb*y)**2) + (apb - Eb)**2*(1 + invGMT*(2*b0pbbar + ct2 + 2*ab0*ainv*(apbbar - Ebbar) - mt**2 + 2*ab1*ainv*apbbar*x + 2*b1pbbar*x - 2*ab1*ainv*Ebbar*x + y + 2*ab2*ainv*apbbar*y + 2*b2pbbar*y - 2*ab2*ainv*Ebbar*y)**2))*DIS(x, y) + 16*ainv**4*(apb - Eb)**2*(apbbar - Ebbar)**2*invGMT**2*(1 + invGMW*(mW**2 - x)**2)*(1 + invGMW*(mW**2 - y)**2)*DIS(x, y)**2 + sign*(4*ainv*invGMT*(1 + invGMW*(mW**2 - x)**2)*(1 + invGMW*(mW**2 - y)**2)*((apbbar - Ebbar)*(2*b0pbbar + ct2 + 2*ab0*ainv*(apbbar - Ebbar) - mt**2 + 2*ab1*ainv*apbbar*x + 2*b1pbbar*x - 2*ab1*ainv*Ebbar*x + y + 2*ab2*ainv*apbbar*y + 2*b2pbbar*y - 2*ab2*ainv*Ebbar*y)*(1 + invGMT*(2*b0pb - ct1 + 2*ab0*ainv*(apb - Eb) + mt**2 - x + 2*ab1*ainv*apb*x + 2*b1pb*x - 2*ab1*ainv*Eb*x + 2*ab2*ainv*apb*y + 2*b2pb*y - 2*ab2*ainv*Eb*y)**2) + (apb - Eb)*(2*b0pb - ct1 + 2*ab0*ainv*(apb - Eb) + mt**2 - x + 2*ab1*ainv*apb*x + 2*b1pb*x - 2*ab1*ainv*Eb*x + 2*ab2*ainv*apb*y + 2*b2pb*y - 2*ab2*ainv*Eb*y)*(1 + invGMT*(2*b0pbbar + ct2 + 2*ab0*ainv*(apbbar - Ebbar) - mt**2 + 2*ab1*ainv*apbbar*x + 2*b1pbbar*x - 2*ab1*ainv*Ebbar*x + y + 2*ab2*ainv*apbbar*y + 2*b2pbbar*y - 2*ab2*ainv*Ebbar*y)**2)) + 16*ainv**3*(apb - Eb)*(apbbar - Ebbar)*invGMT**2*(1 + invGMW*(mW**2 - x)**2)*(1 + invGMW*(mW**2 - y)**2)*(2*apb*b0pbbar + apb*ct2 - 2*b0pbbar*Eb - ct2*Eb + 4*ab0*ainv*(apb - Eb)*(apbbar - Ebbar) - 2*b0pb*Ebbar + ct1*Ebbar - apb*mt**2 + Eb*mt**2 - Ebbar*mt**2 + 2*apb*b1pbbar*x - 2*b1pbbar*Eb*x + Ebbar*x - 4*ab1*ainv*apb*Ebbar*x - 2*b1pb*Ebbar*x + 4*ab1*ainv*Eb*Ebbar*x + apb*y + 2*apb*b2pbbar*y - Eb*y - 2*b2pbbar*Eb*y - 4*ab2*ainv*apb*Ebbar*y - 2*b2pb*Ebbar*y + 4*ab2*ainv*Eb*Ebbar*y + apbbar*(2*b0pb - ct1 + mt**2 - x + 4*ab1*ainv*apb*x + 2*b1pb*x - 4*ab1*ainv*Eb*x + 4*ab2*ainv*apb*y + 2*b2pb*y - 4*ab2*ainv*Eb*y))*DIS(x, y))*sqrt(DIS(x, y))

    return total







## Read event data
    
for event in data.events:
    n_parts = len(event.particles)
    info = np.zeros((n_parts,7), dtype=float)
    for particle in event.particles:
        info[counter1] = [particle.pdgid, particle.energy, particle.px, particle.py, particle.pz, particle.mass, particle.eta]
        counter1 += 1       
    counter1 = 0
    trap += 1

    for i in xrange(2, n_parts):
        if info[i][0] == 6:
            t_data = info[i]
        elif info[i][0] == 24:
            Wbar_data = info[i]
        elif info[i][0] == -6:
            tbar_data = info[i]
        elif info[i][0] == -24:
            W_data = info[i]
        elif info[i][0] == 11 or info[i][0] == 13:
            l_data = info[i]
        elif info[i][0] == -11 or info[i][0] == -13:
            lbar_data = info[i]
        elif info[i][0] == 12 or info[i][0] == 14:
            nu_data = info[i]
        elif info[i][0] == -12 or info[i][0] == -14:
            nubar_data = info[i]
        elif info[i][0] == 5:
            b_data = info[i]
        elif info[i][0] == -5:
            bbar_data = info[i]
        else:
            print "Unknown particle in event %i" % counter1


            
    #print("*************************** Iteration %.0f ***************************" % trap)

            
    ## Build 4-momentum arrays
            
    t = np.array([t_data[1],  t_data[2], t_data[3], t_data[4]])
    tbar = np.array([tbar_data[1],  tbar_data[2], tbar_data[3], tbar_data[4]])
    W = np.array([W_data[1],  W_data[2], W_data[3], W_data[4]])
    Wbar = np.array([Wbar_data[1],  Wbar_data[2], Wbar_data[3], Wbar_data[4]])
    l = np.array([l_data[1],  l_data[2], l_data[3], l_data[4]])
    lbar = np.array([lbar_data[1],  lbar_data[2], lbar_data[3], lbar_data[4]])
    nu = np.array([nu_data[1],  nu_data[2], nu_data[3], nu_data[4]])
    nubar = np.array([nubar_data[1],  nubar_data[2], nubar_data[3], nubar_data[4]])
    b = np.array([b_data[1],  b_data[2], b_data[3], b_data[4]])
    bbar = np.array([bbar_data[1],  bbar_data[2], bbar_data[3], bbar_data[4]])

    j = b + bbar
    j1 = random.choice([b, bbar])
    j2 = j - j1

    
    ## Safety checks

    #lbar = np.array([152.93683997, -85.963040067, 65.473027328, 108.22807147])
    #l = np.array([207.12183734, 113.66221702, -89.014970026, -148.51495223])
    #b = np.array([85.478262991, -60.900573677, -8.924848204, 59.126226452])
    #bbar = np.array([218.51701648, 119.12451464, -157.7349203, -93.041073837])
    #nu = np.array([261.96683593, -93.60235235, 195.12731146, 147.61624261])
    #nubar = np.array([73.979207301, 7.6792344397, -4.9256002579, -73.41451446]) 
    #t = np.array([500.38193888, -240.46596609, 251.67549058, 314.97054053])
    #tbar = np.array([499.61806112, 240.46596609, -251.67549058, -314.97054053])
    #j = b + bbar
    #j1 = random.choice([b, bbar])
    #j2 = j - j1
    

            
    ## Asign values to vector components

    delta = 2*(j[3]*l[2]*lbar[1] - j[2]*l[3]*lbar[1] - j[3]*l[1]*lbar[2] + j[1]*l[3]*lbar[2] + j[2]*l[1]*lbar[3] - j[1]*l[2]*lbar[3])

    alpha[0] = (1/delta)*((2*j[3]*l[2] - 2*j[2]*l[3])*lbar[0] + l[0]*(-2*j[3]*lbar[2] + 2*j[2]*lbar[3]) + (-sqrt(s) + j[0])*(2*l[3]*lbar[2] - 2*l[2]*lbar[3]))
    alpha[1] = (1/delta)*((-2*j[3]*l[1] + 2*j[1]*l[3])*lbar[0] + l[0]*(2*j[3]*lbar[1] - 2*j[1]*lbar[3]) + (-sqrt(s) + j[0])*(-2*l[3]*lbar[1] + 2*l[1]*lbar[3]))
    alpha[2] = (1/delta)*((2*j[2]*l[1] - 2*j[1]*l[2])*lbar[0] + l[0]*(-2*j[2]*lbar[1] + 2*j[1]*lbar[2]) + (-sqrt(s) + j[0])*(2*l[2]*lbar[1] - 2*l[1]*lbar[2]))


    beta0[0] = (1/delta)*((1./2)*(2*(-s + 2*sqrt(s)*j[0] - j[0]**2 + j[1]**2 + j[2]**2 + j[3]**2 + 2*sqrt(s)*lbar[0] - 2*j[0]*lbar[0] + 2*j[1]*lbar[1] + 2*j[2]*lbar[2] + 2*j[3]*lbar[3])*((-l[3])*lbar[2] + l[2]*lbar[3]) - 2*((-j[3])*lbar[2] + j[2]*lbar[3])*(2*sqrt(s)*l[0] - 2*j[0]*l[0] - l[0]**2 + 2*j[1]*l[1] + l[1]**2 + 2*j[2]*l[2] + l[2]**2 + 2*j[3]*l[3] + l[3]**2 - 2*l[0]*lbar[0] + 2*l[1]*lbar[1] + 2*l[2]*lbar[2] + 2*l[3]*lbar[3]) + (2*j[3]*l[2] - 2*j[2]*l[3])*(lbar[0]**2 - lbar[1]**2 - lbar[2]**2 - lbar[3]**2)))
    beta0[1] = (1/delta)*((1./2)*(-2*(-s + 2*sqrt(s)*j[0] - j[0]**2 + j[1]**2 + j[2]**2 + j[3]**2 + 2*sqrt(s)*lbar[0] - 2*j[0]*lbar[0] + 2*j[1]*lbar[1] + 2*j[2]*lbar[2] + 2*j[3]*lbar[3])*((-l[3])*lbar[1] + l[1]*lbar[3]) + 2*((-j[3])*lbar[1] + j[1]*lbar[3])*(2*sqrt(s)*l[0] - 2*j[0]*l[0] - l[0]**2 + 2*j[1]*l[1] + l[1]**2 + 2*j[2]*l[2] + l[2]**2 + 2*j[3]*l[3] + l[3]**2 - 2*l[0]*lbar[0] + 2*l[1]*lbar[1] + 2*l[2]*lbar[2] + 2*l[3]*lbar[3]) + 2*((-j[3])*l[1] + j[1]*l[3])*(lbar[0]**2 - lbar[1]**2 - lbar[2]**2 - lbar[3]**2)))
    beta0[2] = (1/delta)*((1./2)*(2*((-l[2])*lbar[1] + l[1]*lbar[2])*(-s + 2*sqrt(s)*j[0] - j[0]**2 + j[1]**2 + j[2]**2 + j[3]**2 + 2*sqrt(s)*lbar[0] - 2*j[0]*lbar[0] + 2*j[1]*lbar[1] + 2*j[2]*lbar[2] + 2*j[3]*lbar[3]) - 2*((-j[2])*lbar[1] + j[1]*lbar[2])*(2*sqrt(s)*l[0] - 2*j[0]*l[0] - l[0]**2 + 2*j[1]*l[1] + l[1]**2 + 2*j[2]*l[2] + l[2]**2 + 2*j[3]*l[3] + l[3]**2 - 2*l[0]*lbar[0] + 2*l[1]*lbar[1] + 2*l[2]*lbar[2] + 2*l[3]*lbar[3]) + (2*j[2]*l[1] - 2*j[1]*l[2])*(lbar[0]**2 - lbar[1]**2 - lbar[2]**2 - lbar[3]**2)))


    beta1[0] = (1/delta)*((-j[3])*l[2] + j[2]*l[3] + l[3]*lbar[2] - l[2]*lbar[3])
    beta1[1] = (1/delta)*(j[3]*l[1] - j[1]*l[3] - l[3]*lbar[1] + l[1]*lbar[3])
    beta1[2] = (1/delta)*((-j[2])*l[1] + j[1]*l[2] + l[2]*lbar[1] - l[1]*lbar[2])


    beta2[0] = (1/delta)*((-j[3])*lbar[2] - l[3]*lbar[2] + (j[2] + l[2])*lbar[3])
    beta2[1] = (1/delta)*(j[3]*lbar[1] + l[3]*lbar[1] - (j[1] + l[1])*lbar[3])
    beta2[2] = (1/delta)*((-j[2])*lbar[1] - l[2]*lbar[1] + (j[1] + l[1])*lbar[2])


    # Define some useful variables to reduce computation time

    ab0 = np.dot(alpha, beta0)
    ab1 = np.dot(alpha, beta1)
    ab2 = np.dot(alpha, beta2)
    ainv = 1/(1 - np.dot(alpha, alpha))

    d00 = ab0**2 + np.dot(beta0, beta0)*(1 - np.dot(alpha, alpha))
    d10 = 2*ab0*ab1 + 2*np.dot(beta0, beta1)*(1 - np.dot(alpha, alpha))
    d01 = 2*ab0*ab2 + 2*np.dot(beta0, beta2)*(1 - np.dot(alpha, alpha))
    d11 = 2*ab1*ab2 + 2*np.dot(beta1, beta2)*(1 - np.dot(alpha, alpha))
    d20 = ab1**2 + np.dot(beta1, beta1)*(1 - np.dot(alpha, alpha))
    d02 = ab2**2 + np.dot(beta2, beta2)*(1 - np.dot(alpha, alpha))

    mW0 = mW
    mW1_0 = mW
    mW2_0 = mW


    ## Check if starting point in mW0 has DIS > 0

    if (DIS(mW0**2, mW0**2) < 0):
        trap_bad += 1    # counter to store bad points
        continue
    else:
        trap_good += 1   # counter to store good points


    ## Reconstruct t and tbar:   pW = pnu + plbar
    ##                           pWbar = p0 - pW - pj1 - pj2

    for sign in [1, -1]:
        for signb in [1, -1]:
            arg1 = mW1_0**2 + mb**2 + 2*lorentz(lbar + Pnu(mW1_0**2, mW2_0**2, sign), 1./2*(1 + signb)*j1 + 1./2*(1 - signb)*j2)
            arg2 = mW2_0**2 + mb**2 + 2*lorentz(p0 - lbar - j1 - j2 - Pnu(mW1_0**2, mW2_0**2, sign), 1./2*(1 - signb)*j1 + 1./2*(1 + signb)*j2)

            if ( (arg1 < 0) | (arg2 < 0) ):   # Forbid sqrt of negative number
                mt_rec = 0
                mtbar_rec = 0
                chi2_tops = 10**18
            else:
                mt_rec = sqrt(arg1)
                mtbar_rec = sqrt(arg2)
                chi2_tops = ((mt_rec - mt)**2 + (mtbar_rec - mt)**2)/w_t**2
                
            
            ##print("%3.3f, %3.3f, %.1d, %.1d, %3.3f" % (mt_rec, mtbar_rec, sign, signb, chi2_tops))
            
            branch[counter2][0] = mt_rec
            branch[counter2][1] = mtbar_rec
            branch[counter2][2] = sign
            branch[counter2][3] = signb
            branch[counter2][4] = chi2_tops

            counter2 += 1  
            
    counter2 = 0

    
    ## Pick up the matrix element with the min value for chi2_tops
    chi2_min = np.min(branch, axis=0)[4]     
    flex1 = np.where(branch==chi2_min)
    u1 = int(flex1[0])
    u2 = int(flex1[1])
    good_rec = branch[u1]

    
    ## Flag for close branches
    if ((np.abs(branch[0][4] / chi2_min) < 5) & (np.abs(branch[0][4] != chi2_min))):
        flag_branch += 1
    elif ((np.abs(branch[1][4] / chi2_min) < 5) & (np.abs(branch[1][4] != chi2_min))):
        flag_branch += 1
    elif ((np.abs(branch[2][4] / chi2_min) < 5) & (np.abs(branch[2][4] != chi2_min))):
        flag_branch += 1
    elif ((np.abs(branch[3][4] / chi2_min) < 5) & (np.abs(branch[3][4] != chi2_min))):
        flag_branch += 1

    
    ## Pick up the good reconstruction: update mt_rec, mtbar_rec, sign, signb, p
    mt_rec = branch[u1][0]
    mtbar_rec = branch[u1][1]
    sign = branch[u1][2]
    signb = branch[u1][3]
    pb = 1./2*(1 + signb)*j1 + 1./2*(1 - signb)*j2
    pbbar = 1./2*(1 - signb)*j1 + 1./2*(1 + signb)*j2

    ## Flag for bad b branch
    dif_b = pb[0]*b[0] - pb[1]*b[1] - pb[2]*b[2] - pb[3]*b[3]
    dif_bbar = pb[0]*bbar[0] - pb[1]*bbar[1] - pb[2]*bbar[2] - pb[3]*bbar[3]

    if ( dif_bbar < dif_b):
        flag_b += 1
    if ( (dif_bbar < 0) | (dif_b < 0)):
        flag_scalar += 1

    ## Calculate K4 value with truth-level information
    K4_truth = K4(M2(lbar+nu), M2(l+nubar), sign, signb)
    #print("K4 value with truth-level information: %.3f" % K4_truth)

    
    ## Define some useful variables to reduce computation time

    ct1 = mb**2 + 2*lorentz(lbar, pb)
    ct2 = mb**2 + 2*lorentz(p0 - j1 - j2 - lbar, pbbar)

    Eb = pb[0]
    Ebbar = pbbar[0]
    p3b = np.array([pb[1], pb[2], pb[3]])
    p3bbar = np.array([pbbar[1], pbbar[2], pbbar[3]])

    apb = np.dot(alpha, p3b)
    apbbar = np.dot(alpha, p3bbar)

    b0pb = np.dot(beta0, p3b)
    b1pb = np.dot(beta1, p3b)
    b2pb = np.dot(beta2, p3b)
    b0pbbar = np.dot(beta0, p3bbar)
    b1pbbar = np.dot(beta1, p3bbar)
    b2pbbar = np.dot(beta2, p3bbar)

    invGMW = (1./mW**2)*(1./w_W**2)
    invGMT = (1./mt**2)*(1./w_t**2)


    
    ## We don't minimize this time

    mW1 = mW1_0
    mW2 = mW2_0
    

    ## Reconstruct t and tbar masses:   pW = pnu + plbar
    ##                                  pWbar = p0 - pW - pj1 - pj2
    ## W- = W2 = W

    if (DIS(mW1**2, mW2**2) < -1):
        trap_good -= 1
        trap_bad -= 1
        bad += 1
        continue

    arg1 = mW1**2 + mb**2 + 2*lorentz(lbar + Pnu(mW1**2, mW2**2, sign), pb)
    arg2 = mW2**2 + mb**2 + 2*lorentz(p0 - lbar - j1 - j2 - Pnu(mW1**2, mW2**2, sign), pbbar)

    if ( (arg1 < 0) | (arg2 < 0) ):   # Forbid sqrt of negative number
        mt_rec = 0
        mtbar_rec = 0
        trap_good -= 1
        trap_bad -= 1
        bad += 1
        continue
    else:
        mt_rec = sqrt(arg1)
        mtbar_rec = sqrt(arg2)

    
    ## Reconstruct all momenta
    
    pnu = Pnu(mW1**2, mW2**2, sign)
    pnubar = p0 - pnu - lbar - pb - pbbar - l
    pWbar = pnu + lbar
    pW = p0 - pWbar - pb - pbbar
    pt = pWbar + pb
    ptbar = pW + pbbar

    # Check that neutrinos energy is > 0
    if ( (pnu[0] < 0) | (pnubar[0] < 0) ):
        trap_good -= 1
        trap_bad -= 1
        bad += 1
        flag_E += 1
        continue


    ## Reconstruct pseudo-rapidities

    eta_t = np.arctanh(pt[3]/(sqrt(pt[1]**2 + pt[2]**2 + pt[3]**2)))
    eta_tbar = np.arctanh(ptbar[3]/(sqrt(ptbar[1]**2 + ptbar[2]**2 + ptbar[3]**2)))
    eta_W = np.arctanh(W[3]/(sqrt(W[1]**2 + W[2]**2 + W[3]**2)))
    eta_Wbar = np.arctanh(Wbar[3]/(sqrt(Wbar[1]**2 + Wbar[2]**2 + Wbar[3]**2)))
    eta_nu = np.arctanh(nu[3]/(sqrt(nu[1]**2 + nu[2]**2 + nu[3]**2)))
    eta_nubar = np.arctanh(nubar[3]/(sqrt(nubar[1]**2 + nubar[2]**2 + nubar[3]**2)))


    ## Save all the relevant information about the reconstruction of the event
        
    t_data_rec = np.array([t_data[0], pt[0], pt[1], pt[2], pt[3], mt_rec, eta_t])
    tbar_data_rec = np.array([tbar_data[0], ptbar[0], ptbar[1], ptbar[2], ptbar[3], mtbar_rec, eta_tbar])
    W_rec = np.array([W_data[0], pW[0],  pW[1], pW[2], pW[3], mW2, eta_W])
    Wbar_rec = np.array([Wbar_data[0], pWbar[0],  pWbar[1], pWbar[2], pWbar[3], mW1, eta_Wbar])
    nu_data_rec = np.array([nu_data[0], pnu[0],  pnu[1], pnu[2], pnu[3], 0, eta_nu])
    nubar_data_rec = np.array([nubar_data[0], pnubar[0],  pnubar[1], pnubar[2], pnubar[3], 0, eta_nubar])
    l_data_rec = l_data
    lbar_data_rec = lbar_data
    b_data_rec = b_data
    bbar_data_rec = bbar_data

    event_rec = np.array([t_data_rec, tbar_data_rec, W_rec, Wbar_rec, b_data_rec, bbar_data_rec, l_data_rec, lbar_data_rec, nu_data_rec, nubar_data_rec])
    
    event_truth = np.array([t_data, tbar_data, W_data, Wbar_data, b_data, bbar_data, l_data, lbar_data, nu_data, nubar_data])


    ## Save all the information in a file

    reconstructed[trap-1] = event_rec
    truth[trap-1] = event_truth

    rec_good[trap_good-1,:,:] = event_rec
    truth_good[trap_good-1] = event_truth
    K4_good[trap_good-1] = K4_truth
    rec_good.resize((trap_good+1, 10, 7))
    truth_good.resize((trap_good+1, 10, 7))
    K4_good.resize((trap_good+1, 1))
    

    ## Instructions to read:
    #  f = h5py.File('results.hdf5', 'r')   # We first read the file
    #  data_set = f['reconstructed']      # Read dataset
    #  data_set = f['reconstructed'][:]   # Read dataset and store in numpy array
    #  data = data_set[:]  # Store elements of dataset in numpy array.
    #  f.close()



    #print("mW1_truth = %.4f, mW2_truth = %.4f" % (sqrt(M2(Wbar)), sqrt(M2(W))))


    #if ( (abs(sqrt(M2(Wbar)) - mW1) > 10) | (abs(sqrt(M2(W)) - mW2) > 10)):
        #print("mW1_rec = %.4f, mW2_rec = %.4f" % (mW1, mW2))
        #print("mW1_truth = %.4f, mW2_truth = %.4f" % (sqrt(M2(Wbar)), sqrt(M2(W))))
        #break
    

        
    ## Print flags and end the code
    
    if trap == n_evs:
        print("Bad points: %.f" % bad)
        print("Bad branch points: %.f" % flag_branch)
        print("Bad b branches: %.f" % flag_b)
        print("Number of points with DIS < 0: %.f" % trap_bad)
        print("Number of points with scalar product < 0: %.f" % flag_scalar)
        print("Number of points with Enu < 0: %.f" % flag_E)
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))
        f.close()
        break


