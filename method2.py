# We use 10 identical versions of this code for speed/memory purposes

from math import sqrt
import numpy as np
import random
import time
from scipy import optimize as opt
import matplotlib.pyplot as plt
import h5py

start_time = time.time()

# Read input data from lhe file using the lhereader interface. See https://github.com/diptaparna/lhereader
execfile("lhereader.py")

data=readLHEF('big_run_data/unweighted_events_100000_0.lhe')     # instance


## Define physical parameters
s = 10**6
mb = 4.7
mt = 173
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
flag = 0
flag_branch = 0
flag_b = 0
flag_K4 = 0
flag_next = 0
flag_corrected = 0

## Define number of entries in arrays
n_evs = len(data.events)
n_parts = 10


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
K4s_min = np.zeros(2, dtype=float)

## Arrays for gradient descent
m = np.zeros(2, dtype=float)
v = np.zeros(2, dtype=float)


## Open the file to save the data
f = h5py.File('big_run_global_results/results_100000_global_next_1.hdf5', 'w')
reconstructed = f.create_dataset('reconstructed', (n_evs, n_parts, 7), dtype='f4')
truth = f.create_dataset('truth', (n_evs, n_parts, 7), dtype='f4')
rec_good = f.create_dataset('rec_good', (1, n_parts, 7), maxshape=(n_evs, n_parts, 7), dtype='f4')
rec_bad = f.create_dataset('rec_bad', (1, n_parts, 7), maxshape=(n_evs, n_parts, 7), dtype='f4')
truth_good = f.create_dataset('truth_good', (1, n_parts, 7), maxshape=(n_evs, n_parts, 7), dtype='f4')
truth_bad = f.create_dataset('truth_bad', (1, n_parts, 7), maxshape=(n_evs, n_parts, 7), dtype='f4')
K4_good = f.create_dataset('K4_good', (1, 2), maxshape=(n_evs, 2), dtype='f4')
K4_bad = f.create_dataset('K4_bad', (1, 2), maxshape=(n_evs, 2), dtype='f4')


## List of functions that we will use, imported from functions.py

# lorentz(v1, v2)  | Input: two 4-vectors | Output: a scalar
# M2(v1)  | Input: one 4-vector | Output: a scalar
# beta(x, y)  | Input: two squared masses | Output: a 3-component vector
# alphabeta(x, y)  | Input: two squared masses | Output: a scalar
# DIS(x, y)  | Input: two squared masses | Output: a scalar
# DIS_var(x)  | Input: array of two squared masses | Output: a scalar
# DIS_jac(x, y)  | Input: array of two squared masses | Output: jacobian
# Pnu(x, y, sign)  | Input: two squared masses and a discriminant sign | Output: a 4-vector
# K4(x, y, sign, signb)  | Input: two squared masses, sign and sign of b's | Output: a scalar
# F4(x)  | Input: array of two squared masses | Output: a scalar



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

    else:
        print(DIS(x, y))
    
    pnu = Enu*alpha + beta(x, y)

    vector = np.concatenate([Enu, pnu], axis=0)

    return vector


def K4(x, y, sign, signb):   ## Input: two squared masses, sign and sign of b's | Output: a scalar

    # Computes the weight function to minimize

    if (DIS(x, y) > -1):
        # Compute numerator of K4 with the tops
        weight = 10**15
        num_tops = (lorentz(pb, Pnu(x, y, sign)) * lorentz(lbar + Pnu(x, y, sign) + pb, lbar) * lorentz(pbbar, p0 - lbar - pb - pbbar - Pnu(x, y, sign) - l) * lorentz(p0 - lbar - pb - Pnu(x, y, sign), l)) / weight
        
        total = (((1./mW**2)*(1./w_W**2)*(x - mW**2)**2 + 1) * ((1./mW**2)*(1./w_W**2)*(y - mW**2)**2 + 1) * ((1./mt**2)*(1./w_t**2)*(x + mb**2 + 2*lorentz(lbar + Pnu(x, y, sign), 1./2*(1 + signb)*j1 + 1./2*(1 - signb)*j2) - mt**2)**2 + 1) * ((1./mt**2)*(1./w_t**2)*(y + mb**2 + 2*lorentz(p0 - lbar - j1 - j2 - Pnu(x, y, sign), 1./2*(1 - signb)*j1 + 1./2*(1 + signb)*j2) - mt**2)**2 + 1)) / num_tops

    else:
        total = 10**18

    return total


def F4(variables):   ## Input: two squared masses | Output: a scalar

    # K4 defined in a more efficient way

    x, y = variables

    if (DIS(x, y) < 0):
        total = 10**18
    else:
        # Compute numerator of K4 with the tops
        weight = 10**15
        num_tops = (lorentz(pb, Pnu(x, y, sign)) * lorentz(lbar + Pnu(x, y, sign) + pb, lbar) * lorentz(pbbar, p0 - lbar - pb - pbbar - Pnu(x, y, sign) - l) * lorentz(p0 - lbar - pb - Pnu(x, y, sign), l)) / weight
        
        total = ((1 + invGMW*(mW**2 - x)**2)*(1 + invGMW*(mW**2 - y)**2)*(1 + invGMT*(2*b0pb - ct1 + 2*ab0*ainv*(apb - Eb) + mt**2 - x + 2*ab1*ainv*apb*x + 2*b1pb*x - 2*ab1*ainv*Eb*x + 2*ab2*ainv*apb*y + 2*b2pb*y - 2*ab2*ainv*Eb*y)**2)*(1 + invGMT*(2*b0pbbar + ct2 + 2*ab0*ainv*(apbbar - Ebbar) - mt**2 + 2*ab1*ainv*apbbar*x + 2*b1pbbar*x - 2*ab1*ainv*Ebbar*x + y + 2*ab2*ainv*apbbar*y + 2*b2pbbar*y - 2*ab2*ainv*Ebbar*y)**2) + 4*ainv**2*invGMT*(1 + invGMW*(mW**2 - x)**2)*(1 + invGMW*(mW**2 - y)**2)*(4*(apb - Eb)*(apbbar - Ebbar)*invGMT*(2*b0pb - ct1 + 2*ab0*ainv*(apb - Eb) + mt**2 - x + 2*ab1*ainv*apb*x + 2*b1pb*x - 2*ab1*ainv*Eb*x + 2*ab2*ainv*apb*y + 2*b2pb*y - 2*ab2*ainv*Eb*y)*(2*b0pbbar + ct2 + 2*ab0*ainv*(apbbar - Ebbar) - mt**2 + 2*ab1*ainv*apbbar*x + 2*b1pbbar*x - 2*ab1*ainv*Ebbar*x + y + 2*ab2*ainv*apbbar*y + 2*b2pbbar*y - 2*ab2*ainv*Ebbar*y) + (apbbar - Ebbar)**2*(1 + invGMT*(2*b0pb - ct1 + 2*ab0*ainv*(apb - Eb) + mt**2 - x + 2*ab1*ainv*apb*x + 2*b1pb*x - 2*ab1*ainv*Eb*x + 2*ab2*ainv*apb*y + 2*b2pb*y - 2*ab2*ainv*Eb*y)**2) + (apb - Eb)**2*(1 + invGMT*(2*b0pbbar + ct2 + 2*ab0*ainv*(apbbar - Ebbar) - mt**2 + 2*ab1*ainv*apbbar*x + 2*b1pbbar*x - 2*ab1*ainv*Ebbar*x + y + 2*ab2*ainv*apbbar*y + 2*b2pbbar*y - 2*ab2*ainv*Ebbar*y)**2))*DIS(x, y) + 16*ainv**4*(apb - Eb)**2*(apbbar - Ebbar)**2*invGMT**2*(1 + invGMW*(mW**2 - x)**2)*(1 + invGMW*(mW**2 - y)**2)*DIS(x, y)**2 + sign*(4*ainv*invGMT*(1 + invGMW*(mW**2 - x)**2)*(1 + invGMW*(mW**2 - y)**2)*((apbbar - Ebbar)*(2*b0pbbar + ct2 + 2*ab0*ainv*(apbbar - Ebbar) - mt**2 + 2*ab1*ainv*apbbar*x + 2*b1pbbar*x - 2*ab1*ainv*Ebbar*x + y + 2*ab2*ainv*apbbar*y + 2*b2pbbar*y - 2*ab2*ainv*Ebbar*y)*(1 + invGMT*(2*b0pb - ct1 + 2*ab0*ainv*(apb - Eb) + mt**2 - x + 2*ab1*ainv*apb*x + 2*b1pb*x - 2*ab1*ainv*Eb*x + 2*ab2*ainv*apb*y + 2*b2pb*y - 2*ab2*ainv*Eb*y)**2) + (apb - Eb)*(2*b0pb - ct1 + 2*ab0*ainv*(apb - Eb) + mt**2 - x + 2*ab1*ainv*apb*x + 2*b1pb*x - 2*ab1*ainv*Eb*x + 2*ab2*ainv*apb*y + 2*b2pb*y - 2*ab2*ainv*Eb*y)*(1 + invGMT*(2*b0pbbar + ct2 + 2*ab0*ainv*(apbbar - Ebbar) - mt**2 + 2*ab1*ainv*apbbar*x + 2*b1pbbar*x - 2*ab1*ainv*Ebbar*x + y + 2*ab2*ainv*apbbar*y + 2*b2pbbar*y - 2*ab2*ainv*Ebbar*y)**2)) + 16*ainv**3*(apb - Eb)*(apbbar - Ebbar)*invGMT**2*(1 + invGMW*(mW**2 - x)**2)*(1 + invGMW*(mW**2 - y)**2)*(2*apb*b0pbbar + apb*ct2 - 2*b0pbbar*Eb - ct2*Eb + 4*ab0*ainv*(apb - Eb)*(apbbar - Ebbar) - 2*b0pb*Ebbar + ct1*Ebbar - apb*mt**2 + Eb*mt**2 - Ebbar*mt**2 + 2*apb*b1pbbar*x - 2*b1pbbar*Eb*x + Ebbar*x - 4*ab1*ainv*apb*Ebbar*x - 2*b1pb*Ebbar*x + 4*ab1*ainv*Eb*Ebbar*x + apb*y + 2*apb*b2pbbar*y - Eb*y - 2*b2pbbar*Eb*y - 4*ab2*ainv*apb*Ebbar*y - 2*b2pb*Ebbar*y + 4*ab2*ainv*Eb*Ebbar*y + apbbar*(2*b0pb - ct1 + mt**2 - x + 4*ab1*ainv*apb*x + 2*b1pb*x - 4*ab1*ainv*Eb*x + 4*ab2*ainv*apb*y + 2*b2pb*y - 4*ab2*ainv*Eb*y))*DIS(x, y))*sqrt(DIS(x, y))) / num_tops

    return total



################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################



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

    
    ## Safety check: see what happens with this real event

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
    

            
    ## Assign values to vector components

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


    ## Pick starting point in mW0 to have DIS > 0

    if (DIS(mW0**2, mW0**2) < 0):

        flag = 1         # flag to save these points separately
        trap_bad += 1    # counter to store bad points

        step = 2    ## Convergence might be very sensible to the step value!

        x0 = mW0**2
        y0 = mW0**2

        eps = 1e-8   # Adam learning update parameters
        par1 = 0.9
        par2 = 0.999

        counter = 0   # Count iterations in gradient descent

        positive = np.array([x0, y0])

        # Gradient descent
        for i in xrange(1, 10000):
            gradient = DIS_jac(positive)

            param_scale = np.linalg.norm(positive)
            m = par1*m + (1 - par1)*gradient     # We use Adam learning update
            v = par2*v + (1 - par2)*(gradient**2)
            update = step * m/(np.sqrt(v) + eps)
            update_scale = np.linalg.norm(update)
            positive += update   # the actual update
            
            #print(update_scale / param_scale)   # want ~1e-3

            if (update_scale / param_scale < 1e-3):
                step = step*2

            #print("step = %.1f" % step)

            counter += 1

            
            # If discriminant becomes positive, stop
            if ( (DIS_var(positive) > 0) & (positive[0] > 0) & (positive[1] > 0)):
                #print("---------------------------------------------")
                #print("Final (x, y) = (%.4f, %.4f)" % (sqrt(positive[0]), sqrt(positive[1])))
                #print("gradient = (%.4f, %.4f)" % (gradient[0], gradient[1]))
                #print("DIS(x, y) = %.4f" % DIS_var(positive))
                #print("Iterations: %d" % counter)
                #print("---------------------------------------------")
                mW1_0 = sqrt(positive[0])
                mW2_0 = sqrt(positive[1])
                           
                break

            #print(counter)
            #print("gradient = (%.4f, %.4f)" % (gradient[0], gradient[1]))
            #print("DIS(x, y) = %.4f" % DIS_var(positive))
            #print("---")

    else:
        flag = 0          # flag to save good points separately
        trap_good += 1    # counter to store good points


    
    ## Reconstruct t and tbar:   pW = pnu + plbar
    ##                           pWbar = p0 - pW - pj1 - pj2

    for sign in [1, -1]:
        for signb in [1, -1]:
            arg1 = mW1_0**2 + mb**2 + 2*lorentz(lbar + Pnu(mW1_0**2, mW2_0**2, sign), 1./2*(1 + signb)*j1 + 1./2*(1 - signb)*j2)
            arg2 = mW2_0**2 + mb**2 + 2*lorentz(p0 - lbar - j1 - j2 - Pnu(mW1_0**2, mW2_0**2, sign), 1./2*(1 - signb)*j1 + 1./2*(1 + signb)*j2)

            if ( (arg1 < 0) | (arg2 < 0) ):   # Forbid sqrt of negative number
                mt_rec = 0
                mtbar_rec = 0
                den_tops = 10**20
            else:
                mt_rec = sqrt(arg1)
                mtbar_rec = sqrt(arg2)

                # Compute numerator of K4
                weight = 10**15
                num_tops = (lorentz(1./2*(1 + signb)*j1 + 1./2*(1 - signb)*j2, Pnu(mW1_0**2, mW2_0**2, sign)) * lorentz(lbar + Pnu(mW1_0**2, mW2_0**2, sign) + 1./2*(1 + signb)*j1 + 1./2*(1 - signb)*j2, lbar) * lorentz(1./2*(1 - signb)*j1 + 1./2*(1 + signb)*j2, p0 - lbar - j1 - j2 - Pnu(mW1_0**2, mW2_0**2, sign) - l) * lorentz(p0 - lbar - j1 - j2 - Pnu(mW1_0**2, mW2_0**2, sign) + 1./2*(1 - signb)*j1 + 1./2*(1 + signb)*j2, l)) / weight

                # For good points only tops in den, for bad points also Ws
                if ( flag == 0 ):
                    den_tops = (1./mt**4)*(1./w_t**4)*((mt_rec**2 - mt**2)**2 + mt**2 * w_t**2)*((mtbar_rec**2 - mt**2)**2 + mt**2 * w_t**2)
                else:
                    den_tops = (1./mt**4)*(1./w_t**4)*((mt_rec**2 - mt**2)**2 + mt**2 * w_t**2)*((mtbar_rec**2 - mt**2)**2 + mt**2 * w_t**2) * (1./mW**4)*(1./w_W**4)*((mW1_0**2 - mW**2)**2 + mW**2 * w_W**2)*((mW2_0**2 - mW**2)**2 + mW**2 * w_W**2)
                
            
            ##print("%3.3f, %3.3f, %.1d, %.1d, %3.3f" % (mt_rec, mtbar_rec, sign, signb, den_tops))
            
            branch[counter2][0] = mt_rec
            branch[counter2][1] = mtbar_rec
            branch[counter2][2] = sign
            branch[counter2][3] = signb
            branch[counter2][4] = den_tops / num_tops

            counter2 += 1  
            
    counter2 = 0

    
    ## Pick up the matrix element with the min value for den_tops
    den_min = np.min(branch, axis=0)[4]     
    flex1 = np.where(branch==den_min)
    u1 = int(flex1[0])
    u2 = int(flex1[1])
    good_rec = branch[u1]
    keep_good = good_rec

    ## Pick up the matrix element with the second min value for den_tops
    den_nextmin = np.partition(branch[:,4], 1)[1]
    flex2 = np.where(branch==den_nextmin)
    u3 = int(flex2[0])
    u4 = int(flex2[1])
    good_nextrec = branch[u3]

    ## Put both together
    recs = np.array([good_rec, good_nextrec])
    

    ## Flag for close branches
    if ( np.abs(den_nextmin / den_min) < 5 ):
        flag_next = 1
        flag_branch += 1


    ## Reconstruct the event for the min and next to min branches

    if (flag_next == 1):
    
        for i in xrange(2):
        
            ## Pick up the good reconstruction: update mt_rec, mtbar_rec, sign, signb, p
            mt_rec = recs[i][0]
            mtbar_rec = recs[i][1]
            sign = recs[i][2]
            signb = recs[i][3]
            pb = 1./2*(1 + signb)*j1 + 1./2*(1 - signb)*j2
            pbbar = 1./2*(1 - signb)*j1 + 1./2*(1 + signb)*j2     
                
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
            
            
            ## Minimize F4 with Differential-Evolution method (finds global minima!)
            
            bounds = [((mW0 - 15*w_W)**2, (mW0 + 15*w_W)**2), ((mW0 - 15*w_W)**2, (mW0 + 15*w_W)**2)]
            result = opt.differential_evolution(F4, bounds, tol = 1e-3)
            if result.success:
                mW1 = np.sqrt(result.x[0])
                mW2 = np.sqrt(result.x[1])
                K4s_min[i] = K4(mW1**2, mW2**2, sign, signb)
                if ( (K4s_min[0] == 10**20) & (K4s_min[1] == 10**20)):
                    bad += 1
                    trap_good -= 1
                    trap_bad -= 1
                    continue
            else:
                K4s_min[i] = 10**10
                continue
            
        valid = np.min(K4s_min)
        valid_index = np.where(K4s_min == valid)

        # Keep the number of events with the min value for the 2nd branch
        if ( K4s_min[1] < K4s_min[0] ):
            flag_corrected += 1

        # Check if both values are equal
        if ( len(valid_index[0]) == 2 ):
            u1 = int(valid_index[0][0])
        else:
            u1 = int(valid_index[0])

        keep_good = recs[u1]
            
                
    # Reset flag value
    flag_next = 0

    ## Pick up the good reconstruction: update mt_rec, mtbar_rec, sign, signb, p
    mt_rec = keep_good[0]
    mtbar_rec = keep_good[1]
    sign = keep_good[2]
    signb = keep_good[3]
    pb = 1./2*(1 + signb)*j1 + 1./2*(1 - signb)*j2
    pbbar = 1./2*(1 - signb)*j1 + 1./2*(1 + signb)*j2

    ## Flag for bad b branch
    dif_b = pb[0]*b[0] - pb[1]*b[1] - pb[2]*b[2] - pb[3]*b[3]
    dif_bbar = pb[0]*bbar[0] - pb[1]*bbar[1] - pb[2]*bbar[2] - pb[3]*bbar[3]
    
    if ( dif_bbar < dif_b):
        flag_b += 1
        
    ## Calculate K4 value with truth-level information
    K4_truth = K4(M2(lbar+nu), M2(l+nubar), sign, signb)

    
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


    ## Minimize F4 with Differential-Evolution method (finds global minimum!)

    bounds = [((mW0 - 15*w_W)**2, (mW0 + 15*w_W)**2), ((mW0 - 15*w_W)**2, (mW0 + 15*w_W)**2)]
    result = opt.differential_evolution(F4, bounds, tol = 1e-3)
    if result.success:
        mW1 = np.sqrt(result.x[0])
        mW2 = np.sqrt(result.x[1])
        K4_min = K4(mW1**2, mW2**2, sign, signb)
    else:
        K4_min = 10**20
        bad += 1
        trap_good -= 1
        trap_bad -= 1
        continue

    
    ## Warning if it is not the true minimum
    if (5*K4_truth < K4_min):
        flag_K4 += 1
      

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
    
    if (flag == 0):
        rec_good[trap_good-1,:,:] = event_rec
        truth_good[trap_good-1] = event_truth
        K4_good[trap_good-1] = np.array([K4_min, K4_truth])
        rec_good.resize((trap_good+1, 10, 7))
        truth_good.resize((trap_good+1, 10, 7))
        K4_good.resize((trap_good+1, 2))
    else:
        rec_bad[trap_bad-1] = event_rec
        truth_bad[trap_bad-1] = event_truth
        K4_bad[trap_bad-1] = np.array([K4_min, K4_truth])
        rec_bad.resize((trap_bad+1, 10, 7))
        truth_bad.resize((trap_bad+1, 10, 7))
        K4_bad.resize((trap_bad+1, 2))
    

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
        print("Smaller minimum for 2nd branch: %.f" % flag_corrected)
        print("Bad b branches: %.f" % flag_b)
        print("Bad K4 minimum: %.f" % flag_K4)
        print("--- Execution time: %s seconds ---" % (time.time() - start_time))
        f.close()
        break


