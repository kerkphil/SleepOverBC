# -*- coding: utf-8 -*-
"""
This program is used to solve the Economics of Sleep model with utility from 
consumption and leisure over the business cycle.  It uses a linearization
approach.

This is model 1, which has a single representative agent that allocates time 
over work, play and sleep.

The program first solves for parameters that meet specified steady state 
targets (currently choosing chil and chiS to match Nbar and Sbar).

It then solves for all the steady state values.

Linear approximations for the policy and jump functions are solved for using
the LinApp toolkit

A long simulation is run and a subsample of the time-series variables is 
plotted.

Standard deviations, correlations with GDP, and autocorrelations are 
calculated and reported.

Impulse reposnse functions are calculated and plotted.

The model is fit to data over 2003Q1 - 2017Q1.  The shocks that exactly
replicate the observed hours worked series are calculated.  These are then used
to generate a prediction for GDP which is compared with the GDP data.
Predictions for hours slept are also calculated and plotted against ATUS data.
Predictions for all other endogenous variables are also calcuated.  


Code written by Kerk l. Phillips
Oct. 25, 2017

Updated on Nov. 16, 2017
"""

modname = 'BCEconSleepLin1'

import numpy as np
import scipy.optimize as opt
import pickle as pkl
from LinApp_FindSS import LinApp_FindSS
from LinApp_Deriv import LinApp_Deriv
from LinApp_Solve import LinApp_Solve
from LinApp_SSL import LinApp_SSL
from DSGEmoments import calcmom
import pandas as pd

def mdefs(Kp, K, N, S, z, *params):
    '''
    This function calculates definitions given the state and jump
    variables along with the model parameters.
    
    Inputs:
        Kp - capital stock next period
        K - capital stock this period
        N - hours worked this period
        S - hours slept this period
        z - the productivity shock today
        params - list of model parameters
            
        
    Outputs:
        L - leisure
        d - waking pressure f
        b - effectiveness of workers due to sleep
        Y - total output
        w - hourly wage
        r - capital rental rate
        C - household consumption
        I - investment
        uC - consumption utility
        uL - leisure utility
        uS - sleep utility
        u - utility
    
    '''    
    L = 24 - N - S
    d = 2*np.arctan(phi*S+xi)/np.pi
    b = kappa*S**eta
    Y = A*np.exp(z)*K**alpha*(b*N)**(1-alpha)
    w = (1-alpha)*Y/(b*N)
    r = alpha*Y/K
    C = w*b*N + (1+r-delta)*K - Kp
    I = Y - C
    if gamma == 1:
        uC = np.log(C)
    else:
        uC = (C**(1-gamma)-1)/(1-gamma)
    if lambd == 1:
        uL = chiL*np.log(L)
    else:
        uL = chiL*(L**(1-lambd)-1)/(1-lambd)
    uS = chiS*( (48/np.pi)*(np.sin(S*np.pi/24)) + d*(24 - 2*S) )
    u = uC + uL + uS
    
    return L, d, b, Y, w, r, C, I, uC, uL, uS, u


def mdyn(theta, *params):
    '''
    This function calculates Euler deviations given the state and jump
    variables along with the model parameters
    
    Inputs:
        invec - 3*nx+2*ny+2*nz element numpy array of state and jump variables
        params - list of model parameters
            
        
    Outputs:
        E - nx+ny element numpy array of deviations of Euler and other key
            dynamic equations
    '''
    
    # unpack theta
    [Kpp, Kp, K, Np, Sp, N, S, zp, z] = theta
    
    L, d, b, Y, w, r, C, I, uC, uL, uS, u =  \
        mdefs(Kp, K, N, S, z, *params)
    Lp, dp, bp, Yp, wp, rp, Cp, Ip, uCp, uLp, uSp, up = \
        mdefs(Kpp, Kp, Np, Sp, zp, *params)
    
    e1 = C**(-gamma)*b*w - chiL*L**(-lambd)
    e2 = chiL*L**(-lambd) - chiS*( 2*np.cos(S*np.pi/24) - 2*d ) \
         - C**(-gamma)*w*N*eta*S**(eta-1) \
         - chiS*(2*phi*(24-2*S))/(np.pi*(1+S**2))
    e3 = C**(-gamma) - beta*Cp**(-gamma)*(1+rp-delta)
    
    return np.array([e1, e2, e3])


def mdefs_c(Kp, K, chiL, chiS, z, *cparams):
    '''
    This function calculates definitions given the state and jump
    variables along with the model parameters.
    
    Parallel to mdefs, but takes chiL and chiS as variables and treats 
    Nbar and Sbar as parameters.
    '''
    
    L = 24 - Nbar - Sbar
    d = 2*np.arctan(phi*Sbar+xi)/np.pi
    b = kappa*Sbar**eta
    Y = A*np.exp(z)*K**alpha*(b*Nbar)**(1-alpha)
    w = (1-alpha)*Y/(b*Nbar)
    r = alpha*Y/K
    C = w*b*Nbar + (1+r-delta)*K - Kp
    I = Y - C
    if gamma == 1:
        uC = np.log(C)
    else:
        uC = (C**(1-gamma)-1)/(1-gamma)
    if lambd == 1:
        uL = chiL*np.log(L)
    else:
        uL = chiL*(L**(1-lambd)-1)/(1-lambd)
    uS = chiS*( (48/np.pi)*(np.sin(Sbar*np.pi/24)) + d*(24 - 2*Sbar) )
    u = uC + uL + uS
    
    return L, d, b, Y, w, r, C, I, uC, uL, uS, u


def mSS_c(theta, *cparams):
    '''
    This function calculates SS Euler deviations given the state and jump
    variables along with model parameters and SS targets  It is used when
    calibrating the model.
    
    Parallel to mdyn, but only for the steady state.  Takes chiL and chiS as 
    variables and treats Nbar and Sbar as parameters.
    '''
    
    # unpack theta
    [Kbar, chiL, chiS] = theta
    # set zbar
    zbar = 0.
    
    L, d, b, Y, w, r, C, I, uC, uL, uS, u =  \
        mdefs_c(Kbar, Kbar, chiL, chiS, zbar, *cparams)
    
    e1 = C**(-gamma)*b*w - chiL*L**(-lambd)
    e2 = chiL*L**(-lambd) - chiS*( 2*np.cos(Sbar*np.pi/24) - 2*d ) \
         - C**(-gamma)*w*Nbar*eta*Sbar**(eta-1) \
         - chiS*(2*phi*(24-2*Sbar))/(np.pi*(1+Sbar**2))
    e3 = C**(-gamma) - beta*C**(-gamma)*(1+r-delta)
    
    return np.array([e1, e2, e3])


# -----------------------------------------------------------------------------
# CALIBRATION

# declare model parameters
phi = 1.
xi = 0.
kappa = 1.
eta = .92  # sleep volatility and cyclicality no longer responsive to this
alpha = .33
delta = .01
beta = .99
gamma = 1.5
lambd = 1.0
Nbar = 3.16+3.78/2
Sbar = 8.81 #8.97
rho_z = .87
sigma_z = .0044
A = 1.

# put model parameters into a list
cparams = (phi, xi, kappa, eta, alpha, delta, beta, gamma, lambd, \
           Nbar, Sbar, rho_z, sigma_z, A)

# find calibrated SS values
# put guesses into np vector
Kbar = 1000.
chiL = 1.4
chiS = -.05
guess = np.array([Kbar, chiL, chiS])

# set up lambda function
SSfunc = lambda theta: mSS_c(theta, *cparams)
cbar = opt.fsolve(SSfunc, guess)

# unpack cbar
[Kbar, chiL, chiS] = cbar

# check
check = mSS_c(cbar, *cparams)
print('calibration check: ', check)
print('Kbar: ', Kbar)
print('chiL: ', chiL)
print('chiS: ', chiS)
print(' ')

# -----------------------------------------------------------------------------
# STEADY STATE

zbar = 0.
Zbar = np.array([zbar])
nx = 1
ny = 2
nz = 1
guess = np.array([Kbar, Nbar, Sbar])

params = (phi, xi, kappa, eta, alpha, delta, beta, gamma, lambd, \
          chiL, chiS, rho_z, sigma_z, A)

bar = LinApp_FindSS(mdyn, params, guess, Zbar, nx, ny)
# unpack bar
[Kbar, Nbar, Sbar] = bar
# check
thetabar = np.array([Kbar, Kbar, Kbar, Nbar, Sbar, Nbar, Sbar, zbar, zbar])
check = mdyn(thetabar, *params)
print('check', np.max(check))

# find remaining SS values
Lbar, dbar, bbar, Ybar, wbar, rbar, Cbar, Ibar, uCbar, uLbar, uSbar, ubar \
    = mdefs(Kbar, Kbar, Nbar, Sbar, zbar, *params)
    
# create bar list
bars = (Kbar, Nbar, Sbar, Lbar, dbar, bbar, Ybar, wbar, rbar, Cbar, Ibar, 
        uCbar, uLbar, uSbar, ubar )

# print SS values
print('Steady State Values')
print('Kbar: ', Kbar)
print('Nbar: ', Nbar)
print('Sbar: ', Sbar)
print('Lbar: ', Lbar)
print('dbar: ', dbar)
print('bbar: ', bbar)
print('Ybar: ', Ybar)
print('wbar: ', wbar)
print('rbar: ', rbar)
print('Cbar: ', Cbar)
print('Ibar: ', Ibar)
print('uCbar:', uCbar)
print('uLbar:', uLbar)
print('uSbar:', uSbar)
print('ubar: ', ubar)
print(' ')

# -----------------------------------------------------------------------------
# LINEARIZATION

# set autocorrelation matrix
NN = np.array([rho_z])

[AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT] = \
    LinApp_Deriv(mdyn, params, thetabar, nx, ny, nz, False)
             
PP, QQ, UU, RR, SS, VV = \
    LinApp_Solve(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT, NN, \
                 Zbar, False)   

LinCoeffs = (PP, QQ, UU, RR, SS, VV)
        
# save coeffs in pickle file
output = open(modname + '_Coeffs.pkl', 'wb')
pkl.dump(LinCoeffs, output)
output.close()
        


# -----------------------------------------------------------------------------
# RUN SIMULATION

# write a simulation function
def runsim(Zhist, nobs, params, LinCoeffs, bars):
    '''
    This function simulates the model once.
    
    Inputs:
        Zhist - a numpy array with the history of Z
        nobs - the number of observations to simulate
        params - list of model parameters
        LinCoeffs - list of linear coefficients for the policy and jump
            functions (PP, QQ, RR, SS)
        bars - list of steady state values
        
    Outputs:
        Khist - history of capital stock
        Nhist - history of hours worked
        Shist - history of sleep hours 
        Lhist - history of leisure hours
        dhist - history of waking pressure
        bhist - history of effectiveness due to sleep
        Yhist - history of output
        whist - history of hourly wage
        vhist - history of per worker fixed payments
        rhist - history of rental rate
        Chist - history of consumption
        Ihist - history of investment
        uChist - history of conumption utility
        uLhist - history of lesiure utility
        uShist - history of sleep utility
        uhist - history of total utility per period
    '''
    
    # create a history using polynomial coefficients.
    Khist = np.zeros(nobs+1)
    Nhist = np.zeros(nobs)
    Shist = np.zeros(nobs)
    Lhist = np.zeros(nobs)
    dhist = np.zeros(nobs)
    bhist = np.zeros(nobs)
    Yhist = np.zeros(nobs)
    whist = np.zeros(nobs)
    rhist = np.zeros(nobs)
    Chist = np.zeros(nobs)
    Ihist = np.zeros(nobs)
    uChist = np.zeros(nobs)
    uLhist = np.zeros(nobs)
    uShist = np.zeros(nobs)
    uhist = np.zeros(nobs)
            
    # set starting values
    X0 = np.array([Kbar])   
    Y0 = np.array([Nbar, Sbar])
    
    # simulate
    X, Y = LinApp_SSL(X0, Zhist, bar, False, PP, QQ, UU, Y0, RR, SS, VV)
    
    # unpack X & Y
    Khist = X
    Nhist = Y[:,0]
    Shist = Y[:,1]
    for t in range(0,nobs-1):
        Lhist[t], dhist[t], bhist[t], Yhist[t], whist[t], rhist[t], \
            Chist[t], Ihist[t], uChist[t], uLhist[t], uShist[t], uhist[t] = \
            mdefs(Khist[t+1], Khist[t], Nhist[t], Shist[t], zhist[t], *params)
    
    # delete last observation from Khist
    Khist = Khist[0:nobs]

    return Khist, Nhist, Shist, Lhist, dhist, bhist, Yhist, \
        whist, rhist, Chist, Ihist, uChist, uLhist, uShist, uhist


nobs = 1000000

# generate history of z's and y's
zhist = np.zeros(nobs)
yhist = np.zeros(nobs)
epszhist = np.random.normal(0., sigma_z, nobs)
Zhist = np.zeros((nobs,nz))

for t in range(0,nobs):
    if t == 0:
        zhist[t] = epszhist[t]
    else:
        zhist[t] = rho_z*zhist[t-1] + epszhist[t]
    Zhist[t,0] = zhist[t]

Khist, Nhist, Shist, Lhist, dhist, bhist, Yhist, \
    whist, rhist, Chist, Ihist, uChist, uLhist, uShist, uhist = \
    runsim(Zhist, nobs, params, LinCoeffs, bars)

# -----------------------------------------------------------------------------
# PLOT SUBSAMPLE

import matplotlib.pyplot as plt

# write plot function

def plotvars(serlist, start, sample):
    '''
    This function generates a series of time plots for selected model
    varaibles.
    
    Inputs:
        serlist - list of series that can be plotted
        start - begining period for sample to be plotted
        sample - number of periods to plot
        
    Outputs:
        fig1 - plots for GDP and hours
        fig2 - plots foer capital, technology, wages and investment
    '''

    time = range(0, sample)
    
    fig1 = plt.figure()
    plt.subplot(2,2,1)
    plt.plot(time, Yhist[start:start+sample], label='Y')
    plt.title('Y')
    plt.subplot(2,2,2)
    plt.plot(time, Lhist[start:start+sample], label='L')
    plt.title('L')
    plt.subplot(2,2,3)
    plt.plot(time, Nhist[start:start+sample], label='N')
    plt.title('N')
    plt.subplot(2,2,4)
    plt.plot(time, Shist[start:start+sample], label='S')
    plt.title('S')
    plt.show()
    
    fig2 = plt.figure()
    plt.subplot(2,2,1)
    plt.plot(time, Khist[start:start+sample], label='K')
    plt.title('K')
    plt.subplot(2,2,2)
    plt.plot(time, zhist[start:start+sample], label='z')
    plt.title('z')
    plt.subplot(2,2,3)
    plt.plot(time, whist[start:start+sample], label='w')
    plt.title('w')
    plt.subplot(2,2,4)
    plt.plot(time, Ihist[start:start+sample], label='I')
    plt.title('I')
    plt.savefig('test.pdf', format='pdf', dpi=2000)
    plt.show()
    
    return fig1, fig2
    
start = 500
sample = 300
serlist = (Khist, Nhist, Shist, Lhist, dhist, bhist, Yhist, \
    whist, rhist, Chist, Ihist, uChist, uLhist, uShist, uhist, zhist)

fig1, fig2 = plotvars(serlist, start, sample)

# -----------------------------------------------------------------------------
# CALCULATE AND REPORT MOMENTS

Datahist = np.vstack((np.log(Yhist[start:nobs-1]), \
                      np.log(Nhist[start:nobs-1]), \
                      np.log(Shist[start:nobs-1]), \
                      np.log(Lhist[start:nobs-1]), \
                      np.log(whist[start:nobs-1]), \
                      rhist[start:nobs-1], \
                      np.log(Chist[start:nobs-1]), \
                      np.log(Ihist[start:nobs-1]), \
                      uChist[start:nobs-1], \
                      uLhist[start:nobs-1], \
                      uShist[start:nobs-1], \
                      uhist[start:nobs-1]))

varindex = ['log Y', 'log N', 'log S', 'log L', 'log w', 'r', 'log C', \
            'log I', 'uC', 'uL', 'uS', 'u']

report, momindex = calcmom(Datahist.T, means = False, stds = True, \
    relstds = True, corrs = True, autos = True, cvars = False)

writer = pd.ExcelWriter(modname + '_ModMoms.xlsx')

BCdf = pd.DataFrame(report.T)
BCdf.index = varindex
BCdf.columns = momindex
print (BCdf.to_latex())
BCdf.to_excel(writer,'LT')

# -----------------------------------------------------------------------------
# IMPULSE RESPONSE FUNCTIONS

nobs = 81

# z IRF

# generate history of z's and y's
zhist = np.zeros(nobs)
yhist = np.zeros(nobs)
epszhist = np.zeros(nobs)
Zhist = np.zeros((nobs,nz))
epszhist[2] = sigma_z

for t in range(0,nobs):
    if t == 0:
        zhist[t] = epszhist[t]
    else:
        zhist[t] = rho_z*zhist[t-1] + epszhist[t]
    Zhist[t,0] = zhist[t]

Khist, Nhist, Shist, Lhist, dhist, bhist, Yhist, \
    whist, rhist, Chist, Ihist, uChist, uLhist, uShist, uhist = \
    runsim(Zhist, nobs, params, LinCoeffs, bars)

# plot IRFs
start = 0
sample = nobs-1
serlist = (Khist, Nhist, Shist, Lhist, dhist, bhist, Yhist, \
    whist, rhist, Chist, uChist, uLhist, uShist, uhist, zhist, yhist)

fig1, fig2 = plotvars(serlist, start, sample)

time = range(0, sample)
fig4 = plt.figure()
plt.plot(time, 60*(Nhist[start:start+sample] - Nbar), label='Work')
plt.plot(time, 60*(Lhist[start:start+sample] - Lbar), label='Leisure')
plt.plot(time, 60*(Shist[start:start+sample] - Sbar), label='Sleep')
plt.legend()
plt.ylabel('Minutes per Day')
plt.savefig(modname + '_TimeUse.pdf', format='pdf', dpi=2000)
plt.show()

# -----------------------------------------------------------------------------
# CONSTRUCT PREDICTED TIME USE
import statsmodels.api as sm

lfreq = 6
hfreq = 32

HP = False

# load data on output and capital stock
df = pd.read_excel('BCdata1.xlsx')
BCdata = df.as_matrix()

# log detrend and add back model bar values
# capital
K = np.log(BCdata[:,0])
if HP:
    Kdev, Ktrd = sm.tsa.filters.hpfilter(K)
else:
    Kdev, Ktrd = sm.tsa.filters.cffilter(K, low=lfreq, high=hfreq)
K = np.exp(Kdev)*Kbar
Kdev = K - Kbar

nobs = K.size
time = np.arange(2003., 2017., .25)

# GDP
Y = np.log(BCdata[:,1])
if HP:
    Ydev, Ytrd = sm.tsa.filters.hpfilter(Y)
else:
    Ydev, Ytrd = sm.tsa.filters.cffilter(Y, low=lfreq, high=hfreq)
Y = np.exp(Ydev)*Ybar

# consumption
C = np.log(BCdata[:,6])
if HP:
    Cdev, Ctrd = sm.tsa.filters.hpfilter(C)
else:
    Cdev, Ctrd = sm.tsa.filters.cffilter(C, low=lfreq, high=hfreq)
C = np.exp(Cdev)*Cbar

# labor is market and home
N = np.log(BCdata[:,3]+BCdata[:,4]/2)
# filter series
if HP:
    Ndev, Ntrd = sm.tsa.filters.hpfilter(N)
else:
    Ndev, Ntrd = sm.tsa.filters.cffilter(N, low=lfreq, high=hfreq)
N = np.exp(Ndev)*Nbar
Ndev = N - Nbar

# sleep
S = np.log(BCdata[:,2])
# filter series
if HP:
    Sdev, Strd = sm.tsa.filters.hpfilter(S)
else:
    Sdev, Strd = sm.tsa.filters.cffilter(S, low=lfreq, high=hfreq)
S = np.exp(Sdev)*Sbar
Sdev = S - Sbar

# unfiltered hours
Nraw = BCdata[:,3]+BCdata[:,4]/2
Sraw = BCdata[:,2]
Lraw = BCdata[:,5]+BCdata[:,4]/2

# averages of unfiltered hours
Navg = np.mean(Nraw)
Savg = np.mean(Sraw)
Lavg = np.mean(Lraw)

# plot unfiltered hoursa radjusting from per quarter to per day
plt.figure()
plt.plot(time, Nbar*Nraw/Navg, label='Work')
plt.plot(time, Sbar*Sraw/Savg, label='Sleep')
plt.plot(time, (24-Nbar-Sbar)*Lraw/Lavg, label='Leisure')
plt.legend(loc=5, ncol=1)
plt.ylabel('Hours per Day')
plt.savefig(modname + '_raw.pdf', format='pdf', dpi=2000)
plt.show()

# recover z from model jump function for N
z = (Ndev-RR[0,0]*Kdev)/SS[0,0]

# construct N & S from jump functions
Nmod = RR[0,0]*Kdev + SS[0,0]*z + Nbar
Smod = RR[1,0]*Kdev + SS[1,0]*z + Sbar

# reduce observations by one
nobs = nobs - 1 

# contruct other edongenous variables from mdefs using model hours
Lmod = np.zeros(nobs)
dmod = np.zeros(nobs)
bmod = np.zeros(nobs)
Ymod = np.zeros(nobs)
wmod = np.zeros(nobs)
rmod = np.zeros(nobs)
Cmod = np.zeros(nobs)
Imod = np.zeros(nobs)
uCmod = np.zeros(nobs)
uLmod = np.zeros(nobs)
uSmod = np.zeros(nobs)
umod = np.zeros(nobs)
for t in range(0,nobs):
    Lmod[t], dmod[t], bmod[t], Ymod[t], wmod[t], rmod[t], Cmod[t], Imod[t], \
        uCmod[t], uLmod[t], uSmod[t], umod[t] = mdefs(K[t+1], K[t], Nmod[t], \
        Smod[t], z[t], *params)

# contruct other edongenous variables from mdefs using actual hours       
Lmod2 = np.zeros(nobs)
dmod2 = np.zeros(nobs)
bmod2 = np.zeros(nobs)
Ymod2 = np.zeros(nobs)
wmod2 = np.zeros(nobs)
rmod2 = np.zeros(nobs)
Cmod2 = np.zeros(nobs)
Imod2 = np.zeros(nobs)
uCmod2 = np.zeros(nobs)
uLmod2 = np.zeros(nobs)
uSmod2 = np.zeros(nobs)
umod2 = np.zeros(nobs)
for t in range(0,nobs):
    Lmod2[t], dmod2[t], bmod2[t], Ymod2[t], wmod2[t], rmod2[t], Cmod2[t], \
        Imod2[t], uCmod2[t], uLmod2[t], uSmod2[t], umod2[t] = mdefs(K[t+1], \
        K[t], N[t], S[t], z[t], *params)

time2 = time[0:nobs]

# contruct consumption utility using actual hours
if gamma == 1:
    uCact = np.log(C)
else:
    uCact = (C**(1-gamma) - 1)/(1-gamma)

# plot model histories
plt.figure()
plt.plot(time2, N[0:nobs], label='data')
plt.plot(time2, Nmod[0:nobs], label='model')
plt.title('Work Hours')
plt.legend(loc=4, ncol=2)
plt.ylabel('Hours per Week')
plt.savefig(modname + '_Npred.pdf', format='pdf', dpi=2000)
plt.show()

plt.figure()
plt.plot(time2, Y[0:nobs]/Ybar, label='data')
plt.plot(time2, Ymod/Ybar, label='model')
plt.title('GDP')
plt.ylabel('Relative to Model Steady State')
plt.legend(loc=4, ncol=2)
plt.savefig(modname + '_Ypred.pdf', format='pdf', dpi=2000)
plt.show()

plt.figure()
plt.plot(time2, C[0:nobs]/Cbar, label='data')
plt.plot(time2, Cmod/Cbar, label='model')
plt.title('Consumption')
plt.ylabel('Relative to Model Steady State')
plt.legend(loc=4, ncol=2)
plt.savefig(modname + '_Cpred.pdf', format='pdf', dpi=2000)
plt.show()

plt.figure()
plt.plot(time2, S[0:nobs], label='data')
plt.plot(time2, Smod[0:nobs], label='model')
plt.title('Sleep Hours')
plt.ylabel('Hours per Week')
plt.legend(loc=4, ncol=2)
plt.savefig(modname + '_Spred.pdf', format='pdf', dpi=2000)
plt.show()

# find compensating variations for sleep change
uSdev = uSmod - uSbar
if gamma==1:
    Ccomp = np.exp(uCmod-uSdev)
else:
    Ccomp = ((1-gamma)*(uCmod-uSdev)+1)**(1/(1-gamma))
Cperc = Ccomp/Cmod - 1

uSdev2 = uSmod2 - uSbar
if gamma==1:
    Ccomp2 = np.exp(uCmod2-uSdev2)
else:
    Ccomp2 = ((1-gamma)*(uCmod2-uSdev2)+1)**(1/(1-gamma))
Cperc2 = Ccomp2/Cmod2 - 1

if gamma==1:
    Ccomp3 = np.exp(uCact[0:nobs]-uSdev2)
else:
    Ccomp3 = ((1-gamma)*(uCact[0:nobs]-uSdev2)+1)**(1/(1-gamma))
Cperc3 = Ccomp3/C[0:nobs] - 1

rconpc = BCdata[0:nobs,6]
Cdol = rconpc*Cperc
Cdol2 = rconpc*Cperc2
Cdol3 = rconpc*Cperc3

plt.figure()
plt.plot(time2, Cdol, label='model hours')
plt.plot(time2, Cdol2, label='actual hours')
#plt.plot(time2, Cdol3, label='US data')
plt.title('Sleep Compensation')
plt.ylabel('2009 dollars')
plt.legend(loc=4, ncol=3)
plt.savefig(modname + '_comp.pdf', format='pdf', dpi=2000)
plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.plot(time2, uCmod, label='model hours')
plt.plot(time2, uCmod2, label='actual hours')
#plt.plot(time2, uCact[0:nobs], label='US data')
plt.title('Consumption Utility')
plt.legend(loc=4, ncol=3)
plt.xticks([])
plt.subplot(2,1,2)
plt.plot(time2, uSmod)
plt.plot(time2, uSmod2, label='actual model')
plt.title('Sleep Utility')
plt.savefig(modname + '_util.pdf', format='pdf', dpi=2000)
plt.show()

# use this code for comparing results with different hours worked series
#plt.figure()
#plt.plot(time, zManuf, label='Manufacturing')
#plt.plot(time, zNFarm, label='Non_Farm')
#plt.title('Technology Shock')
#plt.legend(loc=4, ncol=2)
#plt.savefig(modname + '_zcompare.pdf', format='pdf', dpi=2000)
#plt.show()