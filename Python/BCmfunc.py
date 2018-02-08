import pandas_datareader.data as web
import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def OLScoeffs(X, Y):
    ''' This function calculates OLS coefficents.  X and Y are organized with
    observations in rows and variables in columns. '''
    (nobs, nvar) = X.shape
    XX = np.dot(X.T, X)
    XXinv = np.linalg.inv(XX)
    XY = np.dot(X.T, Y)
    beta = np.dot(XXinv, XY)
    temp = np.dot(X,np.dot(XXinv, X.T))
    temp = np.dot(Y.T, np.dot((np.identity(nobs) - temp), Y))
    sigsq = temp/(nobs-1)
    
    return beta, sigsq


# set start and end dates
start = datetime.datetime(1950, 1, 1)
end = datetime.datetime(2017, 1, 1)

# get data from FRED, convert to numpy arrays and take logs.  Resample if needed.

# quarterly data
DataQ = web.DataReader(["GDPC1"] , "fred", start, end)   
DataQ.columns = ['GDP']

# monthly data
DataM = web.DataReader(["UNRATE", "CIVPART"], "fred", start, end)
DataM.columns = ['Unemp', 'Part']

# convert to quarterly frequency by averaging
DataM = DataM.resample('3M').mean()

# transform data appropriately logs or ratios to GDP, for example
logY = np.log(DataQ.as_matrix(columns=['GDP']))
Unemp = DataM.as_matrix(columns=['Unemp'])
Unemp = Unemp/100.
Part = DataM.as_matrix(columns=['Part'])
Part = Part/100.

# construct employment percentage
m = Part*(1-Unemp)

# use HP filter
Ytil, Ytrend = sm.tsa.filters.hpfilter(logY, 1600)
nobs = Ytil.size
time = range(0, nobs)

# plot deviations
plt.figure()
plt.plot(time, Ytil, label='Y')
plt.plot(time, m, label='m')
plt.legend()
plt.show()

# run regression
Xdata = np.hstack((np.ones((nobs,1)), np.reshape(Ytil, (nobs, 1))))
Ydata = np.reshape(m, (nobs, 1))
# lag of GDP
beta, sigsq = OLScoeffs(Xdata[0:nobs-1,:], Ydata[1:nobs,:])

print(beta, sigsq)