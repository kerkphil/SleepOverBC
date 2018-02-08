import pandas_datareader.data as web
import pandas as pd
import datetime
import numpy as np
import statsmodels.api as sm

from DSGEmoments import calcmom

modname = 'BCData'

def fdfilter(data):
    (nobs, nvar) = data.shape
    return data[1:nobs,:] - data[0:nobs-1,:]


def ltfilter(data):
    (nobs, nvar) = data.shape
    # regressors are a constant and time trend
    X = np.stack((np.ones(nobs), np.linspace(0, nobs-1, nobs)), axis=1)
    Y = data
    # beta = (X'X)^(-1) X'Y
    beta = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), \
                  np.dot(np.transpose(X), Y))
    # fitted values are X*beta
    return Y - np.dot(X, beta)


# set start and end dates
start = datetime.datetime(1960, 1, 1)
end = datetime.datetime(2017, 1, 1)

# get data from FRED, convert to numpy arrays and take logs.  Resample if needed.

# quarterly data
DataQ = web.DataReader(["GDPC1", "PCECC96", "GPDIC1", "GCEC1", "EXPGSC1", \
    "IMPGSC1", "GDPDEF", "WASCUR", "M318501Q027NBEA", "PRS85006023"] \
    , "fred", start, end)   
DataQ.columns = ['GDP', 'Cons', 'Inv', 'Govt', 'Exp', 'Imp', 'Defl', 'Ecomp', \
    'BSurp', 'NFarmHrs']

# monthly data
DataM = web.DataReader(["PAYEMS", "AWHMAN", "CPIAUCSL", "TB3MS", \
    "M1SL", "UNRATE", "CIVPART"], "fred", start, end)
DataM.columns = ['Emp', 'ManufHrs', 'CPI', '3MTbill', 'M1', 'Unemp', 'Part']

# convert to quarterly frequency by averaging
DataM = DataM.resample('3M').mean()

# transform data appropriately logs or ratios to GDP, for example
logY = np.log(DataQ.as_matrix(columns=['GDP']))

logC = np.log(DataQ.as_matrix(columns=['Cons']))

logI = np.log(DataQ.as_matrix(columns=['Inv']))

logG = np.log(DataQ.as_matrix(columns=['Govt']))

NX_Y = (DataQ.as_matrix(columns=['Exp']) - DataQ.as_matrix(columns=['Imp'])) \
    / DataQ.as_matrix(columns=['GDP'])
    
logX = np.log(DataQ.as_matrix(columns=['Exp']))

logM = np.log(DataQ.as_matrix(columns=['Imp']))

logEmp = np.log(DataM.as_matrix(columns=['Emp']))

logMHrs = np.log(DataM.as_matrix(columns=['ManufHrs']))

logNHrs = np.log(DataQ.as_matrix(columns=['NFarmHrs']))

logMLab = logEmp + logMHrs

logNLab = logEmp + logNHrs

logCPI = np.log(DataM.as_matrix(columns=['CPI']))

logNWag = np.log(DataQ.as_matrix(columns=['Ecomp'])) - logNLab

logRWag = logNWag - logCPI

NInt = DataM.as_matrix(columns=['3MTbill'])
NInt = NInt/100.

Infl = fdfilter(logCPI)
Infl = np.concatenate((np.array([[0.]]), Infl))

RInt = NInt - Infl

logNMon = np.log(DataM.as_matrix(columns=['M1']))

logRMon = logNMon - logCPI

logDfl = np.log(DataQ.as_matrix(columns=['Defl']))

logProd = logY - logNLab

Unemp = DataM.as_matrix(columns=['Unemp'])
Unemp = Unemp/100.

Part = DataM.as_matrix(columns=['Part'])
Part = Part/100.

BD_Y =  - DataQ.as_matrix(columns=['BSurp']) / DataQ.as_matrix(columns=['GDP'])

Data = np.hstack((logY, logC, logI, logG, NX_Y, logX, logM, logEmp, \
   logMHrs, logNHrs, logNLab, logNWag, logRWag, NInt, RInt, logNMon, logRMon, logDfl, \
   logCPI, logProd, Unemp, Part, BD_Y))

varindex = ['logY', 'logC', 'logI', 'logG', 'NX_Y', 'logX', 'logM', 'logEmp', \
   'logMHrs', 'logNHrs', 'logNLab', 'logNWag', 'logRWag', 'NInt', 'RInt', 'logNMon', \
   'logRMon', 'logDfl', 'logCPI', 'logProd', 'Unemp', 'Part', 'BD_Y']

# use linear trend filter
DataLT = ltfilter(Data)
reportLT, momindex = calcmom(DataLT)

# use FD filter
DataFD = fdfilter(Data)
reportFD, momindex = calcmom(DataFD)

# use HP filter
DataHP, trendHP = sm.tsa.filters.hpfilter(Data, 1600)
reportHP, momindex = calcmom(DataHP)

# use BK filter
DataCF, trendCF = sm.tsa.filters.cffilter(Data)
reportCF, momindex = calcmom(DataCF)

writer = pd.ExcelWriter(modname + '_Moms.xlsx')

LTdf = pd.DataFrame(reportLT)
LTdf.columns = varindex
LTdf.index = momindex
LTdf = LTdf.transpose()
print (LTdf.to_latex())
LTdf.to_excel(writer,'LT')

FDdf = pd.DataFrame(reportFD)
FDdf.columns = varindex
FDdf.index = momindex
FDdf = FDdf.transpose()
print (FDdf.to_latex())
FDdf.to_excel(writer,'FD')

HPdf = pd.DataFrame(reportHP)
HPdf.columns = varindex
HPdf.index = momindex
HPdf = HPdf.transpose()
print (HPdf.to_latex())
HPdf.to_excel(writer,'HP')

CFdf = pd.DataFrame(reportCF)
CFdf.columns = varindex
CFdf.index = momindex
CFdf = CFdf.transpose()
print (CFdf.to_latex())
CFdf.to_excel(writer,'CF')

writer.save()