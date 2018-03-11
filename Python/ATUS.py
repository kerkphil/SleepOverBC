# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:46:40 2017

@author: Kerk
"""

#import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from DSGEmoments import calcmom

# load data on output and capital stock
ATUSdata = pd.read_excel('wquartertotals.xls', index_col = 0)
SE = ATUSdata.iloc[:,0].values
N  = ATUSdata.iloc[:,1].values
HE = ATUSdata.iloc[:,2].values
LE = ATUSdata.iloc[:,3].values
SP = ATUSdata.iloc[:,4].values
NP = ATUSdata.iloc[:,5].values
HP = ATUSdata.iloc[:,6].values
LP = ATUSdata.iloc[:,7].values
SU = ATUSdata.iloc[:,8].values
HU = ATUSdata.iloc[:,9].values
LU = ATUSdata.iloc[:,10].values

time = np.linspace(0., N.size-1, N.size)
time = 2003. + time/4.

plt.figure()

plt.subplot(3,1,1)
plt.plot(time, HE, label='Home')
plt.plot(time, LE, label='Leisure')
plt.plot(time, SE, label='Sleep')
plt.plot(time, N , label='Work')
plt.legend()
#plt.ylabel('Minutes per Day')
plt.xticks([])
plt.title('Full-time')

plt.subplot(3,1,2)
plt.plot(time, HP, label='Home')
plt.plot(time, LP, label='Leisure')
plt.plot(time, SP, label='Sleep')
plt.plot(time, NP, label='Work')
plt.legend()
#plt.ylabel('Minutes per Day')
plt.xticks([])
plt.title('Part-time')

plt.subplot(3,1,3)
plt.plot(time, HU, label='Home')
plt.plot(time, LU, label='Leisure')
plt.plot(time, SU, label='Sleep')
plt.legend()
#plt.ylabel('Minutes per Day')
plt.title('Not Working')

plt.savefig('ATUS1.pdf', format='pdf', dpi=2000)
plt.show()


plt.figure()

plt.subplot(2,2,1)
plt.plot(time, N , label='Full-time')
plt.plot(time, NP, label='Part-time')
#plt.legend()
#plt.ylabel('Minutes per Day')
plt.xticks([])
plt.title('Work')

plt.subplot(2,2,2)
plt.plot(time, SE, label='Full-time')
plt.plot(time, SP, label='Part-time')
plt.plot(time, SU, label='Not Working')
#plt.legend()
#plt.ylabel('Minutes per Day')
plt.xticks([])
plt.title('Sleep')

plt.subplot(2,2,3)
plt.plot(time, HE, label='Full-time')
plt.plot(time, HP, label='Part-time')
plt.plot(time, HU, label='Not Working')
#plt.legend()
#plt.ylabel('Minutes per Day')
plt.title('Home')

plt.subplot(2,2,4)
plt.plot(time, LE, label='Full-time')
plt.plot(time, LP, label='Part-time')
plt.plot(time, LU, label='Not Working')
#plt.legend()
#plt.ylabel('Minutes per Day')
plt.title('Leisure')

plt.savefig('ATUS2.pdf', format='pdf', dpi=2000)
plt.show()


# load data on output and capital stock
BCdata = pd.read_excel('BCdata.xlsx')
Y = BCdata.iloc[:,1].values
Nmanuf = BCdata.iloc[:,2].values
Nnonf = BCdata.iloc[:,3].values
Y = Y[0:56]
Nmanuf = Nmanuf[0:56]
Nnonf = Nnonf[0:56]

Data = np.vstack((Y, SE, N, HE, LE, SP, NP, HP, LP, SU, HU, LU, Nmanuf, Nnonf))
Data = np.log(Data.T)

# detrend all series
Datadev, Datatrd = sm.tsa.filters.hpfilter(Data, 1600)

varindex = ['Y', 'SE', 'N', 'HE', 'LE', 'SP', 'NP', 'HP', 'LP', 'SU', \
            'HU', 'LU', 'Nmanuf', 'Nnonf']

report, momindex = calcmom(Datadev, means = False, stds = True, relstds = True, \
    corrs = True, autos = True, cvars = False)

Momdf = pd.DataFrame(report)
Momdf.columns = varindex
Momdf.index = momindex
Momdf = Momdf.transpose()
print (Momdf.to_latex())
writer = pd.ExcelWriter('ATUS_Moms.xlsx')
Momdf.to_excel(writer,'HP')
writer.save()
