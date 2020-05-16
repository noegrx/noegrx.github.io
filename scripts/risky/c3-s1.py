import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
import parameters as par
import requests
import scipy.interpolate as sci
import scipy.optimize as sco
import plotly.graph_objects as go
from plotly.offline import plot

symbols = ['RNO.PA', 'FDJ.PA', 'FP.PA', 'VK.PA', 'AIR.PA'] 
source='yahoo'
past = datetime.timedelta(weeks=52, days=0, hours=0, minutes=0)
start = datetime.date.today() - 5*past
end = datetime.date.today()
images = par.img_path+'c3-s1-'
textfiles  = par.txt_path+'c3-s1-'

df = pd.DataFrame()
for sym in symbols:
    df[sym] = web.DataReader(sym, source, start=start, end=end)['Close']
df.index = pd.to_datetime(df.index)
print(df.tail())

noa = len(symbols)

fig = plt.figure()
(df / df.iloc[0] * 100).plot()
plt.xlabel(" ")
plt.title(symbols[0]+' - '+symbols[1]+' - '+symbols[2]+' - '+symbols[3]+' - '+symbols[4])
plt.savefig(images+'1')

rets = np.log(df / df.shift(1))

file = open(textfiles+'1.txt', 'w')
file.write(str(rets.head()))
file.close()

file = open(textfiles+'2.txt', 'w')
file.write(str(rets.mean() * 252))
file.close()

file = open(textfiles+'3.txt', 'w')
file.write(str(rets.cov() * 252))
file.close()

weights = np.random.random(noa)
weights /= np.sum(weights)

#print(weights)
file = open(textfiles+'4.txt', 'w')
file.write(str(weights))
file.close()

file = open(textfiles+'5.txt', 'w')
file.write(str(np.sum(rets.mean() * weights) * 252))
file.close()

variance = np.dot(weights.T, np.dot(rets.cov() * 252, weights))
file = open(textfiles+'6.txt', 'w')
file.write(str(variance))
file.close()

volatility = np.sqrt(variance)
file = open(textfiles+'7.txt', 'w')
file.write(str(volatility))
file.close()

def port_ret(weights):
    return np.sum(rets.mean() * weights) * 252

def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

prets = []
pvols = []
for p in range (2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(port_ret(weights))  
    pvols.append(port_vol(weights))
prets = np.array(prets)
pvols = np.array(pvols)

fig = plt.figure()
plt.scatter(pvols, prets, c=prets / pvols, marker='.', cmap='cool')
plt.xlabel("Volatilité espérée")
plt.ylabel("Rendement espéré")
plt.colorbar(label="Ratio de Sharpe")
plt.title(symbols[0]+' - '+symbols[1]+' - '+symbols[2]+' - '+symbols[3]+' - '+symbols[4])
fig.savefig(images+'2')

def min_func_sharpe(weights):  
    return -port_ret(weights) / port_vol(weights)  

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
eweights = np.array(noa * [1. / noa,])

#print(min_func_sharpe(eweights))

opts = sco.minimize(min_func_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)

#print (opts)

# print (opts['x'].round(3))
#print(port_ret(opts['x']).round(3))
#print(port_vol(opts['x']).round(3))
#print(port_ret(opts['x']) / port_vol(opts['x']))

optv = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)

#print(optv)
#print(optv['x'].round(3))
#print(port_vol(optv['x']).round(3))
#print(port_ret(optv['x']).round(3))
#print(port_ret(optv['x']) / port_vol(optv['x']))


### EFFICIENT FRONTIER
cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - tret},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

bnds = tuple((0, 1) for x in weights)

trets = np.linspace(0.05, 0.2, 50)
tvols = []
for tret in trets:
    res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

fig = plt.figure()
plt.scatter(pvols, prets, c=prets / pvols, marker='.', cmap='cool')
plt.scatter(tvols, trets, c=trets / tvols, lw=3.0, cmap='cool')
plt.plot(port_vol(opts['x']), port_ret(opts['x']),'*',c='#a6e22e',markersize=15)
plt.plot(port_vol(optv['x']), port_ret(optv['x']),'*',c='#f92672',markersize=15)
plt.xlabel("Volatilité espérée")
plt.ylabel("Rendement espéré")
plt.colorbar(label="Ratio de Sharpe")
plt.title(symbols[0]+' - '+symbols[1]+' - '+symbols[2]+' - '+symbols[3]+' - '+symbols[4])
fig.savefig(images+'3')

### Droite de marche
ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]
tck = sci.splrep(evols, erets)

def f(x):
    ''' Efficient frontier function (splines approximation). '''
    return sci.splev(x, tck, der=0)
def df(x):
    ''' First derivative of efficient frontier function. '''
    return sci.splev(x, tck, der=1)

def equations(p, rf=0.01):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3

opt = sco.fsolve(equations, [0.02, 0.5, 0.15])

print(opt)

print(np.round(equations(opt), 6))

#fig = plt.figure()
#plt.scatter(pvols, prets, c=(prets - 0.02) / pvols, marker='.', cmap='cool')
#plt.plot(evols, erets, '#a6e22e', lw=4)
#cx = np.linspace(0.0, 0.3)
#plt.plot(cx, opt[0] + opt[1] * cx, c='#7fffff',lw=2)
#plt.plot(opt[2], f(opt[2]), '*',c='#a6e22e',markersize=15)
#plt.axhline(0,c='w',ls='--')
#plt.axvline(0,c='w',ls='--')
#plt.xlabel("Volatilité espérée")
#plt.ylabel("Rendement espéré")
#plt.colorbar(label="Ratio de Sharpe")
#plt.title(symbols[0]+' - '+symbols[1]+' - '+symbols[2]+' - '+symbols[3]+' - '+symbols[4])
#fig.savefig(images+'4')

cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - f(opt[2])}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)

file = open(textfiles+'8.txt', 'w')
file.write(str(res['x'].round(3)))
file.close()

parents=[""]

fig = go.Figure(layout=par.layout)
fig.add_pie(labels=symbols, values=res['x'].round(2), hole=.3)
#fig.add_sunburst(labels=symbols, parents=parents, values=res['x'].round(2))
fig.update_layout(title_text=' Portefeuille optimal', showlegend=True)
plot(fig, filename=images+'6.html', config=par.config, auto_open=False)
