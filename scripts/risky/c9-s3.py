import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
import parameters as par
import scipy.interpolate as sci
import scipy.optimize as sco
import plotly.graph_objects as go
from plotly.offline import plot

symbols = ['AAPL', 'AMZN', 'FB', 'GOOGL']
source='yahoo'
past = datetime.timedelta(weeks=52, days=0, hours=0, minutes=0)
start = datetime.date.today() - 5*past
end = datetime.date.today()
images = par.img_path+'c9-s3-'
textfiles  = par.txt_path+'c9-s3-'
red='#f92672'
green='#a6e22e'
blue='#7fffff'

df = pd.DataFrame()
for sym in symbols:
    df[sym] = web.DataReader(sym, source, start=start, end=end)['Close']
df.index = pd.to_datetime(df.index)
print(df.tail())

stock_return = df.apply(lambda x: (x / x[0] ))
print(stock_return)

fig = go.Figure(layout=par.layout)
for i in symbols:
    fig.add_trace(go.Scatter(x=stock_return.index, y=stock_return[i], name=i))
plot(fig, filename=images+'1.html', config=par.config, auto_open=False)

logReturnXi= df.copy() # logarithmic returns
logReturnXi.iloc[0]= float('NaN')
for j in range(1, len(logReturnXi)):
    logReturnXi.iloc[j]= np.log(df.iloc[j]/ df.iloc[j-1]) * 365 \
        / (df.index[j]-df.index[j-1]).days

fig = go.Figure(layout=par.layout)
for i in symbols:
    fig.add_trace(go.Scatter(x=logReturnXi.index, y=logReturnXi[i], name=i))
plot(fig, filename=images+'2.html', config=par.config, auto_open=False)

print('Annualized return, average: ')
print(logReturnXi.mean())
pd.DataFrame.describe(logReturnXi)

r = logReturnXi.mean()
print('returns: r=')
print(r)

# auxiliary quantities
CovMatrix = logReturnXi.cov()          # covariance matrix
CovInv=     np.linalg.inv(CovMatrix)   # its inverse

J= np.size(logReturnXi, axis=1)
a= np.matmul(r.values,   np.matmul(CovInv, r.values));   print(); print('a= ', a)
b= np.matmul(r.values,   np.matmul(CovInv, np.ones(J))); print('b= ', b)
c= np.matmul(np.ones(J), np.matmul(CovInv, np.ones(J))); print('c= ', c)
# print('d= ', a*c-b*b)

print(); print('correllation matrix:')
logReturnXi.corr()

# efficient asset allocation following Markowitz
def xEfficient(mu, r0=None, a=a, b=b, c=c):
    d= a*c- b*b
    if r0==None:
        xs= mu* ( c/d* np.matmul(CovInv,r) -         b/d*np.matmul(CovInv, np.ones(J))) \
                + a/d* np.matmul(CovInv,np.ones(J))- b/d*np.matmul(CovInv, r)
    else:
        mu0= (a-r0*b)/ (b-r0*c)
        xs=  (mu-r0) / (mu0-r0) * (np.matmul(CovInv,r)-r0*np.matmul(CovInv, np.ones(J)))/(b-r0*c)
        xs['risk free']= (mu0-mu)/(mu0-r0)
    return xs


# example
#mu= 0.01
#xsBsp= xEfficient(mu)
#print(xsBsp); print()
#print("σ= ", np.sqrt(np.matmul(xsBsp.values, np.matmul(CovMatrix.values, xsBsp.values.transpose()))))

mux = np.linspace(0.0, .5, 101)
s1= [xEfficient(muy) for muy in mux]
for i in range(len(mux)):
    for s in range(len(symbols)):
        print(s1[i][s])


#fig = go.Figure(layout=par.layout)
#for x in symbols:
#    for x in range(len(mux)):
#        fig.add_trace(go.Scatter( x=mux, y=[s1[i][x] for i in range(len(mux))], name= x))
#        #fig.add_trace(go.Scatter(x=mux, y=(mux*mux*c-2*mux*b+a)/(a*c-b*b), name='hellomu', yaxis='y2'))
#        fig.add_trace(go.Scatter(x=mux, y=logReturnXi[i], name=i))
#    data+= [go.Scatter( x=mux, y= (mux*mux*c-2*mux*b+a)/(a*c-b*b), name= 'σ(μ)', yaxis='y2')]
#plot(fig, filename=images+'3.html', config=par.config, auto_open=False)

# sigma as a function of mu
def sigma(mu, a=a, b=b, c=c):
    return np.sqrt(c*mu*mu-2*mu*b+a)/(a*c-b*b)

# the tangecny portfolio
riskFree= 0.02              # set the risk free rate
muMarket= (a - riskFree*b) / (b - riskFree*c)

print('Tangency portfolio: σ(CAPM)= ', sigma(muMarket))
print('                    μ(CAPM)= ', muMarket)

# plot the efficient frontier and tangency
muy=   np.linspace(.0, muMarket+.1, 201)
data=  [go.Scatter( y=muy, x=[sigma(mu, a, b, c) for mu in muy], name= 'μ(σ)', fill='tozeroy', fillcolor = red)]
data+= [go.Scatter( y=[muMarket, muMarket], x=[sigma(muMarket), sigma(muMarket)], name= 'tangency portfolio', marker=dict(color=green,size=15))]
data+= [go.Scatter( y=muy, x=[sigma(muMarket) * (mu-riskFree)/(muMarket-riskFree) for mu in muy], name= 'capital market line')]
fig = go.Figure(data=data, layout=par.layout)
plot(fig, filename=images+'3.html', config=par.config, auto_open=False)

# Parameter des tangency portfolios
print('marktet portfolio: σ(CAPM)= ', sigma(muMarket))
print('                   μ(CAPM)= ', muMarket)
print('Sharpe ratio:      s(CAPM)= ', (muMarket- riskFree)/ sigma(muMarket))
print('the market portfolio:')
xEfficient(muMarket)



# Visualization One Fund Theorem

#mux = np.linspace(0.0, muMarket+.1, 21)
#s1= [xEfficient(muy, riskFree) for muy in mux]
#minimum= min([min(s1[i]) for i in range(len(s1))])
#maximum= max([max(s1[i]) for i in range(len(s1))])

#data=  [go.Scatter( x=mux, y=[s1[i][x[0]] for i in range(len(mux))], name= x[0]) for x in symbols]
#data+= [go.Scatter( x=mux, y=[s1[i]['risk free'] for i in range(len(mux))], name= 'risk free')]
#data+= [go.Scatter( x=mux, y= (mux*mux*c-2*mux*b+a)/(a*c-b*b), name= 'σ(μ)', yaxis='y2')]
#data+= [go.Scatter( x=[riskFree, riskFree], y=[minimum, maximum], name= 'risk free portfolio')]
#data+= [go.Scatter( x=[muMarket, muMarket], y=[minimum, maximum], name= 'market portfolio')]

#fig = go.Figure(data=data, layout=go.Layout(xaxis=dict(title='μ'), 
#                                              yaxis=dict(title='allocation', zerolinewidth= 6),
#                                              yaxis2=dict(overlaying='y', side='right', showticklabels=False, showgrid=False),
#                                              title=go.layout.Title(text=' Tobin-Separation or „Two Fund Separation“ ')))
#plot(fig, filename=images+'5.html', config=config, auto_open=False)

# print β coefficients
(r-riskFree) / (muMarket-riskFree)

#j=5    # pick the asset
#xs= xEfficient(muMarket)
#xiMarket= [np.matmul(xs.values, row.values) for index, row in logReturnXi.iterrows()]
#xij=      [row[symbols[j][0]] for index, row in logReturnXi.iterrows()]
#betaj=    (r[symbols[j][0]]-riskFree) / (muMarket-riskFree)

#data = [go.Scatter( x=xiMarket, y=xij, mode='markers',  name= symbols[j][0])]
#data+= [go.Scatter( x=[np.nanmin(xiMarket), np.nanmax(xiMarket)], 
#                    y=r[symbols[j][0]] - muMarket*betaj 
#                           + [betaj*np.nanmin(xiMarket), betaj*np.nanmax(xiMarket)], 
#                           name= 'regression')]

#layout = go.Layout(xaxis=dict(title='ξ (market)', zeroline= False),
#                   yaxis=dict(title= 'ξ ('+ symbols[j][0]+')', zeroline=False),
#                   title = symbols[j][0]+'<br>(β=' + str(betaj) +')')

#fig = go.Figure(data=data, layout=layout) 
#plot(fig, filename=images+'6.html', config=config, auto_open=False)
