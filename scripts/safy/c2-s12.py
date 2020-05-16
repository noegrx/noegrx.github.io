import datetime
import parameters as par
import pandas as pd
import pandas_datareader.data as web
import plotly.graph_objects as go
#from iexfinance.stocks import Stock, get_historical_data
from plotly.offline import plot

symbol = 'AMZN'
company="Amazon"
source='yahoo'
past = datetime.timedelta(weeks=2, days=0, hours=0, minutes=0)
start = datetime.date.today() - 1*past
end = datetime.date.today()
images = par.img_path+'c2-s12-'
#company = Stock(symbol, token=token).get_company()

#df = get_historical_data(symbol, output_format='pandas', start=start, end=end, token=token)
df = web.DataReader(symbol, source, start=start, end=end)
df.index = pd.to_datetime(df.index)
print(df.tail())

mygreen='rgba(173,255,47,1)'
mypink='rgba(255,0,76,1)'

fig = go.Figure(layout=par.layout)
fig.update_layout(title_text='Chandeliers japonais - '+company+' ('+symbol+')')
fig.add_candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, increasing=dict(line=dict(color=mygreen)), decreasing=dict(line=dict(color=mypink)))
plot(fig, filename=images+'1.html', config=par.config, auto_open=False)
