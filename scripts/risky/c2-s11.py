import datetime
import parameters as par
import pandas as pd
import pandas_datareader.data as web
import plotly.graph_objects as go
#from iexfinance.stocks import Stock, get_historical_data
from plotly.offline import plot
from plotly.subplots import make_subplots

symbol = 'AIR.PA'
company="Airbus"
source='yahoo'
past = datetime.timedelta(weeks=10, days=0, hours=0, minutes=0)
start = datetime.date.today() - 5*past
end = datetime.date.today()
images = par.img_path+'c2-s11-'
#company = Stock(symbol, token=token).get_company()

#df = get_historical_data(symbol, output_format='pandas', start=start, end=end, token=token)
df = web.DataReader(symbol, source, start=start, end=end)
df.index = pd.to_datetime(df.index)
print(df.tail())

myblue='rgba(127,255,255,1)'
mypink='rgba(255,0,76,1)'

fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.05,row_heights=[2,1])
fig.add_scatter(x=df.index, y=df.Close, name="Prix de clôture",marker=dict(color=myblue),row=1,col=1)
fig.add_bar(x=df.index, y=df.Volume, name="Volume",marker=dict(color=mypink,line=dict(color=mypink)),row=2,col=1)
fig.update_layout(font=dict(color='white'), margin=dict(l=0,r=0,t=25,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title_text='Simple graph - '+company+' ('+symbol+')', yaxis_title="$")
fig.update_xaxes(gridcolor='rgba(.5,.5,.5,.5)')
fig.update_yaxes(gridcolor='rgba(.5,.5,.5,.5)')
plot(fig, filename=images+'1.html', config={'displayModeBar': False}, auto_open=False)

