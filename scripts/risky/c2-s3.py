import datetime
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import parameters as par
#from iexfinance.stocks import Stock
#from iexfinance.stocks import get_historical_data

symbol = 'FB'
company = 'Facebook'
past = datetime.timedelta(weeks=10, days=0, hours=0, minutes=0)
start = datetime.date.today() - 5*past
end = datetime.date.today()
images = par.img_path+'c2-s3-'
#company = Stock(symbol, token=par.token).get_company()
source='yahoo'

#df = get_historical_data(symbol, output_format='pandas', start=start, end=end, token=token, close_only=True)
df = web.DataReader(symbol, source, start=start, end=end)
df.index = pd.to_datetime(df.index)
print(df.tail())

fig = plt.figure()
plt.plot(df.Close, c='#7fffff')
plt.title(company+' ('+symbol+')')
plt.ylabel('$',rotation=0) 
fig.savefig(images+'1')
