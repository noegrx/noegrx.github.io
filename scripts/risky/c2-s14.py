import datetime
import parameters as par
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

images = par.img_path+'c2-s14-'

labels = ['Google','Apple','Facebook','Amazon']
values = [.2,.3,.4,.1]
print(values)

fig = go.Figure(layout=par.layout)
fig.add_pie(labels=labels, values=values)
fig.update_layout(title_text='Pie chart', showlegend=True)
plot(fig, filename=images+'1.html', config=par.config, auto_open=False)
