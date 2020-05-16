import datetime
import parameters as par
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

images = par.img_path+'c2-s18-'

value=170
minvalue=0
maxvalue=200
reference=125

fig = go.Figure(layout=par.layout)
fig.add_indicator(
    mode = "gauge+number+delta",
    value = value,
    domain = {'x': [0, 1], 'y': [0, 1]},
    delta = {'reference': reference, 'increasing': {'color': 'rgb(249,38,114)'}},
    gauge = {
        'axis': {'range': [None, maxvalue], 'tickwidth': 1, 'tickcolor': "white"},
        'bar': {'color': "white"},
        'bgcolor': "red",
        'steps': [
            {'range': [minvalue, maxvalue/2], 'color': 'green'},
            {'range': [maxvalue/2, maxvalue-(.2*maxvalue)], 'color': 'orange'}],
        'threshold': {
            'line': {'color': "white", 'width': 4},
            'thickness': 0.75,
            'value': 490}})

fig.update_layout(font = {'color': "white"})
plot(fig, filename=images+'1.html', config=par.config, auto_open=False)

print(value)
