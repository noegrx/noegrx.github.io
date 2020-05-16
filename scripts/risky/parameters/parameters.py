import matplotlib as mpl
import plotly.graph_objects as go

img_path="../../img/risky/"
txt_path="../../txt/risky/"

mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.labelcolor'] = 'white'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.edgecolor'] = 'white'
mpl.rcParams['figure.figsize'] = 9.6, 7
mpl.rcParams['font.size'] = 14
mpl.rcParams['grid.alpha'] = .3
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.edgecolor'] = '0.8'
mpl.rcParams['legend.loc'] = 'best'
mpl.rcParams['savefig.transparent'] = True
mpl.rcParams['savefig.format'] = 'svg'
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['text.color'] = 'white'
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['ytick.color'] = 'white'

layout=go.Layout(
    titlefont=dict(color='white'),
    legend=dict(font=dict(color='white')),
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(.5,.5,.5,.5)',
        linecolor='white',
        showticklabels=True,
        tickfont=dict(color='white')
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(.5,.5,.5,.5)',
        linecolor='white',
        showticklabels=True,
        tickfont=dict(color='white')
    ),
    autosize=True,
    margin=go.layout.Margin(
        l=0,
        r=0,
        b=0,
        t=24
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

config = {
    'displayModeBar': False,
    'scrollZoom': True,
    'showLink':False
}
