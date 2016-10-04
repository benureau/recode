from ipywidgets import widgets
import numpy as np
import colorsys

from bokeh import io as bkio
from bokeh import plotting as bkp
from bokeh.core.properties import value
from bokeh.models import FixedTicker
from bokeh.models.mappers import LinearColorMapper

def interact(*args, **kwargs):
    widgets.interact(*args, **kwargs)
IntSlider = widgets.IntSlider
FloatSlider = widgets.FloatSlider

    ## Disable autoscrolling

from IPython.display import display, Javascript

disable_js = """
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
"""
display(Javascript(disable_js))

    ## Load bokeh for jupyter

bkp.output_notebook(hide_banner=True)

    ## Routines for adjusting figures

def tweak_fig(fig, grid=False, tree_ticks_enable=False):
    tight_layout(fig)
    if tree_ticks_enable:
        three_ticks(fig)
    disable_minor_ticks(fig)
    if not grid:
        disable_grid(fig)
    fig.toolbar.logo = None

def tight_layout(fig):
    fig.min_border_top    = 35
    fig.min_border_bottom = 35
    fig.min_border_right  = 35
    fig.min_border_left   = 35

def three_ticks(fig):
    x_min, x_max = fig.x_range.start, fig.x_range.end
    y_min, y_max = fig.y_range.start, fig.y_range.end
    x_ticks = [x_min, 0, x_max]
    y_ticks = [x_min, 0, x_max]

    fig.xaxis[0].ticker=FixedTicker(ticks=x_ticks)
    fig.yaxis[0].ticker=FixedTicker(ticks=y_ticks)

def disable_minor_ticks(fig):
    fig.axis.major_label_text_font_size = value('8pt')
    fig.axis.minor_tick_line_color = None
    fig.axis.major_tick_in = 0

def disable_grid(fig):
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None


    ## Figure 1

def curve(xs, ys, ys_dash, title='Child growth curve', y_range=[-0.5, 9], **kwargs):
    """Plots a continuous curve and a dashed one for Fig. 1."""

    fig = bkp.figure(x_range=[-0.5, 25.5], y_range=y_range,
                     title=title, tools="",
                     plot_width=320, plot_height=250, **kwargs)
    tweak_fig(fig, tree_ticks_enable=False, grid=False)
    fig.title.text_font_size = '12pt'
    fig.title.align = 'center'

    line_dash = fig.line(xs, ys_dash, line_width=2, line_dash='dashed')
    line_ref  = fig.line(xs, ys, line_width=2)
    return fig, line_dash.data_source.data


def fig1(xs, g, P_sb, P_sc, I, R=8.4/2):
    """Compute y-values, and lay out Fig. 1 plots."""

    kwargs = {}

    ys_1a        = [g(x, 8.4) for x in xs]
    ys_1a_dashed = [g(x, R)   for x in xs]
    fig1a, data_1a = curve(xs, ys_1a, ys_1a_dashed, y_range=[-0.5, 15],
                           title='Child growth curve', **kwargs)
    fig1a.xaxis.axis_label = "Age (month)"
    fig1a.yaxis.axis_label = "Head radius R (cm)"

    ys_1b        = [P_sb(x, 8.4) for x in xs]
    ys_1b_dashed = [P_sb(x, R)   for x in xs]
    fig1b, data_1b = curve(xs, ys_1b, ys_1b_dashed, y_range=[-0.05, 1.05],
                           title='Birth survival curve', **kwargs)
    fig1b.xaxis.axis_label = "Birth Age T (month)"
    fig1b.yaxis.axis_label = "P(survive birth)"

    ys_1c        = [P_sc(x, I(8.4)) for x in xs]
    ys_1c_dashed = [P_sc(x, I(R))   for x in xs]
    fig1c, data_1c = curve(xs, ys_1c, ys_1c_dashed, y_range=[-0.05, 1.05],
                           title='Childhood survival curve', **kwargs)
    fig1c.xaxis.axis_label = "Time until maturity M (month)"
    fig1c.yaxis.axis_label = "P(survive childhood)"


    fig = bkp.gridplot([[fig1a, fig1b, fig1c]])
    # fig.toolbar.location = None

    handle = bkp.show(fig, notebook_handle=True)
    return handle, (data_1a, data_1b, data_1c)

def update_fig1(datas, g, P_sb, P_sc, r):
    """Refresh plots of Fig 1 for a new value of `r`."""
    handle, figdatas = datas
    for figdata, f in zip(figdatas, [g, P_sb, P_sc]):
        figdata['y'] = [f(x, r) for x in figdata['x']]
    bkio.push_notebook(handle=handle)



    ## Figure 2A

palette_20 = ['#d52b22', '#db2c23', '#e52e24', '#d62b22',
              '#ce2a21', '#ce3021', '#df4b23', '#d15b21',
              '#dc7423', '#e78d25', '#efa626', '#ddab23',
              '#fdda28', '#f4f127', '#e8f825', '#e6f625',
              '#e7f625', '#e1ec5d', '#f2fba0', '#f4f7db']

palette_20 = ([(3*i+198, int(i/19*(3*i+198)), 0) for i in range(20)]
             +[(255, 255, int(255*i/4)) for i in range(4)])
palette_20 = ["#%02x%02x%02x" % rgb for rgb in palette_20]

def fig2a(D, x_max=30, y_max=10, palette=palette_20,
          title='P(survive to adulthood)', show=True,
          plot_width=500, plot_height=400):
    fig = bkp.figure(x_range=(0, x_max), y_range=(0, y_max), webgl=True,
                     plot_width=plot_width, plot_height=plot_height,
                     tools='pan,wheel_zoom,reset,save', title=title)
    fig.title.text_font_size = '12pt'
    fig.title.align = 'center'
    tweak_fig(fig, tree_ticks_enable=False, grid=False)

    cmap = LinearColorMapper(high=1.0, low=0.0, palette=palette)
    Dmat = fig.image(image=[D], x=0, y=0, dw=x_max, dh=y_max, color_mapper=cmap)
    fig.xaxis.axis_label = "Birth age (month)"
    fig.yaxis.axis_label = "Brain size (radius, cm)"

    if show:
        handle = bkp.show(fig, notebook_handle=True)
    else:
        handle = None

    return handle, fig, Dmat.data_source.data

def update_fig2a(fig_data, D, gamma, V):
    """Refresh plots of Fig 1 for a new value of `r`."""
    print('recomputing fig2a...')
    handle, fig, data = fig_data
    data['image'] = [D]
    fig.title.text = 'P(survive to adulthood) gamma={}, V={}'.format(gamma, V)
    bkio.push_notebook(handle=handle)
    print('done.')

    ## Figure 2B

palette_2 = ['#dddddd', '#ffffff']

def line_color():
    """Return a random (bright) color"""
    r, g, b = colorsys.hsv_to_rgb(np.random.random(), 0.65, 0.95)
    return (255*r, 255*g, 255*b)

def fig2b(D, traces, x_max=30, y_max=10):
    handle, fig, data = fig2a(D, x_max=30, y_max=10, palette=palette_2, show=False,
                              plot_width=900, plot_height=720, title='Population dynamics')
    for trace in traces:
        color = line_color()
        R_0, T_0 = trace[0][:2]
        fig.circle([T_0], [R_0], radius=0.2, color=color)
        fig.line([RTI[1] for RTI in trace], [RTI[0] for RTI in trace],
                 line_width=3, line_join='bevel', color=color)
    bkp.show(fig)
