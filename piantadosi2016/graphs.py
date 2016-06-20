from ipywidgets import widgets
import numpy as np

from bokeh import io as bkio
from bokeh import plotting as bkp
from bokeh.core.properties import value
from bokeh.models import FixedTicker


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
    fig.logo = None

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

    line_dash = fig.line(xs, ys_dash, line_width=2, line_dash='dashed')
    line_ref    = fig.line(xs, ys, line_width=2)
    return fig, line_dash.data_source.data


def fig1(xs, g, P_sc, P_sa, r=8.4/2):
    """Compute y-values, and lay out Fig. 1 plots."""

    kwargs = {'title_text_font_size': '12pt'}

    ys_1a        = [g(x, 8.4) for x in xs]
    ys_1a_dashed = [g(x, r)   for x in xs]
    fig1a, data_1a = curve(xs, ys_1a, ys_1a_dashed, y_range=[-0.5, 15],
                           title='Child growth curve', **kwargs)

    ys_1b        = [P_sc(x, 8.4) for x in xs]
    ys_1b_dashed = [P_sc(x, r)   for x in xs]
    fig1b, data_1b = curve(xs, ys_1b, ys_1b_dashed, y_range=[-0.05, 1.05],
                           title='Birth survival curve', **kwargs)

    ys_1c        = [P_sa(x, 8.4) for x in xs]
    ys_1c_dashed = [P_sa(x, r)   for x in xs]
    fig1c, data_1c = curve(xs, ys_1c, ys_1c_dashed, y_range=[-0.05, 1.05],
                           title='Childhood survival curve', **kwargs)

    fig = bkp.gridplot([[fig1a, fig1b, fig1c]])
    fig.toolbar_location = None

    bkp.show(fig)
    return data_1a, data_1b, data_1c

def update_fig1(datas, g, P_sc, P_sa, r):
    """Refresh plots of Fig 1 for a new value of `r`."""
    for data, f in zip(datas, [g, P_sc, P_sa]):
        data['y'] = [f(x, r) for x in data['x']]
    bkio.push_notebook()
