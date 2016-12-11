from ipywidgets import widgets
import numpy as np
import colorsys

from bokeh import io as bkio
from bokeh import plotting as bkp
from bokeh import models as bkm

from bokeh.core.properties import value
from bokeh.models import FixedTicker, LinearAxis, Range1d
from bokeh.models.mappers import LinearColorMapper
from bokeh.plotting import ColumnDataSource
from bokeh.models import HoverTool

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

def fig1a(xs, y1s, y1s_dash, y2s=None, y2s_dash=None, y1_range=None, y2_range=None, title='',
          fig=None, **kwargs):
    """Plots a continuous curve and a dashed one for Fig. 1."""

    if fig is None:
        fig = bkp.figure(y_range=y1_range,
                         title=title, tools="",
                         plot_width=600, plot_height=325, **kwargs)
    tweak_fig(fig, tree_ticks_enable=False, grid=False)
    fig.title.text_font_size = '12pt'
    fig.title.align = 'center'

    fig.line(xs, y1s, line_width=4, color='grey', line_alpha=0.5)
    fig.line(xs, y1s_dash, line_width=1, line_dash='dashed', color='blue')

    if y2s is not None:
        fig.extra_y_ranges = {"y2": Range1d(start=y2_range[0], end=y2_range[1])}
        fig.add_layout(LinearAxis(y_range_name="y2"), 'right')

        fig.line(xs, y2s, line_width=4, color='grey', line_alpha=0.5, y_range_name="y2")
        fig.line(xs, y2s_dash, line_width=1, line_dash='dashed', color='red', y_range_name="y2")

    bkp.show(fig)

def fig1b(xs, y_abs, y_relative=None, title='',
          fig=None, **kwargs):
    """Plots a continuous curve and a dashed one for Fig. 1."""

    if fig is None:
        fig = bkp.figure(title=title, tools="",
                         plot_width=600, plot_height=325, **kwargs)
    tweak_fig(fig, tree_ticks_enable=False, grid=False)
    fig.title.text_font_size = '12pt'
    fig.title.align = 'center'

    fig.line(xs, y_abs, line_width=1, color='blue')
    if y_relative is not None:
        fig.line(xs, y_relative, line_width=2, color='red')

    bkp.show(fig)



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
