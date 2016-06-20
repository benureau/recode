import math

import numpy as np
from bokeh import plotting as bkp
from bokeh.core.properties import value
from bokeh.models import FixedTicker

SIZE = 450


    ## Disable autoscrolling

from IPython.display import display, Javascript

disable_js = """
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
"""
display(Javascript(disable_js))



# load bokeh for jupyter
bkp.output_notebook(hide_banner=True)

#
def show(grid):
    """Display graphs, accepts a list of list of figures.

    The postures(), goals() and effects() functions display their
    graphs automatically unless the keyword argument `show` is set
    to False.
    """
    bkp.show(bkp.gridplot(grid))


def tweak_fig(fig):
    tight_layout(fig)
    three_ticks(fig)
    disable_minor_ticks(fig)
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
    if x_min == y_min and x_max == y_max:
        if x_min <= -10 < 10 <= x_max:
            x_ticks = [-10, 0, 10]
            y_ticks = [-10, 0, 10]
        elif x_min <= -2 < 2 <= x_max:
            x_ticks = [ -2, 0, 2]
            y_ticks = [ -2, 0, 2]
        else:
            x_ticks = [ -1, 0, 1]
            y_ticks = [ -1, 0, 1]

        fig.xaxis[0].ticker=FixedTicker(ticks=x_ticks)
        fig.yaxis[0].ticker=FixedTicker(ticks=y_ticks)

def disable_minor_ticks(fig):
    fig.axis.major_label_text_font_size = value('8pt')
    fig.axis.minor_tick_line_color = None
    fig.axis.major_tick_in = 0

def disable_grid(fig):
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None



    ## Display effect distribution ##

def spread(s_vectors, title='', fig=None,
           color='#1A60A3', alpha=0.5, radius=0.005,
           width=SIZE, height=SIZE,
           x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), **kwargs):

    if fig is None:
        fig = bkp.figure(x_range=x_range, y_range=y_range, title=title,
                         title_text_font_size='12pt', tools="pan,box_zoom,reset,save",
                         plot_width=width, plot_height=height,
                         **kwargs)
        tweak_fig(fig)

    xs, ys = np.array([e[0] for e in s_vectors]), np.array([e[1] for e in s_vectors])
    fig.circle([0], [0], radius=1.00, alpha=0.5, fill_color=None, color='#AAAAAA')
    fig.scatter(xs, ys, radius=radius, fill_color=color, fill_alpha=alpha, line_color=None)
    return fig

def effects(history, fig=None, show=False, **kwargs):
    s_vectors = [h[1] for h in history]
    fig = spread(s_vectors, fig=fig, **kwargs)
    if show:
        bkp.show(fig)
    return fig

def goals(s_vectors, fig=None, show=False, title='goal distribution', **kwargs):
    x_range = min(s[0] for s in s_vectors), max(s[0] for s in s_vectors)
    y_range = min(s[1] for s in s_vectors), max(s[1] for s in s_vectors)
    x_range = min(-1.0, x_range[0]), max(1.0, x_range[1])
    y_range = min(-1.0, y_range[0]), max(1.0, y_range[1])
    x_range = 1.1*x_range[0], 1.1*x_range[1]
    y_range = 1.1*y_range[0], 1.1*y_range[1]

    radius=0.005*((x_range[1]-1)/2+1)
    fig = spread(s_vectors, fig=fig, color='red', radius=radius,
                 x_range=x_range, y_range=y_range, title=title, **kwargs)
    if show:
        bkp.show(fig)
    return fig


    ## Display arm posture ##

def posture(arm, angles, title='', fig=None,
            color='#333333', alpha=0.5, radius_factor=1.0,
            width=SIZE, height=SIZE,
            x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), **kwargs):

    if fig is None:
        fig = bkp.figure(x_range=x_range, y_range=y_range, title=title,
                         title_text_font_size='10pt', tools="pan,box_zoom,reset,save",
                         plot_width=width, plot_height=height,
                         **kwargs)
        tweak_fig(fig)

    arm.execute(angles)
    xs = [p[0] for p in arm.posture]
    ys = [p[1] for p in arm.posture]

    kwargs.update({'line_color'  : color,
                   'line_alpha'  : alpha,
                   'fill_color'  : color,
                   'fill_alpha'  : alpha,
                  })

    fig.line(xs, ys, line_width=radius_factor, line_color=color, line_alpha=alpha)

    fig.circle(xs[  : 1], ys[  : 1], radius=radius_factor*0.015, **kwargs)
    fig.circle(xs[ 1:-1], ys[ 1:-1], radius=radius_factor*0.008, **kwargs)
    fig.circle(xs[-1:  ], ys[-1:  ], radius=radius_factor*0.01, alpha=1.0)

    return fig

def postures(arm, angles_list, fig=None, show=True, disk=False, **kwargs):
    for angles in angles_list:
        fig = posture(arm, angles, fig=fig, **kwargs)
    if disk and fig is not None:
        fig.circle([0], [0], radius=1.0, alpha=0.5, fill_color=None, color='#AAAAAA')
    if show:
        bkp.show(fig)
    else:
        return fig
