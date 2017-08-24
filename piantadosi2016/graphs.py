from ipywidgets import widgets
import numpy as np
import colorsys

from bokeh import io as bkio
from bokeh import plotting as bkp
from bokeh import models as bkm

from bokeh.core.properties import value
from bokeh.models import FixedTicker
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

def tweak_fig(fig, grid=False, three_ticks_enable=False):
    tight_layout(fig)
    if three_ticks_enable:
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
    tweak_fig(fig, three_ticks_enable=False, grid=False)
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
    #fig.logo = None
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
          plot_width=500, plot_height=400, tools=''):
    fig = bkp.figure(x_range=(0, x_max), y_range=(0, y_max),
                     plot_width=plot_width, plot_height=plot_height,
                     tools=tools, title=title)
    fig.title.text_font_size = '12pt'
    fig.title.align = 'center'
    tweak_fig(fig, three_ticks_enable=False, grid=False)

    cmap = LinearColorMapper(high=1.0, low=0.0, palette=palette)
    Dmat = fig.image(image=[D], x=0, y=0, dw=x_max, dh=y_max, color_mapper=cmap)
    fig.xaxis.axis_label = "Birth age (month)"
    fig.yaxis.axis_label = "Brain size (radius, cm)"

    # color_bar = bkm.ColorBar(color_mapper=cmap, label_standoff=12, location=(0,0),
    #                          border_line_color=None, bar_line_color='black', major_tick_line_color='black')
    #
    # fig.add_layout(color_bar, 'right')

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
    return '#{0:02x}{1:02x}{2:02x}'.format(int(255*r), int(255*g), int(255*b))


def fig2b(traces, D, x_max=30, y_max=10):
    return fig2b_aux(traces, D, x_max=30, y_max=10, axes=('T', 'R'),
                     line_width=3, radius=0.2)

def intelligence_radius_fig(traces):
    return fig2b_aux(traces, None, x_max=11, y_max=10, axes=('I', 'R'),
                     line_width=1, radius=0.05, display_D=False,
                     display_start=False, display_end=True)

def fig2b_aux(traces, D, x_max=30, y_max=10, axes=('T', 'R'),  line_width=3, radius=0.2,
              display_D=True, display_start=True, display_end=False):

    x_idx = ('R', 'T', 'I').index(axes[0])
    y_idx = ('R', 'T', 'I').index(axes[1])

    if display_D:
        handle, fig, data = fig2a(D, x_max=x_max, y_max=y_max, palette=palette_2, show=False,
                                  plot_width=900, plot_height=720, title='Population dynamics',
                                  tools='pan,wheel_zoom,reset,save')
    else:
        fig = bkp.figure(x_range=(0, x_max), y_range=(0, y_max),
                         plot_width=660, plot_height=600,
                         tools='pan,wheel_zoom,reset,save', title='Intelligence/Brain radius relationship')
        tweak_fig(fig, three_ticks_enable=False, grid=False)

    fig.xaxis.axis_label = ["Brain size (radius, cm)", "Birth age (month)", "Intelligence"][x_idx]
    fig.yaxis.axis_label = ["Brain size (radius, cm)", "Birth age (month)", "Intelligence"][y_idx]

    colors = []
    for trace in traces:
        color = line_color()
        colors.append(color)
        fig.line([RTI[x_idx] for RTI in trace], [RTI[y_idx] for RTI in trace],
                 line_width=line_width, line_join='bevel', color=color)

    source = ColumnDataSource(
        data=dict(
            x=[trace[0][x_idx] for trace in traces],
            y=[trace[0][y_idx] for trace in traces],
            x_end=[trace[-1][x_idx] for trace in traces],
            y_end=[trace[-1][y_idx] for trace in traces],
            R_start=['{:.2f}'.format(trace[0][0]) for trace in traces],
            T_start=['{:.2f}'.format(trace[0][1]) for trace in traces],
            I_start=['{:.2f}'.format(trace[0][2]) for trace in traces],
            R_end=['{:.2f}'.format(trace[-1][0]) for trace in traces],
            T_end=['{:.2f}'.format(trace[-1][1]) for trace in traces],
            I_end=['{:.2f}'.format(trace[-1][2]) for trace in traces],
            radius=[radius]*len(traces),
            color=colors,
            size=[10]*len(traces),
        )
    )
    if display_start:
        # radius=radius, color=colors
        g_r = fig.circle('x', 'y', color='color', radius='radius', size='size', source=source)
    if display_end:
        g_r = fig.square('x_end', 'y_end', size='size', color='color', source=source)

    g_hover = bkm.HoverTool(renderers=[g_r],
                            tooltips=[("R_start", "@R_start"),
                                      ("R_end",   "@R_end"),
                                      ("T_start", "@T_start"),
                                      ("T_end",   "@T_end"),
                                      ("I_start", "@I_start"),
                                      ("I_end",   "@I_end"),
                                     ])
    fig.add_tools(g_hover)
    bkp.show(fig)
