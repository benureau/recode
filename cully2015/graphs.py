import math

import numpy as np
from bokeh import plotting as bkp
from bokeh.models import PrintfTickFormatter


    ## Disable autoscrolling

from IPython.display import display, Javascript

disable_js = """
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
"""
display(Javascript(disable_js))




bkp.output_notebook(hide_banner=True)

# sns.color_palette("PuBu", 15)

SIZE = 450
BLUE_COLORS = ((245, 238, 246),
               (235, 230, 241),
               (221, 219, 235),
               (207, 208, 229),
               (186, 198, 224),
               (165, 188, 218),
               (140, 178, 212),
               (115, 168, 206),
               ( 83, 156, 199),
               ( 53, 143, 191),
               ( 28, 127, 183),
               (  4, 111, 175),
               (  4, 100, 157),
               (  3,  89, 139),
               (  2,  72, 112))

RED_COLORS = ((255, 245, 181),
              (254, 236, 159),
              (254, 226, 138),
              (254, 216, 117),
              (254, 197,  96),
              (253, 177,  75),
              (253, 158,  67),
              (252, 140,  59),
              (252, 108,  50),
              (251,  76,  41),
              (238,  50,  34),
              (226,  25,  28),
              (207,  12,  33),
              (187,   0,  38),
              (156,   0,  38))


    ## Graphs

def colorbar(colors, inversed=False):
    img = np.zeros((len(colors)-1, 1), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape(img.shape + (4,))

    for i, c in enumerate(colors[:-1]):
        r, g, b = c
        view[i, 0, 0] = r
        view[i, 0, 1] = g
        view[i, 0, 2] = b
        view[i, 0, 3] = 255

    if inversed:
        img = img[::-1]

    return img

def variance_map(perf_map, res, colors=RED_COLORS, title='performance map (angle variance)', **kwargs):
    plot_map(perf_map, res, colors=colors, title=title, scale='log', **kwargs)

def distance_map(perf_map, res, colors=BLUE_COLORS, title='performance map on the intact arm (distance to target)', **kwargs):
    plot_map(perf_map, res, colors=colors, title=title, **kwargs)

EPSILON = 1e-10 # HACK for log scale

def plot_map(perf_map, res, title='performance map', colors=BLUE_COLORS, show=True, scale='default'):
    ps = list(perf_map.values())
    p_min, p_max = np.min(ps), np.max(ps)
    if scale == 'log':
        c_min, c_max = -math.log(-p_min+EPSILON), -math.log(-p_max+EPSILON)
    else:
        c_min, c_max = p_min, p_max

    img = np.zeros((res, res), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape(img.shape + (4,))

    for (i, j), p in perf_map.items():
        if scale == 'log':
            p = -math.log(-p+EPSILON)
        c_idx = int(np.floor((len(colors)-1)*(p - c_min)/(c_max-c_min)))
        r, g, b = colors[c_idx]
        view[j, i, 0] = r
        view[j, i, 1] = g
        view[j, i, 2] = b
        view[j, i, 3] = 255

    plot = bkp.figure(width=SIZE, height=SIZE, x_range=(-0.7, 0.7), y_range=(-0.7, 0.7),
                           title=title, tools = "pan,box_zoom,reset,save")
    plot.title.text_font_size = '12pt'
    plot.image_rgba([img], x=[-0.7],  y=[-0.7], dh=[1.4], dw=[1.4])

    if scale == 'log':
        cbar = colorbar(colors, inversed=True)
        cb_plot = bkp.figure(width=100, height=SIZE, x_range=(0, 1.0), y_axis_type="log", y_range=(-p_max, -p_min))
        cb_plot.image_rgba([cbar], x=[0.0],  y=[-p_max], dw=[1.0], dh=[p_max-p_min])
    else:
        cbar = colorbar(colors)
        cb_plot = bkp.figure(width=100, height=SIZE, x_range=(0, 1.0), y_range=(p_min, p_max))
        cb_plot.image_rgba([cbar], x=[0.0],  y=[p_min], dw=[1.0], dh=[p_max-p_min])

    cb_plot.min_border_right = 25
    cb_plot.xgrid.grid_line_color = None
    cb_plot.xaxis.minor_tick_line_color = None
    cb_plot.xaxis.major_tick_line_color = None
    cb_plot.xaxis.axis_line_color = None
    cb_plot.xaxis[0].formatter = PrintfTickFormatter(format="")
    if scale == 'log':
        cb_plot.yaxis.formatter = PrintfTickFormatter(format="-%1.0e")

    if show:
        bkp.show(bkp.gridplot([[plot, cb_plot]]))
    return [plot, cb_plot]


def plot_maps(perf_maps, res, damage, target, colors=BLUE_COLORS, title='', show=True):
    p_min, p_max = float('inf'), float('-inf')
    for p_map, _ in perf_maps:
        ps = list(p_map.values())
        p_min, p_max = min(p_min, np.min(ps)), max(p_max, np.max(ps))

    cbar = colorbar(colors).T
    c_min, c_max = p_min, p_max
    if p_max + 0.001 > 0: c_max = 0.0 # HACK

    cb_plot = bkp.figure(width=SIZE, height=120, x_range=(p_min, p_max), y_range=(0, 1),
                         title='aquisition function (mu + kappa*sigma^2)',
                         tools="pan,box_zoom,reset,save")
    cb_plot.title.text_font_size = '12pt'
    cb_plot.image_rgba([cbar], x=[p_min],  y=[0.0], dh=[1.0], dw=[p_max-p_min])
    cb_plot.ygrid.grid_line_color = None
    cb_plot.yaxis.minor_tick_line_color = None
    cb_plot.yaxis.major_tick_line_color = None
    cb_plot.yaxis.axis_line_color = None
    cb_plot.yaxis[0].formatter = PrintfTickFormatter(format="")

    plots = [[cb_plot], []]
    for k, (p_map_k, data_k) in enumerate(perf_maps):
        ctrl, _, perf = data_k

        img = np.zeros((res, res), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape(img.shape + (4,))

        for (i, j), p in p_map_k.items():
            c_idx = int(np.floor((len(colors)-1)*(p - p_min)/(p_max-p_min)))
            r, g, b = colors[c_idx]
            view[j, i, 0] = r
            view[j, i, 1] = g
            view[j, i, 2] = b
            view[j, i, 3] = 255

        title = '{}. {:4.1f} cm to target'.format(k+1, -100*perf)
        if k == len(perf_maps) - 1:
            title += ': done!'
        plot = bkp.figure(width=450, height=450, x_range=(-0.7,0.7), y_range=(-0.7,0.7),
                          title=title, tools="pan,box_zoom,reset,save")
        plot.title.text_font_size = '12pt'
        plot.image_rgba([img], x=[-0.7],  y=[-0.7],
                              dh=[1.4], dw=[1.4])

        # next behavior
        if k < len(perf_maps) - 1:
            _, data_k1 = perf_maps[k+1]
            _, behavior_k1, _ = data_k1
            plot.circle([behavior_k1[0]], [behavior_k1[1]], radius=0.01,
                        fill_color='#FFFFFF', fill_alpha=0.75, line_color='#000000')
            #plot.line([behavior[0], behavior_k1[0]], [behavior[1], behavior_k1[1]], line_color='white', line_alpha=0.5)

        # target
        plot.circle([target[0]], [target[1]], radius=0.05,
                    line_color='red', line_alpha=0.5, fill_color='red', fill_alpha=0.25)

        # arms
        arm_intact(ctrl, fig=plot, color='#000000')
        arm_broken(ctrl, damage=damage, fig=plot, color='red')

        if len(plots[-1]) >= 2:
            plots.append([])
        plots[-1].append(plot)
    if show:
        bkp.show(bkp.gridplot(plots))
#    return plot



def arm(angles, title='posture graphs', fig=None,
        color='#666666', alpha=0.5, radius_factor=1.0,
        x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), **kwargs):

    if fig is None:
        fig = bkp.figure(x_range=x_range, y_range=y_range, title=title, **kwargs)
        fig.title.text_font_size = '6pt'


    xs, ys, sum_a, length = [0], [0], 0, 0.62/len(angles)

    for a in angles:
        sum_a += a
        xs.append(xs[-1] + length*math.sin(sum_a))
        ys.append(ys[-1] + length*math.cos(sum_a))

        kwargs.update({'line_color'  : color,
                       'line_alpha'  : alpha,
                       'fill_color'  : color,
                       'fill_alpha'  : alpha,
                      })

        fig.line(xs, ys, line_width=radius_factor, line_color=color, line_alpha=alpha)

    #ys, xs = xs, ys
    fig.circle(xs[  : 1], ys[  : 1], radius=radius_factor*0.015, **kwargs)
    fig.circle(xs[ 1:-1], ys[ 1:-1], radius=radius_factor*0.008, **kwargs)
    fig.circle(xs[-1:  ], ys[-1:  ], radius=radius_factor*0.01, color=color, alpha=alpha)


    return fig

def arm_intact(ctrl, **kwargs):
    angles = [2*math.pi/2*(a-0.5) for a in ctrl]
    return arm(angles, **kwargs)


damages = [((5,), ()), ((4,), ()), ((3,), ()), ((2,), ()), # stuck only
           ((), (5,)), ((), (4,)), ((), (3,)), ((), (2,)), # offset only
           ((2,), (5,)), ((2,), (4,)), ((2,), (3,)),       # stuck then offset
           ((5,), (2,)), ((4,), (2,)), ((3,), (2,))]       # offset then stuck

def arm_broken(ctrl, damage, **kwargs):
    angles = [2*math.pi/2*(a-0.5) for a in ctrl]
    stuck, offset = damage
    for i, a in stuck.items():
        angles[i]  = math.radians(a) # joint stuck at 45 degrees
    for j, a in offset.items():
        angles[j] += math.radians(a) # permanent joint offset of 45 degrees
    return arm(angles, **kwargs)
