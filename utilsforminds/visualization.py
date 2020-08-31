import matplotlib
import pandas as pd
from sys import platform
if platform.lower() != 'darwin' and 'win' in platform.lower():
    matplotlib.use('TkAgg')
else:
    matplotlib.use("MacOSX")
# matplotlib.pyplot.set_cmap('Paired')
import matplotlib.pyplot as plt

# plt.set_cmap('Paired')

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axis3d import Axis

import os
import utilsforminds.helpers as helpers
import random
import numpy as np
import utilsforminds
# from mayavi import mlab # see install manual + brew install vtk
# from mayavi.api import Engine
# import mayavi.tools.pipeline
from scipy import ndimage
import moviepy.editor as mpy
from copy import deepcopy
import tikzplotlib
from itertools import product, combinations

import plotly.graph_objs as graph_objs
import plotly.tools as tls

axis_rotation_dict = {0: 0, 1: 0, 2: 0}
axis_name_dict = {0: 'East', 1: 'North', 2: 'Elevation'}

def savePlotLstOfLsts(lstOfLsts, labelsLst, xlabel, ylabel, title, directory, save_tikz = True):
    '''
        plot multiple lists
    '''
    for valueLst, label in zip(lstOfLsts, labelsLst):
        plt.plot(valueLst, label = label)
    # plt.plot(lst1, label = label1)
    # plt.plot(lst2, label = label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(directory)
    if save_tikz:
        tikzplotlib.save(format_path_extension(directory, '.tex'))

def plot2Ds(planeLst, titleLst, filePath, cbarLabel = 'amount', plotShape = [3, 1], subplot_length = 16., subplot_ratio = (1., 1.), planeMaskLst = None, axis = 2, axisInfo = None, vmin_vmax = None, method = 'imshow', convertXYaxis = False, rotate = 0, specific_value_color_dict = {"value": 0., "color": "white"}, label_font_size = 25, label_positions = [0., 1/3, 2/3], title_font_size = 25, cbar_font_size = 25, save_tikz = True):
    '''
        If you want to provide mask matrix for scatter visualization, sequence should be (original matrix, recovered matrix) or (original matrix, recovered matrix, sparse matrix)

        Parameters
        ----------
        subplot_ration : array-like
            horizontal, vertical size ration of each subplot.
    '''

    assert(len(planeLst) == plotShape[0])
    if filePath.count('/') <= 2:
        print(f"Warning: May be wrong file path: {filePath}")
    else:
        filePath_ = filePath
    
    # label_positions = [0/4, 1/4, 2/4, 3/4, 4/4]
    labels_colorbar_margin_size = 2.5 * (label_font_size / 25.)
    nPlots = len(planeLst)
    whole_figure_size = [subplot_length * subplot_ratio[0] / (subplot_ratio[0] + subplot_ratio[1]) + labels_colorbar_margin_size, subplot_length * subplot_ratio[1] / (subplot_ratio[0] + subplot_ratio[1])]
    whole_figure_size[1] *= nPlots ## increase horizontal length
    fig = plt.figure(figsize = whole_figure_size)

    ## adjust colorbar
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    if vmin_vmax is None:
        vmax_ = planeLst[0].max()
        vmin_ = planeLst[0].min()
    else:
        assert(len(vmin_vmax) == 2)
        vmax_ = vmin_vmax[1]
        vmin_ = vmin_vmax[0]

    assert(axis in (0, 1, 2) and method in ('imshow', 'contour', 'scatter'))

    shape_ = planeLst[0].shape
    for i in range(nPlots):
        assert(planeLst[i].shape == shape_)
    
    plotPlaneLst = []
    plotPlaneMaskLst = []
    for i in range(nPlots):
        # plotPlaneLst.append(np.copy(planeLst[i]))
        plotPlaneLst.append(np.transpose(planeLst[i]))
    if planeMaskLst is not None:
        for i in range(len(planeMaskLst)):
            # plotPlaneMaskLst.append(np.copy(planeMaskLst[i]))
            plotPlaneMaskLst.append(np.transpose(planeMaskLst[i]))
    
    # Set White color for unobserved points
    if specific_value_color_dict is not None:
        current_cmap = plt.get_cmap()
        current_cmap_copied = deepcopy(plt.get_cmap())
        current_cmap.set_bad(color= specific_value_color_dict["color"])
        for i in range(nPlots):
            plotPlaneLst[i] = np.ma.masked_equal(plotPlaneLst[i], specific_value_color_dict["value"])
    
    # if planeMaskLst is not None:
    #     for i in range(len(plotPlaneMaskLst)):
    #         plotPlaneLst[i] = plotPlaneLst[i] * np.where(plotPlaneMaskLst[i] >= 1., 1., 0.)

    # if planeMaskLst is not None:
    #     for i in range(len(plotPlaneMaskLst)):
    #         plotPlaneLst[i] = np.where(plotPlaneMaskLst[i] >= 1., plotPlaneLst[i], np.nan)

    if rotate % 360 != 0:
        for i in range(nPlots):
            plotPlaneLst[i] = ndimage.rotate(plotPlaneLst[i], rotate)
        if planeMaskLst is not None:
            for i in range(len(planeMaskLst)):
                plotPlaneMaskLst[i] = ndimage.rotate(plotPlaneMaskLst[i], rotate)
        shape_ = plotPlaneLst[0].shape

    horiLabelIdc = [min(round(shape_[0] * position_proportion), shape_[0] - 1) for position_proportion in label_positions]
    vertLabelIdc = [min(round(shape_[1] - shape_[1] * position_proportion), shape_[1] - 1) for position_proportion in label_positions]

    if axisInfo is None:
        if method == 'imshow':
            horiLabels = utilsforminds.helpers.reverseLst(horiLabelIdc) ## ??????????????????
            vertLabels = vertLabelIdc
        elif method == 'contour' or method == 'scatter':
            horiLabels = horiLabelIdc
            vertLabels = vertLabelIdc
    else:
        horiAxis, vertAxis = get_xy_axis_from_z(axis)
        horiLabels = [round(axisInfo[horiAxis]["min"] + (axisInfo[horiAxis]["max"] - axisInfo[horiAxis]["min"]) * position_proportion) for position_proportion in label_positions]
        vertLabels = [round(axisInfo[vertAxis]["min"] + (axisInfo[vertAxis]["max"] - axisInfo[vertAxis]["min"]) * position_proportion) for position_proportion in label_positions]
    
    axis_label_names_dict = {0: "East(m)", 1: "North(m)", 2: "Elevation(m)"}
    xlabel = axis_label_names_dict[get_xy_axis_from_z(axis)[0]]
    ylabel = axis_label_names_dict[get_xy_axis_from_z(axis)[1]]

    if convertXYaxis:
        tmp = vertLabels
        vertLabels = horiLabels
        horiLabels = tmp
        tmp = xlabel
        xlabel = ylabel
        ylabel = tmp
    
    if method == 'imshow' or method == 'contour':
        for i in range(nPlots):
            plt.subplot(*(plotShape + [i + 1]))
            plt.title(titleLst[i], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels, fontsize = label_font_size)
            plt.yticks(vertLabelIdc, vertLabels, fontsize = label_font_size)

            if method == 'imshow':
                if vmin_vmax is not None:
                    img = plt.imshow(plotPlaneLst[i], vmin = vmin_, vmax = vmax_, aspect = 'auto')
                else:
                    img = plt.imshow(plotPlaneLst[i], aspect = 'auto')
                cbarInst = plt.colorbar(fraction=0.046 * subplot_ratio[1] * nPlots / subplot_ratio[0], pad=0.04, aspect= 10 * subplot_ratio[1] * nPlots / subplot_ratio[0])
                cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
                cbarInst.ax.tick_params(labelsize= cbar_font_size)
            elif method == 'contour':
                if vmin_vmax is not None:
                    img = plt.contour(np.flipud(plotPlaneLst[i]), vmin = vmin_, vmax = vmax_, linewidths = 2.0, colors = 'black', levels = [vmin_, vmin_ + (vmax_ - vmin_) * 1/8, vmin_ + (vmax_ - vmin_) * 2/8, vmin_ + (vmax_ - vmin_) * 3/8, vmin_ + (vmax_ - vmin_) * 4/8, vmin_ + (vmax_ - vmin_) * 5/8, vmin_ + (vmax_ - vmin_) * 6/8, vmin_ + (vmax_ - vmin_) * 7/8, vmax_])
                else:
                    img = plt.contour(np.flipud(plotPlaneLst[i]), linewidths = 2.0, colors = 'black', levels = [vmin_, vmin_ + (vmax_ - vmin_) * 1/8, vmin_ + (vmax_ - vmin_) * 2/8, vmin_ + (vmax_ - vmin_) * 3/8, vmin_ + (vmax_ - vmin_) * 4/8, vmin_ + (vmax_ - vmin_) * 5/8, vmin_ + (vmax_ - vmin_) * 6/8, vmin_ + (vmax_ - vmin_) * 7/8, vmax_])
    
    elif method == 'scatter':
        assert(nPlots == 2 or nPlots ==3)
        pointSize = 5.0 * (80 * 80 / (shape_[0] * shape_[1])) ** 0.5
        if nPlots == 2: # original, recovered
            plt.subplot(*(plotShape + [1]))
            plt.title(titleLst[0], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels, fontsize = label_font_size)
            plt.yticks(vertLabelIdc, vertLabels, fontsize = label_font_size)

            if planeMaskLst is not None:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneMaskLst[0] >= 1., 1., 0.))
                xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneMaskLst[0] < 1., 1., 0.))
            else:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneLst[0] > 1e-8, 1., 0.))
                xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneLst[0] > 1e-8, 0., 1.))
            minArr = np.ones(shape_) * vmin_

            if vmin_vmax is not None:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], vmin = vmin_, vmax = vmax_, marker = 'x', s = pointSize) # param s = 5.0 sets size of dots for 150 * 150 * 150 mapping
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], vmin=vmin_, vmax=vmax_, marker = 'o', s = pointSize)
            
            else:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x') # param s = 5.0 sets size of dots for 150 * 150 * 150 mapping
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], marker = 'o', s = pointSize)
            
            cbarInst = plt.colorbar(fraction=0.046 * subplot_ratio[1] * nPlots / subplot_ratio[0], pad=0.04, aspect= 10 * subplot_ratio[1] * nPlots / subplot_ratio[0])
            # cbarInst = plt.colorbar(cax = cax)
            cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
            cbarInst.ax.tick_params(labelsize= cbar_font_size)


            plt.subplot(*(plotShape + [2]))
            plt.title(titleLst[1], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels, fontsize = label_font_size)
            plt.yticks(vertLabelIdc, vertLabels, fontsize = label_font_size)

            # xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneLst[0] >= 1., 1., 0.))
            if planeMaskLst is not None:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneMaskLst[0] >= 1., 1., 0.))
                xCounterOnlyRecovered, yCounterOnlyRecovered = np.nonzero(np.where(plotPlaneMaskLst[0] < 1., np.where(plotPlaneMaskLst[1] >= 1., 1., 0.), 0.))
                xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneMaskLst[1] < 1., np.where(plotPlaneMaskLst[0] < 1., 1., 0.), 0.))
            else:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneLst[0] > 1e-8, 1., 0.))
                xCounterOnlyRecovered, yCounterOnlyRecovered = np.nonzero(np.where(plotPlaneLst[0] <= 1e-8, np.where(plotPlaneLst[1] > 1e-8, 1., 0.), 0.))
                xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneLst[1] <= 1e-8, np.where(plotPlaneLst[0] <= 1e-8, 1., 0.), 0.))
            minArr = np.ones(shape_) * vmin_

            if vmin_vmax is not None:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], vmin=vmin_, vmax=vmax_, marker = 'x', s = pointSize)
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], vmin=vmin_, vmax=vmax_, marker = 'o', s = pointSize)
                img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = plotPlaneLst[1][xCounterOnlyRecovered, yCounterOnlyRecovered], vmin=vmin_, vmax=vmax_, marker = '^', s = pointSize)

            else:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x')
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], marker = 'o')
                img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = plotPlaneLst[1][xCounterOnlyRecovered, yCounterOnlyRecovered], marker = '^')

            cbarInst = plt.colorbar(fraction=0.046 * subplot_ratio[1] * nPlots / subplot_ratio[0], pad=0.04, aspect= 10 * subplot_ratio[1] * nPlots / subplot_ratio[0])
            # cbarInst = plt.colorbar(cax = cax)
            cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
            cbarInst.ax.tick_params(labelsize= cbar_font_size)

        else: # nPlots == 3
            plt.subplot(*(plotShape + [1]))
            plt.title(titleLst[0], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels, fontsize = label_font_size)
            plt.yticks(vertLabelIdc, vertLabels, fontsize = label_font_size)

            if planeMaskLst is not None:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneMaskLst[0] >= 1., 1., 0.))
                xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneMaskLst[0] < 1., 1., 0.))
            else:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneLst[0] > 1e-8, 1., 0.))
                xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneLst[0] > 1e-8, 0., 1.))
            minArr = np.ones(shape_) * vmin_

            if vmin_vmax is not None:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], vmin = vmin_, vmax = vmax_, marker = 'x', s = 3.0) # param s = 5.0 sets size of dots for 150 * 150 * 150 mapping
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], vmin=vmin_, vmax=vmax_, marker = 'o', s = 3.0)
            
            else:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x') # param s = 5.0 sets size of dots for 150 * 150 * 150 mapping
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], marker = 'o')
            
            cbarInst = plt.colorbar(fraction=0.046 * subplot_ratio[1] * nPlots / subplot_ratio[0], pad=0.04, aspect= 10 * subplot_ratio[1] * nPlots / subplot_ratio[0])
            # cbarInst = plt.colorbar(cax = cax)
            cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
            cbarInst.ax.tick_params(labelsize= cbar_font_size)

            plt.subplot(*(plotShape + [2]))
            plt.title(titleLst[1], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels, fontsize = label_font_size)
            plt.yticks(vertLabelIdc, vertLabels, fontsize = label_font_size)

            # xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneLst[0] >= 1., 1., 0.))
            if planeMaskLst is not None:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneMaskLst[0] >= 1., 1., 0.))
                xCounterOnlyRecovered, yCounterOnlyRecovered = np.nonzero(np.where(plotPlaneMaskLst[0] < 1., np.where(plotPlaneMaskLst[1] >= 1., 1., 0.), 0.))
                xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneMaskLst[1] < 1., np.where(plotPlaneMaskLst[0] < 1., 1., 0.), 0.))
            else:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneLst[0] > 1e-8, 1., 0.))
                xCounterOnlyRecovered, yCounterOnlyRecovered = np.nonzero(np.where(plotPlaneLst[0] <= 1e-8, np.where(plotPlaneLst[1] > 1e-8, 1., 0.), 0.))
                xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneLst[1] <= 1e-8, np.where(plotPlaneLst[0] <= 1e-8, 1., 0.), 0.))
            minArr = np.ones(shape_) * vmin_

            if vmin_vmax is not None:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], vmin=vmin_, vmax=vmax_, marker = 'x', s = pointSize)
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], vmin=vmin_, vmax=vmax_, marker = 'o', s = pointSize)
                img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = plotPlaneLst[1][xCounterOnlyRecovered, yCounterOnlyRecovered], vmin=vmin_, vmax=vmax_, marker = '^', s = pointSize)

            else:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x', s = pointSize)
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], marker = 'o', s = pointSize)
                img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = plotPlaneLst[1][xCounterOnlyRecovered, yCounterOnlyRecovered], marker = '^', s = pointSize)

            cbarInst = plt.colorbar(fraction=0.046 * subplot_ratio[1] * nPlots / subplot_ratio[0], pad=0.04, aspect= 10 * subplot_ratio[1] * nPlots / subplot_ratio[0])
            # cbarInst = plt.colorbar(cax = cax)
            cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
            cbarInst.ax.tick_params(labelsize= cbar_font_size)

            plt.subplot(*(plotShape + [3]))
            plt.title(titleLst[2], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels, fontsize = label_font_size)
            plt.yticks(vertLabelIdc, vertLabels, fontsize = label_font_size)

            if planeMaskLst is not None:
                xCounterSampled, yCounterSampled = np.nonzero(np.where(plotPlaneMaskLst[2] >= 1., 1., 0.))
                xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneMaskLst[2] < 1., 1., 0.))
            else:
                xCounterSampled, yCounterSampled = np.nonzero(np.where(plotPlaneLst[2] > 1e-8, 1., 0.))
                xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneLst[2] <= 1e-8, 1., 0.))
            minArr = np.ones(shape_) * vmin_

            if vmin_vmax is not None:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], vmin=vmin_, vmax=vmax_, marker = 'x', s = pointSize)
                img = plt.scatter(xCounterSampled, yCounterSampled, c = plotPlaneLst[2][xCounterSampled, yCounterSampled], vmin=vmin_, vmax=vmax_, marker = 'o', s = pointSize)

            else:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x', s = pointSize)
                img = plt.scatter(xCounterSampled, yCounterSampled, c = plotPlaneLst[2][xCounterSampled, yCounterSampled], marker = 'o', s = pointSize)

            cbarInst = plt.colorbar(fraction=0.046 * subplot_ratio[1] * nPlots / subplot_ratio[0], pad=0.04, aspect= 10 * subplot_ratio[1] * nPlots / subplot_ratio[0])
            # cbarInst = plt.colorbar(cax = cax)
            cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
            cbarInst.ax.tick_params(labelsize= cbar_font_size)

    # tikzplotlib.save(format_path_extension(filePath_))
    # plt.savefig(filePath_, bbox_inches='tight')
    plt.tight_layout()
    plt.savefig(filePath)
    if save_tikz:
        tikzplotlib.save(format_path_extension(filePath_, '.tex'))
    
    if specific_value_color_dict is not None:
        plt.set_cmap(current_cmap_copied)
    plt.close('all')
    

def plot3DScatter(npArr, vmin = None, vmax = None, filename = None, axisInfo = None, label_positions = [0., 1/3, 2/3], highest_amount_proportion_threshod = None, small_delta = 1e-8, bar_label = 'gram/ton', default_point_size = 1.0, alpha_min = 0.2, transparent_cbar = False, cbar_font_size = 11, cbar_position = 'center left', label_fontsize = 12, adjust_axis_ratio = True, figsize_default = None, save_tikz = True):
    """Plot the points with amounts of mineral in 3D numpy array.

    Color intensities indicate the amounts.

    Parameters
    ----------
    npArr : numpy array
        3-dimensional numpy array to plot.
    vmin : float
        Threshold to cut the minimum mineral amount. The points with smaller mineral amount of this will not be plotted.
    vmax : float
        Threshold to cut the maximum mineral amount. The points with larger mineral amount of this will not be plotted.
    filename : string
        Path to save the resulted image.
    axisInfo : dictionary
        dinctionary containing axis information.
    maxNth : int
        The rank corresponding the mineral amount of a point which will be plotted as highest color intensity. For example, if maxNth is 50 and you have 1,000 points to plot, 50 points will be plotted with same color indicating highest mineral amount. This will set the highest color range.
    """

    # label_positions = [0., 1/4, 2/4, 3/4, 1.]
    npArr_ = np.where(npArr >= 1e-8, npArr, 0.)
    x, y, z = npArr_.nonzero()

    shape = npArr_.shape
    avg_length = 1
    num_entries = 1
    for i in range(3):
        avg_length *= shape[i]
        num_entries *= shape[i]
    avg_length = avg_length ** (1/3)
    num_obs = np.count_nonzero(npArr_)

    ## Adjust width height ratio
    if adjust_axis_ratio and axisInfo is not None: # https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
        z_range = axisInfo[2]['max'] - axisInfo[2]['min']
        xy_range_avg = (axisInfo[0]['max'] - axisInfo[0]['min'] + axisInfo[1]['max'] - axisInfo[1]['min']) / 2.
        height_by_width_ratio = (z_range / xy_range_avg) ** 0.5
        # fig = plt.figure(figsize= plt.figaspect(z_range / xy_range_avg))
        if figsize_default is None:
            fig = plt.figure(figsize= (8 / height_by_width_ratio, 8 * height_by_width_ratio))
        else:
            fig = plt.figure(figsize= (figsize_default / height_by_width_ratio, figsize_default * height_by_width_ratio))
        ## * xy_range_avg / z_range, multiply this to fig size if you want to enlarge
        ## If you wanna change font size of axis tickers
        # ax.xaxis.set_tick_params(labelsize = 12 * xy_range_avg / z_range)
        # ax.yaxis.set_tick_params(labelsize = 12 * xy_range_avg / z_range)
        # ax.zaxis.set_tick_params(labelsize = 12 * xy_range_avg / z_range)
    else:
        if figsize_default is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize = (figsize_default, figsize_default))
    ax = fig.add_subplot(111, projection='3d')

    if vmin is not None and (vmax is not None or highest_amount_proportion_threshod is not None):
        if highest_amount_proportion_threshod is not None:
            assert(1.0 >= highest_amount_proportion_threshod and highest_amount_proportion_threshod >= 0.)
            # maxNth = (npArr.shape[0] * npArr.shape[1] * npArr.shape[2] * maxNthPercentageFromSmallest)
            vmax = helpers.get_proportional_ranked_value(npArr_, proportion=0.1)
        ax.scatter(x, y, z, zdir='z', c= npArr_[x, y, z], vmin = vmin, vmax = vmax, alpha = max(1. - (num_obs / num_entries) ** (1/3), alpha_min), s = default_point_size * (100/avg_length)) ## parameter s is point size
    else:
        ax.scatter(x, y, z, zdir='z', c= npArr_[x, y, z], alpha = max(1. - (num_obs / num_entries) ** (1/12), alpha_min), s = default_point_size * (100/avg_length))
    
    ## remove BORDER FRAME BOUNDARY line
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ## Make panes transparent
    ax.xaxis.pane.fill = False # Left pane
    ax.yaxis.pane.fill = False # Right pane
    ax.zaxis.pane.fill = False
    ## make the grid lines transparent
    # ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    ## draw frame
    ## ref: https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    # r = [-0.5, 30.5]
    # for s, e in combinations(np.array(list(product(x_minmax, y_minmax, z_minmax))), 2): ## Combination of two points on the grid points of cube(the number of case = 8 * 8 = 64), starting-ending
    #     if np.sum(np.abs(s-e)) == xr[1]-xr[0]:
    #         ax.plot3D(*zip(s, e), color="black")
    x_minmax = [npArr.shape[0] / 60., npArr.shape[0] * (30.65 / 30.)]
    y_minmax = [- npArr.shape[1] / 27.272727272727, npArr.shape[1] * (30.5 / 30.)]
    z_minmax = [npArr.shape[2] / 300, npArr.shape[2] * (30.1 / 30.)]
    for edge in ([[x_minmax[0], y_minmax[0], z_minmax[0]], [x_minmax[0], y_minmax[1], z_minmax[0]]], [[x_minmax[0], y_minmax[0], z_minmax[0]], [x_minmax[0], y_minmax[0], z_minmax[1]]], [[x_minmax[0], y_minmax[1], z_minmax[0]], [x_minmax[1], y_minmax[1], z_minmax[0]]], [[x_minmax[0], y_minmax[1], z_minmax[0]], [x_minmax[0], y_minmax[1], z_minmax[1]]], [[x_minmax[0], y_minmax[1], z_minmax[1]], [x_minmax[0], y_minmax[0], z_minmax[1]]], [[x_minmax[0], y_minmax[1], z_minmax[1]], [x_minmax[1], y_minmax[1], z_minmax[1]]]):
        ax.plot3D(*zip(edge[0], edge[1]), color = "black", linewidth= 1.0)

    ax.grid(b= False) ## Turn off grid lines
    
    ax.set_xlabel('East(m)', fontsize = label_fontsize)
    ax.set_ylabel('North(m)', fontsize = label_fontsize)
    ax.set_zlabel('Elevation(m)', fontsize = label_fontsize)
    ## set label margin
    ax.xaxis.labelpad= 20 * (label_fontsize / 10.)
    ax.yaxis.labelpad= 20 * (label_fontsize / 10.)
    ax.zaxis.labelpad= 8 * (label_fontsize / 10.)
    ## Define plot space
    ax.set_xlim(0, npArr.shape[0])
    ax.set_ylim(0, npArr.shape[1])
    ax.set_zlim(0, npArr.shape[2])

    if axisInfo != None:

        horiLabels = [round(position_proportion * shape[0]) for position_proportion in label_positions]
        vertLabels = [round(position_proportion * shape[1]) for position_proportion in label_positions]
        elevLabels = [round(position_proportion * shape[2]) for position_proportion in label_positions]

        ax.set_xticks(horiLabels)
        ax.set_xticklabels([round(axisInfo[0]["min"] + (axisInfo[0]["max"] - axisInfo[0]["min"]) * position_proportion) for position_proportion in label_positions], fontsize = label_fontsize)
        ax.set_yticks(vertLabels)
        ax.set_yticklabels([round(axisInfo[1]["min"] + (axisInfo[1]["max"] - axisInfo[1]["min"]) * position_proportion) for position_proportion in label_positions], fontsize = label_fontsize)
        ax.set_zticks(elevLabels)
        ax.set_zticklabels([round(axisInfo[2]["min"] + (axisInfo[2]["max"] - axisInfo[2]["min"]) * position_proportion) for position_proportion in label_positions], fontsize = label_fontsize)       

    # ax.view_init(30, view_angle) ## set angle, elev, azimuth angle
    ax.set_proj_type('ortho') ## make z axis vertical: https://stackoverflow.com/questions/26796997/how-to-get-vertical-z-axis-in-3d-surface-plot-of-matplotlib
    # plt.tight_layout()

    #%% colorbar plot
    # bbox_to_anchor = (0.7, 2.4, 1.0, 1.0)
    axins = inset_axes(ax, width = "2%", height = "60%", loc = cbar_position)
    cbar = plt.colorbar(ax.get_children()[0], ax = ax, cax = axins)
    axins.yaxis.set_ticks_position("left")
    cbar.set_label(bar_label, fontsize = cbar_font_size)
    cbar.ax.tick_params(labelsize= cbar_font_size)
    if not transparent_cbar:
        cbar.solids.set(alpha=1)
    # plt.tight_layout()
    # plt.autoscale()
    # plt.savefig(filename, bbox_inches = "tight")
    plt.savefig(filename)
    if save_tikz:
        tikzplotlib.save(filepath = format_path_extension(filename, '.tex'))

def plot3DScatterDistinguish(nparrValue, nparrOriginalMask = None, nparrRecoveredMask = None, vmin = None, vmax = None, filename = None, axisInfo = None, maxNth = None, cbarLabel = 'amount', save_tikz = True):
    # xOriginal, yOriginal, zOriginal = npArrOriginal.nonzero()
    if nparrOriginalMask is not None:
        nparrOriginalMaskPlot = np.where(nparrOriginalMask >= 1., 1., 0.)
        xOriginal, yOriginal, zOriginal = np.nonzero(nparrOriginalMaskPlot)
    else:
        xOriginal, yOriginal, zOriginal = np.nonzero(nparrValue)
    
    if nparrRecoveredMask is not None:
        assert(nparrOriginalMask is not None)
        nparrRecoveredOnlyMask = np.where(nparrOriginalMask < 1., np.where(nparrRecoveredMask >= 1., 1., 0.), 0.)
        xRecovered, yRecovered, zRecovered = np.nonzero(nparrRecoveredOnlyMask)

    shape = nparrValue.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, zdir='z', c= npArr[x, y, z], cmap = 'Reds', vmin = vmin, vmax = vmax, marker = '.')
    if vmin is not None and (vmax is not None or maxNth is not None):
        if maxNth is not None:
            assert(np.count_nonzero(nparrValue) >= maxNth)
            # maxNth = (npArr.shape[0] * npArr.shape[1] * npArr.shape[2] * maxNthPercentageFromSmallest)
            vmax = np.partition(nparrValue.flatten(), -1 * maxNth)[-1 * maxNth]

        if nparrOriginalMask is not None:
            ax.scatter(xOriginal, yRecovered, zRecovered, zdir='z', c= nparrValue[xOriginal, yOriginal, zOriginal], vmin = vmin, vmax = vmax, s = 0.02, marker = 'o') # parameter s is point size
        
        if nparrRecoveredMask is not None:
            ax.scatter(xRecovered, yRecovered, zRecovered, zdir='z', c= nparrValue[xRecovered, yRecovered, zRecovered], vmin = vmin, vmax = vmax, s = 0.02, marker = '*')
            
    else:
        if nparrOriginalMask is not None:
            ax.scatter(xOriginal, yRecovered, zRecovered, zdir='z', c= nparrValue[xOriginal, yOriginal, zOriginal], s = 0.02, marker = 'o') # parameter s is point size
        
        if nparrRecoveredMask is not None:
            ax.scatter(xRecovered, yRecovered, zRecovered, zdir='z', c= nparrValue[xRecovered, yRecovered, zRecovered], s = 0.02, marker = '*')
    
    ax.set_xlabel('East(m)')
    ax.set_ylabel('North(m)')
    ax.set_zlabel('Elevation(m)')

    if axisInfo != None:
        vertLabels = (0, shape[1]*1//4, shape[1]*2//4, shape[1]*3//4, shape[1])
        horiLabels = (0, shape[0]*1//4, shape[0]*2//4, shape[0]*3//4, shape[0])
        elevLabels = (0, shape[2]*1//4, shape[2]*2//4, shape[2]*3//4, shape[2])

        ax.set_xticks(horiLabels)
        ax.set_xticklabels((round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
                round(axisInfo[0]["max"])))
        ax.set_yticks(vertLabels)
        ax.set_yticklabels((round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                round(axisInfo[1]["max"])))
        ax.set_zticks(elevLabels)
        ax.set_zticklabels((round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                round(axisInfo[2]["max"])))

    cbar = plt.colorbar(ax.get_children()[0], ax = ax)
    # cbar.set_label("amount of metal\n \n ", rotation = 270)
    cbar.set_label(cbarLabel)

    plt.savefig(filename)
    if save_tikz:
        tikzplotlib.save(filepath = format_path_extension(filename, '.tex'))

def saveGifFromArr(arr, path, fps = 1, vmin = None, vmax = None, axis = 0, get_dense_slice_threshold = 0, limit_frame = 100, frame_idc = None):
    """Save 3D or 4D Numpy array into gif.
    
    If 4D, last dimension(shape[3]) is regarded as RGB channel, thus shape[3] == 3.\n
    If 3D, values are converted into RGB color intensities following cmap whose default is current cmap = plt.get_cmap().

    Parameters
    ----------
    arr : array
        3D or 4D array to be gif animation.
    path : str
        path to save gif file.
    vmin : float
        Celi cut threshold for color intensities. Values below vmin will be changed to vmin.
    vmax : float
        Floor cut threshold for color intensities. Values above vmax will be changed to vmax.
    axis : int
        Axis to follow to get slices of gif.
    get_dense_slice_threshold : int
        The 
    """
    
    assert(fps >= 1 and axis in [0, 1, 2])
    arrFormatted = np.copy(arr)
    if frame_idc is not None:
        arrFormatted = helpers.get_slices_with_idc(arrFormatted, {axis: frame_idc})
    elif get_dense_slice_threshold > 0:
        nonzero_idx_list = helpers.collect_idx_of_dense_slices_along_axis(arrFormatted, axis = axis, get_dense_slice_threshold = get_dense_slice_threshold)
        arrFormatted = helpers.get_slices_with_idc(arrFormatted, {axis: nonzero_idx_list})
    if vmin is not None:
        arrFormatted_mask = np.where(arrFormatted > vmin, 1., 0.)
    else:
        arrFormatted_mask = np.where(arrFormatted >= 1e-8, 1., 0.)
    arrFormatted = helpers.cutArrayToMinMax(arrFormatted, min = vmin, max = vmax)
    if len(arr.shape) == 4:
        assert(arr.shape[3] == 3)
        arrFormatted = helpers.min_max_scale(arrFormatted, 0, 255).astype(np.uint8)
    elif len(arr.shape) == 3:
        arrFormatted = convert_3Darray_to_4DarrayRGB(arrFormatted)
    if arrFormatted.shape[axis] > limit_frame:
        arrFormatted = helpers.get_slices_with_idc(arrFormatted, {axis: list(range(limit_frame))})
    clip_mask = mpy.VideoClip(make_frame=lambda t: helpers.getSlicesV2(arrFormatted_mask, {axis: int(t)}), duration=arrFormatted.shape[0], ismask=True)
    # clip = mpy.VideoClip(lambda t: arrFormatted[int(t), :, :, :], duration=(arr.shape[0]) / float(fps))
    clip = mpy.VideoClip(make_frame=lambda t: helpers.getSlicesV2(arrFormatted, {axis: int(t)}), duration = arrFormatted.shape[axis])
    clip.set_mask(clip_mask)
    clip.speedx(fps).write_gif(path, fps = fps, logger = "bar")

def convert_3Darray_to_4DarrayRGB(arr_3D, vmin = None, vmax = None, cmap = plt.get_cmap()):
    if vmin is not None and vmax is not None:
        assert(vmax >= vmin)
    if vmin is None and vmax is None:
        arr_3D_copied = np.copy(arr_3D)
    else:
        arr_3D_copied = helpers.cutArrayToMinMax(arr_3D, min = vmin, max = vmax)
    arr_3D_copied = helpers.min_max_scale(arr_3D_copied)
    arr_4D = cmap(arr_3D_copied, bytes = True)
    arr_4D = np.delete(arr_4D, 3, 3)
    return arr_4D

def plot_bar_charts(path_to_save : str, name_numbers : dict, xlabels : list, xtitle = None, ytitle = None, bar_width = 'auto', alpha = 0.8, colors_dict = None, format = 'eps', diagonal_xtickers = 0, name_errors = None, name_to_show_percentage = None, name_not_to_show_percentage_legend = None, fontsize = 10, title = None, figsize = None, ylim = None, fix_legend = True, plot_legend = True, save_tikz = True):
    """
    
    Parameters
    ----------
        name_numbers : dict
            For example, name_numbers['enriched'] == [0.12, 0.43, 0.12] for RMSE
        xlabels : list
            Name of groups, For example ['CNN', 'LR', 'SVR', ..]
    """

    ## Set kwargs parameters
    plt_bars_kwargs_dict = {}
    for name in name_numbers.keys():
        plt_bars_kwargs_dict[name] = {}
        if name_errors is not None and name in name_errors.keys():
            plt_bars_kwargs_dict[name]['yerr'] = name_errors[name]
        if colors_dict is not None:
            plt_bars_kwargs_dict[name]['color'] = colors_dict[name]

    single_key = next(iter(name_numbers))
    n_groups = len(name_numbers[single_key])
    for numbers in name_numbers.values():
        assert(len(numbers) == n_groups)
    assert(len(xlabels) == n_groups)
    xlabels_copied = deepcopy(xlabels)

    if name_to_show_percentage is not None:
        assert(name_to_show_percentage in name_numbers.keys())
        assert(len(name_numbers) >= 2)
        for i in range(len(xlabels_copied)):
            scores_of_group = []
            for name in name_numbers.keys():
                if name != name_to_show_percentage:
                    scores_of_group.append(name_numbers[name][i])
            mean = np.mean(scores_of_group)
            xlabels_copied[i] += f'({(mean - name_numbers[name_to_show_percentage][i]) * 100. / mean:.2f}%)'

    legend_prefix_dict = {name: '' for name in name_numbers.keys()}
    if name_not_to_show_percentage_legend is not None:
        if not isinstance(name_not_to_show_percentage_legend, (list, tuple)):
            name_not_to_show_percentage_legend_copied = [name_not_to_show_percentage_legend]
        else:
            name_not_to_show_percentage_legend_copied = deepcopy(name_not_to_show_percentage_legend)
        avg_number = 0.
        for name in name_not_to_show_percentage_legend_copied:
            assert(name in name_numbers.keys())
            avg_number += sum(name_numbers[name])
        avg_number /= len(name_not_to_show_percentage_legend_copied)
        for name in name_numbers.keys():
            if name not in name_not_to_show_percentage_legend_copied:
                legend_prefix_dict[name] = f" ({round(100. * ((avg_number - sum(name_numbers[name])) / sum(name_numbers[name])))}%)"

    if bar_width == 'auto':
        # bar_width_ = 0.30 * (2 / len(name_numbers))  
        bar_width_ = 0.20 * (2 / len(name_numbers))  
    else:
        bar_width_ = bar_width
    index = np.arange(n_groups)
    # create plot
    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize = figsize)

    rects_list = []
    index_copied = np.copy(index).astype(np.float)
    for name, numbers in name_numbers.items():
        rects_list.append(plt.bar(index_copied, numbers, bar_width_, alpha = alpha, label = name + legend_prefix_dict[name], **plt_bars_kwargs_dict[name])) ## label will be label in legend
        index_copied += bar_width_

    if title is not None:
        plt.title(title, fontsize = fontsize + 2)
    if xtitle is not None:
        plt.xlabel(xtitle, fontsize = fontsize)
    if ytitle is not None:
        plt.ylabel(ytitle, fontsize = fontsize)
    # plt.title('Scores by person')
    if diagonal_xtickers == True:
        plt.xticks(index + (bar_width_/2) * (len(name_numbers)-1), xlabels_copied, fontsize = fontsize, rotation = 45, ha = "right")
    elif diagonal_xtickers == False:
        plt.xticks(index + (bar_width_/2) * (len(name_numbers)-1), xlabels_copied, fontsize = fontsize)
    else:
        plt.xticks(index + (bar_width_/2) * (len(name_numbers)-1), xlabels_copied, fontsize = fontsize, rotation = diagonal_xtickers, ha = "right")
    if fix_legend:
        numbers_tot = []
        for numbers in name_numbers.values():
            numbers_tot += numbers
        plt.ylim([0., np.max(numbers_tot) * (1. + 0.1 * len(name_numbers))])
    
    if ylim is not None:
        assert(len(ylim) == 2)
        if ylim[1] == None:
            plt.ylim(bottom = ylim[0])
        elif ylim[0] == None:
            plt.ylim(top = ylim[1])
        else:
            plt.ylim(ylim)
        
    if plot_legend:
        plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(path_to_save, format = format)
    if save_tikz:
        tikzplotlib.save(filepath = format_path_extension(path_to_save, '.tex'))

def plot_multiple_lists(lists_dict: dict, path_to_save: str, labels : dict = {'x': 'Iteration', 'y': 'Loss'}, format = 'eps', save_tikz = True):
    """
    
    Parameters
    ----------
    lists_dict : dict
        Dictionary of lists, where key is name of constraint or dual and value is losses of them\n
        ex) lists_dict[r'$\Vert Y_l - F_l \Vert _F$'] == [0.1, 30.5, 21, ...]
    """

    single_key = next(iter(lists_dict))
    len_data = len(lists_dict[single_key])
    for list_ in lists_dict.values():
        assert(len(list_) == len_data)
    xs = np.arange(len_data)
    fig = plt.figure()
    ax = plt.subplot(111)
    for key, list_ in lists_dict.items():
        ax.plot(xs, list_, label = key)
    # plt.title('')
    ax.legend()
    plt.grid()
    plt.yscale('log')

    def format_fn(tick_val, tick_pos):
        if int(tick_val) in xs:
            return xs[int(tick_val)]
        else:
            return ''
    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel(labels['x'])
    plt.ylabel(labels['y'])
    plt.savefig(path_to_save, format = format)
    if save_tikz:
        tikzplotlib.save(filepath = format_path_extension(path_to_save, '.tex'))

def format_path_extension(path : str, extension : str = '.tex'):
    """ Format path as tex.
    
    Examples
    --------
        print(format_path_extension('Users/hi/something.eps'))
        print(format_path_extension('abc/.def/ghi.eps'))
        print(format_path_extension('abc/.def/ghi'))
        print(format_path_extension('./abc/def/ghi'))
        ==
        Users/hi/something.tex
        abc/.def/ghi.tex
        abc/.def/ghi.tex
        ./abc/def/ghi.tex
    """

    splitted_path = path.split('/')
    if '.' in splitted_path[-1]:
        for i in range(len(path) - 1):
            if path[-(i + 1)] == '.':
                return path[:-(i + 1)] + extension
    else:
        return path + extension

def plot_group_scatter(group_df, path_to_save, group_column, y_column, color_column = None, colors_rotation = ["red", "navy", "lightgreen", "teal", "violet", "green", "orange", "blue", "coral", "yellowgreen", "sienna", "olive", "maroon", "goldenrod", "darkblue", "orchid", "crimson"], group_sequence = None, xlabel = None, ylabel = None, rotation_xtickers = 0, group_column_xtext_dict = None, order_by = None, num_truncate_small_groups = 0, save_tikz = True):
    group_df_copied = group_df.copy()
    assert("order" not in group_df_copied.columns and "color_temp" not in group_df_copied.columns and "index_temp" not in group_df_copied.columns)
    ## To get most common color : group_df_copied[[group_column, color_column]].groupby(group_column).agg(lambda x:x.value_counts().index[0])

    if group_sequence is None:
        unique_group_names_list = list(group_df_copied[group_column].unique())
        group_sequence_copied = deepcopy(unique_group_names_list)
    else:
        group_sequence_copied = deepcopy(group_sequence)
        # assert(set(unique_group_names_list) == set(group_sequence_copied)) ## The groups should match without ordering
        is_in_group_sequence = group_df_copied[group_column].isin(group_sequence_copied)
        group_df_copied = group_df_copied[is_in_group_sequence]
    
    ## Set the label locations.
    # counts_groups = [0]
    counts_groups = []
    for group in group_sequence_copied:
        series_obj = group_df_copied.apply(lambda x: True if x[group_column] == group else False, axis = 1)
        counts_groups.append(len(series_obj[series_obj == True].index))
    
    if num_truncate_small_groups > 0:
        smallest_groups_idc_list = helpers.get_smallest_largest_idc(counts_groups, num_truncate_small_groups)
        counts_groups = helpers.delete_items_from_list_with_indices(counts_groups, smallest_groups_idc_list, keep_not_remove = False)
        group_sequence_copied = helpers.delete_items_from_list_with_indices(group_sequence_copied, smallest_groups_idc_list, keep_not_remove = False)
        is_in_group_sequence = group_df_copied[group_column].isin(group_sequence_copied)
        group_df_copied = group_df_copied[is_in_group_sequence]
    counts_groups.insert(0, 0)
    
    hori_labels = [] ## position of horizontal labels
    accumulated = 0
    for i in range(len(counts_groups) - 1): ## instead use group_df_copied[[group_column, color_column]].groupby(group_column).count()?
        accumulated += counts_groups[i]
        hori_labels.append(round(counts_groups[i + 1] / 2 + accumulated))

    if color_column is not None:
        assert(group_df_copied[[group_column, color_column]].groupby(group_column).agg(lambda x: len(set(x)) == 1).all(axis = None))
        group_color_dict = dict(group_df_copied[[group_column, color_column]].groupby(group_column)[color_column].apply(lambda x: list(x)[0]))
        assert(set(group_color_dict.keys()) == set(group_sequence_copied))
    else:
        group_color_dict = {}
        for group, i in zip(group_sequence_copied, range(len(group_sequence_copied))):
            group_color_dict[group] = colors_rotation[i % len(colors_rotation)]
        group_df_copied['color_temp'] = group_df_copied[group_column].apply(lambda x: group_color_dict[x])
        
    group_df_copied["order"] = group_df_copied[group_column].apply(lambda x, group_sequence_copied: group_sequence_copied.index(x), args = (group_sequence_copied,))

    if order_by is None:
        group_df_copied.sort_values(by = ["order"], ascending = [True], inplace = True)
    else:
        group_df_copied.sort_values(by = ["order", order_by], ascending = [True, True], inplace = True)
    group_df_copied['index_temp'] = range(1, len(group_df_copied) + 1)
    if color_column is not None:
        ax = group_df_copied.plot.scatter(x = 'index_temp', y = y_column, c = group_df_copied[color_column], s = 0.03, figsize = (8, 2), colorbar = False, fontsize = 6, marker = ',')
    else:
        ax = group_df_copied.plot.scatter(x = 'index_temp', y = y_column, c = group_df_copied["color_temp"], s = 0.03, figsize = (8, 2), colorbar = False, fontsize = 6, marker = ',')
    
    # x_ticks_texts = group_sequence_copied if group_sequence is not None else deepcopy(group_sequence_copied)
    x_ticks_texts = []
    x_ticks_colors = []
    for group, i in zip(group_sequence_copied, range(len(group_sequence_copied))):
        x_ticks_colors.append(group_color_dict[group])
        if group_column_xtext_dict is None:
            x_ticks_texts.append(group)
        else:
            x_ticks_texts.append(group_column_xtext_dict[group])
    # x_ticks_colors_rotation = deepcopy(colors_rotation)
    x_ticks_indices = list(range(len(group_sequence_copied)))

    hori_labels = helpers.delete_items_from_list_with_indices(hori_labels, x_ticks_indices, keep_not_remove = True)
    x_ticks_texts = helpers.delete_items_from_list_with_indices(x_ticks_texts, x_ticks_indices, keep_not_remove = True)
    x_ticks_colors = helpers.delete_items_from_list_with_indices(x_ticks_colors, x_ticks_indices, keep_not_remove = True)
    assert(len(hori_labels) == len(x_ticks_texts))

    x_axis = ax.axes.get_xaxis()
    # x_axis.set_visible(False)
    ax.set_xticks(hori_labels)
    if rotation_xtickers != 0:
        ax.set_xticklabels(x_ticks_texts, rotation = rotation_xtickers, ha = "right")
    else:
        ax.set_xticklabels(x_ticks_texts)
    # ax.tick_params(axis = 'x', colors = x_ticks_colors)
    for i in range(len(hori_labels)):
        ax.get_xticklabels()[i].set_color(x_ticks_colors[i])
    ax.margins(x = 0)
    ax.margins(y = 0)
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(group_column)
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(y_column)
    # cbar = plt.colorbar(mappable = ax)
    # cbar.remove()
    # plt.margins(y=0)
    # plt.tight_layout()
    # plt.grid(which = 'major', linestyle='-', linewidth=2)
    plt.savefig(path_to_save, bbox_inches = "tight")
    if save_tikz:
        tikzplotlib.save(utilsforminds.visualization.format_path_extension(path_to_save))
    plt.cla()

def plot_top_bars_with_rows(data_df, path_to_save : str, color_column = None, colors_rotation = ["red", "navy", "lightgreen", "teal", "violet", "green", "orange", "blue", "coral", "yellowgreen", "sienna", "olive", "maroon", "goldenrod", "darkblue", "orchid", "crimson"], order_by = "weights", x_column = None, group_column = None, xticks_replace_dict = None, xlabel = None, ylabel = None, title = None, num_bars = 10, num_rows = 2, re_range_max_min_proportion = None, show_group_error = True, show_group_size = True, bar_width = 'auto', opacity = 0.8, xticks_fontsize = 10, rotation_xtickers = 0, format = 'eps', save_tikz = True):
    """Plot the pandas df's rows of highest values.
    
    """
    assert(order_by in data_df.columns and 'color_temp' not in data_df.columns and 'index_temp' not in data_df.columns)

    # total_num_elements = num_bars * num_rows
    data_df_copied = data_df.copy()
    data_df_copied['index_temp'] = range(0, len(data_df_copied))
    if group_column is not None:
        assert(x_column is None)
        assert(group_column in data_df_copied.columns)
        unique_group_names_list = list(data_df_copied[group_column].unique())
        if show_group_size:
            counts_groups_dict = {}
            for group in unique_group_names_list:
                series_obj = data_df_copied.apply(lambda x: True if x[group_column] == group else False, axis = 1)
                counts_groups_dict[group] = len(series_obj[series_obj == True].index)
        if color_column is not None:
            assert(data_df_copied[[group_column, color_column]].groupby(group_column).agg(lambda x: len(set(x)) == 1).all(axis = None))
            group_color_dict = dict(data_df_copied[[group_column, color_column]].groupby(group_column)[color_column].apply(lambda x: list(x)[0]))
            data_df_copied = data_df_copied.groupby([group_column], as_index = False).agg({order_by: ['mean', lambda x: x.std(ddof=0)], color_column: 'first', 'index_temp': 'first'})
        else:
            group_color_dict = {}
            for group, i in zip(unique_group_names_list, range(len(unique_group_names_list))):
                group_color_dict[group] = colors_rotation[i % len(colors_rotation)]
            data_df_copied['color_temp'] = data_df_copied[group_column].apply(lambda x: group_color_dict[x])
            data_df_copied = data_df_copied.groupby([group_column], as_index = False).agg({order_by: ['mean', lambda x: x.std(ddof=0)], 'color_temp': 'first', 'index_temp': 'first'})
        data_df_copied.columns = [group_column, order_by, 'std', 'color_temp', 'index_temp']
        # assert(len(data_df_copied) >= total_num_elements)
        index_column = group_column
    else:
        assert(x_column is not None)
        index_column = x_column
        if color_column is not None:
            data_df_copied = data_df_copied.rename(columns= {color_column: 'color_temp'})
        else:
            data_df_copied['color_temp'] = data_df_copied['index_temp'].apply(lambda x: colors_rotation[x % len(colors_rotation)])
    
    data_df_copied = data_df_copied.sort_values(by = order_by, ascending = False, inplace = False)

    total_num_elements = len(data_df_copied)
    num_for_each_row_list = []
    for i in range(num_rows):
        if total_num_elements > num_bars:
            num_for_each_row_list.append(num_bars)
            total_num_elements -= num_bars
        else:
            num_for_each_row_list.append(total_num_elements)
            break

    # fig = plt.figure(figsize = (7, 3 * len(num_for_each_row_list)))
    fig = plt.figure()

    axes = []
    num_plots_accumulated = 0
    for row_idx in range(len(num_for_each_row_list)):
        top_names = list(data_df_copied.loc[:, index_column][num_plots_accumulated:(num_plots_accumulated + num_for_each_row_list[row_idx])])
        if xticks_replace_dict is not None:
            top_names_ = []
            for name in top_names:
                top_names_.append(xticks_replace_dict[name])
        else:
            top_names_ = deepcopy(top_names)
        if show_group_size:
            assert(group_column is not None)
            for idx, name in zip(range(len(top_names)), top_names):
                top_names_[idx] = str(top_names_[idx]) + f"({counts_groups_dict[name]})"
        top_weights = list(data_df_copied.loc[:, order_by][num_plots_accumulated:(num_plots_accumulated + num_for_each_row_list[row_idx])])
        top_colors = list(data_df_copied.loc[:, "color_temp"][num_plots_accumulated:(num_plots_accumulated + num_for_each_row_list[row_idx])])
        if show_group_error and group_column is not None:
            top_errors = list(data_df_copied.loc[:, 'std'][num_plots_accumulated:(num_plots_accumulated + num_for_each_row_list[row_idx])])
        if bar_width == 'auto':
            # bar_width_ = 1. / num_for_each_row_list[row_idx]
            bar_width_ = 0.6 / num_for_each_row_list[row_idx]
        else:
            bar_width_ = bar_width
        num_plots_accumulated += num_for_each_row_list[row_idx]
        
        ## create plot
        axes.append(plt.subplot(len(num_for_each_row_list), 1, row_idx + 1))
        index_ = np.arange(num_for_each_row_list[row_idx])

        ## set range
        if re_range_max_min_proportion is not None:
            assert(re_range_max_min_proportion[0] < 1. and re_range_max_min_proportion[1] >= 1.)
            min_, max_ = np.min(top_weights), np.max(top_weights)
            plt.ylim([re_range_max_min_proportion[0] * min_, re_range_max_min_proportion[1] * max_])

        if show_group_error and group_column is not None:
            plt.bar(index_, top_weights, alpha = opacity, color = top_colors, yerr = top_errors)
        else:
            plt.bar(index_, top_weights, alpha = opacity, color = top_colors)
        if xlabel is not None:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(index_column)
        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel(order_by)
        
        if rotation_xtickers != 0:
            plt.xticks(index_ + (bar_width_/2) * (1-1), top_names_, rotation = rotation_xtickers, ha = "right")
        else:
            plt.xticks(index_ + (bar_width_/2) * (1-1), top_names_)

        for obj in axes[row_idx].get_xticklabels():
            obj.set_fontsize(xticks_fontsize)

    if title is not None:
        plt.title(title)

    fig.tight_layout()
    plt.savefig(path_to_save, format = format)
    if save_tikz:
        tikzplotlib.save(utilsforminds.visualization.format_path_extension(path_to_save))
    plt.clf()

def plot_xy_lines(x, y_dict_list : list, path_to_save : str, title = None, x_label = None, y_label = None, figsize= (17, 5), label_fontsize = 20, format = 'eps', save_tikz = True):
    """
    
    Examples
    --------
    plot_xy_lines([1, 2, 3, 4], [{"label": "my label", "ydata": [3.4, 3.2, 1.1, 0.3]}], "dummy.eps")
    """
    x_arr_copied = np.array(x)
    x_arr_sorted_ind = np.argsort(x_arr_copied)
    x_arr_copied = x_arr_copied[x_arr_sorted_ind]

    y_dict_list_copied = deepcopy(y_dict_list)
    for y_dict_idx in range(len(y_dict_list_copied)):
        for required_key in ["label", "ydata"]:
            assert(required_key in y_dict_list_copied[y_dict_idx].keys())
        y_dict_list_copied[y_dict_idx]["ydata"] = np.array(y_dict_list_copied[y_dict_idx]["ydata"])
        assert(y_dict_list_copied[y_dict_idx]["ydata"].shape == x_arr_copied.shape)
        y_dict_list_copied[y_dict_idx]["ydata"] = y_dict_list_copied[y_dict_idx]["ydata"][x_arr_sorted_ind]

    plt.figure(figsize=figsize)
    for y_dict in y_dict_list_copied:
        y_dict_no_ydata = deepcopy(y_dict)
        y_dict_no_ydata.pop("ydata", None)
        plt.plot(x_arr_copied, y_dict["ydata"], **y_dict_no_ydata)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if title is not None:
        plt.title(title, fontsize = int(label_fontsize * 1.5))
    if x_label is not None:
        plt.xlabel(x_label, fontsize = label_fontsize)
    if y_label is not None:
        plt.ylabel(y_label, fontsize = label_fontsize)
    plt.savefig(path_to_save, format = format)
    if save_tikz:
        tikzplotlib.save(utilsforminds.visualization.format_path_extension(path_to_save))
    plt.clf()

def get_xy_axis_from_z(zaxis = 0):
    assert(zaxis in [0, 1, 2])
    # return (zaxis + 1) % 3, (zaxis + 2) % 3
    if zaxis == 0:
        return 1, 2
    elif zaxis == 1:
        return 0, 2
    elif zaxis == 2:
        return 0, 1
    else:
        raise Exception(ValueError)

def plot_3D_plotly(nparr_3D, path_to_save_static : str, do_save_html : bool = True, kinds_to_plot : list = None, marker_kwargs : dict = None, vmin = None, vmax = None, alpha_shape_kwargs : dict = None, points_decider = lambda x: x > 1e-8, mask_nparr_3D = None, title= None, points_legends : dict = None, alpha_shape_legend = "", scene_kwargs : dict = None, xyz_tickers = None, layout_kwargs : dict = None, figsize_ratio : dict = None, camera = None, showgrid = False, zeroline = True, showline = False, transparent_bacground = True, colorbar_kwargs = None):
    """
    
    Parameters
    ----------
    alphahull : float
        The larger alphahull results the more sharp(shrinked) alpha-shape. alpha = 1 / alphahull
    """

    assert(len(nparr_3D.shape) == 3)
    input_shape = deepcopy(nparr_3D.shape)
    
    ## assign default values
    title_copied = "" if title is None else title
    if kinds_to_plot is None:
        kinds_to_plot = ["scatter"]
    else:
        for kind in kinds_to_plot:
            assert(kind in ["scatter", "alphashape"])
        kinds_to_plot = deepcopy(kinds_to_plot)

    marker_kwargs_local = {"colorscale": 'Viridis', "size": 2.}
    if vmin is not None:
        marker_kwargs_local["cmin"] = vmin
    if vmax is not None:
        marker_kwargs_local["cmax"] = vmax

    marker_kwargs_local = utilsforminds.containers.merge_dictionaries([marker_kwargs_local, marker_kwargs])
    # marker_kwargs_local["colorbar"] = utilsforminds.containers.merge_dictionaries([{"title": "colorbar", "xpad": 0.0}, colorbar_kwargs])
    marker_kwargs_local["colorbar"] = utilsforminds.containers.merge_dictionaries([{"title": "colorbar"}, colorbar_kwargs])
    alpha_shape_kwargs_local = utilsforminds.containers.merge_dictionaries([{"color": "orange", "opacity": 0.3}, alpha_shape_kwargs])
    points_legends_local = utilsforminds.containers.merge_lists([["Added to Mask", "Mask"], points_legends])
    scene_kwargs_local = utilsforminds.containers.merge_dictionaries([{"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z"}, scene_kwargs])

    # print(f"marker_kwargs: {marker_kwargs}")
    # print(f"colorbar_kwargs_local: {colorbar_kwargs_local}")
    if xyz_tickers is None:
        xyz_tickers_copied = {
            "x": {"tickvals": range(input_shape[0] // 5, input_shape[0], input_shape[0] // 5), "ticktext": range(input_shape[0] // 5, input_shape[0], input_shape[0] // 5)},
            "y": {"tickvals": range(input_shape[1] // 5, input_shape[1], input_shape[1] // 5), "ticktext": range(input_shape[1] // 5, input_shape[1], input_shape[1] // 5)},
            "z": {"tickvals": range(input_shape[2] // 5, input_shape[2], input_shape[2] // 5), "ticktext": range(input_shape[2] // 5, input_shape[2], input_shape[2] // 5)}
        }
    else:
        for axis in ["x", "y", "z"]:
            assert(len(xyz_tickers[axis]["tickvals"]) == len(xyz_tickers[axis]["ticktext"]))
        xyz_tickers_copied = deepcopy(xyz_tickers)
    layout_kwargs_local = {} if layout_kwargs is None else deepcopy(layout_kwargs)

    camera_copied = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=1.25)
    ) ## default camera setting
    if camera is not None:
        camera_copied.update(camera)
    
    ## Generates objects to plot
    plot_objects = []
    nparr_3D_filtered = np.where(points_decider(nparr_3D), 1., 0.)
    if mask_nparr_3D is None:
        mask_nparr_3D_added = np.zeros(input_shape)
    else:
        mask_nparr_3D_added = np.where(nparr_3D_filtered == 0., np.where(points_decider(mask_nparr_3D), 1., 0.), 0.)
    if "scatter" in kinds_to_plot:
        for is_main, mask_to_plot, marker_symbol, points_legend in zip([True, False], [nparr_3D_filtered, mask_nparr_3D_added], ["circle", "cross"], [points_legends_local[1], points_legends_local[0]]):
            x, y, z = mask_to_plot.nonzero()
            colors_arr = utilsforminds.numpy_array.push_arr_to_range(nparr_3D[x, y, z], vmin = vmin, vmax = vmax) ## don't need maybe, because of cmin and cmax.
            if is_main: ## Only plot one colorbar.
                marker_kwargs_copied = marker_kwargs_local
            else:
                marker_kwargs_copied = utilsforminds.containers.copy_dict_and_delete_element(marker_kwargs_local, ["colorbar"])
            plot_objects.append(graph_objs.Scatter3d(mode = 'markers', name = points_legend, x = x, y = y, z = z, marker = graph_objs.Marker(color = colors_arr, symbol = marker_symbol, **marker_kwargs_copied)))
    if "alphashape" in kinds_to_plot:
        mask_nparr_3D_alphashape = nparr_3D_filtered
        x, y, z = mask_nparr_3D_alphashape.nonzero()
        plot_objects.append(graph_objs.Mesh3d(name = alpha_shape_legend, x = x, y = y, z = z, **alpha_shape_kwargs_local))

    scene = graph_objs.Scene(xaxis = {"range": [0, input_shape[0]], "tickvals": xyz_tickers_copied["x"]["tickvals"], "ticktext": xyz_tickers_copied["x"]["ticktext"], "showgrid": showgrid, "zeroline": zeroline, "showline": showline, "zerolinecolor": "black", "backgroundcolor": "rgb(255, 255, 255)"},
    yaxis = {"range": [0, input_shape[1]], "tickvals": xyz_tickers_copied["y"]["tickvals"], "ticktext":  xyz_tickers_copied["y"]["ticktext"], "showgrid": showgrid, "zeroline": zeroline, "showline": showline, "zerolinecolor": "black", "backgroundcolor": "rgb(255, 255, 255)"},
    zaxis = {"range": [0, input_shape[2]], "tickvals": xyz_tickers_copied["z"]["tickvals"], "ticktext":  xyz_tickers_copied["z"]["ticktext"], "showgrid": showgrid, "zeroline": zeroline, "showline": showline, "zerolinecolor": "black", "backgroundcolor": "rgb(255, 255, 255)"}, **scene_kwargs_local)

    # layout = graph_objs.Layout(title = title_copied, width = figsize_copied["width"], height = figsize_copied["height"], scene = scene, scene_camera = camera_copied)
    layout = graph_objs.Layout(title = title_copied, scene = scene, scene_camera = camera_copied, **layout_kwargs_local)

    fig = graph_objs.Figure(data = graph_objs.Data(plot_objects), layout = layout)

    ## Set grid line and zero line
    if figsize_ratio is not None:
        fig.update_layout(scene_aspectmode='manual', scene_aspectratio=figsize_ratio) ## scene_aspectmode='auto' is default argument
    if transparent_bacground:
        fig.update_layout(paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)')

    ## Save the result
    fig.write_image(path_to_save_static)
    if do_save_html:
        fig.write_html(utilsforminds.strings.format_extension(path_to_save_static, "html"))


if __name__ == '__main__':
    pass
    # plot_bar_charts('dummy', {'Frank':[12.7, 0.4, 4.4, 5.3, 7.1, 3.2], 'Guido':[6.3, 10.3, 10, 0.3, 5.3, 2.9]}, ['RR', 'Lasso', 'SVR', 'CNN', 'SVR', 'LR'], ytitle="RMSE of Prediction of TRIAILB-A")
    test_list = []
    for i in range(15):
        test_list.append({"label": f"my label_{i}", "ydata": [3.4, 3.2, 1.1, 0.3]})
    plot_xy_lines([1, 2, 3, 4], test_list, "dummy.eps")