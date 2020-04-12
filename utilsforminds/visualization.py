# import plotly.plotly as py
# import plotly.graph_objs as go
# import plotly
import matplotlib
import pandas as pd
from sys import platform
if platform.lower() != 'darwin' and 'win' in platform.lower():
    matplotlib.use('TkAgg')
else:
    matplotlib.use("MacOSX")
# matplotlib.pyplot.set_cmap('Paired')

import matplotlib.pyplot as plt
plt.set_cmap('Paired')
import os
import utilsforminds.helpers as helpers
import random
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utilsforminds
from mayavi import mlab # see install manual + brew install vtk
from mayavi.api import Engine
import mayavi.tools.pipeline
from scipy import ndimage
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import AxesGrid
import moviepy.editor as mpy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from copy import deepcopy
import tikzplotlib

axis_rotation_dict = {0: 90, 1: 0, 2: 0}

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

def plot2Ds(planeLst, titleLst, filePath, cbarLabel = 'amount', plotShape = [3, 1], figSize = (8, 24), planeMaskLst = None, axis = 2, axisInfo = None, vmin_vmax = None, method = 'imshow', convertXYaxis = False, rotate = 0, label_font_size = 15, title_font_size = 18, cbar_font_size = 15, save_tikz = True):
    '''
        If you want to provide mask matrix for scatter visualization, sequence should be (original matrix, recovered matrix) or (original matrix, recovered matrix, sparse matrix)
    '''

    if filePath.count('/') <= 2:
        filePath_ = utilsforminds.helpers.getExecPath() + '/current_results/' + filePath
    else:
        filePath_ = filePath
    fig = plt.figure(figsize = figSize)
    
    nPlots = len(planeLst)
    if vmin_vmax is None:
        vmax_ = planeLst[0].max()
        vmin_ = planeLst[0].min()
    else:
        assert(len(vmin_vmax) == 2)
        vmax_ = vmin_vmax[1]
        vmin_ = vmin_vmax[0]
    
    shape_ = planeLst[0].shape
    for i in range(nPlots):
        assert(planeLst[i].shape == shape_)

    assert(axis in (0, 1, 2) and method in ('imshow', 'contour', 'scatter'))

    # Set White color for unobserved points
    # current_cmap = matplotlib.cm.get_cmap()
    # current_cmap.set_bad(color='white')

    plotPlaneLst = []
    plotPlaneMaskLst = []
    for i in range(nPlots):
        plotPlaneLst.append(np.copy(planeLst[i]))
    if planeMaskLst is not None:
        for i in range(len(planeMaskLst)):
            plotPlaneMaskLst.append(np.copy(planeMaskLst[i]))

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


    horiLabelIdc = (0, shape_[1]*1//4, shape_[1]*2//4, shape_[1]*3//4, shape_[1] - 1)
    vertLabelIdc = (0, shape_[0]*1//4, shape_[0]*2//4, shape_[0]*3//4, shape_[0] - 1)
    if axisInfo is None:
        if method == 'imshow':
            horiLabels = utilsforminds.helpers.reverseLst(horiLabelIdc)
            vertLabels = vertLabelIdc
        elif method == 'contour' or method == 'scatter':
            horiLabels = horiLabelIdc
            vertLabels = vertLabelIdc
    else:
        horiAxis = (axis + 2) % 3
        vertAxis = (axis + 1) % 3
        if method == 'imshow':
            horiLabels = (round(axisInfo[horiAxis]["max"]), 
            round(axisInfo[horiAxis]["min"] + (axisInfo[horiAxis]["max"]-axisInfo[horiAxis]["min"])*3/4),
            round(axisInfo[horiAxis]["min"] + (axisInfo[horiAxis]["max"]-axisInfo[horiAxis]["min"])*2/4),
            round(axisInfo[horiAxis]["min"] + (axisInfo[horiAxis]["max"]-axisInfo[horiAxis]["min"])*1/4),
            round(axisInfo[horiAxis]["min"]))
        elif method == 'contour' or method == 'scatter':
            horiLabels = (round(axisInfo[horiAxis]["min"]), round(axisInfo[horiAxis]["min"] + (axisInfo[horiAxis]["max"]-axisInfo[horiAxis]["min"])*1/4), 
                    round(axisInfo[horiAxis]["min"] + (axisInfo[horiAxis]["max"]-axisInfo[horiAxis]["min"])*2/4),
                    round(axisInfo[horiAxis]["min"] + (axisInfo[horiAxis]["max"]-axisInfo[horiAxis]["min"])*3/4),
                    round(axisInfo[horiAxis]["max"]))
        vertLabels = (round(axisInfo[vertAxis]["min"]), round(axisInfo[vertAxis]["min"] + (axisInfo[vertAxis]["max"]-axisInfo[vertAxis]["min"])*1/4), 
            round(axisInfo[vertAxis]["min"] + (axisInfo[vertAxis]["max"]-axisInfo[vertAxis]["min"])*2/4),
            round(axisInfo[vertAxis]["min"] + (axisInfo[vertAxis]["max"]-axisInfo[vertAxis]["min"])*3/4),
            round(axisInfo[vertAxis]["max"]))
    if convertXYaxis:
        tmp = vertLabels
        vertLabels = horiLabels
        horiLabels = tmp
    
    if axis == 0:
        if convertXYaxis:
            xlabel = 'Elevation(m)'
            ylabel = 'North(m)'
        else:
            xlabel = 'North(m)'
            ylabel = 'Elevation(m)'
    elif axis == 1:
        if convertXYaxis:
            xlabel = 'East(m)'
            ylabel = 'Elevation(m)'
        else:
            xlabel = 'Elevation(m)'
            ylabel = 'East(m)'
    elif axis == 2:
        if convertXYaxis:
            xlabel = 'North(m)'
            ylabel = 'East(m)'
        else:
            xlabel = 'East(m)'
            ylabel = 'North(m)'
    
    if method == 'imshow' or method == 'contour':
        for i in range(nPlots):
            plt.subplot(*(plotShape + [i + 1]))
            plt.title(titleLst[i], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels)
            plt.yticks(vertLabelIdc, vertLabels)

            if method == 'imshow':
                if vmin_vmax is not None:
                    img = plt.imshow(plotPlaneLst[i], vmin = vmin_, vmax = vmax_)
                else:
                    img = plt.imshow(plotPlaneLst[i])
                cbarInst = plt.colorbar()
                cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
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

            plt.xticks(horiLabelIdc, horiLabels)
            plt.yticks(vertLabelIdc, vertLabels)

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
            
            cbarInst = plt.colorbar()
            cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)


            plt.subplot(*(plotShape + [2]))
            plt.title(titleLst[1], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels)
            plt.yticks(vertLabelIdc, vertLabels)

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

                cbarInst = plt.colorbar()
                cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
            else:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x')
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], marker = 'o')
                img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = plotPlaneLst[1][xCounterOnlyRecovered, yCounterOnlyRecovered], marker = '^')

                cbarInst = plt.colorbar()
                cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
        else: # nPlots == 3
            plt.subplot(*(plotShape + [1]))
            plt.title(titleLst[0], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels)
            plt.yticks(vertLabelIdc, vertLabels)

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
            
            cbarInst = plt.colorbar()
            cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)


            plt.subplot(*(plotShape + [2]))
            plt.title(titleLst[1], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels)
            plt.yticks(vertLabelIdc, vertLabels)

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

                cbarInst = plt.colorbar()
                cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
            else:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x', s = pointSize)
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], marker = 'o', s = pointSize)
                img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = plotPlaneLst[1][xCounterOnlyRecovered, yCounterOnlyRecovered], marker = '^', s = pointSize)

                cbarInst = plt.colorbar()
                cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)


            plt.subplot(*(plotShape + [3]))
            plt.title(titleLst[2], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels)
            plt.yticks(vertLabelIdc, vertLabels)

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

                cbarInst = plt.colorbar()
                cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
            else:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x', s = pointSize)
                img = plt.scatter(xCounterSampled, yCounterSampled, c = plotPlaneLst[2][xCounterSampled, yCounterSampled], marker = 'o', s = pointSize)

                cbarInst = plt.colorbar()
                cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)

    # tikzplotlib.save(format_path_extension(filePath_))
    plt.savefig(filePath_, bbox_inches='tight')
    if save_tikz:
        tikzplotlib.save(format_path_extension(filePath_, '.tex'))
    # tikz_save('fig.tikz')
    plt.close('all')
    

def plot3DScatter(npArr, vmin = None, vmax = None, filename = None, axisInfo = None, highest_amount_proportion_threshod = None, small_delta = 1e-8, bar_label = 'gram/ton', default_point_size = 1.0, alpha_min = 0.2, transparent_cbar = False, cbar_font_size = 10, adjust_axis_ratio = True, save_tikz = True):
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

    if adjust_axis_ratio and axisInfo is not None: # https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
        z_range = axisInfo[2]['max'] - axisInfo[2]['min']
        xy_range_avg = (axisInfo[0]['max'] - axisInfo[0]['min'] + axisInfo[1]['max'] - axisInfo[1]['min']) / 2.
        fig = plt.figure(figsize=plt.figaspect(z_range / xy_range_avg))
        ## * xy_range_avg / z_range, multiply this to fig size if you want to enlarge
        ## If you wanna change font size of axis tickers
        # ax.xaxis.set_tick_params(labelsize = 12 * xy_range_avg / z_range)
        # ax.yaxis.set_tick_params(labelsize = 12 * xy_range_avg / z_range)
        # ax.zaxis.set_tick_params(labelsize = 12 * xy_range_avg / z_range)
    else:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(x, y, z, zdir='z', c= npArr[x, y, z], cmap = 'Reds', vmin = vmin, vmax = vmax, marker = '.')
    if vmin is not None and (vmax is not None or highest_amount_proportion_threshod is not None):
        if highest_amount_proportion_threshod is not None:
            assert(1.0 >= highest_amount_proportion_threshod and highest_amount_proportion_threshod >= 0.)
            # maxNth = (npArr.shape[0] * npArr.shape[1] * npArr.shape[2] * maxNthPercentageFromSmallest)
            vmax = helpers.get_proportional_ranked_value(npArr_, proportion=0.1)
        ax.scatter(x, y, z, zdir='z', c= npArr_[x, y, z], vmin = vmin, vmax = vmax, alpha = max(1. - (num_obs / num_entries) ** (1/3), alpha_min), s = default_point_size * (100/avg_length)) ## parameter s is point size
    else:
        ax.scatter(x, y, z, zdir='z', c= npArr_[x, y, z], alpha = max(1. - (num_obs / num_entries) ** (1/12), alpha_min), s = default_point_size * (100/avg_length))
    
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

    #%% colorbar plot
    axins = inset_axes(ax, width = "1%", height = "60%", loc = 'upper left')
    cbar = plt.colorbar(ax.get_children()[0], ax = ax, cax = axins)
    axins.yaxis.set_ticks_position("left")
    cbar.set_label(bar_label, fontsize = cbar_font_size)
    if not transparent_cbar:
        cbar.solids.set(alpha=1)
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
    clip.speedx(fps).write_gif(path, fps = fps)

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

def plot_bar_charts(path_to_save : str, name_numbers : dict, xlabels : list, xtitle = None, ytitle = None, bar_width = 'auto', alpha = 0.8, colors_dict = None, format = 'eps', diagonal_xtickers = False, name_errors = None, name_to_show_percentage = None, fontsize = 10, title = None, fix_legend = True, save_tikz = True):
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
    if bar_width == 'auto':
        bar_width_ = 0.30 * (2 / len(name_numbers))   
    else:
        bar_width_ = bar_width

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)

    rects_list = []
    index_copied = np.copy(index).astype(np.float)
    for name, numbers in name_numbers.items():
        rects_list.append(plt.bar(index_copied, numbers, bar_width_, alpha = alpha, label = name, **plt_bars_kwargs_dict[name]))
        index_copied += bar_width_

    if title is not None:
        plt.title(title)
    if xtitle is not None:
        plt.xlabel(xtitle, fontsize = fontsize)
    if ytitle is not None:
        plt.ylabel(ytitle, fontsize = fontsize)
    # plt.title('Scores by person')
    if diagonal_xtickers:
        plt.xticks(index + (bar_width_/2) * (len(name_numbers)-1), xlabels_copied, fontsize = fontsize, rotation = 45, ha = "right")
    else:
        plt.xticks(index + (bar_width_/2) * (len(name_numbers)-1), xlabels_copied, fontsize = fontsize)
    if fix_legend:
        numbers_tot = []
        for numbers in name_numbers.values():
            numbers_tot += numbers
        plt.ylim([0., np.max(numbers_tot) * (1. + 0.1 * len(name_numbers))])
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

def plot_group_scatter(group_df, path_to_save, group_column, y_column, color_column = None, colors_rotation = ["red", "navy", "lightgreen", "teal", "violet", "green", "orange", "blue", "coral", "yellowgreen", "sienna", "olive", "maroon", "goldenrod", "darkblue", "orchid", "crimson"], group_sequence = None, xlabel = None, ylabel = None, diagonal_xtickers = False, group_column_xtext_dict = None, order_by = None, num_truncate_small_groups = 0, save_tikz = True):
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
    if diagonal_xtickers:
        ax.set_xticklabels(x_ticks_texts, rotation = 45, ha = "right")
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

def plot_top_bars_with_rows(reordered_SNPs_info_df, path_to_save : str, color_column = None, order_by = "weights", num_bars = 10, num_rows = 2, bar_width = 'auto', opacity = 0.8, format = 'eps', xticks_fontsize = 6, diagonal_xtickers = False):
    """
    
    Parameters
    ----------
        name_numbers : dict
            For example, name_numbers['enriched'] == [0.12, 0.43, 0.12] for RMSE
        xlabels : list
            Name of groups, For example ['CNN', 'LR', 'SVR', ..]
    """

    reordered_SNPs_info_df_copied = reordered_SNPs_info_df.copy()
    reordered_SNPs_info_df_copied = reordered_SNPs_info_df_copied.sort_values(by = 'weights', ascending = False, inplace = False)
    top_20_SNPs_names = list(reordered_SNPs_info_df_copied.loc[:, "SNP"][:20])
    top_20_SNPs_weights = list(reordered_SNPs_info_df_copied.loc[:, "weights"][:20])
    top_20_SNPs_colors = list(reordered_SNPs_info_df_copied.loc[:, "color_chr"][:20])
    fig = plt.figure(figsize = (7, 4))

    n_groups = 10

    if bar_width == 'auto':
        bar_width_ = 0.1

    ## create plot
    ax_1 = plt.subplot(2, 1, 1)
    index = np.arange(n_groups)

    ## set range
    min_, max_ = np.min(top_20_SNPs_weights), np.max(top_20_SNPs_weights)
    plt.ylim([0.5 * min_, 1.2 * max_])

    rects_list = []
    plt.bar(np.arange(10), top_20_SNPs_weights[:10], alpha = opacity, color = top_20_SNPs_colors[:10])
    plt.xlabel("")
    plt.ylabel("Weights")
    if diagonal_xtickers:
        plt.xticks(index + (bar_width_/2) * (1-1), top_20_SNPs_names[:10], rotation = 45, ha = "right")
    else:
        plt.xticks(index + (bar_width_/2) * (1-1), top_20_SNPs_names[:10])
    # plt.legend()
    plt.title('Top-20 SNPs')
    for obj in ax_1.get_xticklabels():
        obj.set_fontsize(xticks_fontsize)
    
    ax_2 = plt.subplot(2, 1, 2)
    plt.ylim([0.5 * min_, 1.2 * max_])
    plt.bar(np.arange(10), top_20_SNPs_weights[10:], alpha = opacity, color = top_20_SNPs_colors[10:])
    plt.xlabel("")
    plt.ylabel("Weights")
    if diagonal_xtickers:
        plt.xticks(index + (bar_width_/2) * (1-1), top_20_SNPs_names[10:], rotation = 45, ha = "right")
    else:
        plt.xticks(index + (bar_width_/2) * (1-1), top_20_SNPs_names[10:])
    for obj in ax_2.get_xticklabels():
        obj.set_fontsize(xticks_fontsize)
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    plt.savefig(path_to_save, format = format)
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

if __name__ == '__main__':
    pass
    # plot_bar_charts('dummy', {'Frank':[12.7, 0.4, 4.4, 5.3, 7.1, 3.2], 'Guido':[6.3, 10.3, 10, 0.3, 5.3, 2.9]}, ['RR', 'Lasso', 'SVR', 'CNN', 'SVR', 'LR'], ytitle="RMSE of Prediction of TRIAILB-A")
    test_list = []
    for i in range(15):
        test_list.append({"label": f"my label_{i}", "ydata": [3.4, 3.2, 1.1, 0.3]})
    plot_xy_lines([1, 2, 3, 4], test_list, "dummy.eps")