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
import random
import numpy as np

import utilsforminds
import utilsforminds.decorators as decorators
import utilsforminds.helpers as helpers

from itertools import cycle
# from mayavi import mlab # see install manual + brew install vtk
# from mayavi.api import Engine
# import mayavi.tools.pipeline
from tqdm import tqdm
from scipy import ndimage
import moviepy.editor as mpy
from copy import deepcopy
import tikzplotlib
from itertools import product, combinations

import plotly.graph_objs as graph_objs
import plotly.graph_objs as go
import plotly.tools as tls
import plotly
import plotly.express as px
import pyvista as pv

# import logging
# alphashape_log = logging.getLogger("alphashape")
# alphashape_log.setLevel(logging.CRITICAL)
# alphashape_log.addHandler(logging.NullHandler())
# alphashape_log.propagate = False

import alphashape
from alphashape import optimizealpha as optimizealpha_original

import shapely
from shapely.geometry import MultiPoint
import trimesh

from utilsforminds.containers import merge_dictionaries
from utilsforminds.containers import merge

from sklearn import cluster
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

from scipy import interp

from math import degrees, cos, radians
# from meshplex import MeshTri

axis_rotation_dict = {0: 0, 1: 0, 2: 0}
axis_name_dict = {0: 'Easting', 1: 'Northing', 2: 'Elevation'}

def savePlotLstOfLsts(lstOfLsts, labelsLst, xlabel, ylabel, title, directory, save_tikz = False):
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

def plot2Ds(planeLst, titleLst, filePath, cbarLabel = 'amount', plotShape = [3, 1], subplot_length = 16., subplot_ratio = (1., 1.), planeMaskLst = None, axis = 2, axisInfo = None, vmin_vmax = None, method = 'imshow', convertXYaxis = False, rotate = 0, specific_value_color_dict = {"value": 0., "color": "white"}, label_font_size = 25, label_positions = [0., 1/3, 2/3], title_font_size = 25, cbar_font_size = 25, save_tikz = False, plot_obj_kwargs = None, plot_zeros_in_scatter = False):
    '''
        If you want to provide mask matrix for scatter visualization, sequence should be (original matrix, recovered matrix) or (original matrix, recovered matrix, sampled matrix), or (original matrix)

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
    if plot_obj_kwargs is None: plot_obj_kwargs = {}
    
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
        if method == 'imshow' or method == 'contour': plotPlaneLst.append(np.transpose(planeLst[i]))
        elif method == "scatter": plotPlaneLst.append(np.fliplr(planeLst[i]))
    if planeMaskLst is not None:
        for i in range(len(planeMaskLst)):
            if method == 'imshow' or method == 'contour': plotPlaneMaskLst.append(np.transpose(planeMaskLst[i]))
            elif method == "scatter": plotPlaneMaskLst.append(np.fliplr(planeMaskLst[i]))
    
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

    if method == "scatter": vertLabelIdc = [shape_[1] - idx for idx in reversed(vertLabelIdc)]
    if axisInfo is None:
        if method == 'imshow':
            horiLabels = list(reversed(vertLabelIdc)) ## ??????????????????
            vertLabels = vertLabelIdc
        elif method == 'contour' or method == 'scatter':
            horiLabels = horiLabelIdc
            vertLabels = vertLabelIdc
    else:
        horiAxis, vertAxis = get_xy_axis_from_z(axis)
        horiLabels = [round(axisInfo[horiAxis]["min"] + (axisInfo[horiAxis]["max"] - axisInfo[horiAxis]["min"]) * position_proportion) for position_proportion in label_positions]
        vertLabels = [round(axisInfo[vertAxis]["min"] + (axisInfo[vertAxis]["max"] - axisInfo[vertAxis]["min"]) * position_proportion) for position_proportion in label_positions]
    if method == "scatter": vertLabels = list(reversed(vertLabels))
    
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
                    img = plt.imshow(plotPlaneLst[i], vmin = vmin_, vmax = vmax_, aspect = 'auto', **plot_obj_kwargs)
                else:
                    img = plt.imshow(plotPlaneLst[i], aspect = 'auto')
                cbarInst = plt.colorbar(fraction=0.046 * subplot_ratio[1] * nPlots / subplot_ratio[0], pad=0.04, aspect= 10 * subplot_ratio[1] * nPlots / subplot_ratio[0], **plot_obj_kwargs)
                cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
                cbarInst.ax.tick_params(labelsize= cbar_font_size)
            elif method == 'contour':
                if vmin_vmax is not None:
                    img = plt.contour(np.flipud(plotPlaneLst[i]), vmin = vmin_, vmax = vmax_, linewidths = 2.0, colors = 'black', levels = [vmin_, vmin_ + (vmax_ - vmin_) * 1/8, vmin_ + (vmax_ - vmin_) * 2/8, vmin_ + (vmax_ - vmin_) * 3/8, vmin_ + (vmax_ - vmin_) * 4/8, vmin_ + (vmax_ - vmin_) * 5/8, vmin_ + (vmax_ - vmin_) * 6/8, vmin_ + (vmax_ - vmin_) * 7/8, vmax_], **plot_obj_kwargs)
                else:
                    img = plt.contour(np.flipud(plotPlaneLst[i]), linewidths = 2.0, colors = 'black', levels = [vmin_, vmin_ + (vmax_ - vmin_) * 1/8, vmin_ + (vmax_ - vmin_) * 2/8, vmin_ + (vmax_ - vmin_) * 3/8, vmin_ + (vmax_ - vmin_) * 4/8, vmin_ + (vmax_ - vmin_) * 5/8, vmin_ + (vmax_ - vmin_) * 6/8, vmin_ + (vmax_ - vmin_) * 7/8, vmax_], **plot_obj_kwargs)
    
    elif method == 'scatter':
        assert(nPlots ==3)
        pointSize = 12.0 * (80 * 80 / (shape_[0] * shape_[1])) ** 0.5
        # shape_ = (shape_[1], shape_[0])
        if nPlots == 2: # original, recovered
            raise Exception("Deprecated Option")
            plt.subplot(*(plotShape + [1]))
            plt.title(titleLst[0], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels, fontsize = label_font_size)
            plt.yticks(vertLabelIdc, vertLabels, fontsize = label_font_size)

            if planeMaskLst is not None:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneMaskLst[0] >= 1., 1., 0.))
                if plot_zeros_in_scatter: xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneMaskLst[0] < 1., 1., 0.))
            else:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneLst[0] > 1e-8, 1., 0.))
                if plot_zeros_in_scatter: xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneLst[0] > 1e-8, 0., 1.))
            minArr = np.ones(shape_) * vmin_

            if vmin_vmax is not None:
                if plot_zeros_in_scatter: img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], vmin = vmin_, vmax = vmax_, marker = 'x', s = pointSize, **plot_obj_kwargs) # param s = 5.0 sets size of dots for 150 * 150 * 150 mapping
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], vmin=vmin_, vmax=vmax_, marker = 'o', s = pointSize, **plot_obj_kwargs)
            
            else:
                if plot_zeros_in_scatter: img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x', **plot_obj_kwargs) # param s = 5.0 sets size of dots for 150 * 150 * 150 mapping
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], marker = 'o', s = pointSize, **plot_obj_kwargs)
            
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
                if plot_zeros_in_scatter: xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneMaskLst[1] < 1., np.where(plotPlaneMaskLst[0] < 1., 1., 0.), 0.))
            else:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneLst[0] > 1e-8, 1., 0.))
                xCounterOnlyRecovered, yCounterOnlyRecovered = np.nonzero(np.where(plotPlaneLst[0] <= 1e-8, np.where(plotPlaneLst[1] > 1e-8, 1., 0.), 0.))
                if plot_zeros_in_scatter: xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneLst[1] <= 1e-8, np.where(plotPlaneLst[0] <= 1e-8, 1., 0.), 0.))
            minArr = np.ones(shape_) * vmin_

            if vmin_vmax is not None:
                if plot_zeros_in_scatter: img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], vmin=vmin_, vmax=vmax_, marker = 'x', s = pointSize, **plot_obj_kwargs)
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], vmin=vmin_, vmax=vmax_, marker = 'o', s = pointSize, **plot_obj_kwargs)
                img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = plotPlaneLst[1][xCounterOnlyRecovered, yCounterOnlyRecovered], vmin=vmin_, vmax=vmax_, marker = '^', s = pointSize, **plot_obj_kwargs)

            else:
                if plot_zeros_in_scatter: img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x', **plot_obj_kwargs)
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], marker = 'o', **plot_obj_kwargs)
                img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = plotPlaneLst[1][xCounterOnlyRecovered, yCounterOnlyRecovered], marker = '^', **plot_obj_kwargs)

            cbarInst = plt.colorbar(fraction=0.046 * subplot_ratio[1] * nPlots / subplot_ratio[0], pad=0.04, aspect= 10 * subplot_ratio[1] * nPlots / subplot_ratio[0])
            # cbarInst = plt.colorbar(cax = cax)
            cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
            cbarInst.ax.tick_params(labelsize= cbar_font_size)

        else: # nPlots == 3
            ## Prepare array.
            if planeMaskLst is not None:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneMaskLst[0] >= 1., 1., 0.))

                if len(plotPlaneMaskLst) == 1:
                    xCounterOnlyRecovered, yCounterOnlyRecovered = np.nonzero(np.where(plotPlaneMaskLst[0] < 1., np.where(plotPlaneLst[2] > 1e-8, 1., 0.), 0.))
                elif len(plotPlaneMaskLst) == 2 or len(plotPlaneMaskLst) == 3:
                    xCounterOnlyRecovered, yCounterOnlyRecovered = np.nonzero(np.where(plotPlaneMaskLst[0] < 1., np.where(plotPlaneMaskLst[1] >= 1., 1., 0.), 0.))
                
                if len(plotPlaneMaskLst) == 3:
                    xCounterSampled, yCounterSampled = np.nonzero(np.where(plotPlaneMaskLst[2] >= 1., 1., 0.))
                elif len(plotPlaneMaskLst) == 1 or len(plotPlaneMaskLst) == 2: 
                    # xCounterSampled, yCounterSampled = np.nonzero(np.where(plotPlaneLst[1] > 0, 1., 0.))
                    xCounterSampled, yCounterSampled = xCounterOriginal, yCounterOriginal
                if plot_zeros_in_scatter: xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneMaskLst[0] < 1., 1., 0.))
            else:
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(plotPlaneLst[0] > 1e-8, 1., 0.))
                xCounterOnlyRecovered, yCounterOnlyRecovered = np.nonzero(np.where(plotPlaneLst[0] <= 1e-8, np.where(plotPlaneLst[2] > 1e-8, 1., 0.), 0.))
                xCounterSampled, yCounterSampled = np.nonzero(np.where(plotPlaneLst[1] > 1e-8, 1., 0.))
                if plot_zeros_in_scatter: xZeroCounter, yZeroCounter = np.nonzero(np.where(plotPlaneLst[1] <= 1e-8, np.where(plotPlaneLst[0] <= 1e-8, 1., 0.), 0.))

            minArr = np.ones(shape_) * vmin_

            ## Plot Original
            plt.subplot(*(plotShape + [1]))
            plt.title(titleLst[0], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels, fontsize = label_font_size)
            plt.yticks(vertLabelIdc, vertLabels, fontsize = label_font_size)

            plt.xlim(0, shape_[0])
            plt.ylim(0, shape_[1])

            if vmin_vmax is not None:
                if plot_zeros_in_scatter: img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], vmin = vmin_, vmax = vmax_, marker = 'x', s = 3.0, **plot_obj_kwargs) # param s = 5.0 sets size of dots for 150 * 150 * 150 mapping
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], vmin=vmin_, vmax=vmax_, marker = 'o', s = 3.0, **plot_obj_kwargs)
            
            else:
                if plot_zeros_in_scatter: img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x', **plot_obj_kwargs) # param s = 5.0 sets size of dots for 150 * 150 * 150 mapping
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], marker = 'o', **plot_obj_kwargs)
            
            cbarInst = plt.colorbar(fraction=0.046 * subplot_ratio[1] * nPlots / subplot_ratio[0], pad=0.04, aspect= 10 * subplot_ratio[1] * nPlots / subplot_ratio[0])
            # cbarInst = plt.colorbar(cax = cax)
            cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
            cbarInst.ax.tick_params(labelsize= cbar_font_size)

            ## Plot Sampled

            plt.subplot(*(plotShape + [2]))
            plt.title(titleLst[1], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels, fontsize = label_font_size)
            plt.yticks(vertLabelIdc, vertLabels, fontsize = label_font_size)

            plt.xlim(0, shape_[0])
            plt.ylim(0, shape_[1])

            if vmin_vmax is not None:
                if plot_zeros_in_scatter: img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], vmin=vmin_, vmax=vmax_, marker = 'x', s = pointSize, **plot_obj_kwargs)
                img = plt.scatter(xCounterSampled, yCounterSampled, c = plotPlaneLst[1][xCounterSampled, yCounterSampled], vmin=vmin_, vmax=vmax_, marker = 'o', s = pointSize, **plot_obj_kwargs)

            else:
                if plot_zeros_in_scatter: img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x', s = pointSize, **plot_obj_kwargs)
                img = plt.scatter(xCounterSampled, yCounterSampled, c = plotPlaneLst[1][xCounterSampled, yCounterSampled], marker = 'o', s = pointSize, **plot_obj_kwargs)

            cbarInst = plt.colorbar(fraction=0.046 * subplot_ratio[1] * nPlots / subplot_ratio[0], pad=0.04, aspect= 10 * subplot_ratio[1] * nPlots / subplot_ratio[0])
            # cbarInst = plt.colorbar(cax = cax)
            cbarInst.set_label(cbarLabel, fontsize = cbar_font_size)
            cbarInst.ax.tick_params(labelsize= cbar_font_size)

            ## Plot Recovered
            plt.subplot(*(plotShape + [3]))
            plt.title(titleLst[2], fontsize = title_font_size)

            plt.xlabel(xlabel, fontsize = label_font_size)
            plt.ylabel(ylabel, fontsize = label_font_size)

            plt.xticks(horiLabelIdc, horiLabels, fontsize = label_font_size)
            plt.yticks(vertLabelIdc, vertLabels, fontsize = label_font_size)

            plt.xlim(0, shape_[0])
            plt.ylim(0, shape_[1])

            if vmin_vmax is not None:
                if plot_zeros_in_scatter: img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], vmin=vmin_, vmax=vmax_, marker = 'x', s = pointSize, **plot_obj_kwargs)
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], vmin=vmin_, vmax=vmax_, marker = 'o', s = pointSize, **plot_obj_kwargs)
                img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = plotPlaneLst[2][xCounterOnlyRecovered, yCounterOnlyRecovered], vmin=vmin_, vmax=vmax_, marker = '^', s = pointSize, **plot_obj_kwargs)

            else:
                if plot_zeros_in_scatter: img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x', s = pointSize, **plot_obj_kwargs)
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], marker = 'o', s = pointSize, **plot_obj_kwargs)
                img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = plotPlaneLst[2][xCounterOnlyRecovered, yCounterOnlyRecovered], marker = '^', s = pointSize, **plot_obj_kwargs)

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
    

def plot3DScatter(npArr, vmin = None, vmax = None, filename = None, axisInfo = None, label_positions = [0., 1/3, 2/3], highest_amount_proportion_threshod = None, small_delta = 1e-8, bar_label = 'gram/ton', default_point_size = 1.0, alpha_min = 0.2, transparent_cbar = False, cbar_font_size = 11, cbar_position = 'center left', label_fontsize = 12, adjust_axis_ratio = True, figsize_default = None, save_tikz = False):
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

def plot3DScatterDistinguish(nparrValue, nparrOriginalMask = None, nparrRecoveredMask = None, vmin = None, vmax = None, filename = None, axisInfo = None, maxNth = None, cbarLabel = 'amount', save_tikz = False):
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

def plot_bar_charts(path_to_save : str, name_numbers : dict, xlabels : list = None, xlabels_for_names : dict = None, xtitle = None, ytitle = None, bar_width = 'auto', alpha = 0.8, colors_dict = None, format = 'eps', diagonal_xtickers = 0, name_errors = None, name_to_show_percentage = None, name_not_to_show_percentage_legend = None, fontsize = 10, title = None, figsize = None, ylim = None, fix_legend = True, plot_legend = True, save_tikz = False, horizontal_bars = False):
    """Plot bars.

    For example, following the below case, enriched-original(for CNN)    enriched-original(for LR)    enriched-original(for SVR)    ...
    
    Parameters
    ----------
        name_numbers : dict
            Length of bars. For example, name_numbers = {'enriched': [0.12, 0.43, 0.12], 'original': [0.15, 0.35, 0.15]} for RMSE.
        xlabels : list
            Name of groups, For example ['CNN', 'LR', 'SVR', ..]. 
        name_errors: dict
            Variance or standard deviation of each model (key in name_numbers), for example, name_errors= {'enriched': [0.03, 0.15, 0.07], 'original': [0.05, 0.06, 0.12]}
        diagonal_xtickers : bool/float
            Angle of xtickers. If True, it becomes diagonal with 45 degree.
        ylim : float
            y-axis cut range.
    """

    ## Input sanity check
    for check_target in list(filter(lambda x: True if x is not None else False, [name_numbers, name_errors])):
        for numbers_list in check_target.values():
            if any(pd.isna(numbers_list)): ## At least one element is nan/inf.
                print(f"WARNING: This function plot_bar_charts encounters invalid numbers such as nan/inf, so this call will be ignored and do nothing.")
                return None

    names = list(name_numbers.keys())
    ## Set kwargs parameters
    plt_bars_kwargs_dict = {}
    for name in names:
        plt_bars_kwargs_dict[name] = {}
        if name_errors is not None and name in name_errors.keys():
            plt_bars_kwargs_dict[name]['yerr'] = name_errors[name]
        if colors_dict is not None:
            plt_bars_kwargs_dict[name]['color'] = colors_dict[name]

    single_key = next(iter(name_numbers))
    n_groups = len(name_numbers[single_key])
    for numbers in name_numbers.values():
        assert(len(numbers) == n_groups)
    if xlabels is not None:
        assert(len(xlabels) == n_groups)
    xlabels_copied = deepcopy(xlabels)

    if xlabels_for_names is not None:
        assert(len(xlabels_for_names) == len(name_numbers))
    xlabels_for_names_list = [xlabels_for_names[name] for name in names] if xlabels_for_names is not None else None

    if name_to_show_percentage is not None and xlabels is not None:
        assert(name_to_show_percentage in names)
        assert(len(name_numbers) >= 2)
        for i in range(len(xlabels_copied)):
            scores_of_group = []
            for name in names:
                if name != name_to_show_percentage:
                    scores_of_group.append(name_numbers[name][i])
            mean = np.mean(scores_of_group)
            xlabels_copied[i] += f'({(mean - name_numbers[name_to_show_percentage][i]) * 100. / mean:.2f}%)'

    legend_prefix_dict = {name: '' for name in names}
    if name_not_to_show_percentage_legend is not None:
        if not isinstance(name_not_to_show_percentage_legend, (list, tuple)):
            name_not_to_show_percentage_legend_copied = [name_not_to_show_percentage_legend]
        else:
            name_not_to_show_percentage_legend_copied = deepcopy(name_not_to_show_percentage_legend)
        avg_number = 0.
        for name in name_not_to_show_percentage_legend_copied:
            assert(name in names)
            avg_number += sum(name_numbers[name])
        avg_number /= len(name_not_to_show_percentage_legend_copied)
        for name in names:
            if name not in name_not_to_show_percentage_legend_copied:
                legend_prefix_dict[name] = f" ({round(100. * ((avg_number - sum(name_numbers[name])) / sum(name_numbers[name])))}%)"

    if bar_width == 'auto':
        # bar_width_ = 0.30 * (2 / len(name_numbers))  
        bar_width_ = 0.20 * (2 / len(name_numbers))  
    else:
        bar_width_ = 0.20 * (2 / len(name_numbers)) * bar_width
    index = np.arange(n_groups)

    ## Horizontal bars support:
    plt_dicts = {}
    if not horizontal_bars:
        plt_dicts["xlabel"] = plt.xlabel
        plt_dicts["ylabel"] = plt.ylabel
        plt_dicts["xticks"] = plt.xticks
        plt_dicts["yticks"] = plt.yticks
        plt_dicts["ylim"] = plt.ylim
        plt_dicts["bar"] = plt.bar
    else:
        plt_dicts["xlabel"] = plt.ylabel
        plt_dicts["ylabel"] = plt.xlabel
        plt_dicts["xticks"] = plt.yticks
        plt_dicts["yticks"] = plt.xticks
        plt_dicts["ylim"] = plt.xlim
        plt_dicts["bar"] = plt.barh

    # create plot
    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize = figsize)
    
    # if horizontal_bars: ax.invert_yaxis()  # labels read top-to-bottom

    rects_list = []
    index_copied = np.copy(index).astype(np.float)
    for name in names:
        rects_list.append(plt_dicts["bar"](index_copied, name_numbers[name], bar_width_, alpha = alpha, label = name + legend_prefix_dict[name], **plt_bars_kwargs_dict[name])) ## label will be label in legend
        index_copied += bar_width_

    if title is not None:
        plt.title(title, fontsize = fontsize)
    if xtitle is not None:
        plt_dicts["xlabel"](xtitle, fontsize = fontsize)
    if ytitle is not None:
        plt_dicts["ylabel"](ytitle, fontsize = fontsize)
    # plt.title('Scores by person')

    if xlabels_copied is not None:
        if diagonal_xtickers == True:
            plt_dicts["xticks"](index + (bar_width_/2) * (len(name_numbers)-1), xlabels_copied, fontsize = fontsize, rotation = 45, ha = "right")
        elif diagonal_xtickers == False:
            plt_dicts["xticks"](index + (bar_width_/2) * (len(name_numbers)-1), xlabels_copied, fontsize = fontsize)
        else:
            plt_dicts["xticks"](index + (bar_width_/2) * (len(name_numbers)-1), xlabels_copied, fontsize = fontsize, rotation = diagonal_xtickers, ha = "right")
        if fix_legend:
            numbers_tot = []
            for numbers in name_numbers.values():
                numbers_tot += numbers
            plt_dicts["ylim"]([0., np.max(numbers_tot) * (1. + 0.1 * len(name_numbers))])
    
    if xlabels_for_names is not None:
        local_index = np.arange(len(name_numbers)) * bar_width_
        for i in range(0, n_groups):
            if diagonal_xtickers == True:
                plt_dicts["xticks"](local_index + bar_width_ * 0. + i, xlabels_for_names_list, fontsize = fontsize, rotation = 45, ha = "right")
            elif diagonal_xtickers == False:
                plt_dicts["xticks"](local_index + bar_width_ * 0. + i, xlabels_for_names_list, fontsize = fontsize)
            else:
                plt_dicts["xticks"](local_index + bar_width_ * 0. + i, xlabels_for_names_list, fontsize = fontsize, rotation = diagonal_xtickers, ha = "right")
    
    if ylim is not None:
        assert(len(ylim) == 2)
        if ylim[1] == None:
            plt_dicts["ylim"](bottom = ylim[0])
        elif ylim[0] == None:
            plt_dicts["ylim"](top = ylim[1])
        else:
            plt_dicts["ylim"](ylim)
    
    plt_dicts["yticks"](fontsize = fontsize * 0.6)
        
    if plot_legend:
        plt.legend(fontsize = fontsize * 0.6)

    plt.tight_layout()
    # plt.show()
    plt.savefig(path_to_save, format = format)
    if save_tikz:
        tikzplotlib.save(filepath = format_path_extension(path_to_save, '.tex'))

def plot_multiple_lists(lists_dict: dict, path_to_save: str, labels : dict = {'x': 'Iteration', 'y': 'Loss'}, format = 'eps', fontsize = 15, save_tikz = False):
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
    plt.xlabel(labels['x'], fontsize = fontsize)
    plt.ylabel(labels['y'], fontsize = fontsize)
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

def plot_group_scatter(group_df, path_to_save, group_column, y_column, color_column = None, colors_rotation = ["red", "navy", "lightgreen", "teal", "violet", "green", "orange", "blue", "coral", "yellowgreen", "sienna", "olive", "maroon", "goldenrod", "darkblue", "orchid", "crimson"], group_sequence = None, xlabel = None, ylabel = None, rotation_xtickers = 0, group_column_xtext_dict = None, order_by = None, num_truncate_small_groups = 0, save_tikz = False):
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

def plot_top_bars_with_rows(data_df, path_to_save : str, color_column = None, colors_rotation = ["red", "navy", "lightgreen", "teal", "violet", "green", "orange", "blue", "coral", "yellowgreen", "sienna", "olive", "maroon", "goldenrod", "darkblue", "orchid", "crimson"], order_by = "weights", x_column = None, group_column = None, xticks_replace_dict = None, xlabel = None, ylabel = None, title = None, num_bars = 10, num_rows = 2, re_range_max_min_proportion = None, show_group_error = True, show_group_size = True, bar_width = 'auto', opacity = 0.8, xticks_fontsize = 10, rotation_xtickers = 0, format = 'eps', save_tikz = False):
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

def plot_xy_lines(x, y_dict_list : list, path_to_save : str, title = None, x_label = None, y_label = None, figsize= (17, 5), label_fontsize = 20, font_size_proportion = 1.0, showlegend= True, format = 'eps', save_tikz = False):
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
        for required_key in ["ydata"]:
            assert(required_key in y_dict_list_copied[y_dict_idx].keys())
        y_dict_list_copied[y_dict_idx]["ydata"] = np.array(y_dict_list_copied[y_dict_idx]["ydata"])
        assert(y_dict_list_copied[y_dict_idx]["ydata"].shape == x_arr_copied.shape)
        y_dict_list_copied[y_dict_idx]["ydata"] = y_dict_list_copied[y_dict_idx]["ydata"][x_arr_sorted_ind]

    plt.figure(figsize=figsize)
    for y_dict in y_dict_list_copied:
        y_dict_no_ydata = deepcopy(y_dict)
        y_dict_no_ydata.pop("ydata", None)
        plt.plot(x_arr_copied, y_dict["ydata"], **y_dict_no_ydata)

    if showlegend: plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize= 17 * font_size_proportion)
    if title is not None:
        plt.title(title, fontsize = int(label_fontsize * 1.5 * font_size_proportion))
    if x_label is not None:
        plt.xlabel(x_label, fontsize = label_fontsize * font_size_proportion)
    if y_label is not None:
        plt.ylabel(y_label, fontsize = label_fontsize * font_size_proportion)
    plt.xticks(fontsize= 14 * font_size_proportion)
    plt.yticks(fontsize= 14 * font_size_proportion)

    plt.tight_layout()
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

def get_volume_of_alpha_shape(points, alpha):
    """
    
    Parameters
    ----------
    points : ndarray
        For example, [[1, 2, 1], [3, 13, 61], ...].
    
    Returns
    -------
    volume : float
    """

    cloud = pv.PolyData(points) # set up the pyvista point cloud structure
    mesh = cloud.delaunay_3d(alpha= alpha, progress_bar= True)
    boundary_mesh = mesh.extract_geometry()
    return boundary_mesh.volume

def _testalpha(points, alpha: float, value_flat_for_alphashape_optimization: np.ndarray):
    """
    Evaluates an alpha parameter.

    This helper function creates an alpha shape with the given points and alpha
    parameter.  It then checks that the produced shape is a Polygon and that it
    intersects all the input points.
    ref: https://alphashape.readthedocs.io/en/latest/_modules/alphashape/optimizealpha.html

    Args:
        points: data points
        alpha: alpha value

    Returns:
        bool: True if the resulting alpha shape is a single polygon that
            intersects all the input data points.
    """

    try:
        polygon = alphashape.alphashape(points, alpha = alpha)
    except Exception as e:
        print(f"Exception while creating alphashape for _testalpha: {e}")
        return False
    value_sum = 0
    if isinstance(polygon, (shapely.geometry.polygon.Polygon, shapely.geometry.multipolygon.MultiPolygon)):
        points_multi = MultiPoint(list(points))
        # return all([polygon.intersects(point) for point in points])
        for point_idx in range(len(points_multi)):
            if polygon.intersects(points_multi[point_idx]): value_sum += value_flat_for_alphashape_optimization[point_idx]
        # return value_sum / max(polygon.area, 1e-16)
        return value_sum / max(get_volume_of_alpha_shape(points, alpha), 1e-16)
    elif isinstance(polygon, trimesh.base.Trimesh) and len(polygon.faces) > 0:
        distances = trimesh.proximity.signed_distance(polygon, list(points))
        for point_idx in range(len(distances)):
            if distances[point_idx] >= 0: value_sum += value_flat_for_alphashape_optimization[point_idx]
        return value_sum / (polygon.area ** (3/2) * np.mean(polygon.area_faces))
    else:
        return False

def optimizealpha(points, value_flat_for_alphashape_optimization,
                  max_iterations: int = 100, lower: float = 0.,
                  upper: float = 1000., search_epsilon = 0.01, verbose: int = 2):
    """
    Solve for the alpha parameter.

    Attempt to determine the alpha parameter that best wraps the given set of
    points in one polygon without dropping any points.

    Note:  If the solver fails to find a solution, a value of zero will be
    returned, which when used with the alphashape function will safely return a
    convex hull around the points.
    ref: https://alphashape.readthedocs.io/en/latest/_modules/alphashape/optimizealpha.html

    Args:

        points: an iterable container of points
        max_iterations (int): maximum number of iterations while finding the
            solution
        lower: lower limit for optimization
        upper: upper limit for optimization
        verbose: verbosity

    Returns:

        float: The optimized alpha parameter

    """
    # return optimizealpha_original(points= np.array(points), max_iterations= 100, lower= 0., upper= 100, silent= False)
    # Convert to a shapely multipoint object if not one already
    # if USE_GP and isinstance(points, geopandas.GeoDataFrame):
    #     points = points['geometry']

    # Set the bounds
    assert lower >= 0, "The lower bounds must be at least 0"
    # Ensure the upper limit bounds the solution

    if not _testalpha(points, lower, value_flat_for_alphashape_optimization = value_flat_for_alphashape_optimization):
        # if verbose >= 1:
        #     logging.error('the max float value does not bound the alpha '
        #                   'parameter solution, that is the best alpha exists above the upper bound.')
        if verbose >= 1:
            print("Error: The lowest alpha constructs the disconnected alpha shape.")
        return lower

    # Begin the bisection loop
    counter = 0
    best_value = 0.
    best_alpha = lower
    while (upper - lower) > search_epsilon:
        # Bisect the current bounds
        test_alpha = (upper + lower) * .5

        # Update the bounds to include the solution space
        current_value = _testalpha(points, test_alpha, value_flat_for_alphashape_optimization = value_flat_for_alphashape_optimization)
        if current_value:
            lower = test_alpha
            if current_value > best_value:
                if verbose >= 2:
                    print(f"Best alpha {test_alpha} found with value {current_value}.")
                best_alpha = test_alpha
                best_value = current_value
        else:
            upper = test_alpha

        # Handle exceeding maximum allowed number of iterations
        counter += 1
        if counter > max_iterations:
            if verbose >= 2:
                logging.warning('maximum allowed iterations reached while '
                                'optimizing the alpha parameter')
            lower = 0.
            break
    return best_alpha

@decorators.grid_of_functions(param_to_grid= "alphahull", param_formatter_dict= {"path_to_save_static": lambda **kwargs: kwargs["path_to_save_static"].split(".")[0] + "_" + str(kwargs["alphahull"] * 100)}, grid_condition= lambda **kwargs: True if ("kinds_to_plot" in kwargs.keys() and "alphashape" in kwargs["kinds_to_plot"]) else False)
def plot_3D_plotly(nparr_3D, path_to_save_static : str, do_save_html : bool = True, grid_formatter_to_save_tri = None, save_info_txt = False, kinds_to_plot : list = None, marker_kwargs : dict = None, vmin = None, vmax = None, alpha_shape_kwargs : dict = None, alphahull: float = 0.65, points_decider = lambda x: x > 1e-8, observation_mask_nparr_3D = None, title= None, points_legends : dict = None, alpha_shape_legend = "alpha-shape", scene_kwargs : dict = None, xyz_tickers = None, axis_kwargs : dict = None, layout_kwargs : dict = None, figsize_ratio : dict = None, camera = None, showgrid = False, zeroline = True, showline = False, transparent_bacground = True, colorbar_kwargs = None, get_hovertext = None, alpha_shape_clustering = False, use_pyvista_alphashape = True, additional_gos = None, coordinate_info= None, layout_legend = None, value_arr_for_alphashape_optimization = None, marker_symbols = None, pyvista_alpha_model_kwargs = None, **kwargs):
    """
    
    Parameters
    ----------
    alphahull : float
        The larger alphahull results the more sharp(shrinked) alpha-shape. alpha = 1 / alphahull
    """

    assert(len(nparr_3D.shape) == 3)
    assert(use_pyvista_alphashape) ## optimization of alpha currently only implemented over pyvista.
    input_shape = deepcopy(nparr_3D.shape)
    tri1 = None ## To check whether triangulation is already calculated.
    
    ## assign default values
    suffix = 0
    if os.path.exists(utilsforminds.strings.format_extension(path_to_save_static, 'png')):
        while os.path.exists(utilsforminds.strings.format_extension(f'{path_to_save_static}_{suffix}', 'png')):
            suffix += 1
            if suffix > 100:
                raise Exception("Exceeds the number of tries while generating file name, please check code.")
        path_to_save = f"{path_to_save_static}_{suffix}"
    path_to_save = path_to_save_static
    title_copied = "" if title is None else title
    if kinds_to_plot is None:
        kinds_to_plot = ["scatter"]
    else:
        for kind in kinds_to_plot:
            assert(kind in ["scatter", "alphashape"])
        kinds_to_plot = deepcopy(kinds_to_plot)

    marker_kwargs_local = {"colorscale": 'Rainbow', "size": 2.}
    if vmin is not None:
        marker_kwargs_local["cmin"] = vmin
    if vmax is not None:
        marker_kwargs_local["cmax"] = vmax

    marker_kwargs_local = utilsforminds.containers.merge_dictionaries([marker_kwargs_local, marker_kwargs])
    # marker_kwargs_local["colorbar"] = utilsforminds.containers.merge_dictionaries([{"title": "colorbar", "xpad": 0.0}, colorbar_kwargs])
    marker_kwargs_local["colorbar"] = utilsforminds.containers.merge_dictionaries([{"title": "colorbar"}, colorbar_kwargs])
    alpha_shape_kwargs_local = utilsforminds.containers.merge_dictionaries([{"color": "orange", "opacity": 0.3, "showlegend": True}, alpha_shape_kwargs])
    alpha_shape_kwargs_local.update({"alphahull": alphahull})
    points_legends_local = utilsforminds.containers.merge_lists([["Added to Mask", "Mask"], points_legends])
    scene_kwargs_local = utilsforminds.containers.merge_dictionaries([{"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z"}, scene_kwargs])
    axis_kwargs_local = utilsforminds.containers.merge_dictionaries([{"showgrid": showgrid, "zeroline": zeroline, "showline": showline, "zerolinecolor": "black", "backgroundcolor": "rgb(255, 255, 255)"}, axis_kwargs])
    layout_legend_local = utilsforminds.containers.merge_dictionaries([dict(orientation= "h"), layout_legend])
    marker_symbols_local = merge(dict(nonmask= "circle", mask= "cross"), marker_symbols)

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

    camera_copied = merge_dictionaries([dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.25, y=1.25, z=1.25), projection= dict(type= "orthographic")), camera]) ## projection= dict(type= "orthographic") gives vertial elevation axis, which looks professinal for geologist.

    if additional_gos is None: additional_gos = []
    
    ## Generates objects to plot
    plot_objects = deepcopy(additional_gos)
    nparr_3D_whole = np.where(points_decider(nparr_3D), 1., 0.)
    if observation_mask_nparr_3D is None:
        observation_mask_nparr_3D_added = np.zeros(input_shape)
        nparr_3D_non_mask = nparr_3D_whole
    else:
        # observation_mask_nparr_3D_added = np.where(nparr_3D_whole == 1., np.where(points_decider(observation_mask_nparr_3D), 0., 1.), 0.)
        observation_mask_nparr_3D_added = np.where(nparr_3D_whole == 1., np.where(observation_mask_nparr_3D, 0., 1.), 0.)
        nparr_3D_non_mask = np.where(nparr_3D_whole, np.where(observation_mask_nparr_3D_added, 0., 1.), 0.)
    if "scatter" in kinds_to_plot:
        for is_main, mask_to_plot, marker_symbol, points_legend in zip([True, False], [nparr_3D_non_mask, observation_mask_nparr_3D_added], [marker_symbols_local["nonmask"], marker_symbols_local["mask"]], [points_legends_local[1], points_legends_local[0]]):
            x, y, z = mask_to_plot.nonzero()
            colors_arr = utilsforminds.numpy_array.push_arr_to_range(nparr_3D[x, y, z], vmin = vmin, vmax = vmax) ## don't need maybe, because of cmin and cmax.
            if get_hovertext is not None:
                hovertext = get_hovertext(x = x, y = y, z = z, value = colors_arr)
                hoverinfo = "text"
            else:
                hovertext = None
                hoverinfo = None
            if is_main: ## Only plot one colorbar.
                marker_kwargs_copied = marker_kwargs_local
            else:
                marker_kwargs_copied = utilsforminds.containers.copy_dict_and_delete_element(marker_kwargs_local, ["colorbar"])
            plot_objects.append(graph_objs.Scatter3d(mode = 'markers', name = points_legend, x = x, y = y, z = z, marker = graph_objs.Marker(color = colors_arr, symbol = marker_symbol, **marker_kwargs_copied), hovertext = hovertext, hoverinfo = hoverinfo, showlegend = True)) ## parameter, e.g. x, y, .. can be used in hovertemplate.
    if "alphashape" in kinds_to_plot:
        if use_pyvista_alphashape:
            alpha = 1. / max(alpha_shape_kwargs_local["alphahull"], 1e-16)
            del alpha_shape_kwargs_local["alphahull"]
            pyvista_alpha_model_kwargs_loc = merge(dict(model = "optimize_alpha", kwargs = dict()), pyvista_alpha_model_kwargs)
        mask_nparr_3D_alphashape = nparr_3D_whole
        x, y, z = mask_nparr_3D_alphashape.nonzero()
        # if do_normalize_coordinates:
        #     def min_max_scale(arr):
        #         max_ = arr.max()
        #         min_ = arr.min()
        #         if max_ > min_:
        #             return (arr - min_) / (max_ - min_)
        #         else:
        #             return arr
        #     x = min_max_scale(x)
        #     y = min_max_scale(y)
        #     z = min_max_scale(z)
        def optimize_alpha(x, y, z):
            if value_arr_for_alphashape_optimization is not None and x.shape[0] > 10:
                points = np.array(list(zip(x, y, z)))
                value_flat_for_alphashape_optimization = value_arr_for_alphashape_optimization[x, y, z]
                best_alpha = optimizealpha(points = points, value_flat_for_alphashape_optimization = value_flat_for_alphashape_optimization, lower = 0, upper = alpha * 100)
            else:
                best_alpha = alpha
            with open(utilsforminds.strings.format_extension(path_to_save, "txt"), "a") as text_file:
                text_file.write(f"alpha: {best_alpha}\n")
            return best_alpha

        alphas_radius_list = []
        def get_alpha(x_loc, y_loc, z_loc):
            if pyvista_alpha_model_kwargs_loc["model"] == "optimize_alpha":
                return optimize_alpha(x_loc, y_loc, z_loc)
            elif pyvista_alpha_model_kwargs_loc["model"] == "weighted_alpha":
                if pyvista_alpha_model_kwargs_loc["kwargs"]["kinds"] == "local_density":
                    ## Calculate the array of local densities
                    window = pyvista_alpha_model_kwargs_loc["kwargs"]["window"]
                    window_size = (window[0][1] + window[0][0]) * (window[1][1] + window[1][0]) * (window[2][1] + window[2][0])
                    global_density = np.count_nonzero(nparr_3D_whole) / (nparr_3D_whole.shape[0] * nparr_3D_whole.shape[1] * nparr_3D_whole.shape[2])
                    local_densities = np.zeros(shape= nparr_3D_whole.shape)
                    num_points = x_loc.shape[0]
                    for point_idx in range(num_points):
                        point = [x_loc[point_idx], y_loc[point_idx], z_loc[point_idx]]
                        local_densities[point[0], point[1], point[2]] += np.count_nonzero(nparr_3D_whole[max(0, point[0] - window[0][0]):min(nparr_3D_whole.shape[0], point[0] + window[0][1]), max(0, point[1] - window[1][0]):min(nparr_3D_whole.shape[1], point[1] + window[1][1]), max(0, point[2] - window[2][0]):min(nparr_3D_whole.shape[2], point[2] + window[2][1])])
                def get_alpha_value(indices_four_points, circumradius):
                    """The callable which will be given to alpha shape construction function.

                    This function is passed to alphashape package. Currently alpha = circumradius / cosine(theta)
                    
                    Parameters
                    ----------
                    indices_four_points : np array
                        Array of four (three in 2D points input) integers indicating the indices of four points constituting the simplex (triangle in 2D, tetrahedron in 3D), for example [3, 4, 0, 2].
                    circumradius : float
                        The circumradius of circumcircle (https://artofproblemsolving.com/wiki/index.php/Circumradius).
                    
                    Returns
                    -------
                    alphahull : float, 1 / alpha_radius
                    """

                    if pyvista_alpha_model_kwargs_loc["kwargs"]["kinds"] == "angular":
                        if "theta" in pyvista_alpha_model_kwargs_loc["kwargs"].keys():
                            theta = pyvista_alpha_model_kwargs_loc["kwargs"]["theta"]
                        else:
                            theta = 60
                        
                        # alpha_radius = circumradius / cos(radians(theta))
                        alpha_radius = circumradius
                        alphahull = 1 / alpha_radius
                    
                    elif pyvista_alpha_model_kwargs_loc["kwargs"]["kinds"] == "local_density":
                        num_points = 0
                        for idx in indices_four_points:
                            assert(local_densities[x_loc[idx], y_loc[idx], z_loc[idx]] > 0)
                            assert(nparr_3D_whole[x_loc[idx], y_loc[idx], z_loc[idx]] != 0)
                            num_points += local_densities[x_loc[idx], y_loc[idx], z_loc[idx]]
                        density_ratio = ((num_points / (window_size * 4)) / global_density) ** (1/3)
                        alpha_radius = max(pyvista_alpha_model_kwargs_loc["kwargs"]["alpha_radius"] / max(density_ratio, 1e-16), 0.5)
                        # print(f"alpha_radius: {pyvista_alpha_model_kwargs_loc['kwargs']['alpha_radius'] / max(density_ratio, 1e-16)}")
                        alphahull = 1 / alpha_radius
                        
                    elif pyvista_alpha_model_kwargs_loc["kwargs"]["kinds"] == "constant":
                        if "alpha_radius" in pyvista_alpha_model_kwargs_loc["kwargs"].keys():
                            alpha_radius = pyvista_alpha_model_kwargs_loc["kwargs"]["alpha_radius"]
                        else:
                            alpha_radius = 1.1
                        alphahull = 1 / alpha_radius
                    else:
                        raise Exception(f"Unsupported kinds: {pyvista_alpha_model_kwargs_loc['kwargs']['kinds']}")
                    
                    ## Constraint the maximum alpha radius.
                    # alpha_radius = min(alpha_radius, 5)
                    # alphahull = 1 / alpha_radius
                    return alphahull ## 1 / * because alphashape package use alpha parameter in the meaning of alphahull not alpha-radius, in my guess..
                return get_alpha_value


        if get_hovertext is not None:
            hovertext = get_hovertext(x = x, y = y, z = z, value = nparr_3D[x, y, z])
            hoverinfo = "text"
        else:
            hovertext = None
            hoverinfo = None
        if alpha_shape_clustering:
            positions = np.stack([x, y, z], axis = -1)
            number_of_clusters_to_plot = 5

            # cluster_labels = cluster.OPTICS(min_samples = min(positions.shape[0], 20)).fit_predict(positions) ## Change here if you wanna use different clustering.
            cluster_labels = cluster.Birch(n_clusters= number_of_clusters_to_plot).fit_predict(positions)

            smallest_number_of_points_in_cluster = -1
            cluster_labels_to_plot = []
            number_of_points_in_clusters = []
            for cluster_label in set(cluster_labels):
                number_of_points_in_cluster = np.sum(np.where(cluster_labels == cluster_label, 1., 0.))
                if len(cluster_labels_to_plot) < number_of_clusters_to_plot:
                    cluster_labels_to_plot.append(cluster_label)
                    number_of_points_in_clusters.append(number_of_points_in_cluster)
                else:
                    smallest_number_of_points_in_cluster = min(number_of_points_in_clusters)
                    if smallest_number_of_points_in_cluster < number_of_points_in_cluster:
                        arg_smallest_number_of_points_in_cluster = np.argmin(number_of_points_in_clusters)
                        cluster_labels_to_plot[arg_smallest_number_of_points_in_cluster] = cluster_label
                        number_of_points_in_clusters[arg_smallest_number_of_points_in_cluster] = number_of_points_in_cluster

            showscale = True ## Only the first plot has colorbar.
            for cluster_label in cluster_labels_to_plot:
                x_this_cluster = x[cluster_labels == cluster_label]
                y_this_cluster = y[cluster_labels == cluster_label]
                z_this_cluster = z[cluster_labels == cluster_label]
                best_alpha = get_alpha(x_this_cluster, y_this_cluster, z_this_cluster)
                if use_pyvista_alphashape:
                    tri1, tri2, tri3, volume = get_triangles_of_alpha_shape(x_this_cluster, y_this_cluster, z_this_cluster, alpha = best_alpha, model= pyvista_alpha_model_kwargs_loc["model"])
                    plot_objects.append(graph_objs.Mesh3d(name = f"{alpha_shape_legend}-cluster-{cluster_label}", x = x_this_cluster, y = y_this_cluster, z = z_this_cluster, hovertext = hovertext, hoverinfo = hoverinfo, intensity = utilsforminds.numpy_array.push_arr_to_range(nparr_3D[x_this_cluster, y_this_cluster, z_this_cluster], vmin = vmin, vmax = vmax), colorbar = marker_kwargs_local["colorbar"], showscale = showscale, i = tri1, j = tri2, k = tri3, **alpha_shape_kwargs_local))
                else:
                    plot_objects.append(graph_objs.Mesh3d(name = f"{alpha_shape_legend}-cluster-{cluster_label}", x = x_this_cluster, y = y_this_cluster, z = z_this_cluster, hovertext = hovertext, hoverinfo = hoverinfo, intensity = utilsforminds.numpy_array.push_arr_to_range(nparr_3D[x_this_cluster, y_this_cluster, z_this_cluster], vmin = vmin, vmax = vmax), colorbar = marker_kwargs_local["colorbar"], showscale = showscale, **alpha_shape_kwargs_local))
                showscale = False ## Only the first plot has colorbar.
        else:
            best_alpha = get_alpha(x, y, z)
            if use_pyvista_alphashape:
                tri1, tri2, tri3, volume = get_triangles_of_alpha_shape(x, y, z, alpha = best_alpha, model= pyvista_alpha_model_kwargs_loc["model"])

                ## Plot the histgram of alpha radiuses.
                # if pyvista_alpha_model_kwargs_loc["kwargs"]["kinds"] == "angular":
                #     fig_histo = go.Figure(data=[go.Histogram(x= alpha_radius_list, xbins=dict(start= 0, end= 20, size= 0.5))])
                # else:
                #     fig_histo = go.Figure(data=[go.Histogram(x= alpha_radius_list)])
                # fig_histo.update_layout(xaxis= dict(tickfont= dict(size= 30)))
                # fig_histo.update_layout(yaxis= dict(tickfont= dict(size= 30)))
                # # fig_histo.show()
                # fig_histo.write_html(utilsforminds.strings.format_extension(path_to_save.split(".")[:-1][0] + "_histo" , "html"))

                plot_objects.append(graph_objs.Mesh3d(name = alpha_shape_legend, x = x, y = y, z = z, hovertext = hovertext, hoverinfo = hoverinfo, intensity = utilsforminds.numpy_array.push_arr_to_range(nparr_3D[x, y, z], vmin = vmin, vmax = vmax), colorbar = marker_kwargs_local["colorbar"], i = tri1, j = tri2, k = tri3, **alpha_shape_kwargs_local))
            else:
                plot_objects.append(graph_objs.Mesh3d(name = alpha_shape_legend, x = x, y = y, z = z, hovertext = hovertext, hoverinfo = hoverinfo, intensity = utilsforminds.numpy_array.push_arr_to_range(nparr_3D[x, y, z], vmin = vmin, vmax = vmax), colorbar = marker_kwargs_local["colorbar"], **alpha_shape_kwargs_local))

            if grid_formatter_to_save_tri is not None: ## Saving clustered alpha-shape will be implemented in the future in case we need.
                if tri1 is None:
                    tri1, tri2, tri3, volume = get_triangles_of_alpha_shape(x, y, z, alpha = best_alpha, model= pyvista_alpha_model_kwargs_loc["model"])
                with open(utilsforminds.strings.format_extension(path_to_save, "tri"), "w") as text_file:
                    for triangle_idx in range(tri1.shape[0]):
                        line = ""
                        for triangle in [tri1, tri2, tri3]:
                            for position, position_idx in zip([x, y, z], range(3)):
                                line = line + f"{grid_formatter_to_save_tri(position = position[triangle[triangle_idx]], axis = position_idx)} "
                        text_file.write(line[:-1] + "\n")
        
        if save_info_txt and use_pyvista_alphashape and not alpha_shape_clustering: ## calculate and print alpha-shape information.
            ## calculate the volume of alpha-shape, https://stackoverflow.com/questions/61638966/find-volume-of-object-given-a-triangular-mesh
            if coordinate_info is not None: 
                x_physical, y_physical, z_physical= x * coordinate_info[0]["grid"] + coordinate_info[0]["min"], y * coordinate_info[1]["grid"] + coordinate_info[1]["min"], z * coordinate_info[2]["grid"] + coordinate_info[2]["min"]
                volume = volume * coordinate_info[0]["grid"] * coordinate_info[1]["grid"] * coordinate_info[2]["grid"]
            else:
                x_physical, y_physical, z_physical= x, y, z

            faces= np.concatenate([[tri1], [tri2], [tri3]], axis=0).T
            verts= np.concatenate([[x_physical], [y_physical], [z_physical]], axis=0).T
            # volume = sum(MeshTri(verts, faces).cell_volumes) ## analyze mesh: https://github.com/mikedh/trimesh

            surface_area = 0.
            indices_of_unique_vertices = []
            for triangle_idx in range(tri1.shape[0]):
                x1, y1, z1, x2, y2, z2, x3, y3, z3 = x_physical[tri1[triangle_idx]], y_physical[tri1[triangle_idx]], z_physical[tri1[triangle_idx]], x_physical[tri2[triangle_idx]], y_physical[tri2[triangle_idx]], z_physical[tri2[triangle_idx]], x_physical[tri3[triangle_idx]], y_physical[tri3[triangle_idx]], z_physical[tri3[triangle_idx]]
                surface_area += (1/2) * ((x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3) ** 2. + (x2 * z1 - x3 * z1 - x1 * z2 + x3 * z2 + x1 * z3 - x2 * z3) ** 2. + (y2 * z1 - y3 * z1 - y1 * z2 + y3 * z2 + y1 * z3 - y2 * z3) ** 2.) ** 0.5 ## https://stackoverflow.com/questions/59597399/area-of-triangle-using-3-sets-of-coordinates and https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
                for triangles in [tri1, tri2, tri3]:
                    if triangles[triangle_idx] not in indices_of_unique_vertices: indices_of_unique_vertices.append(triangles[triangle_idx])
            unique_amounts = nparr_3D[x[indices_of_unique_vertices], y[indices_of_unique_vertices], z[indices_of_unique_vertices]]
            with open(utilsforminds.strings.format_extension(path_to_save, "txt"), "a") as text_file:
                text_file.write(f"total grade: {np.sum(unique_amounts)}\navg grade: {np.mean(unique_amounts)}\nnumber of points: {unique_amounts.shape[0]}\nvolume: {volume}\nsurface area: {surface_area}\nnumber of triangulations: {tri1.shape[0]}\ngrade per volume: {np.sum(unique_amounts) / max(volume, 1e-8)}\nalpha optimization: {value_arr_for_alphashape_optimization is not None or value_arr_for_alphashape_optimization}")           

    scene = graph_objs.layout.Scene(xaxis = {"range": [0, input_shape[0]], "tickvals": xyz_tickers_copied["x"]["tickvals"], "ticktext": xyz_tickers_copied["x"]["ticktext"], "tickangle": -90, **axis_kwargs_local},
    yaxis = {"range": [0, input_shape[1]], "tickvals": xyz_tickers_copied["y"]["tickvals"], "ticktext":  xyz_tickers_copied["y"]["ticktext"], "tickangle": -90, **axis_kwargs_local},
    zaxis = {"range": [0, input_shape[2]], "tickvals": xyz_tickers_copied["z"]["tickvals"], "ticktext":  xyz_tickers_copied["z"]["ticktext"], "tickangle": 0, **axis_kwargs_local}, **scene_kwargs_local)

    # layout = graph_objs.Layout(title = title_copied, width = figsize_copied["width"], height = figsize_copied["height"], scene = scene, scene_camera = camera_copied)
    layout = graph_objs.Layout(title = title_copied, scene = scene, scene_camera = camera_copied, **layout_kwargs_local)

    fig = graph_objs.Figure(data = plot_objects, layout = layout)

    ## Set grid line and zero line
    if figsize_ratio is not None:
        fig.update_layout(scene_aspectmode='manual', scene_aspectratio=figsize_ratio) ## scene_aspectmode='auto' is default argument
    if transparent_bacground:
        fig.update_layout(paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)')
    fig.update_layout(legend= layout_legend_local)

    ## Save the result
    if do_save_html:
        fig.write_html(utilsforminds.strings.format_extension(path_to_save, "html"))
    fig.write_image(utilsforminds.strings.format_extension(path_to_save, "png"))

def get_triangles_of_alpha_shape(x, y, z, alpha, model= "optimize_alpha"):
    """
    
    Returns
    -------
    tri1, tri2, tri3: 1D numpy array of indices of points.
        tri1[j], tri2[j], tri3[j] indicate the first, second, third point of j-th triangle. For example, (x[tri1[j]], y[tri1[j]], z[tri1[j]]) is the first point of triangle.
    """

    assert(model in ["weighted_alpha", "optimize_alpha"])

    if model == "optimize_alpha":
        cloud = pv.PolyData(np.array(list(zip(x, y, z)))) # set up the pyvista point cloud structure
        #Extract the total simplicial structure of the alpha shape defined via pyvista
        # mesh = cloud.delaunay_3d(alpha= 1. / alpha_shape_kwargs_local["alphahull"]) ## (pyvista alpha = 1/alphahull; alphahull is an attribute of the Plotly Mesh3d)
        mesh = cloud.delaunay_3d(alpha= alpha, progress_bar= True)

        # #and select its simplexes of dimension 0, 1, 2, 3:
        # unconnected_points3d = []  #isolated 0-simplices
        # edges = [] # isolated edges, 1-simplices
        # faces = []  # triangles that are not faces of some tetrahedra
        # tetrahedra = []  # 3-simplices
        # for k  in tqdm(mesh.offset):  #HERE WE CAN ACCESS mesh.offset
        #     length = mesh.cells[k] 
        #     if length == 2:
        #         edges.append(list(mesh.cells[k+1: k+length+1]))
        #     elif length ==3:
        #         faces.append(list(mesh.cells[k+1: k+length+1]))
        #     elif length == 4:
        #         tetrahedra.append(list(mesh.cells[k+1: k+length+1]))
        #     elif length == 1:
        #         unconnected_points3d.append(mesh.cells[k+1])

        # get faces of the mesh
        boundary_mesh = mesh.extract_geometry()
        boundary_faces = boundary_mesh.faces.reshape((-1,4))[:, 1:]  
        # get indices from mesh triangles
        # boundary_points= points3d[boundary_faces]
        tri1, tri2, tri3 = boundary_faces.T ## x1, y1, z1 is index (in x, y, z) of point of triangle, so there can be duplication, because it is combination of three indices.
        return tri1, tri2, tri3, boundary_mesh.volume
    elif model == "weighted_alpha":
        positions = np.stack([x, y, z], axis = -1)
        alpha_shape = alphashape.alphashape(points= positions, alpha= alpha)
        faces = alpha_shape.faces ## faces are indices for "alpha_shape.vertices" NOT for "positions", don't confuse this.

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces)
        # plt.show()
        # alpha_shape.show()

        vertices = alpha_shape.vertices
        index_map_vertices_to_positions = np.argwhere(np.all(vertices.reshape(vertices.shape[0],1,-1) == positions,2)) ## ref: https://stackoverflow.com/questions/55612617/match-rows-of-two-2d-arrays-and-get-a-row-indices-map-using-numpy
        assert(np.all(index_map_vertices_to_positions[:, 0] == np.arange(vertices.shape[0])))
        index_map_vertices_to_positions = index_map_vertices_to_positions[:, 1]
        return index_map_vertices_to_positions[faces[:, 0]], index_map_vertices_to_positions[faces[:, 1]], index_map_vertices_to_positions[faces[:, 2]], alpha_shape.volume

def get_new_position_from_origin_and_displacement(origin, displacement, whole_space_shape = None, off_limit_position = "boundary"):
    """
        Get new position from origin displaced as much as displacement.

    Examples
    --------
    print(get_new_position_from_origin_and_displacement([1, 2, 0], [0, -3, 3], [3, 3, 3]))
        [1, 0, 2]
    """

    # assert(off_limit_position in ["boundary", "origin"])
    dimensions = len(origin)
    new_position = []
    for d in range(dimensions):
        new_position.append(origin[d] + displacement[d])
    if whole_space_shape is not None:
        if off_limit_position == "boundary":
            for d in range(dimensions):
                if new_position[d] >= whole_space_shape[d]:
                    new_position[d] = whole_space_shape[d] - 1
                elif new_position[d] < 0:
                    new_position[d] = 0
        elif off_limit_position == "origin":
            for d in range(dimensions):
                if new_position[d] >= whole_space_shape[d] or new_position[d] < 0:
                    return deepcopy(origin)
    return tuple(new_position)

def deconv_smoothness_3D(nparr, deconv_list_of_displacement_and_proportion, mask_nparr = None, keep_origin = True, overwrite = False):
    """Smooth the surface of points in 3D.
    
    Deconvolution is one to many mapping, and it will spread each point out.

    Parameters
    ----------
    deconv_list_of_displacement_and_proportion : list
        Indicates the positions and values of deconvolved (spread out) points. For example, [{"displacement": [1, 0, 0], "proportion": 1.0}, {"displacement": [0, 1, 0], "proportion": 1.0}, {"displacement": [0, 0, 1], "proportion": 1.0}, {"displacement": [-1, 0, 0], "proportion": 1.0}, {"displacement": [0, -1, 0], "proportion": 1.0}, {"displacement": [0, 0, -1], "proportion": 1.0}].
    keep_origin : bool
        Whether to keep the original points before deconvolution.
    
    Return
    ------
    nparr_deconv : array
        3D array in same shape as nparr.
    """

    shape = nparr.shape
    nparr_deconv = np.zeros(shape)
    if mask_nparr is None:
        mask_loc = np.where(nparr > 1e-8, 1., 0.)
    else:
        assert(nparr.shape == mask_nparr.shape)
        mask_loc = np.where(mask_nparr >= 1., 1., 0.)
    x_arr, y_arr, z_arr = mask_loc.nonzero()

    for x, y, z in zip(x_arr, y_arr, z_arr):
        origin = [x, y, z]
        for deconv_dict in deconv_list_of_displacement_and_proportion:
            new_position = get_new_position_from_origin_and_displacement(origin = origin, displacement = deconv_dict["displacement"], whole_space_shape = shape)
            if overwrite:
                nparr_deconv[new_position] = nparr[x, y, z] * deconv_dict["proportion"]
            else:
                nparr_deconv[new_position] += nparr[x, y, z] * deconv_dict["proportion"]
    if keep_origin:
        nparr_deconv = (1. - mask_loc) * nparr_deconv + mask_loc * nparr
    return nparr_deconv

def plotly_2D_contour(nparr, path_to_save, arr_filter = None, vmin = None, vmax = None, layout_kwargs = None, figsize_ratio = None, contour_kwargs = None, scene_kwargs = None, axis_kwargs = None, colorbar_kwargs = None, fill_out_small_values_with_nan= True, points_to_plot = None, do_save_html = False, outline_boundary = False, white_on_min= True):
    """Plot contours from nparr.

    Parameters
    ----------
    figsize_ratio : array-like = None
        This looks not work.
    """
    colorscale = deepcopy(plotly.colors.sequential.Rainbow)
    if white_on_min:
        colorscale[0] = 'rgb(255,255,255)' ## Set white color for zero value.
        if contour_kwargs is not None and "colorscale" in contour_kwargs.keys(): contour_kwargs["colorscale"][0] = 'rgb(255,255,255)'

    if arr_filter is not None: nparr_local = arr_filter(nparr)
    else: nparr_local = nparr
    if fill_out_small_values_with_nan: nparr_local = np.where(nparr_local < vmin, np.nan, nparr_local)

    vmin = max(0., nparr_local.min()) if vmin is None else vmin
    vmax = max(0., nparr_local.max()) if vmax is None else vmax
    ## Set local keywords arguments
    axis_kwargs_local = {} if axis_kwargs is None else deepcopy(axis_kwargs)
    # scene_kwargs_local = merge_dictionaries([{"xaxis": {"title": "x", "range": [0, nparr.shape[0]], "tickvals": range(nparr.shape[0] // 5, nparr.shape[0], nparr.shape[0] // 5), "ticktext": range(nparr.shape[0] // 5, nparr.shape[0], nparr.shape[0] // 5), **axis_kwargs_local}, "yaxis": {"title": "y", "range": [0, nparr.shape[1]], "tickvals": range(nparr.shape[1] // 5, nparr.shape[1], nparr.shape[1] // 5), "ticktext": range(nparr.shape[1] // 5, nparr.shape[1], nparr.shape[1] // 5), **axis_kwargs_local}}, scene_kwargs])
    scene_kwargs_local = merge_dictionaries([dict(), scene_kwargs])
    colorbar_kwargs_local = merge_dictionaries([{"titlefont": {"size": 30}, "title": "value"}, colorbar_kwargs])
    contour_kwargs_local = merge_dictionaries([{"colorscale": colorscale, "colorbar": colorbar_kwargs_local}, contour_kwargs]) ## "contours": {"start": vmin, "end": vmax},
    layout_kwargs_local = merge_dictionaries([{"title": None, "xaxis": {"title": "x", "tickvals" : [i * (nparr.shape[1] // 5) for i in range(1, 5)], "ticktext": [i * (nparr.shape[1] // 5) for i in range(1, 5)], "showgrid": False, "zeroline": False, "showline": False, "zerolinecolor": "black"}, "yaxis": {"title": "y", "tickvals" : [i * (nparr.shape[0] // 5) for i in range(1, 5)], "ticktext": [i * (nparr.shape[0] // 5) for i in range(1, 5)], "showgrid": False, "zeroline": False, "showline": False, "zerolinecolor": "black"}}, layout_kwargs]) ## 0 axis goes to vertical, 1 axis goes to horizontal, do not use range parameter as it create weird margins.

    data = []
    data.append(go.Contour(z = nparr_local, **contour_kwargs_local)) ## You may comment this line.
    if points_to_plot is not None:
        x_, y_ = np.nonzero(points_to_plot)
        data.append(go.Scatter(x=y_, y=x_,
                    mode='markers',
                    marker_symbol='cross', marker_color= "black", marker_size= 4))
    scene = go.Scene(**scene_kwargs_local)
    layout = go.Layout(scene = scene, **layout_kwargs_local)
    # fig = go.Figure(data = go.Scatter(x=y_, y=x_,
    #                 mode='markers',
    #                 marker_symbol= "cross", marker_color= "black", marker_size= 6), layout = layout)
    fig = go.Figure(data = data, layout = layout)

    if outline_boundary:
        ## https://plotly.com/python/axes/#styling-and-coloring-axes-and-the-zeroline , https://stackoverflow.com/questions/42096292/outline-plot-area-in-plotly-in-python
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror = True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror = True)

    if figsize_ratio is not None:
        raise Exception(NotImplementedError)
        fig.update_layout(scene_aspectmode= 'manual', scene_aspectratio= figsize_ratio)
    fig.update_layout(paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)')
    
    if do_save_html:
        fig.write_html(utilsforminds.strings.format_extension(path_to_save, "html"))
    fig.write_image(path_to_save)

def plot_ROC(path_to_save, y_true, list_of_y_pred, list_of_model_names = None, list_of_class_names = None, title = 'Receiver operating characteristic', xlabel = 'False Positive Rate', ylabel = 'True Positive Rate', colors = None, linewidth = 1, extension = "eps", fontsize_ratio = 1.0):
    n_classes = y_true.shape[1]
    if colors is None: colors = cycle(["red", "navy", "lightgreen", "teal", "violet", "green", "orange", "blue", "coral", "yellowgreen", "sienna", "olive", "maroon", "goldenrod", "darkblue", "orchid", "crimson"])
    if list_of_model_names is None:
        list_of_model_names = ["" for i in range(len(list_of_y_pred))]
    else:
        list_of_model_names = [f"{name}, " for name in list_of_model_names]
    if list_of_class_names is None:
        list_of_class_names = ["" for i in range(n_classes)]
    else:
        assert(len(list_of_class_names) == n_classes)
        list_of_class_names = [f"{name}, " for name in list_of_class_names]

    plt.figure()

    fpr_list = []
    tpr_list = []
    roc_auc_list = []

    for model_idx in range(len(list_of_y_pred)):
        assert(list_of_y_pred[model_idx].shape[1] == n_classes)
        assert(y_true.shape[0] == list_of_y_pred[model_idx].shape[0])
        ## Compute ROC curve and ROC area for each class
        fpr_list.append(dict())
        tpr_list.append(dict())
        roc_auc_list.append(dict())
        for i in range(n_classes):
            fpr_list[model_idx][i], tpr_list[model_idx][i], _ = roc_curve(y_true[:, i], list_of_y_pred[model_idx][:, i])
            roc_auc_list[model_idx][i] = auc(fpr_list[model_idx][i], tpr_list[model_idx][i])
        
        ## Compute micro-average ROC curve and ROC area
        fpr_list[model_idx]["micro"], tpr_list[model_idx]["micro"], _ = roc_curve(y_true.ravel(), list_of_y_pred[model_idx].ravel())
        roc_auc_list[model_idx]["micro"] = auc(fpr_list[model_idx]["micro"], tpr_list[model_idx]["micro"])

        if n_classes == 2: ## Plot of a ROC curve for a specific class
            plt.plot(fpr_list[model_idx][0], tpr_list[model_idx][0], color = next(colors), lw= linewidth, label= f"{list_of_model_names[model_idx]}area = {roc_auc_list[model_idx][0]:0.2f}")
        
        else: ## Compute macro-average ROC curve and ROC area
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr_list[model_idx][i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr_list[model_idx][i], tpr_list[model_idx][i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr_list[model_idx]["macro"] = all_fpr
            tpr_list[model_idx]["macro"] = mean_tpr
            roc_auc_list[model_idx]["macro"] = auc(fpr_list[model_idx]["macro"], tpr_list[model_idx]["macro"])

            # Plot all ROC curves
            plt.plot(fpr_list[model_idx]["micro"], tpr_list[model_idx]["micro"],
                    label=f'micro, {list_of_model_names[model_idx]}area = {roc_auc_list[model_idx]["micro"]:0.2f}',
                    color= next(colors), linewidth=1) # linestyle=':',
            
            plt.plot(fpr_list[model_idx]["macro"], tpr_list[model_idx]["macro"],
                    label=f'macro, {list_of_model_names[model_idx]}area = {roc_auc_list[model_idx]["macro"]:0.2f}',
                    color= next(colors), linewidth=1) # linestyle=':',

            ## Plot ROC curve for each class:
            # for i, color in zip(range(n_classes), colors):
            #     plt.plot(fpr_list[model_idx][i], tpr_list[model_idx][i], color=color, lw=linewidth,
            #             label=f'{list_of_class_names[i]}area = {roc_auc_list[model_idx][i]:0.2f}')
    ## Plot auxilaries
    plt.plot([0, 1], [0, 1], lw= linewidth, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel, fontsize= 15 * fontsize_ratio)
    plt.ylabel(ylabel, fontsize= 15 * fontsize_ratio)
    if title is not None: plt.title(title, fontsize= 10 * fontsize_ratio)
    plt.legend(loc="lower right", fontsize= 15 * fontsize_ratio)
    plt.tight_layout()
    plt.savefig(path_to_save + "." + extension, format = extension)

def plot_multiple_matrices(container_of_matrices, path_to_save: str, imshow_kwargs = None):
    """
        https://plotly.com/python/imshow/

    Parameters
    ----------
    container_of_matrices : list of arrays.
        For example, [np.random.rand(60, 3), np.random.rand(60, 1), np.random.rand(60, 20), np.random.rand(60, 27), np.random.rand(60, 15), np.random.rand(60, 80), np.random.rand(60, 54), np.random.rand(60, 2), np.random.rand(60, 15)]
    """

    imshow_kwargs_local = merge(dict(labels= dict(x= "x", y= "y")), 
    imshow_kwargs)
    if isinstance(container_of_matrices, list): num_matrices = len(container_of_matrices)
    else: raise Exception("Unsupported input type")
    if len(container_of_matrices[0].shape) == 2:
        container_of_matrices_local = container_of_matrices
    elif len(container_of_matrices[0].shape) == 1:
        container_of_matrices_local = [vec.reshape((1, vec.shape[0])) for vec in container_of_matrices]

    width = 1
    height = 0
    matrices_ranges = []
    
    for matrix_idx in range(num_matrices):
        matrix = container_of_matrices_local[matrix_idx]
        matrices_ranges.append(dict(x0= width, x1= width + matrix.shape[0], y0= 1, y1= 1 + matrix.shape[1]))
        width += matrix.shape[0] + 1
        if matrix.shape[1] > height: height= matrix.shape[1]
    height += 2 ## For padding.
    merged_matrices = np.zeros((width, height)) * np.nan
    for matrix_idx in range(num_matrices):
        merged_matrices[matrices_ranges[matrix_idx]["x0"]:matrices_ranges[matrix_idx]["x1"], matrices_ranges[matrix_idx]["y0"]:matrices_ranges[matrix_idx]["y1"]] = container_of_matrices_local[matrix_idx]

    fig = px.imshow(merged_matrices, **imshow_kwargs_local)

    # Shape defined programatically
    for matrix_range in matrices_ranges:
        fig.add_shape(
            type='rect',
            x0= matrix_range["y0"]-0.5, x1= matrix_range["y1"]-0.5, y0= matrix_range["x0"]-0.5, y1= matrix_range["x1"]-0.5,
            xref='x', yref='y',
            line_color='cyan'
        )
    ## Define dragmode, newshape parameters, amd add modebar buttons
    fig.update_layout(
        dragmode='drawrect',
        newshape=dict(line_color='cyan'),
        plot_bgcolor= "rgba(0, 0, 0, 0)", ## Set white background https://community.plotly.com/t/having-a-transparent-background-in-plotly-express/30205
        paper_bgcolor= "rgba(0, 0, 0, 0)", ## Set white background https://community.plotly.com/t/having-a-transparent-background-in-plotly-express/30205
        )

    # fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # fig.show()
    fig.write_html(utilsforminds.strings.format_extension(path_to_save, "html"))

if __name__ == '__main__':
    pass
    import plotly.graph_objects as go

    import numpy as np
    np.random.seed(1)

    x = np.random.randn(500).tolist()

    fig = go.Figure(data=[go.Histogram(x=x)])
    fig.show()

