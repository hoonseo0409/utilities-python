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
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import moviepy.editor as mpy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def savePlotTwoLst(lst1, lst2, label1, label2, xlabel, ylabel, title, directory):
    '''
        Deprecated, use utils.helpers.savePlotLstOfLsts instead.
        plot two lists
    '''
    plt.plot(lst1, label = label1)
    plt.plot(lst2, label = label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(directory)

def savePlotLstOfLsts(lstOfLsts, labelsLst, xlabel, ylabel, title, directory):
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

def plot2Ds(planeLst, titleLst, filePath, cbarLabel = 'amount', plotShape = [3, 1], figSize = (8, 24), planeMaskLst = None, axis = 2, axisInfo = None, vmin_vmax = None, method = 'imshow', convertXYaxis = False, rotate = 0):
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
            plt.title(titleLst[i])

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.xticks(horiLabelIdc, horiLabels)
            plt.yticks(vertLabelIdc, vertLabels)

            if method == 'imshow':
                if vmin_vmax is not None:
                    img = plt.imshow(plotPlaneLst[i], vmin = vmin_, vmax = vmax_)
                else:
                    img = plt.imshow(plotPlaneLst[i])
                cbarInst = plt.colorbar()
                cbarInst.set_label(cbarLabel)
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
            plt.title(titleLst[0])

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

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
            cbarInst.set_label(cbarLabel)


            plt.subplot(*(plotShape + [2]))
            plt.title(titleLst[1])

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

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
                cbarInst.set_label(cbarLabel)
            else:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x')
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], marker = 'o')
                img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = plotPlaneLst[1][xCounterOnlyRecovered, yCounterOnlyRecovered], marker = '^')

                cbarInst = plt.colorbar()
                cbarInst.set_label(cbarLabel)
        else: # nPlots == 3
            plt.subplot(*(plotShape + [1]))
            plt.title(titleLst[0])

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

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
            cbarInst.set_label(cbarLabel)


            plt.subplot(*(plotShape + [2]))
            plt.title(titleLst[1])

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

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
                cbarInst.set_label(cbarLabel)
            else:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x', s = pointSize)
                img = plt.scatter(xCounterOriginal, yCounterOriginal, c = plotPlaneLst[0][xCounterOriginal, yCounterOriginal], marker = 'o', s = pointSize)
                img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = plotPlaneLst[1][xCounterOnlyRecovered, yCounterOnlyRecovered], marker = '^', s = pointSize)

                cbarInst = plt.colorbar()
                cbarInst.set_label(cbarLabel)


            plt.subplot(*(plotShape + [3]))
            plt.title(titleLst[2])

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

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
                cbarInst.set_label(cbarLabel)
            else:
                img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], marker = 'x', s = pointSize)
                img = plt.scatter(xCounterSampled, yCounterSampled, c = plotPlaneLst[2][xCounterSampled, yCounterSampled], marker = 'o', s = pointSize)

                cbarInst = plt.colorbar()
                cbarInst.set_label(cbarLabel)
    
    plt.savefig(filePath_, bbox_inches='tight')
    plt.close('all')
    


def plotPlanesOSR(originalNpArr, recoveredNpArr, sparsedNpArr, density, sparsedNMAE, recoveredNMAE, axis, axisInfo, requestPosition, showScreen, isSave, fileName, toNormalize = True, rotate = 0, labelAllStartings = True, method = 'imshow', vmin_ = None, vmax_ = None, originalCounterMat = None, recoveredCounterMat = None, convertXYaxis = False): #ranges = (minX, maxX, minY, maxY, minZ, maxZ)
    """
        WARNING: plotPlanesOSR is deprecated, use plot2Ds instead.
        ranges = (minX, maxX, minY, maxY, minZ, maxZ), two of them should be None
    """
    print('WARNING: plotPlanesOSR is deprecated, use plot2Ds instead.')
    # fig = plt.figure(figsize=(8, 24))
    fig = plt.figure(figsize=(36, 12))


    # grid = AxesGrid(fig, 111,
    #             nrows_ncols=(3, 1),
    #             axes_pad=0.05,
    #             cbar_mode='single',
    #             cbar_location='right',
    #             cbar_pad=0.1
    #             )
    # if toNormalize:
    #     originalNpArr, min, max = getScaledMinMax(originalNpArr)
    #     recoveredNpArr = applyMinMax(recoveredNpArr, min, max)
    #     sparsedNpArr = applyMinMax(sparsedNpArr, min, max) 
    if (vmin_ is not None or vmax_ is not None):
        assert(toNormalize)

    if toNormalize:
        # max = -1 * sys.float_info.max
        # min = sys.float_info.max
        # for npArr2D in (originalNpArr, recoveredNpArr, sparsedNpArr):
        #     if max <= npArr2D.max():
        #         max = npArr2D.max()
        #     if min >= npArr2D.min():
        #         min = npArr2D.min()
        if (vmin_ is not None and vmax_ is not None):
            min = vmin_
            max = vmax_
        else:
            max = originalNpArr.max()
            min = originalNpArr.min()
        # _, min, max = getScaledMinMax(originalNpArr)

    assert(originalNpArr.shape == sparsedNpArr.shape and sparsedNpArr.shape == recoveredNpArr.shape)
    assert(axis in ('x', 'y', 'z') or axis in (0, 1, 2))

    # TODO: change axis xyz into 012

    if labelAllStartings:
        assert(axisInfo != None)
    shape = originalNpArr.shape

    horiLabels = (0, shape[1]*1//4, shape[1]*2//4, shape[1]*3//4, shape[1] - 1)
    vertLabels = (0, shape[0]*1//4, shape[0]*2//4, shape[0]*3//4, shape[0] - 1)

    # horiLabels = (0, shape[0]*1/4, shape[0]*2/4, shape[0]*3/4, shape[0])
    # vertLabels = (0, shape[1]*1/4, shape[1]*2/4, shape[1]*3/4, shape[1])
    if rotate != 0:
        originalNpArr = ndimage.rotate(originalNpArr, rotate)
        sparsedNpArr = ndimage.rotate(sparsedNpArr, rotate)
        recoveredNpArr = ndimage.rotate(recoveredNpArr, rotate)
        if originalCounterMat is not None:
            originalCounterMat = ndimage.rotate(originalCounterMat, rotate)
        if recoveredCounterMat is not None:
            recoveredCounterMat = ndimage.rotate(recoveredCounterMat, rotate)

    if axisInfo == None:
        if axis == 'x':
            plt.subplot(3, 1, 1)

            if rotate:
                plt.ylabel('z')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst(horiLabels))
                elif method == 'contour':
                    plt.yticks(horiLabels, horiLabels)
                plt.xlabel('y')
                plt.xticks(vertLabels, vertLabels)
            else:
                plt.xlabel('z')
                plt.xticks(horiLabels, horiLabels)
                plt.ylabel('y')
                plt.yticks(vertLabels, vertLabels)
            plt.title('x = {}, Original Density = {}\n'.format(requestPosition, density))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(originalNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(originalNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(originalNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(originalNpArr, cmap='Greens')

            plt.subplot(3, 1, 2)

            if rotate:
                plt.ylabel('z')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst(horiLabels))
                elif method == 'contour':
                    plt.yticks(horiLabels, horiLabels)
                plt.xlabel('y')
                plt.xticks(vertLabels, vertLabels)
            else:
                plt.xlabel('z')
                plt.xticks(horiLabels, horiLabels)
                plt.ylabel('y')
                plt.yticks(vertLabels, vertLabels)
            plt.title('Recovered NMAE = {}\n'.format(recoveredNMAE))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(recoveredNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(recoveredNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(recoveredNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(recoveredNpArr, cmap='Greens')

            plt.subplot(3, 1, 3)

            if rotate:
                plt.ylabel('z')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst(horiLabels))
                elif method == 'contour':
                    plt.yticks(horiLabels, horiLabels)
                plt.xlabel('y')
                plt.xticks(vertLabels, vertLabels)
            else:
                plt.xlabel('z')
                plt.xticks(horiLabels, horiLabels)
                plt.ylabel('y')
                plt.yticks(vertLabels, vertLabels)
            plt.title('Sparsed NMAE = {}\n'.format(sparsedNMAE))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(sparsedNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(sparsedNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(sparsedNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(sparsedNpArr, cmap='Greens')

            if isSave:
                plt.savefig(utilsforminds.helpers.getExecPath() + '/current_results/' + fileName)
            if showScreen:
                plt.show()
        
        elif axis == 'y':
            
            plt.subplot(3, 1, 1)

            if rotate:
                plt.ylabel('z')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst(horiLabels))
                elif method == 'contour':
                    plt.yticks(horiLabels, horiLabels)
                plt.xlabel('x')
                plt.xticks(vertLabels, vertLabels)
            else:
                plt.xlabel('z')
                plt.xticks(horiLabels, horiLabels)
                plt.ylabel('x')
                plt.yticks(vertLabels, vertLabels)
            plt.title('y = {}, Original Density = {}\n'.format(requestPosition, density))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(originalNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(originalNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(originalNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(originalNpArr, cmap='Greens')
            
            plt.subplot(3, 1, 2)

            if rotate:
                plt.ylabel('z')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst(horiLabels))
                elif method == 'contour':
                    plt.yticks(horiLabels, horiLabels)
                plt.xlabel('x')
                plt.xticks(vertLabels, vertLabels)
            else:
                plt.xlabel('z')
                plt.xticks(horiLabels, horiLabels)
                plt.ylabel('x')
                plt.yticks(vertLabels, vertLabels)
            plt.title('Recovered NAME = {}\n'.format(recoveredNMAE))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(recoveredNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(recoveredNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(recoveredNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(recoveredNpArr, cmap='Greens')

            plt.subplot(3, 1, 3)

            if rotate:
                plt.ylabel('z')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst(horiLabels))
                elif method == 'contour':
                    plt.yticks(horiLabels, horiLabels)
                plt.xlabel('x')
                plt.xticks(vertLabels, vertLabels)
            else:
                plt.xlabel('z')
                plt.xticks(horiLabels, horiLabels)
                plt.ylabel('x')
                plt.yticks(vertLabels, vertLabels)
            plt.title('Sparsed NMAE = {}\n'.format(sparsedNMAE))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(sparsedNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(sparsedNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(sparsedNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(sparsedNpArr, cmap='Greens')

            if isSave:
                plt.savefig(utilsforminds.helpers.getExecPath() + '/current_results/' + fileName)
            if showScreen:
                plt.show()

        else: # axis == 'z':
            plt.subplot(3, 1, 1)

            if rotate:
                plt.ylabel('y')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst(horiLabels))
                elif method == 'contour':
                    plt.yticks(horiLabels, horiLabels)
                plt.xlabel('x')
                plt.xticks(vertLabels, vertLabels)
            else:
                plt.xlabel('y')
                plt.xticks(horiLabels, horiLabels)
                plt.ylabel('x')
                plt.yticks(vertLabels, vertLabels)
            plt.title('z = {}, Original Density = {}\n'.format(requestPosition, density))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(originalNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(originalNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(originalNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(originalNpArr, cmap='Greens')

            plt.subplot(3, 1, 2)

            if rotate:
                plt.ylabel('y')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst(horiLabels))
                elif method == 'contour':
                    plt.yticks(horiLabels, horiLabels)
                plt.xlabel('x')
                plt.xticks(vertLabels, vertLabels)
            else:
                plt.xlabel('y')
                plt.xticks(horiLabels, horiLabels)
                plt.ylabel('x')
                plt.yticks(vertLabels, vertLabels)
            plt.title('Recovered NMAE = {}\n'.format(recoveredNMAE))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(recoveredNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(recoveredNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(recoveredNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(recoveredNpArr, cmap='Greens')

            plt.subplot(3, 1, 3)

            if rotate:
                plt.ylabel('y')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst(horiLabels))
                elif method == 'contour':
                    plt.yticks(horiLabels, horiLabels)
                plt.xlabel('x')
                plt.xticks(vertLabels, vertLabels)
            else:
                plt.xlabel('y')
                plt.xticks(horiLabels, horiLabels)
                plt.ylabel('x')
                plt.yticks(vertLabels, vertLabels)
            plt.title('Sparsed NMAE = {}\n'.format(sparsedNMAE))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(sparsedNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(sparsedNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(sparsedNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(sparsedNpArr, cmap='Greens')

            if isSave:
                plt.savefig(utilsforminds.helpers.getExecPath() + '/current_results/' + fileName)
            if showScreen:
                plt.show()
            
    else:
        if axis == 'x':
            plt.subplot(3, 1, 1)

            if rotate:
                plt.ylabel('Elevation(m)')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst((round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                    round(axisInfo[2]["max"]))))
                elif method == 'contour':
                    plt.yticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                    round(axisInfo[2]["max"])))
                plt.xlabel('North(m)')
                plt.xticks(vertLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                round(axisInfo[1]["max"])))
            else:
                plt.xlabel('Elevation(m)')
                plt.xticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                round(axisInfo[2]["max"])))
                plt.ylabel('North(m)')
                plt.yticks(vertLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                round(axisInfo[1]["max"])))
            if labelAllStartings:
                plt.title('x_slice = {}, y = {}, z = {}\n Original Density = {}\n'.format(requestPosition, axisInfo['x']['min'], axisInfo['z']['min'], density))
            else:
                plt.title('x = {}, Original Density = {}\n'.format(requestPosition, density))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(originalNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(originalNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(originalNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(originalNpArr, cmap='Greens')

            plt.subplot(3, 1, 2)
            
            if rotate:
                plt.ylabel('Elevation(m)')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst((round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                    round(axisInfo[2]["max"]))))
                elif method == 'contour':
                    plt.yticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                round(axisInfo[2]["max"])))
                plt.xlabel('North(m)')
                plt.xticks(vertLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                round(axisInfo[1]["max"])))
            else:
                plt.xlabel('Elevation(m)')
                plt.xticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                round(axisInfo[2]["max"])))
                plt.ylabel('North(m)')
                plt.yticks(vertLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                round(axisInfo[1]["max"])))
            plt.title('Recovered NMAE = {}\n'.format(recoveredNMAE))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(recoveredNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(recoveredNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(recoveredNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(recoveredNpArr, cmap='Greens')

            plt.subplot(3, 1, 3)
            if rotate:
                plt.ylabel('Elevation(m)')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst((round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                    round(axisInfo[2]["max"]))))
                elif method == 'contour':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst((round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                    round(axisInfo[2]["max"]))))
                plt.xlabel('North(m)')
                plt.xticks(vertLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                round(axisInfo[1]["max"])))
            else:
                plt.xlabel('Elevation(m)')
                plt.xticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                round(axisInfo[2]["max"])))
                plt.ylabel('North(m)')
                plt.yticks(vertLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                round(axisInfo[1]["max"])))
            plt.title('Sparsed NMAE = {}\n'.format(sparsedNMAE))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(sparsedNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(sparsedNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(sparsedNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(sparsedNpArr, cmap='Greens')

            if isSave:
                plt.savefig(utilsforminds.helpers.getExecPath() + '/current_results/' + fileName)
            if showScreen:
                plt.show()
        elif axis == 'y':
            
            plt.subplot(3, 1, 1)

            if rotate:
                plt.ylabel('Elevation(m)')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst((round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                    round(axisInfo[2]["max"]))))
                elif method == 'contour':
                    plt.yticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                    round(axisInfo[2]["max"])))
                plt.xlabel('East(m)')
                plt.xticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
                round(axisInfo[0]["max"])))
            else:
                plt.xlabel('Elevation(m)')
                plt.xticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                round(axisInfo[2]["max"])))
                plt.ylabel('East(m)')
                plt.yticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
                round(axisInfo[0]["max"])))
            if labelAllStartings:
                plt.title('x = {}, y_slice = {}, z = {}\n Original Density = {}\n'.format(axisInfo['x']['min'], requestPosition, axisInfo['z']['min'], density))
            else:
                plt.title('y = {}, Original Density = {}\n'.format(requestPosition, density))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(originalNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(originalNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(originalNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(originalNpArr, cmap='Greens')
            
            plt.subplot(3, 1, 2)

            if rotate:
                plt.ylabel('Elevation(m)')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst((round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                    round(axisInfo[2]["max"]))))
                elif method == 'contour':
                    plt.yticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                round(axisInfo[2]["max"])))
                plt.xlabel('East(m)')
                plt.xticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
                round(axisInfo[0]["max"])))
            else:
                plt.xlabel('Elevation(m)')
                plt.xticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                round(axisInfo[2]["max"])))
                plt.ylabel('East(m)')
                plt.yticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
                round(axisInfo[0]["max"])))
            plt.title('Recovered NAME = {}\n'.format(recoveredNMAE))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(recoveredNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(recoveredNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(recoveredNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(recoveredNpArr, cmap='Greens')

            plt.subplot(3, 1, 3)

            if rotate:
                plt.ylabel('Elevation(m)')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst((round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                    round(axisInfo[2]["max"]))))
                elif method == 'contour':
                    plt.yticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                    round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                    round(axisInfo[2]["max"])))
                plt.xlabel('East(m)')
                plt.xticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
                round(axisInfo[0]["max"])))
            else:
                plt.xlabel('Elevation(m)')
                plt.xticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
                round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
                round(axisInfo[2]["max"])))
                plt.ylabel('East(m)')
                plt.yticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
                round(axisInfo[0]["max"])))
            plt.title('Sparsed NMAE = {}\n'.format(sparsedNMAE))
            
            if method == 'imshow':
                if toNormalize:
                    plt.imshow(sparsedNpArr, cmap='Greens', vmin=min, vmax=max)
                else:
                    plt.imshow(sparsedNpArr, cmap='Greens')
            elif method == 'contour':
                if toNormalize:
                    # plt.contour(np.where(sparsedNpArr > (max - min) * 0.001 + min, sparsedNpArr, 0.), cmap = 'Greens', vmin = min, vmax = max)
                    plt.contour(np.flipud(sparsedNpArr), cmap = 'Greens', vmin = min, vmax = max)
                else:
                    plt.imshow(sparsedNpArr, cmap='Greens')

            if isSave:
                plt.savefig(utilsforminds.helpers.getExecPath() + '/current_results/' + fileName)
            if showScreen:
                plt.show()
        else: # axis == 'z':
            # plt.subplot(3, 1, 1)
            plt.subplot(1, 2, 1)

            # if rotate:
            #     plt.ylabel('North(m)')
            #     if method == 'imshow':
            #         plt.yticks(horiLabels, utils.helpers.reverseLst((round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
            #         round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
            #         round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
            #         round(axisInfo[1]["max"]))))
            #     elif method == 'contour':
            #         plt.yticks(horiLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
            #         round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
            #         round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
            #         round(axisInfo[1]["max"])))
            #     plt.xlabel('East(m)')
            #     plt.xticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
            #     round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
            #     round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
            #     round(axisInfo[0]["max"])))
            # else:
            #     plt.xlabel('North(m)')
            #     plt.xticks(horiLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
            #     round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
            #     round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
            #     round(axisInfo[1]["max"])))
            #     plt.ylabel('East(m)')
            #     plt.yticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
            #     round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
            #     round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
            #     round(axisInfo[0]["max"])))


            if convertXYaxis:
                plt.ylabel('North(m)')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst((round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                    round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                    round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                    round(axisInfo[1]["max"]))))
                elif method == 'contour' or method == 'scatter' :
                    plt.yticks(horiLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                    round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                    round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                    round(axisInfo[1]["max"])))
                plt.xlabel('East(m)')
                plt.xticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
                round(axisInfo[0]["max"])))
            else:
                plt.xlabel('North(m)')
                plt.xticks(horiLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                round(axisInfo[1]["max"])))
                plt.ylabel('East(m)')
                plt.yticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
                round(axisInfo[0]["max"])))


            if labelAllStartings:
                plt.title('x = {}, y = {}, z_slice = {}\n Original Density = {}\n'.format(axisInfo['x']['min'], axisInfo['y']['min'], requestPosition, density))
            else:
                plt.title('z = {}, Original Density = {}\n'.format(requestPosition, density))
            
            if method == 'imshow':
                if toNormalize:
                    img = plt.imshow(originalNpArr, vmin=min, vmax=max)
                    cbarInst = plt.colorbar()
                    cbarInst.set_label("gram/ton")
                else:
                    img = plt.imshow(originalNpArr)
                    cbarInst = plt.colorbar()
                    cbarInst.set_label("gram/ton")
            elif method == 'contour':
                if toNormalize:
                    # img = plt.contour(np.flipud(originalNpArr), vmin = min, vmax = max, linewidths = 2.0, colors = 'black')
                    img = plt.contour(np.flipud(originalNpArr), vmin = min, vmax = max, linewidths = 2.0, colors = 'black', levels = [min, min + (max - min) * 1/8, min + (max - min) * 2/8, min + (max - min) * 3/8, min + (max - min) * 4/8, min + (max - min) * 5/8, min + (max - min) * 6/8, min + (max - min) * 7/8, max])
                else:
                    img = plt.contour(np.flipud(originalNpArr), linewidths = 2.0, colors = 'black', levels = [min, min + (max - min) * 1/8, min + (max - min) * 2/8, min + (max - min) * 3/8, min + (max - min) * 4/8, min + (max - min) * 5/8, min + (max - min) * 6/8, min + (max - min) * 7/8, max])
            elif method == 'scatter':
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(originalCounterMat >= 1., 1., 0.))
                # xCounterRecovered, yCounterRecovered = np.nonzero(np.where(recoveredCounterMat >= 1., 1., 0.))
                # xCounterOnlyRecovered, yCounterOnlyRecovered = np.nonzero(np.where(originalCounterMat < 1., np.where(recoveredCounterMat >= 1., 1., 0.), 0.))
                xZeroCounter, yZeroCounter = np.nonzero(np.where(originalCounterMat < 1., 1, 0.))
                minArr = np.ones(originalNpArr.shape) * min

                if toNormalize:
                    # img = plt.scatter(x = x, y = y, c = recoveredNpArr[x, y], vmin=min, vmax=max)
                    img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], vmin=min, vmax=max, marker = 'x', s = 5.0)
                    img = plt.scatter(xCounterOriginal, yCounterOriginal, c = originalNpArr[xCounterOriginal, yCounterOriginal], vmin=min, vmax=max, marker = 'o', s = 5.0)
                    # img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = recoveredNpArr[xCounterOnlyRecovered, yCounterOnlyRecovered], vmin=min, vmax=max, marker = '^')

                    cbarInst = plt.colorbar()
                    cbarInst.set_label("gram/ton")
                """
                    fix below later
                """
                # else:
                #     img = plt.scatter(x = x, y = y, c = recoveredNpArr[x, y])
                #     cbarInst = plt.colorbar()
                #     cbarInst.set_label("gram/ton")

            # ax = plt.subplot(3, 1, 2)
            plt.subplot(1, 2, 2)

            if convertXYaxis:
                plt.ylabel('North(m)')
                if method == 'imshow':
                    plt.yticks(horiLabels, utilsforminds.helpers.reverseLst((round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                    round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                    round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                    round(axisInfo[1]["max"]))))
                elif method == 'contour' or method == 'scatter' :
                    plt.yticks(horiLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                    round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                    round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                    round(axisInfo[1]["max"])))
                plt.xlabel('East(m)')
                plt.xticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
                round(axisInfo[0]["max"])))
            else:
                plt.xlabel('North(m)')
                plt.xticks(horiLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
                round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
                round(axisInfo[1]["max"])))
                plt.ylabel('East(m)')
                plt.yticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
                round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
                round(axisInfo[0]["max"])))
            plt.title('Recovered NMAE = {}\n'.format(recoveredNMAE))
            
            if method == 'imshow':
                if toNormalize:
                    img = plt.imshow(recoveredNpArr, vmin=min, vmax=max)
                    cbarInst = plt.colorbar()
                    cbarInst.set_label("gram/ton")
                else:
                    img = plt.imshow(recoveredNpArr)
                    cbarInst = plt.colorbar()
                    cbarInst.set_label("gram/ton")
            elif method == 'contour':
                if toNormalize:
                    # img = plt.contour(np.flipud(recoveredNpArr), cmap = 'Greens', vmin = min, vmax = max)
                    # img = plt.contour(np.flipud(recoveredNpArr), vmin = min, vmax = max, linewidths = 2.0, colors = 'black')
                    img = plt.contour(np.flipud(recoveredNpArr), vmin = min, vmax = max, linewidths = 2.0, colors = 'black', levels = [min, min + (max - min) * 1/8, min + (max - min) * 2/8, min + (max - min) * 3/8, min + (max - min) * 4/8, min + (max - min) * 5/8, min + (max - min) * 6/8, min + (max - min) * 7/8, max])
                else:
                    # img = plt.imshow(recoveredNpArr, cmap='Greens')
                    img = plt.contour(np.flipud(recoveredNpArr), linewidths = 2.0, colors = 'black', levels = [min, min + (max - min) * 1/8, min + (max - min) * 2/8, min + (max - min) * 3/8, min + (max - min) * 4/8, min + (max - min) * 5/8, min + (max - min) * 6/8, min + (max - min) * 7/8, max])
            elif method == 'scatter':
                xCounterOriginal, yCounterOriginal = np.nonzero(np.where(originalCounterMat >= 1., 1., 0.))
                # xCounterRecovered, yCounterRecovered = np.nonzero(np.where(recoveredCounterMat >= 1., 1., 0.))
                xCounterOnlyRecovered, yCounterOnlyRecovered = np.nonzero(np.where(originalCounterMat < 1., np.where(recoveredCounterMat >= 1., 1., 0.), 0.))
                xZeroCounter, yZeroCounter = np.nonzero(np.where(recoveredCounterMat < 1., np.where(originalCounterMat < 1., 1., 0.), 0.))
                minArr = np.ones(originalNpArr.shape) * min

                if toNormalize:
                    # img = plt.scatter(x = x, y = y, c = recoveredNpArr[x, y], vmin=min, vmax=max)
                    img = plt.scatter(xZeroCounter, yZeroCounter, c = minArr[xZeroCounter, yZeroCounter], vmin=min, vmax=max, marker = 'x', s = 5.0)
                    img = plt.scatter(xCounterOriginal, yCounterOriginal, c = originalNpArr[xCounterOriginal, yCounterOriginal], vmin=min, vmax=max, marker = 'o', s = 5.0)
                    img = plt.scatter(xCounterOnlyRecovered, yCounterOnlyRecovered, c = recoveredNpArr[xCounterOnlyRecovered, yCounterOnlyRecovered], vmin=min, vmax=max, marker = '^', s = 5.0)

                    cbarInst = plt.colorbar()
                    cbarInst.set_label("gram/ton")
                # else:
                #     img = plt.scatter(x = x, y = y, c = recoveredNpArr[x, y])
                #     cbarInst = plt.colorbar()
                #     cbarInst.set_label("gram/ton")
            

            # img = plt.gca().get_children()[0]
            # colorbarInst = fig.colorbar(img, ax = ax)





            # plt.subplot(3, 1, 3)
            # # x_, y_ = np.mgrid[0:1:sparsedNpArr.shape[0], 0:1:sparsedNpArr.shape[1]]
            # if rotate:
            #     plt.ylabel('North(m)')
            #     if method == 'imshow':
            #         plt.yticks(horiLabels, utils.helpers.reverseLst((round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
            #         round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
            #         round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
            #         round(axisInfo[1]["max"]))))
            #     elif method == 'contour':
            #         plt.yticks(horiLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
            #         round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
            #         round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
            #         round(axisInfo[1]["max"])))
            #     plt.xlabel('East(m)')
            #     plt.xticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
            #     round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
            #     round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
            #     round(axisInfo[0]["max"])))
            # else:
            #     plt.xlabel('North(m)')
            #     plt.xticks(horiLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
            #     round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
            #     round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
            #     round(axisInfo[1]["max"])))
            #     plt.ylabel('East(m)')
            #     plt.yticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
            #     round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
            #     round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
            #     round(axisInfo[0]["max"])))
            # plt.title('Sparsed NMAE = {}\n'.format(sparsedNMAE))

            # if method == 'imshow':
            #     if toNormalize:
            #         img = plt.imshow(sparsedNpArr, vmin=min, vmax=max)
            #         cbarInst = plt.colorbar()
            #         cbarInst.set_label("gram/ton")
            #     else:
            #         img = plt.imshow(sparsedNpArr)
            #         cbarInst = plt.colorbar()
            #         cbarInst.set_label("gram/ton")
            # elif method == 'contour':
            #     if toNormalize:
            #         img = plt.contour(np.flipud(sparsedNpArr), vmin = min, vmax = max, linewidths = 2.0, colors = 'black', levels = [min, min + (max - min) * 1/8, min + (max - min) * 2/8, min + (max - min) * 3/8, min + (max - min) * 4/8, min + (max - min) * 5/8, min + (max - min) * 6/8, min + (max - min) * 7/8, max])
            #         # plt.clabel(img, inline = True, fontsize = 8)
            #         # [min, min + (max - min) * 1/5, min + (max - min) * 2/5, min + (max - min) * 3/5, min + (max - min) * 4/5, max]
            #         # [min, min + (max - min) * 1/8, min + (max - min) * 2/8, min + (max - min) * 3/8, min + (max - min) * 4/8, min + (max - min) * 5/8, min + (max - min) * 6/8, min + (max - min) * 7/8, max]
            #         # img = plt.contour(np.flipud(sparsedNpArr), vmin = min, vmax = max, linewidths = 2.0, colors = 'black')

            #         # img = plt.contour(np.flipud(sparsedNpArr), vmin = min, vmax = max, linewidths = 2.0)
            #         # plt.imshow(sparsedNpArr, vmin=min, vmax=max, origin = 'lower', alpha = 0.2)

            #         # img = plt.contourf(np.flipud(sparsedNpArr), vmin = min, vmax = max)

            #         # colorbarInst = plt.colorbar()
            #         # colorbarInst.set_label("gram/ton")

            #         # plt.clabel(img, inline = True, fontsize = 8)
            #         # # plt.imshow(sparsedNpArr, cmap='Greens', vmin=min, vmax=max, origin = 'lower', alpha = 0.5)
            #         # colorbarInst = plt.colorbar()
            #         # colorbarInst.set_label("amount of metal")
            #     else:
            #         img = plt.contour(np.flipud(sparsedNpArr), linewidths = 2.0, colors = 'black', levels = [min, min + (max - min) * 1/8, min + (max - min) * 2/8, min + (max - min) * 3/8, min + (max - min) * 4/8, min + (max - min) * 5/8, min + (max - min) * 6/8, min + (max - min) * 7/8, max])

            # cbar = ax.cax.colorbar(img)
            # cbar = grid.cbar_axes[0].colorbar(img)

            # cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
            # cbar.ax.set_yticklabels(['low', 'medium', 'high'])

            # img = plt.gca().get_children()[0]
            # fig.colorbar(img)
            # colorbarInst.set_label("gram/ton")
            # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d km'))





            if isSave:
                # plt.savefig(utils.helpers.getExecPath() + '/current_results/' + fileName)
                plt.savefig(utilsforminds.helpers.getExecPath() + '/current_results/' + fileName, bbox_inches='tight')
            if showScreen:
                plt.show()

    plt.close('all')




# def plotPlane(npArr, axis, axisInfo, requestPosition, showScreen, isSave, fileName): #ranges = (minX, maxX, minY, maxY, minZ, maxZ)
#     """
#         WARNING: plotPlane is deprecated, use plot2Ds instead.
#         ranges = (minX, maxX, minY, maxY, minZ, maxZ), two of them should be None
#     """
#     print('WARNING: plotPlane is deprecated, use plot2Ds instead.')
#     # if axis == 'x':
#     #     assert(axisInfo[0]["min"] == None and axisInfo[0]["max"] == None and
#     #     axisInfo[1]["min"] <= axisInfo[1]["max"] and axisInfo[2]["min"] <= axisInfo[2]["max"])
#     # elif axis == 'y':
#     #     assert(axisInfo[1]["min"] == None and axisInfo[1]["max"] == None and
#     #     axisInfo[0]["min"] <= axisInfo[0]["max"] and axisInfo[2]["min"] <= axisInfo[2]["max"])
#     # else:
#     #     assert(axisInfo[2]["min"] == None and axisInfo[2]["max"] == None and
#     #     axisInfo[0]["min"] <= axisInfo[0]["max"] and axisInfo[1]["min"] <= axisInfo[1]["max"])

#     # plt.tick_params(labeltop=True, labelleft=True, labelright=True, labelbottom=True)
#     # plt.bar(np.arange(0, 500, 0.5), np.arange(0, 500, 0.5))
#     shape = npArr.shape
#     horiLabels = (0, shape[0]*1/4, shape[0]*2/4, shape[0]*3/4, shape[0])
#     vertLabels = (0, shape[1]*1/4, shape[1]*2/4, shape[1]*3/4, shape[1])
#     # plt.xticks(horiLabels, (317.5, 455.5, 599., 800.))

#     assert(axis in ('x', 'y', 'z'))
#     if axis == 'x':
#         plt.xlabel('Elevation(m)')
#         plt.xticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
#         round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
#         round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
#         round(axisInfo[2]["max"])))

#         plt.ylabel('North(m)')
#         plt.yticks(vertLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
#         round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
#         round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
#         round(axisInfo[1]["max"])))

#         plt.title('Where East = {}, Density = {}\n'.format(requestPosition, utils.helpers.getDataDensity(npArr)))
#     elif axis == 'y':
#         plt.xlabel('Elevation(m)')
#         plt.xticks(horiLabels, (round(axisInfo[2]["min"]), round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*1/4), 
#         round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*2/4),
#         round(axisInfo[2]["min"] + (axisInfo[2]["max"]-axisInfo[2]["min"])*3/4),
#         round(axisInfo[2]["max"])))

#         plt.ylabel('East(m)')
#         plt.yticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
#         round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
#         round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
#         round(axisInfo[0]["max"])))
        
#         plt.title('Where North = {}, Density = {}\n'.format(requestPosition, utils.helpers.getDataDensity(npArr)))
#     else:
#         plt.xlabel('North(m)')
#         plt.xticks(horiLabels, (round(axisInfo[1]["min"]), round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*1/4), 
#         round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*2/4),
#         round(axisInfo[1]["min"] + (axisInfo[1]["max"]-axisInfo[1]["min"])*3/4),
#         round(axisInfo[1]["max"])))

#         plt.ylabel('East(m)')
#         plt.yticks(vertLabels, (round(axisInfo[0]["min"]), round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*1/4), 
#         round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*2/4),
#         round(axisInfo[0]["min"] + (axisInfo[0]["max"]-axisInfo[0]["min"])*3/4),
#         round(axisInfo[0]["max"])))

#         plt.title('Where Elevation = {}, Density = {}\n'.format(requestPosition, utils.helpers.getDataDensity(npArr)))

#     # plt.plot(npArr)
#     plt.imshow(npArr, cmap='Greens')
#     if isSave:
#         plt.savefig(utils.helpers.getExecPath() + '/current_results/' + fileName)
#     if showScreen:
#         plt.show()

def compare3DTensorAndPlot(originalTensor, originalCounterTensor, recoveredTensor, sparsedTensor, axis, noOfSlices, axisInfo, newDirectoryName, metalStr, samplingProb):
    assert(originalTensor.shape == recoveredTensor.shape
    and recoveredTensor.shape == sparsedTensor.shape
    and len(recoveredTensor.shape) == 3
    and axis in (0, 1, 2)
    and noOfSlices > 0)

    '''
        plot 3D numpy arrays
    '''

    slicesIdx = random.sample(range(recoveredTensor.shape[0]), noOfSlices)

    for idx in slicesIdx:
        originalMat = helpers.getSlicesV2(originalTensor, {axis: idx})
        originalCounterMat = helpers.getSlicesV2(originalCounterTensor, {axis: idx})
        recoveredMat = helpers.getSlicesV2(recoveredTensor, {axis: idx})
        sparsedMat = helpers.getSlicesV2(sparsedTensor, {axis: idx})

        sparsedNMAE = utilsforminds.helpers.getNMAE(originalMat, sparsedMat, originalCounterMat)
        recoveredNMAE = utilsforminds.helpers.getNMAE(originalMat, recoveredMat, originalCounterMat)

        plotPlanesOSR(originalMat, recoveredMat, sparsedMat, utilsforminds.helpers.getDataDensity(originalCounterMat), sparsedNMAE, recoveredNMAE, axis, axisInfo
            , idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"], False, True, newDirectoryName + '/' + metalStr + "_" + helpers.axisMap[axis] + "{}_{}".format(round(idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"]), round(samplingProb*100)))

    # if axis == 0:
    #     slicesIdx = random.sample(range(recoveredTensor.shape[0]), noOfSlices)

    #     for idx in slicesIdx:
    #         originalMat = originalTensor[idx, :, :]
    #         originalCounterMat = originalCounterTensor[idx, :, :]
    #         recoveredMat = recoveredTensor[idx, :, :]
    #         sparsedMat = sparsedTensor[idx, :, :]

    #         sparsedNMAE = utils.helpers.getNMAE(originalMat, sparsedMat, originalCounterMat)
    #         recoveredNMAE = utils.helpers.getNMAE(originalMat, recoveredMat, originalCounterMat)

    #         plotPlanesOSR(originalMat, recoveredMat, sparsedMat, utils.helpers.getDataDensity(originalCounterMat), sparsedNMAE, recoveredNMAE, axis, axisInfo
    #             , idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"], False, True, newDirectoryName + '/' + metalStr + "_" + axis + "{}_{}".format(round(idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"]), round(samplingProb*100)))

    # elif axis == 1:
    #     slicesIdx = random.sample(range(recoveredTensor.shape[1]), noOfSlices)

    #     for idx in slicesIdx:
    #         originalMat = originalTensor[:, idx, :]
    #         originalCounterMat = originalCounterTensor[:, idx, :]
    #         recoveredMat = recoveredTensor[:, idx, :]
    #         sparsedMat = sparsedTensor[:, idx, :]

    #         sparsedNMAE = utils.helpers.getNMAE(originalMat, sparsedMat, originalCounterMat)
    #         recoveredNMAE = utils.helpers.getNMAE(originalMat, recoveredMat, originalCounterMat)

    #         plotPlanesOSR(originalMat, recoveredMat, sparsedMat, utils.helpers.getDataDensity(originalCounterMat), sparsedNMAE, recoveredNMAE, axis, axisInfo
    #             , idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"], False, True, newDirectoryName + '/' + metalStr + "_" + axis + "{}_{}".format(round(idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"]), round(samplingProb*100)))
    
    # elif axis == 2:
    #     slicesIdx = random.sample(range(recoveredTensor.shape[2]), noOfSlices)

    #     for idx in slicesIdx:
    #         originalMat = originalTensor[:, :, idx]
    #         originalCounterMat = originalCounterTensor[:, :, idx]
    #         recoveredMat = recoveredTensor[:, :, idx]
    #         sparsedMat = sparsedTensor[:, :, idx]

    #         sparsedNMAE = utils.helpers.getNMAE(originalMat, sparsedMat, originalCounterMat)
    #         recoveredNMAE = utils.helpers.getNMAE(originalMat, recoveredMat, originalCounterMat)

    #         if axisInfo == None: # ?
    #             plotPlanesOSR(originalMat, recoveredMat, sparsedMat, utils.helpers.getDataDensity(originalCounterMat), sparsedNMAE, recoveredNMAE, axis, axisInfo
    #             , idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"], False, True, newDirectoryName + '/' + metalStr + "_" + axis + "{}_{}".format(round(idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"]), round(samplingProb*100)))
    #         plotPlanesOSR(originalMat, recoveredMat, sparsedMat, utils.helpers.getDataDensity(originalCounterMat), sparsedNMAE, recoveredNMAE, axis, axisInfo
    #             , idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"], False, True, newDirectoryName + '/' + metalStr + "_" + axis + "{}_{}".format(round(idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"]), round(samplingProb*100)))
    
    
    txtFile = open(utilsforminds.helpers.getExecPath() + '/current_results/' + newDirectoryName + "/summary.txt", "a")
    txtContents = "{} with {}:\n3D recovered RMAE : {}\n3D sparsed RMAE: {}\n".format(metalStr, helpers.axisMap[axis], utilsforminds.helpers.getNMAE(originalTensor, recoveredTensor, originalCounterTensor), utilsforminds.helpers.getNMAE(originalTensor, sparsedTensor, originalCounterTensor))
    txtFile.write(txtContents)

    txtFile.close()

def compare4DTensorAndPlot(originalTensor, originalCounterTensor, recoveredTensor, sparsedTensor, axis, noOfSlices, axisInfo, newDirectoryName, samplingProb, methods = ['imshow'], recoveredTensorConter = None, scaleSync = False, compress = None, imgFormat = 'png', vmaxRankPercentile = 0.1):
    assert(originalTensor.shape == recoveredTensor.shape
    and recoveredTensor.shape == sparsedTensor.shape
    and len(recoveredTensor.shape) == 4
    and axis in (0, 1, 2)
    and noOfSlices > 0)

    samplingProbNum = 1.
    if type(samplingProb) == type([]):
        for proportion in samplingProb:
            samplingProbNum = samplingProbNum * (1 - proportion ** 3.)
    else:
        samplingProbNum = samplingProb

    shape = recoveredTensor.shape

    if recoveredTensorConter is not None:
        originalTensor = originalTensor * np.where(originalCounterTensor >= 1., 1., 0.) # array is deep copied here, so you don't need to rename it.
        recoveredTensor = recoveredTensor * np.where(recoveredTensorConter >= 1., 1., 0.)
        sparsedTensor = np.where(sparsedTensor >= 1e-8, sparsedTensor, 0.)

    
    slicesIdx = random.sample(range(recoveredTensor.shape[axis]), noOfSlices)

    for idx in slicesIdx:
        for feature in range(shape[3]):
            if scaleSync:
                rank = int(np.count_nonzero(originalCounterTensor[:, :, :, feature]) * vmaxRankPercentile)
                vmin = np.amin(helpers.getSlicesV2(originalTensor, {3: feature})) + 1e-8
                vmax = np.partition(helpers.getSlicesV2(originalTensor, {3: feature}).flatten(), -1 * rank)[-1 * rank]
                if abs(vmax - vmin) <= 1e-8:
                    vmax = np.amax(helpers.getSlicesV2(originalTensor, {3: feature}))
            originalMat = helpers.getSlicesV2(originalTensor, {axis: idx, 3: feature})
            originalCounterMat = helpers.getSlicesV2(originalCounterTensor, {axis: idx, 3: feature})
            recoveredMat = helpers.getSlicesV2(recoveredTensor, {axis: idx, 3: feature})
            sparsedMat = helpers.getSlicesV2(sparsedTensor, {axis: idx, 3: feature})

            sparsedNMAE = utilsforminds.helpers.getNMAE(originalMat, sparsedMat, originalCounterMat)
            recoveredNMAE = utilsforminds.helpers.getNMAE(originalMat, recoveredMat, originalCounterMat)

            recoveredCounterMat = helpers.getSlicesV2(recoveredTensorConter, {axis: idx, 3: feature})
            if compress is None:
                originalMat = helpers.getSlicesV2(originalTensor, {axis: idx, 3: feature})
                recoveredMat = helpers.getSlicesV2(recoveredTensor, {axis: idx, 3: feature})
                sparsedMat = helpers.getSlicesV2(sparsedTensor, {axis: idx, 3: feature})

                sparsedNMAE = utilsforminds.helpers.getNMAE(originalMat, sparsedMat, originalCounterMat)
                recoveredNMAE = utilsforminds.helpers.getNMAE(originalMat, recoveredMat, originalCounterMat)
            else:
                assert(shape[axis] > compress)
                originalTmp = []
                recoveredTmp = []
                sparsedTmp = []
                if idx + compress >= shape[axis]:
                    for i in range(compress):
                        originalTmp.append(helpers.getSlicesV2(originalTensor, {axis: shape[axis] - compress + i, 3: feature}))
                        recoveredTmp.append(helpers.getSlicesV2(recoveredTensor, {axis: shape[axis] - compress + i, 3: feature}))
                        sparsedTmp.append(helpers.getSlicesV2(sparsedTensor, {axis: shape[axis] - compress + i, 3: feature}))
                else:
                    for i in range(compress):
                        originalTmp.append(helpers.getSlicesV2(originalTensor, {axis: idx + i, 3: feature}))
                        recoveredTmp.append(helpers.getSlicesV2(recoveredTensor, {axis: idx + i, 3: feature}))
                        sparsedTmp.append(helpers.getSlicesV2(sparsedTensor, {axis: idx + i, 3: feature}))
                originalMat = utilsforminds.helpers.compressNparrLst(originalTmp)
                recoveredMat = utilsforminds.helpers.compressNparrLst(recoveredTmp)
                sparsedMat = utilsforminds.helpers.compressNparrLst(sparsedTmp)

                sparsedNMAE = 0.
                recoveredNMAE = 0.  

            if axisInfo is not None:
                positions = [axisInfo[0]['min'], axisInfo[1]['min'], axisInfo[2]['min']]
                positions[axis] = positions[axis] + axisInfo[axis]['grid'] * idx
            else:
                positions = [0, 0, 0]
                positions[axis] = positions[axis] + idx
            titleLst = ['x = {}, y = {}, z = {}\n density = {}'.format(positions[0], positions[1], positions[2], utilsforminds.helpers.getDataDensity(originalCounterMat)), 'NMAE recovered = {}'.format(recoveredNMAE), 'NMAE sampled = {}'.format(sparsedNMAE)]

            if scaleSync:
                if axisInfo is None:
                    for method in methods:
                        if method == 'scatter':
                            plot2Ds([originalMat, recoveredMat, sparsedMat], titleLst, newDirectoryName + '/' + str(feature) + "_" + helpers.axisMap[axis] + "_{}_{}_{}".format(idx, round(samplingProbNum*100), method) + '.' + imgFormat, cbarLabel = 'amount', plotShape = [3, 1], 
                            figSize = (8, 24), planeMaskLst = None, axis = axis, axisInfo = axisInfo, vmin_vmax = [vmin, vmax], method = method, rotate = 180, convertXYaxis = False)
                            # [originalCounterMat, recoveredCounterMat, np.ones(recoveredCounterMat.shape)]
                        else:
                            plot2Ds([originalMat, recoveredMat, sparsedMat], titleLst, newDirectoryName + '/' + str(feature) + "_" + helpers.axisMap[axis] + "_{}_{}_{}".format(idx, round(samplingProbNum*100), method) + '.' + imgFormat, cbarLabel = 'amount', plotShape = [3, 1], 
                            figSize = (8, 24), planeMaskLst = None, axis = axis, axisInfo = axisInfo, vmin_vmax = [vmin, vmax], method = method, rotate = 270, convertXYaxis = False)
                else:
                    for method in methods:
                        # ---------- change here -------------
                        if method == 'scatter':
                            plot2Ds([originalMat, recoveredMat, sparsedMat], titleLst, newDirectoryName + '/' + str(feature) + "_" + helpers.axisMap[axis] + "_{}_{}_{}".format(round(idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"]), round(samplingProbNum*100), method) + '.' + imgFormat, cbarLabel = 'amount', plotShape = [3, 1], 
                            figSize = (8, 24), planeMaskLst = None, axis = axis, axisInfo = axisInfo, vmin_vmax = [vmin, vmax], method = method, rotate = 180, convertXYaxis = False)
                            # [originalCounterMat, recoveredCounterMat, np.ones(recoveredCounterMat.shape)]
                        else:
                            plot2Ds([originalMat, recoveredMat, sparsedMat], titleLst, newDirectoryName + '/' + str(feature) + "_" + helpers.axisMap[axis] + "_{}_{}_{}".format(round(idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"]), round(samplingProbNum*100), method) + '.' + imgFormat, cbarLabel = 'amount', plotShape = [3, 1], 
                            figSize = (8, 24), planeMaskLst = None, axis = axis, axisInfo = axisInfo, vmin_vmax = [vmin, vmax], method = method, rotate = 270, convertXYaxis = False)

            else:
                if axisInfo is None:
                    for method in methods:
                        if method == 'scatter':
                            plot2Ds([originalMat, recoveredMat, sparsedMat], titleLst, newDirectoryName + '/' + str(feature) + "_" + helpers.axisMap[axis] + "_{}_{}_{}".format(idx, round(samplingProbNum*100), method) + '.png', cbarLabel = 'amount', plotShape = [3, 1], 
                            figSize = (8, 24), planeMaskLst = None, axis = axis, axisInfo = axisInfo, method = method, rotate = 180, convertXYaxis = False)
                            
                        else:
                            plot2Ds([originalMat, recoveredMat, sparsedMat], titleLst, newDirectoryName + '/' + str(feature) + "_" + helpers.axisMap[axis] + "_{}_{}_{}".format(idx, round(samplingProbNum*100), method) + '.png', cbarLabel = 'amount', plotShape = [3, 1], 
                            figSize = (8, 24), planeMaskLst = None, axis = axis, axisInfo = axisInfo, method = method, rotate = 270, convertXYaxis = False)
                else:
                    for method in methods:
                        if method == 'scatter':
                            plot2Ds([originalMat, recoveredMat, sparsedMat], titleLst, newDirectoryName + '/' + str(feature) + "_" + helpers.axisMap[axis] + "_{}_{}_{}".format(round(idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"]), round(samplingProbNum*100), method) + '.' + imgFormat, cbarLabel = 'amount', plotShape = [3, 1], 
                            figSize = (8, 24), planeMaskLst = None, axis = axis, axisInfo = axisInfo, method = method, rotate = 180, convertXYaxis = False)
                        else:
                            plot2Ds([originalMat, recoveredMat, sparsedMat], titleLst, newDirectoryName + '/' + str(feature) + "_" + helpers.axisMap[axis] + "_{}_{}_{}".format(round(idx * axisInfo[axis]["grid"] + axisInfo[axis]["min"]), round(samplingProbNum*100), method) + '.' + imgFormat, cbarLabel = 'amount', plotShape = [3, 1], 
                            figSize = (8, 24), planeMaskLst = None, axis = axis, axisInfo = axisInfo, method = method, rotate = 270, convertXYaxis = False)
    
    txtFile = open(utilsforminds.helpers.getExecPath() + '/current_results/' + newDirectoryName + "/summary.txt", "a")
    txtContents = "axis {}:\n4D recovered RMAE : {}\n4D sparsed RMAE: {}\n".format(axis, utilsforminds.helpers.getNMAE(originalTensor, recoveredTensor, originalCounterTensor), utilsforminds.helpers.getNMAE(originalTensor, sparsedTensor, originalCounterTensor))
    txtFile.write(txtContents)

    txtFile.close()

def plot3D(npArr,  vmin, vmax, filename = None):
    '''
        plotting 3D numpy array as scatter, deprecated, use plot3DScatter instead
    '''
    assert(len(npArr.shape) == 3)
    print('plotting 3D numpy array as scatter, deprecated, use plot3DScatter instead')
    # https://stackoverflow.com/questions/12414619/creating-a-3d-plot-from-a-3d-numpy-array
    # z, x, y = npArr.nonzero()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, zdir='z', c=npArr[x, y, z])
    # plt.savefig("./current_results/demo.png")

    # engine = Engine()
    # engine.start()
    # fig = mlab.figure(figure = None, engine = engine)

    obj = mlab.contour3d(npArr, transparent=True, vmin = vmin, vmax = vmax, contours = 15, reset_zoom = False)
    # obj = mlab.points3d(npArr, transparent = True, scale_factor = 1, scale_mode = 'scalar', figure = fig)
    # obj = mayavi.tools.pipeline.scalar_scatter(npArr)

    # scene = engine.scenes[0]

    # https://stackoverflow.com/questions/37146143/mayavi-contour-3d
    # get a handle for the plot
    # iso_surface = scene.children[0].children[0].children[0]

    # the following line will print you everything that you can modify on that object
    # iso_surface.contour.print_traits()

    # mlab.draw()
    if filename != None:
        # mlab.show()
        mlab.savefig(filename = filename, figure = obj)
    else:
        mlab.show()

    mlab.close()
    # mlab.quiver3d(npArr)
    # mlab.show()

def plot3DScatter(npArr, vmin = None, vmax = None, filename = None, axisInfo = None, highest_amount_proportion_threshod = None, small_delta = 1e-8, bar_label = 'gram/ton', default_point_size = 1.0, alpha_min = 0.1, transparent_cbar = False, adjust_axis_ratio = True):
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
        ax.scatter(x, y, z, zdir='z', c= npArr_[x, y, z], vmin = vmin, vmax = vmax, alpha = max(1. - (num_obs / num_entries) ** (1/12), alpha_min), s = default_point_size * (100/avg_length)) ## parameter s is point size
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
    cbar.set_label(bar_label)
    if not transparent_cbar:
        cbar.solids.set(alpha=1)

    plt.savefig(filename)

def plot3DScatterDistinguish(nparrValue, nparrOriginalMask = None, nparrRecoveredMask = None, vmin = None, vmax = None, filename = None, axisInfo = None, maxNth = None, cbarLabel = 'amount'):
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


# xyz=np.array(np.random.random((100,3)))
# x=xyz[:,0]
# y=xyz[:,1]
# z=xyz[:,2]*100

# print('test end')

def plotCube(big_cube, cube, dirPath = None):

    data = np.ones([big_cube.shape[0],big_cube.shape[1],big_cube.shape[2]])
    n_voxels = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=bool)

    for step in range(len(cube)):
        #ax = make_ax(True)
        n_voxels = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=bool)
        for small_cube in cube[step].keys():
            x1 = int(small_cube[0][0]) #x-min
            y1 = int(small_cube[0][1]) #y-min
            z1 = int(small_cube[0][2]) #z-min
            x2 = int(small_cube[1][0]) #x-max
            y2 = int(small_cube[1][1]) #y-max
            z2 = int(small_cube[1][2]) #z-max

            for i in range(x1,x2):
                for j in range(y1,y2):
                    for k in range(z1,z2):
                        n_voxels[i,j,k] = True

        #facecolors1 = np.where(n_voxels, '#FFD65DC0', '#1f77b430')
        facecolors1 = np.where(n_voxels, 'red', '#1f77b430')
        edgecolors = np.where(n_voxels, '#BFAB6E', '#7D84A6')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(n_voxels, facecolors=facecolors1, edgecolors=facecolors1)

        if dirPath != None:
            plt.savefig(dirPath + os.path.sep + f'frame_{step}.png')
        else:
            plt.show()

# def plotCubes2(npArr, listOfSteps, dirPath = None):
#     assert(len(listOfSteps) < 100)

#     # data = np.ones([big_cube.shape[0],big_cube.shape[1],big_cube.shape[2]])
#     # n_voxels = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=bool)

#     for step in range(len(listOfSteps)): 
#         #ax = make_ax(True)
#         toColor = np.zeros(npArr.shape)
#         # n_voxels = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=bool)
#         # for small_cube in cube[step].keys():
#         #     x1 = int(small_cube[0][0]) #x-min
#         #     y1 = int(small_cube[0][1]) #y-min
#         #     z1 = int(small_cube[0][2]) #z-min
#         #     x2 = int(small_cube[1][0]) #x-max
#         #     y2 = int(small_cube[1][1]) #y-max
#         #     z2 = int(small_cube[1][2]) #z-max

#         #     for i in range(x1,x2):
#         #         for j in range(y1,y2):
#         #             for k in range(z1,z2):
#         #                 n_voxels[i,j,k] = True

#         for cube in listOfSteps[step].keys():
#             for i in range(cube[0][0], cube[1][0] + 1):
#                 for j in range(cube[0][1], cube[1][1] + 1):
#                     for k in range(cube[0][2], cube[1][2] + 1):
#                         toColor[i, j, k] = True

#         #facecolors1 = np.where(n_voxels, '#FFD65DC0', '#1f77b430')
#         facecolors1 = np.where(toColor, 'red', '#1f77b430')
#         # edgecolors = np.where(toColor, '#BFAB6E', '#7D84A6')
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         ax.voxels(toColor, facecolors=facecolors1, edgecolors=facecolors1)

#         if dirPath != None:
#             plt.savefig(dirPath + os.path.sep + f'frame_{step}.png')
#         else:
#             plt.show()

# def plotWellbores(holes):
#     # fig = matplotlib.pyplot.figure()
#     # ax  = fig.add_subplot(111, projection = '3d')
#     data = []

#     count = 0
#     for hole in holes.keys():
#         count = count + 1
#         # get x y z values for hole
#         pts = holes[hole].getListOfXyzPairFromLength(0,holes[hole].length)
#         # plt.plot(pts[0][0][0],pts[0][0][1],pts[0][0][2],marker='+',color='g')
#         print(hole)
#         wellhead = True
#         for p in range(len(pts)):
#             x = pd.Series([ pts[p][0][0], pts[p][1][0] ])
#             y = pd.Series([ pts[p][0][1], pts[p][1][1] ])
#             z = pd.Series([ pts[p][0][2], pts[p][1][2] ])

#             trace = go.Scatter3d(
#                 x=x, y=y, z=z,
#                 marker = dict(
#                     size=1,
#                     color='#1f77b4'
#                 ),
#                 line=dict(
#                     color='#1f77b4',
#                     width=1
#                 )
#             )
#             data.append(trace)
#         if count == 2000:
#             break
    
#     layout = dict(
#         width=900,
#         height=800,
#         autosize=False,
#         title='Eskay Creek wellbores',
#         scene=dict(
#             xaxis=dict(
#                 gridcolor='rgb(255, 255, 255)',
#                 zerolinecolor='rgb(255, 255, 255)',
#                 showbackground=True,
#                 backgroundcolor='rgb(230, 230,230)'
#             ),
#             yaxis=dict(
#                 gridcolor='rgb(255, 255, 255)',
#                 zerolinecolor='rgb(255, 255, 255)',
#                 showbackground=True,
#                 backgroundcolor='rgb(230, 230,230)'
#             ),
#             zaxis=dict(
#                 gridcolor='rgb(255, 255, 255)',
#                 zerolinecolor='rgb(255, 255, 255)',
#                 showbackground=True,
#                 backgroundcolor='rgb(230, 230,230)'
#             ),
#             camera=dict(
#                 up=dict(
#                     x=0,
#                     y=0,
#                     z=1
#                 ),
#                 eye=dict(
#                     x=-1.7428,
#                     y=1.0707,
#                     z=0.7100,
#                 )
#             ),
#             aspectratio = dict( x=1, y=1, z=0.7 ),
#             aspectmode = 'manual'
#         ),
#         showlegend = False
#     )

#     fig = go.Figure(data=data, layout=layout)
#     plotly.offline.plot(fig)
#     # fig.show()
#     # py.iplot(fig, filename='wellbores', height=700)

# def plotWellbores(holes):
#     fig = matplotlib.pyplot.figure(figsize=(15,15))
#     ax = fig.gca(projection='3d')
#     # ax.set_aspect('equal')
#     count = 0
#     for hole in holes.keys():
#         count = count + 1
#         numSegments = len(holes[hole].distributionData)
#         wellhead = True
#         for s in range(numSegments):
#             x_1, y_1, z_1 = holes[hole].getXyzFromLength(holes[hole].distributionData[s]['from'])
#             x_2, y_2, z_2 = holes[hole].getXyzFromLength(holes[hole].distributionData[s]['to'])

#             if wellhead:
#                 ax.scatter(x_1, y_1, z_1, marker='+', color='lime',s=14)
#                 wellhead = False
            
#             au_dist = holes[hole].distributionData[s]['au']
#             if au_dist is not None:
#                 ax.plot([x_1, x_2],
#                         [y_1, y_2],
#                         [z_1, z_2],color='darkslategray',linewidth=0.5)
#             else:
#                 ax.plot([x_1, x_2],
#                         [y_1, y_2],             
#                         [z_1, z_2],color='darkorchid',linewidth=0.5)

#         # if count > 2:
#         #     break
#     ax.set_xlabel('X')        
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     for ii in range(0,360,60):
#         ax.view_init(elev=10., azim=ii)
#         plt.savefig("/Users/danielamachnik/projects/py/lofting/wellbores_360/view%d.png" % ii,bbox_inches='tight', pad_inches=0)                          

def plotWellbores(holes):
    fig = matplotlib.pyplot.figure(figsize=(15,15))
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')
    # get x y z values for hole
    for hole in holes.keys():
        print(hole)
        pts = holes[hole].getListOfXyzPairFromLength(0,holes[hole].length)
        wellhead = True
        for p in range(len(pts)):
            if wellhead:
                ax.scatter(pts[p][0][0], pts[p][0][1], pts[p][0][2], marker='+', color='lime',s=16)
                wellhead = False

            ax.plot([pts[p][0][0],pts[p][1][0]],
                        [pts[p][0][1],pts[p][1][1]],
                        [pts[p][0][2],pts[p][1][2]],color='darkslategray',linewidth=0.5)

    ax.set_xlabel('X')        
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for ii in range(0,360,60):
        ax.view_init(elev=10., azim=ii)
        plt.savefig("/Users/danielamachnik/projects/py/lofting/wellbores_360/view%d.png" % ii,bbox_inches='tight', pad_inches=0) 

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

def plot_bar_charts(path_to_save : str, name_numbers : dict, xlabels : list, xtitle = None, ytitle = None, bar_width = 'auto', alpha = 0.8, colors_dict = None, format = 'eps', diagonal_xtickers = False, name_errors = None, name_to_show_percentage = None, fontsize = 10, title = None, fix_legend = True):
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
        if name_errors is not None:
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
    
# if __name__ == '__main__':
    # plot_bar_charts('dummy', {'Frank':[12.7, 0.4, 4.4, 5.3, 7.1, 3.2], 'Guido':[6.3, 10.3, 10, 0.3, 5.3, 2.9]}, ['RR', 'Lasso', 'SVR', 'CNN', 'SVR', 'LR'], ytitle="RMSE of Prediction of TRIAILB-A")