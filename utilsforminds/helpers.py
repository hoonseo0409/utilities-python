import numpy as np
import random
import os
import sys
import math
import numbers
import pickle
import moviepy.editor as mpy
from copy import deepcopy
import pandas as pd

# from sklearn.grid_search import ParameterGrid # moved in recent version of sklearn
from sklearn.model_selection import ParameterGrid
import inspect

import shutil
from PIL import Image, ImageSequence
import tensorflow as tf
import utilsforminds.visualization as visualization

axisMap = {0: 'x', 1: 'y', 2: 'z', 'x': 0, 'y': 1, 'z': 2}

def getExecPath():
    '''
        ref: https://stackoverflow.com/questions/606561/how-to-get-filename-of-the-main-module-in-python
    '''
    try:
        sFile = os.path.abspath(sys.modules['__main__'].__file__)
    except:
        sFile = sys.executable
    return os.path.dirname(sFile)

def getNewDirectoryName(parentDir, newDir):
    '''
        To get new directory name to save results while avoiding duplication
    '''

    if parentDir[0] != '/':
        parentDir = '/' + parentDir
    if parentDir[-1] != '/':
        parentDir = parentDir + '/'

    assert(getExecPath() + parentDir)

    duplicatedNameNum = 0
    while(os.path.isdir(getExecPath() + parentDir + newDir + str(duplicatedNameNum))):
        duplicatedNameNum = duplicatedNameNum + 1
    newDir = newDir + str(duplicatedNameNum)

    return newDir

# def getPlaneAverageData(npArr, normalDirection):  # https://stackoverflow.com/questions/37142135/sum-numpy-ndarray-with-3d-array-along-a-given-axis-1
#     assert(normalDirection in ('x', 'y', 'z'))
#     if normalDirection == 'x':
#         return np.average(npArr, axis=0)
#     elif normalDirection == 'y':
#         return np.average(npArr, axis=1)
#     else:
#         return np.average(npArr, axis=2)

def getDataDensity(npArr):
    """Get the proportion of non-zero elements.

    Works for any dimension
    
    """

    noEntries = 1
    for length in npArr.shape:
        noEntries = noEntries * length
    return 1. - np.count_nonzero(npArr == 0.)/float(noEntries)

# def getMostObsIdx(npArr3D, axis):
#     assert(axis in ('x', 'y', 'z'))
#     maxCount = -1
#     maxIdx = 0
#     if axis == 'x':
#         for i in range(npArr3D.shape[0]):
#             if maxCount <= np.count_nonzero(npArr3D[i, :, :]):
#                 maxIdx = i
#                 maxCount = np.count_nonzero(npArr3D[i, :, :])
#         return maxIdx
#     elif axis == 'y':
#         for i in range(npArr3D.shape[1]):
#             if maxCount <= np.count_nonzero(npArr3D[:, i, :]):
#                 maxIdx = i
#                 maxCount = np.count_nonzero(npArr3D[:, i, :])
#         return maxIdx
#     else: # axis == 'z':
#         for i in range(npArr3D.shape[2]):
#             if maxCount <= np.count_nonzero(npArr3D[:, :, i]):
#                 maxIdx = i
#                 maxCount = np.count_nonzero(npArr3D[:, :, i])
#         return maxIdx


def getNMAE(ground_truth_amount_arr, recovered_arr, ground_truth_counter_arr):
    """Get Normalized Mean Absolute Error.

    Works for any dimension.
    
    """

    assert(ground_truth_amount_arr.shape == recovered_arr.shape and recovered_arr.shape == ground_truth_counter_arr.shape)
    rMax = ground_truth_amount_arr.max()
    rMin = ground_truth_amount_arr.min()
    groundTruthCounterTensor01 = np.where(ground_truth_counter_arr >= 1., 1., 0.)
    if np.count_nonzero(groundTruthCounterTensor01) == 0. or rMax == rMin:
        return 0.
    return np.sum(np.absolute(ground_truth_amount_arr - recovered_arr) * groundTruthCounterTensor01) / (np.sum(groundTruthCounterTensor01) * (rMax - rMin))

def getSparsedDataCounterRandom(npArrData, npArrCounter, samplingProb):
    """Get regularly sampled mineral amount array and counter array.

    Parameters
    ----------
    npArrData : Numpy array
        Array of mineral amounts.
    npArrCounter : Numpy array
        Array of the number of observations.
    samplingProb : float
        Probability to KEEP the observations. So large samplingProb gives more dense sampled array.
    
    Returns
    -------
    npArrData * boolArr : Numpy array
        Sampled mineral amount array.
    npArrCounter * boolArr : Numpy array
        Sample counter array.
    """

    assert(samplingProb <= 1. and samplingProb >= 0.)
    assert(npArrCounter.shape == npArrData.shape)
    
    shape_ = npArrData.shape
    randomArr = np.random.uniform(0., 1., size = shape_)
    boolArr = randomArr < samplingProb
    return npArrData * boolArr, npArrCounter * boolArr

def getSparsedDataCounterCuvicHole(npArrData, npArrCounter, lstOfProportions):
    """Get irregularly(cubic shape) sampled mineral amount array and counter array.

    Works only for 3D or 4D array.
    Parameters
    ----------
    npArrData : Numpy array
        Array of mineral amounts.
    npArrCounter : Numpy array
        Array of the number of observations.
    lstOfProportions : list of floats
        Volume proportions of REMOVAL cubes. 
        For example, [0.1, 0.1, 0.1] will create three blanks with 10% volume proportions.
        So large proportions give more sparse sampled array.
    
    Returns
    -------
    npArrData * samplingArr : Numpy array
        Sampled mineral amount array.
    npArrCounter * samplingArr : Numpy array
        Sample counter array.
    """

    # assert(sum(lstOfProportions) <= 1. and sum(lstOfProportions) >= 0.)
    for proportion in lstOfProportions:
        assert(proportion > 0. and proportion <= 1.)
    assert((len(npArrData.shape) == 3 or len(npArrData.shape) == 4) and npArrData.shape == npArrCounter.shape)

    shape_ = npArrData.shape
    samplingArr = np.ones(shape_)

    for proportion in lstOfProportions:
        proportion = proportion ** (1/3)
        cubeSize = (math.floor(shape_[0] * proportion), math.floor(shape_[1] * proportion), math.floor(shape_[2] * proportion))
        startingIdc = []
        for axis in range(3):
            startingIdc.append(random.randint(0, shape_[axis] - cubeSize[axis]))
        if len(npArrData.shape) == 3:
            samplingArr[startingIdc[0]: startingIdc[0] + cubeSize[0], startingIdc[1]: startingIdc[1] + cubeSize[1], startingIdc[2]: startingIdc[2] + cubeSize[2]] = 0.
        else:
            samplingArr[startingIdc[0]: startingIdc[0] + cubeSize[0], startingIdc[1]: startingIdc[1] + cubeSize[1], startingIdc[2]: startingIdc[2] + cubeSize[2], :] = 0.
    print(f'Proportion of sampled entries: {np.count_nonzero(npArrCounter * samplingArr)/np.count_nonzero(npArrCounter)}')

    return npArrData * samplingArr, npArrCounter * samplingArr

# def getScaledMinMax(npArr2D):
#     """
#         Deprecated
#     """
#     shape = npArr2D.shape
#     assert(len(shape) == 2)
#     max = npArr2D.max()
#     min = npArr2D.min()
#     scaled = np.zeros(shape)

#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             scaled[i, j] = (npArr2D[i, j] - min)/(max - min)
    
#     return scaled, min, max

def min_max_scale(arr, vmin = None, vmax = None):
    """Apply min-max scaling
    
    The range becomes [vmin, vmax] if vmin and vmax are both given, 
    [vmin, vmin + 1] if only vmin is given, 
    [vmax - 1, vmax] if only vmax is given,
    [0, 1] if vmin and vmax are both not given.
    """

    max_ = np.max(arr)
    min_ = np.min(arr)
    if max_ == min_:
        arr_01_scaled = np.ones(arr.shape) * 0.5
    else:
        arr_01_scaled = (arr - min_) / (max_ - min_)
    if vmax is not None and vmin is not None:
        assert(vmax >= vmin)
        return arr_01_scaled * (vmax - vimn) + vmin
    elif vmax is not None and vmin is None:
        return arr_01_scaled + (vmax - 1.)
    elif vmax is None and vmin is not None:
        return arr_01_scaled + vmin
    else:
        return arr_01_scaled
    

def reverseMinMaxScale(arr, min_, max_, onlyPositive = False):
    if max_ > min_:
        reversedScaled = arr * (max_ - min_) + min_
        if onlyPositive:
            return np.where(reversedScaled >= 0., reversedScaled, 0.)
        else:
            return reversedScaled
    elif max_ == min_:
        print(f'WARNING: min_: {min_} == max_: {max_}')
        return np.where(arr == min_, 0.5, 0.)
    else:
        raise Exception(f'max: {max_} is smaller than min: {min_}')

def is_small_container(container, length_limit = 20):
    if (type(container) == type({}) or type(container) == type([]) or type(container) == type(tuple([3, 3]))) and len(container) < length_limit:
        if type(container) == type({}):
            for key, value in container.items():
                if not (isinstance(key, (numbers.Number, type('a'), type(True), type(None))) and isinstance(value, (numbers.Number, type('a'), type(True), type(None)))):
                    return False
            return True
        if type(container) == type([]) or type(container) == type(tuple(3, 3)):
            for item in container:
                if not (isinstance(item, (numbers.Number, type('a'), type(True), type(None)))):
                    return False
            return True
    else:
        False

def load_csv_columns_into_list(path_to_csv: str):
    result_dict = {}
    with open(path_to_csv, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter = ',')
        headers = csv_reader.fieldnames
        for header in headers:
            result_dict[header] = []
        for line in csv_reader:
            for header in line.keys():
                result_dict[header].append(line[header])
    del result_dict['']
    return result_dict

def paramDictToStr(param_dict):
    result = ""
    for key in param_dict.keys():
        if (isinstance(key, (numbers.Number, type('a'), type(True), type(None))) or is_small_container(key)) and (isinstance(param_dict[key], (numbers.Number, type('a'), type(True), type(None))) or is_small_container(param_dict[key])):
            result = result + str(key) + " : " + str(param_dict[key]) + "\n"
    return result

def gridSearch(function, params_grid):
    duplicatedNameNum = 0
    while(os.path.isfile(os.path.dirname(__file__) + '/gridSearchResults/' + function.__name__ + '_' + str(duplicatedNameNum) + ".txt")):
        duplicatedNameNum = duplicatedNameNum + 1
    txtFile = open(os.path.dirname(__file__) + '/gridSearchResults/' + function.__name__ + '_' + str(duplicatedNameNum) + ".txt", "w")

    grid = ParameterGrid(params_grid)
    bestParams = {}
    bestScore = -1. * sys.float_info.max
    for params in grid:
        largerBetterScore = function(**params)
        if largerBetterScore >= bestScore:
            bestScore = largerBetterScore
            bestParams = params
            print(str(bestParams) + "\n\t" + str(bestScore) + "\n")
        txtFile.write(str(params) + "\n\t" + str(largerBetterScore) + "\n")
    
    assert(bestScore >= 0.)
    # txtContents = txtContents + "\n\n--best result--\n" + str(bestParams) + "\n\t" + str(round(100 * bestScore))
    # txtFile = open(os.path.dirname(__file__) + "/gridSearchResults/" + function.__name__ + '_' + str(round(100 * bestScore)) + ".txt", "w")
    # txtFile.write(txtContents)
    txtFile.write("\n\n--best result--\n" + str(bestParams) + "\n\t" + bestScore)
    txtFile.close()

def makeTestArr(shape):
    assert(len(shape) == 4)
    result = []
    for i in range(shape[0]):
        result.append([])
        for j in range(shape[1]):
            result[i].append([])
            for k in range(shape[2]):
                result[i][j].append([])
                for l in range(shape[3]):
                    result[i][j][k].append(str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l))
    return np.array(result)

def splitRange(isNoPartitions, number, start, end):
    """
    
    Examples
    --------
    helpers.splitRange(True, 3, 2, 15) : 
        [(2, 5), (7, 10), (12, 15)]
    helpers.splitRange(False, 3, 2, 15) :
        [(2, 3), (5, 6), (8, 9), (11, 12), (14, 15)]
    helpers.splitRange(False, 6, 2, 15) : 
        [(2, 6), (8, 12), (14, 15)]
    helpers.splitRange(True, 6, 2, 15) : 
        [(2, 3), (5, 6), (8, 9), (11, 12), (14, 15)]
    """

    assert(number > 0 and end > start and end > 0 and start >= 0)
    if not isNoPartitions:
        assert(number < end - start)
    if isNoPartitions:
        partSize = math.ceil((end - start) / number)
    else:
        partSize = number
    result = []
    startIdx = start
    endIdx = partSize
    while(True):
        if(startIdx + partSize >= end):
            result.append((startIdx, end))
            break
        result.append((startIdx, endIdx))
        startIdx = startIdx + partSize
        endIdx = endIdx + partSize
    return result

def splitList(isNoPartitions, number, lst):
    """
    
    Examples
    --------
    helpers.splitList(True, 5, list(range(31))) : 
        [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27], [28, 29, 30]]
    helpers.splitList(True, 5, list(range(30))) :
        [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29]]
    helpers.splitList(True, 5, list(range(29))) : 
        [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28]]
    helpers.splitList(False, 5, list(range(31))) : 
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29], [30]]
    """

    assert(number > 0 and len(lst) > 0)
    if not isNoPartitions:
        number < len(lst)
    ranges = splitRange(isNoPartitions, number, 0, len(lst))
    result = []
    for rng in ranges:
        result.append(lst[rng[0]:rng[1]])
    return result

def fromDictToDirName(dict, limitLength = 150):
    dirName = ''
    for key in dict.keys():
        if (isinstance(key, numbers.Number) or isinstance(key, str)) and (isinstance(dict[key], numbers.Number) or isinstance(dict[key], str)):
            dirName = dirName + str(key).replace('.', 'p') + '-' +str(dict[key]).replace('.', 'p') + '_'
    assert(dirName != '')
    dirName = dirName[:-1]
    if len(dirName) > limitLength:
        return dirName[:limitLength]
    else:
        return dirName

def getDirSize(pathStart): # https://stackoverflow.com/questions/20267220/how-to-check-size-of-the-files-in-a-directory-with-python
    bytes = 0
    for f in os.listdir(pathStart):
        path = os.path.join(pathStart, f)
        if os.path.isfile(path):
            bytes = bytes + os.path.getsize(path)
    return bytes / (1000 * 1000 * 1000) # return in GB unit

# def saveGeoTensor(geoTensor, suffix = ''):
#     sizeCurr = getDirSize('mappings')
#     print('print size of mapping files: ', str(sizeCurr) + 'GB')
#     noOfEntries = geoTensor.shape[0] * geoTensor.shape[1] * geoTensor.shape[2]
#     assert(noOfEntries <= 300 * 700 * 300 and sizeCurr < 20.) # to prevent too large disk consumption
#     if geoTensor.sampling_rate is not None:
#         sampling_rate = round(geoTensor.sampling_rate * 100)
#     else:
#         sampling_rate = None

#     # geoTensorFile = open(getExecPath() + '/mappings/' + 'shape-' + str(geoTensor.shape[0]) + '_' + str(geoTensor.shape[1]) + '_' + str(geoTensor.shape[2]) + '_' + 'mapping-' + '' + geoTensor.mapping + '.obj', 'wb')
#     geoTensorFile = open(getExecPath() + f'/mappings/shape-{str(geoTensor.shape[0])}_{str(geoTensor.shape[1])}_{str(geoTensor.shape[2])}_mapping-{geoTensor.mapping}_sr-{str(sampling_rate)}_sm-{str(geoTensor.sampling_method)}'+suffix+'.obj', 'wb')
#     pickle.dump(geoTensor, geoTensorFile)

# def loadGeoTensor(shape, mapping, sampling_rate = None, sampling_method = None, suffix = ''):
#     sizeCurr = getDirSize('mappings')
#     print('print size of mapping files: ', str(sizeCurr) + 'GB')

#     if sampling_method == 'irregular':
#         total_sampling_rate = 1.
#         for proportion in sampling_rate:
#             total_sampling_rate = total_sampling_rate * (1 - proportion)
#         total_sampling_rate = round(total_sampling_rate * 100)
#     elif sampling_method == 'regular':
#         total_sampling_rate = round(sampling_rate * 100)
#     else:
#         total_sampling_rate = None
#     # geoTensorFileOpened = open(getExecPath() + '/mappings/' + 'shape-' + str(shape[0]) + '_' + str(shape[1]) + '_' + str(shape[2]) + '_' + 'mapping-' + mapping + '_sr-' + str(sampling_rate) + '.obj', 'rb')
#     geoTensorFileOpened = open(getExecPath() + f'/mappings/shape-{str(shape[0])}_{str(shape[1])}_{str(shape[2])}_mapping-{mapping}_sr-{str(total_sampling_rate)}_sm-{str(sampling_method)}'+suffix+'.obj', 'rb')
#     # print('size of loaded obj: ', sys.getsizeof(geoTensorFileOpened))
#     return pickle.load(geoTensorFileOpened)

# def keepAvgOnUnobs4D(dataAfterImputation, maskAfterImputation, maskBeforeImputation, avgBeforeImputation):
#     assert(len(dataAfterImputation.shape) == 4 and dataAfterImputation.shape == maskAfterImputation.shape and maskAfterImputation.shape == maskBeforeImputation.shape)
#     mask01AfterImputation = np.where(maskAfterImputation >= 1., 1., 0.)
#     mask01BeforeImputation = np.where(maskBeforeImputation >= 1., 1., 0.)
#     mask01ForImputedEntries = mask01AfterImputation - mask01BeforeImputation

#     avgForImputedEntries = np.sum(mask01ForImputedEntries * dataAfterImputation) / np.count_nonzero(mask01ForImputedEntries)

#     return dataAfterImputation * mask01BeforeImputation + dataAfterImputation * mask01ForImputedEntries * (avgBeforeImputation / (avgForImputedEntries + 1e-8))

# def keepAvgOnUnobs4DForEachFeature(dataAfterImputation, maskAfterImputation, maskBeforeImputation, avgLst):
#     '''
#         Make average of imputed entries same as each feature of avgLst
#     '''
#     assert(len(dataAfterImputation.shape) == 4 and dataAfterImputation.shape == maskAfterImputation.shape and maskAfterImputation.shape == maskBeforeImputation.shape and len(avgLst) == dataAfterImputation.shape[3])

#     dataResult = np.copy(dataAfterImputation)

#     # mask01AfterImputationLst = []
#     # mask01BeforeImputationLst = []
#     # mask01ForImputedEntriesLst = []

#     for feature in range(len(avgLst)):
#         # mask01AfterImputationLst.append(np.where(maskAfterImputation >= 1., 1., 0.))
#         # mask01BeforeImputationLst.append(np.where(maskBeforeImputation >= 1., 1., 0.))
#         mask01AfterImputation = np.where(maskAfterImputation[:, :, :, feature] >= 1., 1., 0.)
#         mask01BeforeImputation = np.where(maskBeforeImputation[:, :, :, feature] >= 1., 1., 0.)
#         mask01ForImputedEntries = mask01AfterImputation - mask01BeforeImputation

#         avgForImputedEntries = np.sum(mask01ForImputedEntries * dataResult[:, :, :, feature]) / (np.count_nonzero(mask01ForImputedEntries) + 1e-8)
#         dataResult[:, :, :, feature] = dataResult[:, :, :, feature] * mask01BeforeImputation + dataResult[:, :, :, feature] * mask01ForImputedEntries * (avgLst[feature] / (avgForImputedEntries + 1e-8))

#     return dataResult

def getAvgLstFrom4DArr(dataArr, counterArr):
    '''
        Get average for each feature on entries whose counterArr > 1.
    '''

    assert(len(dataArr.shape) == 4 and dataArr.shape == counterArr.shape)
    avgLst = []
    for feature in range(dataArr.shape[3]):
        mask01 = np.where(counterArr[:, :, :, feature] >= 1., 1., 0.)
        avgLst.append(np.sum(dataArr[:, :, :, feature] * mask01) / (np.count_nonzero(mask01) + 1e-8))
    return avgLst
    
def getRMSE(groundTruthValueArr, targetValueArr, groundTruthMaskArr):
    """Get Root Mean Squared Error

    Works for any dimension.
    
    """
    groundTruthMaskArr01 = np.where(groundTruthMaskArr >= 1., 1., 0.)
    return np.sum(groundTruthMaskArr01 * (groundTruthValueArr - targetValueArr) ** 2.) ** 0.5 / np.sum(groundTruthMaskArr01)

def reverseLst(lst):
    resultLst = []
    for i in range(len(lst)):
        resultLst.append(lst[len(lst) - 1 - i])
    return resultLst

def compressNparrLst(lst):
    assert(type(lst) == type([]) and len(lst) > 0)
    shape_ = lst[0].shape
    counterNparr = np.zeros(shape_)
    valueNparr = np.zeros(shape_)
    with np.errstate(divide='ignore',invalid='ignore'):
        for idx in range(len(lst)):
            assert(lst[idx].shape == shape_)
            counterNparr = np.where(lst[idx] > 1e-8, counterNparr + 1, counterNparr)
            valueNparr = np.where(counterNparr > 1e-8, np.where(lst[idx] > 1e-8, (valueNparr * (counterNparr - 1) + lst[idx]) / counterNparr, valueNparr), valueNparr)
    
    return valueNparr

# def getSlices(npArr, dimIdxDict):
#     """Deprecated. Use getSlicesV2 instead.

#         example: getSlices(arr, {2: 3, 0: 1}) == arr[1, :, 3]
#         getSlices(arr, {2: [1, 3], 0:1}) == arr[1, :, 1:3]
#     """
#     dims = []
#     idxs = []
#     for dim in dimIdxDict.keys():
#         dims.append(dim)
#         if type(dimIdxDict[dim]) == type([]) or type(dimIdxDict[dim]) == type(()):
#             assert(len(dimIdxDict[dim]) == 2)
#             assert(npArr.shape[dim] >= dimIdxDict[dim][1])
#             idxs.append(list(range(dimIdxDict[dim][0], dimIdxDict[dim][1])))
#         else:
#             assert(npArr.shape[dim] > dimIdxDict[dim])
#             idxs.append(dimIdxDict[dim])
    
#     movedDims = list(range(len(dims)))
#     return np.moveaxis(npArr, dims, movedDims)[tuple(idxs)]

def getSlicesV2(npArr, dimIdxDict):
    """Get slices given tuple.

    ref: https://stackoverflow.com/questions/39474396/building-a-tuple-containing-colons-to-be-used-to-index-a-numpy-array
    
    Examples
    --------
        getSlicesV2(arr, {2: 3, 0: 1}) == arr[1, :, 3]
        getSlicesV2(arr, {2: [1, 3], 0:1}) == arr[1, :, 1:3]

    """

    indices = []
    for dim in range(len(npArr.shape)):
        if dim not in dimIdxDict.keys():
            indices.append(slice(None))
        elif type(dimIdxDict[dim]) == type([]) or type(dimIdxDict[dim]) == type(()):
            if len(dimIdxDict[dim]) == 2:
                indices.append(slice(dimIdxDict[dim][0], dimIdxDict[dim][1]))
            elif len(dimIdxDict[dim]) == 2:
                indices.append(slice(dimIdxDict[dim][0], dimIdxDict[dim][1], dimIdxDict[dim][2]))
            else:
                raise Exception(f'wrong length of slice:{len(dimIdxDict[dim])}, this should be 2 or 3')
        else:
            indices.append(dimIdxDict[dim])
    return npArr[tuple(indices)]

def get_slices_with_idc_rec(arr, dim_idc_dict):
    """ Helper recursive function for get_slices_with_idc().

    Don't use this directly.
    
    """

    if len(dim_idc_dict) == 0:
        return arr
    elif len(dim_idc_dict) == 1:
        single_key = next(iter(dim_idc_dict))
        return np.take(a = arr, indices = dim_idc_dict[single_key], axis = single_key)
    else:
        single_key = next(iter(dim_idc_dict))
        single_item = dim_idc_dict[single_key]
        del dim_idc_dict[single_key]
        return get_slices_with_idc_rec(np.take(arr, indices = single_item, axis = single_key), dim_idc_dict)

def get_slices_with_idc(arr, dim_idc_dict):
    """ Slicing or Indexing Numpy array with given dlictionary {axis: indices}.

    Examples
    --------
    get_slices_with_idc(arr, {0: [2, 3], 3: [4, 1, 0]}) == arr[(2, 3),:,:,(4, 1, 0)]\n
    arr.shape == (2, 4, 3) => get_slices_with_idc(arr, 1: [2]).shape == (2, 1, 3), not (2, 3).
    
    """

    dim_idc_dict_copied = deepcopy(dim_idc_dict)
    return get_slices_with_idc_rec(arr, dim_idc_dict_copied)

def is_number_repl_isdigit(s):
    """Returns True is string is a digit. 

    ref : https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
    
    Parameters
    ----------
    s : str
        Input string.
    returns
    -------
    : bool
        Whether s is digit or not.
    """

    return s.replace('.','',1).isdigit()

# def isProperNumFunctEskayCreek(string):
#     if type(string) == type(1.2):
#         if string < 0.:
#             return False
#         else:
#             return string
#     elif string == '':
#         return False
#     elif is_number_repl_isdigit(string):
#         if float(string) < 0.:
#             return False
#         else:
#             return float(string)
#     else:
#         return False

# def isProperLabelFunctEskayCreek(string):
#     if string [0:2] != 'U-':
#         return string
#     else:
#         return False

def deleteKeysFromDct(dct, keys):
    """Delete elements from dictionary dct whose key is in keys
    
    Returns
    -------
    deletedDct : dict
    """

    deletedDct = dict(dct)
    for key in keys:
        del deletedDct[key]
    return deletedDct

def mergeDcts(dct1, dct2):
    """Merge two dictionaries into one dictionary
    
    Returns
    -------
    : dict
        Merged dictionary
    """

    return {**dct1, **dct2}

def popFromLstWithIdc(lst, idc):
    """Pop multiple elements from lst
    
    Parameters
    ----------
    lst : list
        list to be poped
    idc : iterable
        indices
    
    Returns
    -------
    poped : list
        list of poped elements

    Side effects
    ------------
    lst will be changed
    """

    idc = sorted(idc, reverse = True)
    poped = []
    for idx in idc:
        poped.append(lst.pop(idx))
    return poped

def getNMAE_window_rec(groundTruthTensor, groundTruthCounterTensor, recoveredTensor, window, stride = None):
    """Recursive function to get window-wise absolute error, number of observations, max, and min
    
    This is designed to be used in getNMAE_window function

    Parameters
    ----------
    groundTruthTensor : Numpy array
        array of ground truth values
    groundTruthCounterTensor : Numpy array
        array of the number of observations
    recoveredTensor : Numpy array
        array of the imputed values
    window : Numpy array
        window which will be applied on the values of data as convolution
    stride : list or tuple
        step sizes for each axis

    Returns
    -------
    sumNumObsv : float
        window-wise absolute error
    sumNumObsv : float
        window-wise number of observations
    rMax : float
        window-wise maximum value
    rMin : float
        window-wise minimum value
    """

    shape_ = groundTruthTensor.shape
    assert(len(groundTruthCounterTensor.shape) == len(shape_))
    assert(len(recoveredTensor.shape) == len(shape_))
    assert(len(window.shape) == len(shape_))
    if stride is None:
        stride = []
        for axis in range(len(shape_)):
            stride.append(1)
    assert(len(stride) == len(shape_))

    rMax = - sys.float_info.max
    rMin = sys.float_info.min

    axisToMove = 0
    for axis in range(len(shape_) - 1):
        if shape_[axisToMove] != window.shape[axisToMove]:
            break
        else:
            axisToMove = axisToMove + 1
    if axisToMove == len(shape_) - 1: # Base case: last dim
        sumNMAE = 0.
        sumNumObsv = 0.
        position = 0
        while(position + window.shape[axisToMove] <= shape_[axisToMove]):
            groundTruthConv = np.sum(window * getSlicesV2(groundTruthTensor, {len(shape_) - 1 : [position, position + window.shape[axisToMove]]}) * getSlicesV2(groundTruthCounterTensor, {len(shape_) - 1 : [position, position + window.shape[axisToMove]]}))
            recoveredConv = np.sum(window * getSlicesV2(recoveredTensor, {len(shape_) - 1 : [position, position + window.shape[axisToMove]]}) * getSlicesV2(groundTruthCounterTensor, {len(shape_) - 1 : [position, position + window.shape[axisToMove]]}))
            numObsvConv = np.sum(window * getSlicesV2(groundTruthCounterTensor, {len(shape_) - 1 : [position, position + window.shape[axisToMove]]}))
            sumNMAE = sumNMAE + abs(groundTruthConv - recoveredConv)
            sumNumObsv = sumNumObsv + numObsvConv
            if rMax < groundTruthConv:
                rMax = groundTruthConv
            if rMin > groundTruthConv:
                rMin = groundTruthConv
            position = position + stride[axisToMove]
        return sumNMAE, sumNumObsv, rMax, rMin
    else: # Recursive case
        sumNMAE = 0.
        sumNumObsv = 0.
        position = 0
        while(position + window.shape[axisToMove] <= shape_[axisToMove]):            
            partialNMAE, partialNumObsv, partial_rMax, partial_rMin = getNMAE_window_rec(getSlicesV2(groundTruthTensor, {axisToMove:[position, position + window.shape[axisToMove]]}), 
            getSlicesV2(groundTruthCounterTensor, {axisToMove:[position, position + window.shape[axisToMove]]}), 
            getSlicesV2(recoveredTensor, {axisToMove:[position, position + window.shape[axisToMove]]}), window, stride)

            sumNMAE = sumNMAE + partialNMAE
            sumNumObsv = sumNumObsv + partialNumObsv

            if rMax < partial_rMax:
                rMax = partial_rMax
            if rMin > partial_rMin:
                rMin = partial_rMin

            position = position + stride[axisToMove]
        return sumNMAE, sumNumObsv, rMax, rMin

def getNMAE_window(groundTruthTensor, recoveredTensor, groundTruthCounterTensor, window, stride = None):
    """Get window-wise NMAE.

    Works on any dimensional Numpy array.
    
    Parameters
    ----------
    groundTruthTensor : Numpy array
        array of ground truth values
    groundTruthCounterTensor : Numpy array
        array of the number of observations
    recoveredTensor : Numpy array
        array of the imputed values
    window : Numpy array
        window which will be applied on the values of data as convolution
    stride : list or tuple
        step sizes for each axis
    
    Returns
    -------
    NMAE : float
        window-wise NMAE
    """
    if type(window) == type(()) or type(window) == type([]):
        window_ = np.array(window)
    else:
        window_ = window 
    shape_ = groundTruthTensor.shape
    assert(len(groundTruthCounterTensor.shape) == len(shape_))
    assert(len(recoveredTensor.shape) == len(shape_))
    assert(len(window_.shape) == len(shape_))
    if stride is None:
        stride = []
        for axis in range(len(shape_)):
            stride.append(1)
    assert(len(stride) == len(shape_))

    sumNMAE, sumNumObsv, rMax, rMin = getNMAE_window_rec(groundTruthTensor, groundTruthCounterTensor, recoveredTensor, window_, stride)
    return sumNMAE / ((sumNumObsv + 1e-8) * max((rMax - rMin), 1e-8))

# test_groundTruthTensor = np.random.rand(300, 300, 300, 9) * 10
# test_groundTruthCounterTensor = np.random.rand(300, 300, 300, 9) * 2
# test_recoveredTensor = np.random.rand(300, 300, 300, 9) * 10
# window = np.ones((2, 2, 2, 9))
# stride = [1, 1, 1, 1]

# print(getNMAE_window(test_groundTruthTensor, test_groundTruthCounterTensor, test_recoveredTensor, window, stride))

# ref: https://stackoverflow.com/questions/2651874/embed-bash-in-python
def run_script(script, stdin=None):
    """Returns (stdout, stderr), raises error on non-zero return code"""
    import subprocess
    # Note: by using a list here (['bash', ...]) you avoid quoting issues, as the 
    # arguments are passed in exactly this order (spaces, quotes, and newlines won't
    # cause problems):
    proc = subprocess.Popen(['bash', '-c', script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise ScriptException(proc.returncode, stdout, stderr, script)
    return stdout, stderr

class ScriptException(Exception):
    def __init__(self, returncode, stdout, stderr, script):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        Exception.__init__('Error in script')

def removeDirsInLst(dirLst):
    for dir in dirLst:
        if os.path.isdir(dir):
            shutil.rmtree(dir)
        elif os.path.isfile(dir):
            os.remove(dir)
        else:
            print('WARNING: {} not exists, removing passed'.format(dir))

def gifToArr(gifPath):
    with Image.open(gifPath) as img:
        return np.array([np.array(frame.copy().convert('RGB').getdata(),dtype=np.float32).reshape(frame.size[1],frame.size[0],3) for frame in ImageSequence.Iterator(img)])

# def cutArrayToMinMax(arr, min = None, max = None):
#     """Cut given array with given ceil and floor
    
#     This is not scaling but cutting.
#     """

#     if min is not None and max is not None:
#         assert(max >= min)
#         return np.where(arr > max, max, np.where(arr < min, min, arr))
#     elif min is None and max is not None:
#         return np.where(arr > max, max, arr)
#     elif min is not None and max is None:
#         return np.where(arr < min, min, arr)
#     else:
#         return arr

# def saveGifFromArr(arr, path, fps = 5, vmin = None, vmax = None, axis = 0):
#     """Save 4D Numpy array into gif.
    
#     Last dimension(shape[3]) is regarded as RGB channel.
#     """
    
#     assert(fps >= 1 and axis in [0, 1, 2])
#     arrFormatted = cutArrayToMinMax(arr, min = vmin, max = vmax)
#     if len(arr.shape) == 4:
#         assert(arr.shape[3] == 3)
#         arrFormatted = min_max_scale(arr, 0, 255).astype(np.uint8)
#     elif len(arr.shape) == 3:
#         arrFormatted = visualization.convert_3Darray_to_4DarrayRGB(arr)
#     if vmin is not None:
#         arrFormatted_mask = np.where(arrFormatted >= vmin, 1., 0.)
#     else:
#         arrFormatted_mask = np.where(arrFormatted >= 1e-8, 1., 0.)
#     clip_mask = mpy.VideoClip(make_frame=lambda  t: getSlicesV2(arrFormatted_mask, {axis: int(t)}), duration=arr.shape[0], ismask=True)
#     # clip = mpy.VideoClip(lambda t: arrFormatted[int(t), :, :, :], duration=(arr.shape[0]) / float(fps))
#     clip = mpy.VideoClip(lambda t: getSlicesV2(arrFormatted, {axis: int(t)}), duration = arr.shape[0])
#     clip.set_mask(clip_mask)
#     clip.speedx(fps).write_gif(path, fps = fps)

# def convert_3Darray_to_4DarrayRGB(arr_3D, vmin = None, vmax = None, cmap = plt.get_cmap('jet')):
#     if vmin is not None and vmax is not None:
#         assert(vmax >= vmin)
#     if vmin is None and vmax is None:
#         arr_3D_copied = np.copy(arr_3D)
#     else:
#         arr_3D_copied = cutArrayToMinMax(arr_3D, min = vmin, max = vmax)
#     arr_3D_copied = min_max_scale(arr_3D_copied)
#     arr_4D = cmap(arr_3D_copied, bytes = True)
#     arr_4D = np.delete(arr_4D, 3, 3)
#     return arr_4D

def get_proportional_ranked_value(amount_arr, counter_arr = None, proportion = 0.1):
    assert(proportion >= 0. and proportion <= 1.)
    if counter_arr is not None:
        num_of_nonzero = np.count_nonzero(np.where(counter_arr >= 1e-8, 1., 0.))
    else:
        num_of_nonzero = np.count_nonzero(np.where(amount_arr >= 1e-8, 1., 0.))
    rank = int(proportion * num_of_nonzero)
    return np.partition(amount_arr.flatten(), -1 * rank)[-1 * rank]

def collect_idx_of_dense_slices_along_axis(arr, axis, get_dense_slice_threshold = 0, nonzero_threshold = 1e-8):
    """Get list of indices of dense(the number of nonzero entries are larger than get_dense_slice_threshold) slices along axis
    
    """

    nonzero_idx_list = []
    if get_dense_slice_threshold > 0:
        for idx in range(arr.shape[axis]):
            sliced = get_slices_with_idc(arr, {axis:[idx]})
            if np.count_nonzero(np.where(sliced > nonzero_threshold, 1., 0.)) >= get_dense_slice_threshold:
                nonzero_idx_list.append(idx)
        return nonzero_idx_list
    else:
        return list(range(arr.shape[axis]))
# def tfAssertionAll(conditionTensor):
#     tf.debugging.Assert(tf.reduce_all(conditionTensor))

# def tfAllClose(aTensor, bTensor, rtol = 1e-5, atol = 1e-8):
#     return tf.reduce_all(tf.abs(aTensor - bTensor) <= tf.abs(bTensor) * rtol + atol)

def evaluate_mineral_correlation(geoTensor):
    xcoord, ycoord, zcoord = np.nonzero(geoTensor.counterTensorDict['au'])
    auC = geoTensor.counterTensorDict['au']
    agC = geoTensor.counterTensorDict['ag']
    pbC = geoTensor.counterTensorDict['pb']
    cuC = geoTensor.counterTensorDict['cu']
    znC = geoTensor.counterTensorDict['zn']
    asC = geoTensor.counterTensorDict['as']
    hgC = geoTensor.counterTensorDict['hg']
    sbC = geoTensor.counterTensorDict['sb']
    AU_vals = []
    AG_vals = []
    PB_vals = []
    CU_vals = []
    ZN_vals = []
    AS_vals = []
    HG_vals = []
    SB_vals = []

    for x,y,z in zip(xcoord,ycoord,zcoord):
        if auC[x,y,z] > 0 and agC[x,y,z] > 0 and pbC[x,y,z] > 0 and cuC[x,y,z] > 0 and znC[x,y,z] > 0 and asC[x,y,z] > 0 and hgC[x,y,z] > 0 and sbC[x,y,z] > 0:
            AU_vals.append(geoTensor.dataTensorDict['au'][x][y][z])
            AG_vals.append(geoTensor.dataTensorDict['ag'][x][y][z])
            PB_vals.append(geoTensor.dataTensorDict['pb'][x][y][z])
            CU_vals.append(geoTensor.dataTensorDict['cu'][x][y][z])
            ZN_vals.append(geoTensor.dataTensorDict['zn'][x][y][z])
            AS_vals.append(geoTensor.dataTensorDict['as'][x][y][z])
            HG_vals.append(geoTensor.dataTensorDict['hg'][x][y][z])
            SB_vals.append(geoTensor.dataTensorDict['sb'][x][y][z])

    AU_vals = np.asarray(AU_vals)
    AG_vals = np.asarray(AG_vals)
    PB_vals = np.asarray(PB_vals)
    CU_vals = np.asarray(CU_vals)
    ZN_vals = np.asarray(ZN_vals)
    AS_vals = np.asarray(AS_vals)
    HG_vals = np.asarray(HG_vals)
    SB_vals = np.asarray(SB_vals)
    
    # calculate covariance
    X = np.stack((AU_vals, AG_vals, PB_vals, CU_vals, ZN_vals, AS_vals, HG_vals, SB_vals), axis=0)
    corr = np.corrcoef(X)
    print(corr.shape)
    np.savetxt("mineral_correlation.csv", corr, delimiter=",")
def categoricalTensor(dataTensorDict, counterTensorDict, numberOfClasses, get_avg = False):
    for key in dataTensorDict.keys():
        assert(len(dataTensorDict[key]) == len(counterTensorDict[key]))
        shape_ = dataTensorDict[key].shape
    shape_ = list(shape_)
    shape_.append(numberOfClasses)
    shape_ = tuple(shape_) # convert 3D shape to 4D shoape with number of classes
    mineralLst = dataTensorDict.keys()
    categoricalTensorDict = {}
    range_list_dict = {}
    avg_list_dict = {}
    for mineral in mineralLst:
        range_list_dict[mineral] = [0] * numberOfClasses
        avg_list_dict[mineral] = [0] * numberOfClasses
    for mineral in mineralLst:
        categoricalTensorDict[mineral] = np.zeros(shape_)
    amountsDict = {}
    for mineral in mineralLst:
        amountsDict[mineral] = np.sort(dataTensorDict[mineral][counterTensorDict[mineral].nonzero()])
        for classIdx in range(numberOfClasses):
            range_list_dict[mineral][classIdx] = amountsDict[mineral][int(amountsDict[mineral].shape[0] / numberOfClasses) * classIdx]
            avg_list_dict[mineral][classIdx] = amountsDict[mineral][int(amountsDict[mineral].shape[0] / numberOfClasses) * classIdx + int(amountsDict[mineral].shape[0] * 0.5 / numberOfClasses)]
    # 0.0 ~ 0.7 -> class 0, X , 0.7 ~ 3.2 -> class 0
    #
    for mineral in dataTensorDict.keys():
        nonzerosIndices = counterTensorDict[mineral].nonzero()
        for positionIdx in range(nonzerosIndices[0].shape[0]):
            for category in range(numberOfClasses):
                if category == len(range_list_dict[mineral]) - 1 or dataTensorDict[mineral][nonzerosIndices[0][positionIdx], nonzerosIndices[1][positionIdx], nonzerosIndices[2][positionIdx]] < range_list_dict[mineral][category + 1]:
                    categoricalTensorDict[mineral][nonzerosIndices[0][positionIdx], nonzerosIndices[1][positionIdx], nonzerosIndices[2][positionIdx], category] = 1
                    break
    if get_avg:
        return categoricalTensorDict, avg_list_dict
    else:                        
        return categoricalTensorDict

def get_dummy_arr_list(shape_, functions_list, relation_matrix, origin = 'center', noise = {'mean':0., 'std': 0.1}, min_max_scaled = True):
    """
    
    Parameters
    ----------
    origin : str or length-3 list
        Substract origin from input index of function.
    relation_matrix : 2-D array
        i-th row and j-th column indicates the weights or influence of j-th array on i-th array.
    
    """
    if origin == 'center':
        translate_substraction_ijk = [shape_[0] // 2, shape_[1] // 2, shape_[2] // 2,]
    else:
        translate_substraction_ijk = deepcopy(origin)

    assert(len(functions_list) == relation_matrix.shape[0])
    assert(len(relation_matrix.shape) == 2 and relation_matrix.shape[0] == relation_matrix.shape[1])
    result_arr_list_before_relation = []
    for function, idx in zip(functions_list, range(len(functions_list))):
        arr_of_function = np.empty(shape_)
        for i in range(shape_[0]):
            for j in range(shape_[1]):
                for k in range(shape_[2]):
                    arr_of_function[i, j, k] = function(i - translate_substraction_ijk[0], j - translate_substraction_ijk[1], k - translate_substraction_ijk[2])
        arr_of_function = arr_of_function + np.random.normal(loc = noise['mean'], scale = noise['std'], size = shape_)
        result_arr_list_before_relation.append(arr_of_function)
    result_arr_list_after_relation = []
    for arr_before_relation, row_idx in zip(result_arr_list_before_relation, range(len(result_arr_list_before_relation))):
        array_after_relation = np.copy(arr_before_relation)
        for col_idx in range(relation_matrix.shape[1]):
            if col_idx != row_idx:
                array_after_relation += relation_matrix[row_idx, col_idx] * result_arr_list_before_relation[col_idx]
        result_arr_list_after_relation.append(min_max_scale(array_after_relation))
    return result_arr_list_after_relation

# def sample_z_uniform(min, max, shape):
#     # assert(len(shape) >= 1)
#     return np.random.uniform(min, max, size = shape).astype(np.float32)

# def sample_z_normal(mean, std, shape):
#     assert(len(shape) >= 1)
#     return np.random.normal(loc = mean, scale = std, size = shape).astype(np.float32)

# def mask_prob(shape, p):
#     A = np.random.uniform(0., 1., size = shape)
#     B = A < p
#     return (1. * B).astype(np.float32)

def get_y_x_dictionary(x, y, num_samples_each_category = 1000):
    assert(x.shape[0] == y.shape[0])
    y_x_dict = {}
    for idx in range(x.shape[0]):
        if y[idx] in y_x_dict.keys():
            if len(y_x_dict[y[idx]]) < num_samples_each_category:
                y_x_dict[y[idx]].append(x[idx])
        else:
            y_x_dict[y[idx]] = []
    for y_ in y_x_dict.keys():
        if len(y_x_dict[y_]) < num_samples_each_category:
            raise Exception(f"The number of samples for category {y_x_dict[y_]} = {len(y_x_dict[y_])} is not enough for required {num_samples_each_category}")
        else:
            y_x_dict[y_] = np.stack(y_x_dict[y_], axis = 0).astype(np.float32)
    return y_x_dict

def random_pick_items(item_length_dict, pick_keep_probs_dict, keep_non_prob_item= True):
    """

    Examples
    --------
    item_length_dict = {0: 3, 1: 4, 4: 2}\n
    pick_keep_probs_dict = {-1: 0.3, 0: 0.5}\n
    keep_non_prob_item = True\n
    print(random_pick_items(item_length_dict, pick_keep_probs_dict, keep_non_prob_item))\n
    {0: [2, 1], 1: [1, 2], 4: [1]}\n
    """

    picked_idc_dict = {}
    for idx_0 in item_length_dict.keys():
        picked_idc_dict[idx_0] = []
        done_idx= []
        for key, prob in pick_keep_probs_dict.items():
            key_positive = key if key>= 0 else item_length_dict[idx_0] + key
            if key_positive in done_idx:
                raise Exception(f"Ambiguous pick with index: {key}")
            done_idx.append(key_positive)
            rand_num= random()
            if rand_num<= prob:
                picked_idc_dict[idx_0].append(key_positive)
        if keep_non_prob_item:
            for idx_1 in range(item_length_dict[idx_0]):
                if idx_1 not in done_idx and idx_1 not in picked_idc_dict[idx_0]:
                    picked_idc_dict[idx_0].append(idx_1)
    return picked_idc_dict

def random_pick_items(item_length_dict, pick_keep_probs_dict, keep_non_prob_item= True):
    """

    Examples
    --------
    random_pick_items(item_length_dict= 0: 3, 1: 4, 4: 2}, pick_keep_probs_dict= {-1: 0.3, 0: 0.5}, keep_non_prob_item= True)
        Returns
        {0: [2, 1], 1: [1, 2], 4: [1]}
    """

    picked_idc_dict = {}
    for idx_0 in item_length_dict.keys():
        picked_idc_dict[idx_0] = []
        done_idx= []
        for key, prob in pick_keep_probs_dict.items():
            key_positive = key if key>= 0 else item_length_dict[idx_0] + key
            if key_positive in done_idx:
                raise Exception(f"Ambiguous pick with index: {key}")
            done_idx.append(key_positive)
            rand_num= random()
            if rand_num<= prob:
                picked_idc_dict[idx_0].append(key_positive)
        if keep_non_prob_item:
            for idx_1 in range(item_length_dict[idx_0]):
                if idx_1 not in done_idx and idx_1 not in picked_idc_dict[idx_0]:
                    picked_idc_dict[idx_0].append(idx_1)
    return picked_idc_dict

def delete_items_from_list_with_indices(list_to_filter, indices, keep_not_remove = False):
    base_idx = 0
    sorted_indices = sorted(indices, reverse = False)
    sorted_indices_to_remove = []
    if keep_not_remove:
        for i in range(len(list_to_filter)):
            if i not in sorted_indices:
                sorted_indices_to_remove.append(i)
    else:
        sorted_indices_to_remove = deepcopy(sorted_indices)
    list_to_filter_copied = deepcopy(list_to_filter)
    for idx in sorted_indices_to_remove:
        del list_to_filter_copied[idx - base_idx]
        base_idx += 1
    return list_to_filter_copied

def cutArrayToMinMax(arr, min = None, max = None):
    """Cut given array with given ceil and floor
    
    This is not scaling but cutting.
    """

    if min is not None and max is not None:
        assert(max >= min)
        return np.where(arr > max, max, np.where(arr < min, min, arr))
    elif min is None and max is not None:
        return np.where(arr > max, max, arr)
    elif min is not None and max is None:
        return np.where(arr < min, min, arr)
    else:
        return arr
        
# @decorators.redirect_function(decorators)
# def test_func(number):
#     return number + 1

if __name__ == '__main__':
    pass