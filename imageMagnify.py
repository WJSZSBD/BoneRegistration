#!/home/wangjiashun/anaconda3 python
# -*- coding: UTF-8 -*-
"""
@Project : BoneRegistration 
@File : imageMagnify.py
@Author : wangjiashun    
@Date : 2022/12/6 21:51 
@Brief : Magnify image.
"""

import numpy as np
import nibabel as nib
import scipy
from scipy import interpolate
from util import see


def imageMagnify(image, ratio):
    # see(image)
    reso_in = len(image)
    reso_out = int(reso_in * ratio)
    kernelOut = np.zeros((reso_out), np.uint8)

    x = np.array(range(reso_in))
    y = np.array(range(reso_in))
    z = image

    xx = np.linspace(x.min(), x.max(), reso_out)
    yy = np.linspace(y.min(), y.max(), reso_out)

    newKernel = interpolate.RectBivariateSpline(x, y, z)

    kernelOut = newKernel(xx, yy)
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = kernelOut[i + int((reso_out - reso_in) / 2), j + int((reso_out - reso_in) / 4)]

    # see(result)
    return result


if __name__ == '__main__':
    kernelIn = np.array([
        [0, -2, 0],
        [-2, 11, -2],
        [0, -2, 0]])

    outtest = imageMagnify(kernelIn, 2.0)
