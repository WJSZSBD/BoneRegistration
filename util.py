# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:37:22 2021

@author: lenovo
"""

import SimpleITK as sitk
import numpy as np
from astra import data3d, create_sino3d_gpu
import pylab as plt


def see(data):
    plt.imshow(data,cmap='gray')
    plt.show()


def field_correction(data, struct):
    dark = np.squeeze(struct['datanode']['dark'])
    light = np.squeeze(struct['datanode']['light'])
    ma_ratio = struct['datanode']['ma_ratio']

    divisor = ma_ratio * (light - dark)
    divisor[divisor < ma_ratio] = ma_ratio

    if not data.shape == dark.shape:
        data = resample(data, dark.shape)

    data -= dark
    data /= divisor

    return data


def cal_v(xml, prj_order):
    SD = xml['datanode']['SDD'][0, prj_order, 0]
    sysm = xml['datanode']['SysM'][0, prj_order, 0]
    SO = SD / sysm

    py = xml['datanode']['pY'][0, prj_order, 0]
    detz = xml['datanode']['DetZ'][0, prj_order, 0]

    prj_pixel_size = xml['datanode']['prj_pixel_size']
    theta = xml['datanode']['angles'][0, prj_order, 0]

    # Source (x,y,z) unit: mm
    s = np.zeros(3, dtype=np.float32)
    s[0] = SO * np.sin(theta)
    s[1] = -SO * np.cos(theta)
    s[2] = (py - detz) / sysm * prj_pixel_size
    # s[2] = 0

    return -s / np.sqrt(np.sum(s * s))


def resample_display(image, t, theta, center=None, interpolator=sitk.sitkLinear):
    simg = sitk.GetImageFromArray(image)
    if not center:
        center = simg.TransformContinuousIndexToPhysicalPoint(np.array(simg.GetSize()) / 2.0)
        # print(center)

    euler3d = sitk.Euler3DTransform(center, *theta, t)

    resampled_image = sitk.Resample(simg, euler3d, interpolator=interpolator)
    return resampled_image


def projection(img, proj_geom, vol_geom):
    # img = img[:,::-1,:]
    sino_id, proj = create_sino3d_gpu(img, proj_geom, vol_geom)
    data3d.delete(sino_id)
    return np.squeeze(proj)


def NCC(im1, im2, eps=1e-8):
    var1 = im1 - im1.mean()
    var2 = im2 - im2.mean()
    ncc = np.sum(var1 * var2) / (np.sqrt(np.sum(var1 ** 2)) * np.sqrt(np.sum(var2 ** 2)) + eps)
    return ncc


def cal_theta(dx1, dy1, g1, dx2, dy2, g2, undef=0):
    cos_theta = (dy1 * dy2 + dx1 * dx2) / (g1 * g2)

    cos_theta[np.logical_and(g1 == 0, g2 == 0)] = 1
    # cos_theta[np.logical_and(g1==0, g2!=0)] = 0.2
    cos_theta[np.isnan(cos_theta)] = undef
    cos_theta[cos_theta > 1] = 1
    cos_theta[cos_theta < -1] = -1

    theta = np.arccos(cos_theta)

    return theta


def GA(theta, v=1032 * 1032 * 2 * np.pi, mask=None, weight=1):
    if mask is not None:
        theta = theta * (mask * weight + 1)
    ga = np.sum(theta) / v
    ga = 1 - np.exp(-ga)

    return ga


def PI(im1, im2, sigma=10):
    im1 *= im2.max() / im1.max()
    dif = im1 - im2
    ss = np.zeros(dif.shape, dtype=np.float32)
    s2 = sigma ** 2
    h, w = dif.shape
    for i in range(-1, 2):
        for j in range(-1, 2):
            ss[1:-1, 1:-1] += s2 / (s2 + (dif[1:-1, 1:-1] - dif[1 + i:h - 1 + i, 1 + j:w - 1 + j]) ** 2)
    return ss.sum() / 9


def fit_transform(ts, init_t=[0] * 6, deg=1):
    if ts is None or len(ts) == 0:
        return init_t
    elif len(ts) == 1:
        return ts[0]
    else:
        estimate_t = np.zeros((6,))
        for i in range(6):
            estimate_t[i] = fit_sim(ts[:, i], deg)

        return estimate_t


def fit_sim(ss, deg=1):
    if ss is None or len(ss) < 1:
        return np.inf
    else:
        l = len(ss)
        # x = list(range(min(l, deg)))
        # order = x[-1]
        # coeff = np.polyfit(x, ss[-order-1:], order)
        coeff = np.polyfit(list(range(l)), ss, deg)

        f = np.poly1d(coeff)
        return f(l)
