# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:24:45 2022

@author: lenovo
"""

import os
import time
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import pickle

from matplotlib import pyplot as plt
from numpy import cos, sin
from parse_geom import geometry
from scipy.optimize import minimize
from scipy.spatial import KDTree
from skimage.morphology import dilation, cube
from skimage.transform import rescale
from fit import draw_edge, fit
from load_xml import parse_xml
from astra import create_vol_geom, create_proj_geom
from util import projection, NCC, fit_transform, resample_display, cal_v, field_correction, fit_sim, see

best = {'x': [], 'cost': np.inf}


class Registration:
    def __init__(self, bones_3d, fixed_3d, proj_geom, vol_geom):
        self.bones_3d = bones_3d
        self.fixed_3d = fixed_3d
        self.proj_geom = proj_geom
        self.vol_geom = vol_geom


# 根据变换参数生成投影几何
def transform_projgeom(pg, t, rotate_m):
    v = pg['Vectors'].copy().reshape((4, 3)).T

    v = np.dot(rotate_m, v)
    # translation
    t = t[::-1]
    for i in range(3):
        v[i, :2] -= t[i]

    proj_geom = create_proj_geom('cone_vec', h, w, v.T.reshape((1, 12)))
    return proj_geom


# 根据旋转角度生成旋转变换矩阵
def get_rotate_matrix(r):
    # ZYX/Rx*Ry*Rz
    # m = np.array([[cos(r[1])*cos(r[2]), -cos(r[1])*sin(r[2]), sin(r[1])],
    #           [sin(r[0])*sin(r[1])*cos(r[2])+cos(r[0])*sin(r[2]),-sin(r[0])*sin(r[1])*sin(r[2])+cos(r[0])*cos(r[2]),-sin(r[0])*cos(r[1])],
    #           [-cos(r[0])*sin(r[1])*cos(r[2])+sin(r[0])*sin(r[2]),cos(r[0])*sin(r[1])*sin(r[2])+sin(r[0])*cos(r[2]),cos(r[0])*cos(r[1])]])

    # XYZ/Rz*Ry*Rx
    m = np.array([[cos(r[1]) * cos(r[2]), sin(r[0]) * sin(r[1]) * cos(r[2]) - cos(r[0]) * sin(r[2]),
                   cos(r[0]) * sin(r[1]) * cos(r[2]) + sin(r[0]) * sin(r[2])],
                  [cos(r[1]) * sin(r[2]), sin(r[0]) * sin(r[1]) * sin(r[2]) + cos(r[0]) * cos(r[2]),
                   cos(r[0]) * sin(r[1]) * sin(r[2]) - sin(r[0]) * cos(r[2])],
                  [-sin(r[1]), sin(r[0]) * cos(r[1]), cos(r[0]) * cos(r[1])]])

    return m


# 优化目标函数1    
def f1(x, moving_3d, fixed, proj_geom, vol_geom):
    moving_3d_sitk = resample_display(moving_3d, x[:3], x[3:])
    moving_3d_t = sitk.GetArrayFromImage(moving_3d_sitk)
    moving = projection(moving_3d_t, proj_geom, vol_geom)

    # 仅用于查看
    # for view in range(300):
    #     moving[:, view, :].tofile('H:/Views/%d_float32_1024x1024.raw' % view)

    # NCC
    # ncc = NCC(g1, g2)
    ncc = NCC(moving, fixed)
    ncc = abs(ncc)

    # cost = ga*(1-ncc)
    cost = 1 - ncc

    if best['cost'] > cost:
        best['cost'] = cost
        best['x'] = x

    return cost


# 优化目标函数2
def f2(x, moving_3d, m, fixed, proj_geom, vol_geom, kdt, xml, order, v, t_mean, g):
    r = x[3:]
    rotate_m = get_rotate_matrix(r)

    new_v = np.dot(v, rotate_m)
    edge_error = fit(g, fixed, kdt, xml, order, new_v, nu=1, t=x[:3], rotate_m=rotate_m)

    if np.isnan(t_mean[0]):
        move_cost = 0
    else:
        diff = abs((x - t_mean) / 100)
        diff[3:] /= 0.17
        move_cost = sum(diff)

    cost = edge_error + move_cost

    # print(x, cost)

    if f1_best['cost'] > cost:
        f1_best['cost'] = cost
        f1_best['x'] = x

    return cost


# 计算降采样的mask和volume
def cal_vols(vol, mask, scale=[0.25, 0.5, 1], sigma=[0.4, 0.1, 0]):
    s_num, b_num = len(scale), mask.max()
    vols = [[0] * s_num for _ in range(b_num)]

    for i in range(b_num):
        tmp = (mask == (i + 1)).astype(np.uint8) * vol
        # tmp = erosion(tmp, cube(2))
        for j in range(s_num):
            if scale[j] == 1:
                vols[i][j] = tmp
            else:
                vols[i][j] = rescale(tmp, scale[j], clip=False)
                # vols[i][j] = filters.gaussian(vols[i][j], sigma[j])

    return vols, [rescale(vol, s, clip=False, anti_aliasing=True) for s in scale]


# 对投影raw图进行预处理
def preprocess_proj(proj_path, shape, xml):
    fixed = np.fromfile(proj_path, dtype=np.uint16).reshape(shape).astype(np.float32)
    fixed = field_correction(fixed, xml)
    fixed[fixed > 1] = 1
    fixed[fixed < 1e-5] = 1e-5
    fixed = -np.log(fixed.T)
    fixed[fixed < 1e-2] = 0

    # fixed = filters.median(fixed, disk(3))
    return fixed


# 校正投影方向
def prj_orientation(prj, ori):
    prj_new = prj.copy()

    if prj.ndim == 3:
        for i in range(prj.shape[0]):
            prj_new[i] = prj_orientation(prj[i], ori)
    elif prj.ndim == 2:
        if 1 == ori:
            pass
        elif 2 == ori:
            prj_new = prj[:, ::-1]
        elif 3 == ori:
            prj_new = prj.T[::-1]
        elif 4 == ori:
            prj_new = prj.T[::-1, ::-1]
        elif 5 == ori:
            prj_new = prj[::-1, ::-1]
        elif 6 == ori:
            prj_new = prj[::-1]
        elif 7 == ori:
            prj_new = prj.T[:, ::-1]
        elif 8 == ori:
            prj_new = prj.T
        else:
            raise ValueError("The orientation should be 1-8")
    else:
        raise ValueError("The dimension of input should be 2 or 3")

    return prj_new


# 显示模拟投影与实际投影
def show(fixed, moving, warp_moving, edge, sm, i=0, j=0, save=True):
    plt.subplot(231)
    plt.imshow(moving, cmap='gray')
    plt.subplot(232)
    plt.imshow(fixed - moving, cmap='gray', vmax=2.6)
    plt.subplot(233)
    plt.imshow(fixed, cmap='gray', vmin=1.3, vmax=2.8)
    plt.subplot(234)
    plt.imshow(warp_moving, cmap='gray')
    plt.subplot(235)
    plt.imshow(fixed - warp_moving, cmap='gray', vmax=2.6)
    plt.subplot(236)
    plt.imshow(edge, cmap='gray', vmin=1.3, vmax=2.8)
    if save:
        plt.savefig('E:/CBCT/BoneRegistration/data/%d/%d/%d.png' % (sm, j, i))
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # read xml & parse xml
    recon_path = 'H:/CBCT/BoneRegistration/data/recon.nii.gz'
    mask_path = 'H:/CBCT/BoneRegistration/data/mask.nii.gz'
    DRs_path = 'H:/CBCT/BoneRegistration/data/2D_DRs.nii.gz'
    xml_path = 'H:/CBCT/BoneRegistration/config/recon.xml'
    xml = parse_xml(xml_path)

    # cal vol_geom & proj_geom
    v, vol_geom = geometry(xml, False)
    h, w = xml['datanode']['prj_height'], xml['datanode']['prj_width']

    # 选择初始投影角度
    order = 170  # order = 170 为正面
    proj_geom = create_proj_geom('cone_vec', h, w, v[order:order + 1])

    # proj_geom = create_proj_geom('cone_vec', h, w, v)

    # read 3D CT & mask
    vol_size = np.array([xml['datanode']['vol_size'], xml['datanode']['vol_size'], xml['datanode']['vol_height']])
    # moving_3d = np.fromfile('H:/CBCT/BoneRegistration/data/3D.raw', dtype=np.uint16).reshape(vol_size).astype(
    #     np.float32)[::-1, :, ::-1]
    moving_3d_pre = nib.load(recon_path)
    moving_3d_pre = moving_3d_pre.get_fdata()
    moving_3d_pre = moving_3d_pre.transpose(1, 0, 2)
    moving_3d = moving_3d_pre.copy()
    for i in range(512):
        moving_3d[:, i, :] = moving_3d_pre[:, i, ::-1].T
    moving_3d = (moving_3d / xml['datanode']['linearHU0']).astype(np.float32)
    moving_3d = moving_3d.transpose(0, 2, 1)
    moving_3d = moving_3d[:, :, ::-1]
    moving_3d = moving_3d[::-1, :, :]

    # 设置运动序列数量
    prj_num = 47
    # mask = np.fromfile('H:/CBCT/BoneRegistration/data/mask.raw', dtype=np.uint8).reshape(vol_size)[::-1, :, ::-1]
    mask_pre = nib.load(mask_path)
    mask_pre = mask_pre.get_fdata()
    mask_pre = mask_pre.transpose(1, 0, 2)
    mask = mask_pre.copy()
    for i in range(512):
        mask[:, i, :] = mask_pre[:, i, ::-1].T
    mask = mask.transpose(0, 2, 1)
    mask = mask[:, :, ::-1]
    mask = mask[::-1, :, :]

    mask = dilation(mask, cube(2))

    # read 2D projections
    fixed_3d = nib.load(DRs_path).get_fdata().astype(np.float32)
    fixed_3d = fixed_3d.transpose(2, 1, 0)
    # fixed_3d = fixed_3d[:, ::-1, :]

    # for i in range(prj_num):
    #     # prj_1x1 = preprocess_proj('E:/CBCT/BoneRegistration/data/2D/ProcessedImagex%06d_3072X3072.raw'%i, (3*h,3*w), xml)
    #     # fixed_3d[i] = np.lib.stride_tricks.as_strided(prj_1x1, shape=(1024,1024,3,3), strides=(36864,12,12288,4)).sum((2,3))/9
    #     # fixed_3d[i].tofile('E:/CBCT/BoneRegistration/data/2D/ProcessedImagex%06d_1024X1024.raw'%i)
    #
    #     fixed_3d[i] = preprocess_proj('H:/CBCT/BoneRegistration/data/2D/ProcessedImagex%06d_1024X1024.raw' % (i + 1),
    #                                   (h, w),
    #                                   xml)

    # fixed_3d = prj_orientation(fixed_3d, xml['datanode']['prj_orientation'])
    # fixed_3d = np.fromfile('E:/CBCT/BoneRegistration/data/proc_2D.raw', dtype=np.float32).reshape((prj_num,h,w))/9

    fixed = fixed_3d[0]
    ###################################################
    # 初始化整体的变换参数
    outTx = [0] * 6

    bones_3d_pickle = []
    # bone_num = int(np.max(mask))
    bone_num = 3
    bones_3d_pickle.append(moving_3d * (mask == 1))
    bones_3d_pickle.append(moving_3d * (mask == 2))
    bones_3d_pickle.append(moving_3d * (mask > 2))
    fixed_3d_pickle = fixed_3d / 500
    ideal_proj_v = np.zeros((1, 12), dtype='float64')  # ideal front angle
    # src
    ideal_proj_v[0, 0] = 617.0
    ideal_proj_v[0, 1] = 0.0
    ideal_proj_v[0, 2] = 0.0
    # det
    ideal_proj_v[0, 3] = 617.0 - 1140.0
    ideal_proj_v[0, 4] = 0.0
    ideal_proj_v[0, 5] = 0.0
    # vector from detector pixel(0, 0) to(0, 1)
    ideal_proj_v[0, 6] = 0.0
    ideal_proj_v[0, 7] = 0.417
    ideal_proj_v[0, 8] = 0.0
    # vector from detector pixel(0, 0) to(1, 0)
    ideal_proj_v[0, 9] = 0.0
    ideal_proj_v[0, 10] = 0.0
    ideal_proj_v[0, 11] = -0.417

    proj_geom_pickle = create_proj_geom('cone_vec', xml['datanode']['prj_height'], xml['datanode']['prj_width'],
                                        ideal_proj_v)
    vol_geom_pickle = vol_geom.copy()
    Reg_ideal = Registration(bones_3d_pickle, fixed_3d_pickle, proj_geom_pickle, vol_geom_pickle)
    with open('H:/CBCT/BoneRegistration/data/pickle/IdealFront.pickle', 'wb') as f_pickle:
        f_pickle.write(pickle.dumps(Reg_ideal))

    exit(-1)  # turn to `registration.py`
    res = minimize(f1, outTx, args=(moving_3d, fixed, proj_geom, vol_geom), method='Powell', \
                   bounds=((-30, 0), (25, 125), (-30, 0), (-0.7, 0.0), (-0.7, 0), (0, 0.7)), \
                   options={'xtol': 1e-2, 'ftol': 1e-2, 'maxiter': 6, 'disp': True})

    t, theta = res.x[:3], res.x[3:]
    ###################################################
    bone_num = mask.max()

    # downsample vol
    vols, ms = cal_vols(moving_3d, mask, scale=[1])
    del moving_3d, mask

    # 初始化每一帧的变换参数
    init_t = [[0] * 6 for _ in range(bone_num)]

    if os.path.isfile('H:/CBCT/BoneRegistration/data/trans.raw'):
        trans = np.fromfile('H:/CBCT/BoneRegistration/data/trans.raw', dtype=np.float64).reshape((prj_num, bone_num, 6))
    else:
        trans = np.array([init_t for _ in range(prj_num)], dtype=np.float64)
    ss = np.ones((prj_num, bone_num), dtype=np.float64)
    # ss = np.fromfile('temp_ss.raw', dtype=np.float64).reshape((prj_num, bone_num))

    # 配准的变换参数的搜索范围
    param_range = np.array([5, 2, 5, 0.04, 0.04, 0.04])
    param_range2 = np.array([10, 2, 10, 0.05, 0.07, 0.1])

    # 计算投影方向的单位向量
    v = cal_v(xml, order)

    N = h * w * 2 * np.pi

    # 读取2D CT raw图提取的edge图
    edge_3d = np.fromfile('H:/CBCT/BoneRegistration/data/edge.raw', dtype='>b').reshape([prj_num, h, w])

    thre1, thre2 = 0.91, 0.84
    sm = 7
    for i in range(0, prj_num):
        # w_3d = np.zeros(vol_size, dtype=np.float32)

        fixed = fixed_3d[i]
    # 生成边缘图像的KD树
    edge = edge_3d[i] != 0
    ps = np.where(edge)
    kdt = KDTree(np.c_[ps[0], ps[1]])

    count = j = 0
    m_iter = 6 if i == 0 else 4
    tmp_t = np.zeros((2, 6), dtype=np.float64)
    tmp_s = np.ones((2,))

    # 分别对mask中的每块骨骼与每一张投影图像进行配准
    while j < bone_num:
        # 根据前几帧的投影图像中的同一块骨骼的变换参数进行一阶插值以预测当前帧的变换参数
        if all(trans[i][j][k] == init_t[j][k] for k in range(6)) or count > 0:
            fit_t = fit_transform(trans[max(0, i - 3):i, j, :], np.array(init_t[j]), max(0, min(1, i - 1)))
            # fit_t *= [1]*6 if count %2 == 0 else [1,1,1,0,0,0]
        else:
            fit_t = trans[i][j]
        fit_s = fit_sim(ss[0:i, j], 0)

        t0 = time.time()
        outTx = fit_t.copy()
        # outTx[:3]/=2
        # xml['datanode']['vol_pixel_size']*=2

        vol_geom_s = vol_geom.copy()
        vol_geom_s['GridSliceCount'], vol_geom_s['GridRowCount'], vol_geom_s['GridColCount'] = vols[j][0].shape

        f1_best = {'x': outTx, 'cost': np.inf}

        if j == 0:
            # t_mean = fit_t#/2
            t_mean = [np.nan]
        else:
            t_mean = trans[i, :j].mean(axis=0)  # /2

        dz, dy, dx = np.gradient(vols[j][0].astype(np.float32))
        g_len = dx ** 2 + dy ** 2 + dz ** 2

        # 排除vol法向量为0的点和内部点
        g_len[g_len < np.percentile(g_len, 99.6)] = 0

        res = minimize(f2, outTx,
                       args=(ms[0], vols[j][0], fixed, proj_geom, vol_geom_s, kdt, xml, order, v, t_mean,
                             (g_len, dz, dy, dx)),
                       method='SLSQP',
                       bounds=((outTx[0] - param_range2[0], outTx[0] + param_range2[0]),
                               (outTx[1] - param_range2[1], outTx[1] + param_range2[1]),
                               (outTx[2] - param_range2[2], outTx[2] + param_range2[2]),
                               (outTx[3] - param_range2[3], outTx[3] + param_range2[3] / 3),
                               (outTx[4] - param_range2[4] / 100, outTx[4] + param_range2[4] / 3),
                               (outTx[5] - param_range2[5] / 4, outTx[5] + param_range2[5] / 5)),
                       options={'ftol': 1e-5,
                                'eps': 1e-3,
                                'finite_diff_rel_step': [1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2],
                                'maxiter': 20, 'disp': True})

        # 防止变换参数偏差过大
        if all(outTx[k] - param_range2[k] <= f1_best['x'][k] <= outTx[k] + param_range2[k] for k in range(6)):
            outTx = f1_best['x']
        else:
            outTx = res.x
        # outTx = res.x

        t1 = time.time()
        print("---------------------")
        print("proj%d bone%d costs %d s" % (i, j, t1 - t0))
        print(outTx[:3], outTx[3:])
        print("---------------------")

        trans[i, j] = outTx
        ss[i, j] = f1_best['cost']

        moving = projection(vols[j][-1], proj_geom, vol_geom)
        # w_3d_ = sitk.GetArrayFromImage(resample_display(vols[j][-1],trans[i,j][:3],trans[i,j][3:])).astype(np.float32)
        r = trans[i, j][3:]
        rotate_m = get_rotate_matrix(-r[::-1])
        new_pg = transform_projgeom(proj_geom, trans[i, j][:3], rotate_m)
        warp_moving = projection(vols[j][-1], new_pg, vol_geom)

        new_v = np.dot(v, rotate_m)

        # 判断当前变换参数是否偏差过大
        if thre1 * ss[i, j] < fit_s and thre2 * ss[i, j] < ss[max(0, i - 1), j]:
            t = fixed + edge * 0.5
            proj = draw_edge((g_len, dz, dy, dx), t, xml, trans[i, j][:3], new_v, order, get_rotate_matrix(r))
            show(t, moving, warp_moving, proj, sm, i, j, True)
            j += 1
            count = 0

            # w_3d += w_3d_
            # del w_3d_
        else:
            tmp_t[count] = outTx
            tmp_s[count] = f1_best['cost']
            if count == 1:
                t = np.argmin(tmp_s)
                trans[i, j] = tmp_t[t]
                ss[i, j] = tmp_s[t]

                # w_3d_ = sitk.GetArrayFromImage(resample_display(vols[j][-1],trans[i,j][:3],trans[i,j][3:])).astype(np.float32)
                r = trans[i, j][3:]
                rotate_m = get_rotate_matrix(-r[::-1])
                new_pg = transform_projgeom(proj_geom, trans[i, j][:3], rotate_m)
                warp_moving = projection(vols[j][-1], new_pg, vol_geom)
                # del w_3d_
                new_v = np.dot(v, rotate_m)
                t = fixed + edge * 0.5
                proj = draw_edge((g_len, dz, dy, dx), t, xml, trans[i, j][:3], new_v, order, get_rotate_matrix(r))
                # show(t, moving, warp_moving, proj, sm, i, j, True)

                count = 0
                j += 1
            else:
                count += 1

        # moving = projection(w_3d, proj_geom, vol_geom)
        # w_3d.tofile('F:/bone/3/warp1_%d.raw'%(i))

    trans.tofile('H:/CBCT/BoneRegistration/data/trans.raw')
