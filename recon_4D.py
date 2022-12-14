# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 12:05:23 2022

@author: lenovo
"""

import SimpleITK as sitk
import numpy as np

from astra import create_proj_geom
from scipy.ndimage import gaussian_filter1d
from load_xml import parse_xml
from parse_geom import geometry
from util import resample_display, projection

# from scipy.optimize import curve_fit

xml_path = 'F:/bone/6/recon.xml'
xml = parse_xml(xml_path)

# cal vol_geom & proj_geom    
h, w = xml['datanode']['prj_height'], xml['datanode']['prj_width']
v, vol_geom = geometry(xml, False)

# 选择初始投影角度
order = 170
proj_geom = create_proj_geom('cone_vec', h, w, v[order:order+1])

# read 3D CT & mask
vol_size = np.array([xml['datanode']['vol_size'],xml['datanode']['vol_size'],xml['datanode']['vol_height']])
moving_3d = np.fromfile('F:/bone/6/3D.raw', dtype=np.uint16).reshape(vol_size).astype(np.float32)[::-1,:,::-1]
moving_3d = (moving_3d/xml['datanode']['linearHU0']).astype(np.float32)

prj_num = 36
fixed_3d = np.fromfile('F:/bone/6/proc_2D.raw', dtype=np.float32).reshape((prj_num,h,w))

mask = np.fromfile('F:/bone/6/mask.raw', dtype=np.uint8).reshape(vol_size)[::-1,:,::-1]

bone_num = mask.max()
trans = np.fromfile('F:/bone/6/trans.raw', dtype=np.float64).reshape((prj_num, bone_num, 6))
sim_p = np.zeros((prj_num, h, w), dtype=np.float32)
prj_num2 = 18
t = range(prj_num2)
k = 9
fit_trans = trans.copy()
# 高斯滤波平滑相邻帧的变换参数
for i in range(bone_num):
    for j in range(6):
        fit_trans[:prj_num2,i,j] = gaussian_filter1d(trans[:prj_num2,i,j], 1)

for i in range(prj_num2):
    # 根据变换参数生成对应的3D CBCT并生成模拟投影
    w_3d = np.zeros(vol_size, dtype=np.float32)
    for j in range(bone_num):
        w_3d_ = sitk.GetArrayFromImage(resample_display(moving_3d*(mask==(j+1)),-fit_trans[i,j][:3][::-1],-fit_trans[i,j][3:][::-1])).astype(np.float32)
        w_3d += w_3d_
    w_moving = projection(w_3d, proj_geom, vol_geom)
    sim_p[i] = w_moving
    # 保存配准后的3D volume
    w_3d.tofile('F:/bone/6/warp1_%d.raw'%(i))
# 保存模拟投影
sim_p.tofile('F:/bone/6/simulate_proj.raw')
