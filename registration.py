#!/home/wangjiashun/anaconda3 python
# -*- coding: UTF-8 -*-
"""
@Project : BoneRegistration 
@File : registration.py
@Author : wangjiashun    
@Date : 2022/12/5 21:32 
@Brief : Registration from pickle generated by `calculate_parameter.py`.
"""

import SimpleITK as sitk
import pickle
import numpy as np
import nibabel as nib
from scipy import interpolate
import matplotlib

from imageMagnify import imageMagnify
from util import projection, resample_display, see

# pickle structure
class Registration:
    def __init__(self, bones_3d, fixed_3d, proj_geom, vol_geom):
        self.bones_3d = bones_3d
        self.fixed_3d = fixed_3d
        self.proj_geom = proj_geom
        self.vol_geom = vol_geom

if __name__ == '__main__':
    file = open('H:/CBCT/BoneRegistration/data/pickle/IdealFront.pickle', 'rb')
    pickle_data = pickle.load(file)
    file.close()
    bones_3d = pickle_data.bones_3d
    fixed_3d = pickle_data.fixed_3d
    proj_geom = pickle_data.proj_geom
    vol_geom = pickle_data.vol_geom
    del pickle_data

    bone_num = len(bones_3d)
    prj_num = fixed_3d.shape[0]
    # trans = [0] * 6
    trans = [0, 10, -62, -0.52, 0, -0.2]
    bone_iter = 0
    fixed_iter = 5

    RAI = np.eye(4)
    RAI[0, 0] = -0.54
    RAI[1, 1] = -0.54
    RAI[2, 2] = 0.54

    while True:
        # moving_3d = bones_3d[0] + bones_3d[1] + bones_3d[2]
        while True:
            dump_flag = False
            recon4D_flag = False

            moving_3d = bones_3d[bone_iter]
            fixed = fixed_3d[fixed_iter]

            fixed = imageMagnify(fixed, 1.7)
            moving_3d_trans = sitk.GetArrayFromImage(resample_display(moving_3d, trans[:3], trans[3:]))
            moving = projection(moving_3d_trans, proj_geom, vol_geom)
            print('%d\t%d\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf\n' % (
                bone_iter, fixed_iter, trans[0], trans[1], trans[2], trans[3], trans[4], trans[5]))

            see(moving + fixed)
            if dump_flag:
                moving_3d_trans = moving_3d_trans[::-1, :, :]
                moving_3d_trans = moving_3d_trans[:, :, ::-1]
                moving_3d_trans = moving_3d_trans.transpose(0, 2, 1)
                # xml['datanode']['linearHU0'] = 109000
                moving_3d_trans = (moving_3d_trans * 109000).astype('int16')
                moving_3d_trans_post = moving_3d_trans.copy()
                for i in range(512):
                    moving_3d_trans_post[:, i, :] = moving_3d_trans[:, i, :].T
                    moving_3d_trans_post[:, i, :] = moving_3d_trans_post[:, i, ::-1]
                moving_3d_trans_post = moving_3d_trans_post.transpose(1, 0, 2)

                nib.save(nib.Nifti1Image(moving_3d_trans_post, RAI),
                         'H:/CBCT/BoneRegistration/result/result%d/%02d.nii.gz' % (bone_iter + 1, fixed_iter + 1))
                matplotlib.image.imsave(
                    'H:/CBCT/BoneRegistration/result/result%d/%02d.png' % (bone_iter + 1, fixed_iter + 1),
                    moving + fixed, cmap='gray')
                matplotlib.image.imsave(
                    'H:/CBCT/BoneRegistration/result/magnified_DRs/%02d.png' % (fixed_iter + 1),
                    fixed, cmap='gray')

            if recon4D_flag:
                data = None
                affine = None
                for iter in range(bone_num):
                    file = nib.load('H:/CBCT/BoneRegistration/result/result%d/%02d.nii.gz' % (iter + 1, fixed_iter + 1))
                    if iter == 0:
                        data = file.get_fdata()
                        affine = file.affine
                    else:
                        data += file.get_fdata()
                data = data.astype('int16')
                nib.save(nib.Nifti1Image(data, affine),
                         'H:/CBCT/BoneRegistration/result/result_all/%02d.nii.gz' % (fixed_iter + 1))

                data = data.transpose(1, 0, 2)
                data_post = data.copy()
                for i in range(512):
                    data_post[:, i, :] = data[:, i, ::-1].T
                # xml['datanode']['linearHU0'] = 109000
                data_post = (data_post / 109000).astype(np.float32)
                data_post = data_post.transpose(0, 2, 1)
                data_post = data_post[:, :, ::-1]
                data_post = data_post[::-1, :, :]

                simu_proj = projection(data_post, proj_geom, vol_geom)
                simu_proj.tofile('H:/CBCT/BoneRegistration/result/result_all/%02d_float32_1024x1024.raw' % (fixed_iter + 1))
                matplotlib.image.imsave(
                    'H:/CBCT/BoneRegistration/result/result_all/%02d.png' % (fixed_iter + 1),
                    simu_proj, cmap='gray')
