# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:34:33 2021

@author: lenovo
"""

import numpy as np
from astra import create_vol_geom, create_proj_geom


def geometry(struct, return_proj_geom=True):
    vol_pixel_size = struct['datanode']['vol_pixel_size']
    vol_size = struct['datanode']['vol_size']
    vol_height = struct['datanode']['vol_height']

    x_size = (vol_size / 2) * vol_pixel_size
    z_size = (vol_height / 2) * vol_pixel_size

    vol_geom = create_vol_geom(
        vol_size, vol_size, vol_height, -x_size, x_size, -x_size, x_size, -z_size, z_size)

    angles = np.squeeze(struct['datanode']['angles'])
    # distance from the source to the detector
    SDD = np.squeeze(struct['datanode']['SDD'])
    # SDD/distance from source to the axis of rotation of th source
    SysM = np.squeeze(struct['datanode']['SysM'])
    pX, pY = np.squeeze(struct['datanode']['pX']), np.squeeze(
        struct['datanode']['pY'])
    DetZ = np.squeeze(struct['datanode']['DetZ'])

    prj_width = struct['datanode']['prj_width']
    prj_height = struct['datanode']['prj_height']
    prj_pixel_size = struct['datanode']['prj_pixel_size']

    SOD = SDD / SysM
    ODD = SDD - SOD

    vectors_full = np.zeros((angles.shape[0], 12))

    # source
    vectors_full[:, 0] = np.sin(angles) * SOD
    vectors_full[:, 1] = -np.cos(angles) * SOD
    vectors_full[:, 2] = (pY - DetZ) / SysM * prj_pixel_size

    # center of detector
    vectors_full[:, 3] = np.cos(angles) * (prj_width / 2 - 0.5 - pX) * \
                         prj_pixel_size - np.sin(angles) * ODD
    vectors_full[:, 4] = np.sin(angles) * (prj_width / 2 - 0.5 - pX) * \
                         prj_pixel_size + np.cos(angles) * ODD
    # vectors_full[:, 5] = vectors_full[:, 3] + (DetZ-prj_height/2+0.5)*prj_pixel_size
    vectors_full[:, 5] = -(prj_height / 2 + 0.5 - (pY - DetZ) / SysM - DetZ) * prj_pixel_size

    # vector from detector pixel(0, 0) to(0, 1)
    vectors_full[:, 6] = np.cos(angles) * prj_pixel_size
    vectors_full[:, 7] = np.sin(angles) * prj_pixel_size
    vectors_full[:, 8] = 0

    # vector from detector pixel(0, 0) to(1, 0)
    vectors_full[:, 9] = 0
    vectors_full[:, 10] = 0
    vectors_full[:, 11] = -prj_pixel_size

    if return_proj_geom:
        proj_geom = create_proj_geom(
            'cone_vec', prj_height, prj_width, vectors_full)

        return proj_geom, vol_geom
    else:
        return vectors_full, vol_geom
