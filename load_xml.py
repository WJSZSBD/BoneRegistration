# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:34:17 2021

@author: lenovo
"""

import os
import xml
import numpy as np

from xml.dom import minidom


def parse_path(node):
    keys = [n.nodeName for n in node.childNodes if isinstance(
        n, xml.dom.minidom.Element)]
    vs = [n.firstChild.data for n in node.childNodes if isinstance(
        n, xml.dom.minidom.Element)]
    vs = [p.replace('\\', '/') for p in vs]

    return dict(zip(keys, vs))


def parse_data(node):
    keys = [n.nodeName for n in node.childNodes if isinstance(
        n, xml.dom.minidom.Element)]
    vs = [n.firstChild.data for n in node.childNodes if isinstance(
        n, xml.dom.minidom.Element)]
    # vs = [float(v) for v in vs]
    vs = [int(v) if v.isdigit() else float(v) for v in vs]

    datas = dict(zip(keys, vs))
    datas['imgingROI'] = [datas.get('imagingROI'+str(i)) for i in range(4)]
    return datas


def parse_geo(node):
    keys = [n.nodeName for n in node.childNodes if isinstance(
        n, xml.dom.minidom.Element)]
    vs = [n.firstChild.data for n in node.childNodes if isinstance(
        n, xml.dom.minidom.Element)]
    # vs = [float(v) for v in vs]
    vs = [int(v) if v.isdigit() else float(v) for v in vs]

    geos = dict(zip(keys, vs))
    geos['start_angle'] = geos.get('start_angle')*np.pi/180
    return geos


def parse_recon(node):
    recons = {}
    # recons['linearHU'] = [int(node.getElementsByTagName(
    #     "linearHU"+str(i))[0].firstChild.data) for i in range(2)]
    # recons['device_type'] = node.getElementsByTagName("device_type")[
    #     0].firstChild.data
    recons['method'] = node.getElementsByTagName("method")[0].firstChild.data
    recons['method_para'] = node.getElementsByTagName("method_para")[
        0].firstChild.data

    return recons


def load_data(file_path, img_width, img_height, img_N, data_type, offset):
    with open(file_path, 'rb') as f:
        f.seek(offset, os.SEEK_SET)
        img = np.fromfile(f, dtype=data_type, count=img_width*img_height*img_N)

    img = img.reshape((img_height, img_width, img_N)).astype(np.float32)
    return img


def load_dead_pixel_map(file_path):
    with open(file_path, 'rb') as f:
        length = int(f.read(4))
        f.seek(length, os.SEEK_SET)
        img = np.fromfile(f, dtype=np.float32)

    img = img.reshape((9, length))
    return img


def load_geo_data(struct):
    prj_num = struct['datanode']['prj_num']
    prj_width = struct['datanode']['prj_width']
    prj_height = struct['datanode']['prj_height']

    # load files
    struct['datanode']['angles'] = load_data(
        struct['pathnode']['angle'], prj_num, 1, 1, np.float32, 0)
    struct['datanode']['DetY'] = load_data(
        struct['pathnode']['sor_x0'], prj_num, 1, 1, np.float32, 0)
    struct['datanode']['DetZ'] = load_data(
        struct['pathnode']['sor_y0'], prj_num, 1, 1, np.float32, 0)
    struct['datanode']['InRot'] = load_data(
        struct['pathnode']['detector_Y'], prj_num, 1, 1, np.float32, 0)
    struct['datanode']['pX'] = load_data(
        struct['pathnode']['p_x'], prj_num, 1, 1, np.float32, 0)
    struct['datanode']['pY'] = load_data(
        struct['pathnode']['p_y'], prj_num, 1, 1, np.float32, 0)
    struct['datanode']['SDD'] = load_data(
        struct['pathnode']['SD'], prj_num, 1, 1, np.float32, 0)
    struct['datanode']['SysM'] = load_data(
        struct['pathnode']['SysM'], prj_num, 1, 1, np.float32, 0)
    struct['datanode']['angles'] += struct['datanode']['start_angle']

    # fields and deadpixels
    struct['datanode']['dark'] = load_data(
        struct['pathnode']['dark'], prj_width, prj_height, 1, np.uint16, 2048)
    struct['datanode']['light'] = load_data(
        struct['pathnode']['flat'], prj_width, prj_height, 1, np.uint16, 2048)
    # struct['datanode']['deadpix'] = loadDeadPixelMap(
    #     struct['pathnode']['deadpixel'])

    return struct


def parse_xml(file_path):
    dirname = os.path.dirname(file_path)
    tmp_path = os.path.join(dirname, 'tmp.xml')
    with open(tmp_path, 'w') as tmp:
        tmp.write('<root>\n')
        with open(file_path, 'r') as f:
            lines = f.read().replace('[', '').replace(']', '').replace("<?xml version='1.0' encoding='utf-8' ?>",'')
            tmp.write(lines)
        tmp.write('</root>')

    dom_tree = minidom.parse(tmp_path)
    root = dom_tree.documentElement

    paths = parse_path(root.getElementsByTagName('PATH')[0])
    datas = parse_data(root.getElementsByTagName('DATA')[0])
    recons = parse_recon(root.getElementsByTagName('RECON')[0])

    struct = {'pathnode': paths, 'datanode': datas,
              'reconnode': recons}

    struct['datanode']['start_angle'] = float(struct['pathnode']['start_angle'])
    struct = load_geo_data(struct)

    os.remove(tmp_path)

    return struct


# parse_xml(r'F:/data/WD_DSA_0/FICBCT_fR001.000.xml')
