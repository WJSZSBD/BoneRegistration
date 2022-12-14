# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:24:15 2021

@author: lenovo
"""

from collections import OrderedDict

import numpy as np


def vol2world(p, shape, vps, t, rotate_m):
    # vps vol pixel size
    d, h, w = shape
    z,y,x = p
    z = (z-d/2)*vps+t[0]
    y = (y-h/2)*vps+t[1]
    x = (x-w/2)*vps+t[2]
    
    wc = np.array([z,y,x]).T
    wc = np.dot(wc, rotate_m)
    wc = wc[(wc[:,0]<d/2)&(wc[:,1]>-d/2)&(wc[:,1]<h/2)&(wc[:,1]>-h/2)&(wc[:,2]<w/2)&(wc[:,2]>-w/2)]
    return wc


def cal_marks(g_len,dz,dy,dx, v=[1,0,0], thre=0.1):
    # v - 投影方向
    
    # 计算vol法向量与投影向量的余弦值
    cos_v = np.abs(dz*v[2]+dy*v[1]+dx*v[0])/np.sqrt(g_len)

    # 确定vol中法向量与投影向量基本正交的点(边缘)
    mask = cos_v < thre
    
    return np.array(np.where(mask), dtype=np.int16)#, \
        #np.transpose(np.array([dz,dy,dx], dtype=np.float32), [1,2,3,0])
    

def world2proj(marks, xml, prj_order=0):
    SD = xml['datanode']['SDD'][0,prj_order,0]
    sysm = xml['datanode']['SysM'][0,prj_order,0]
    SO = SD/sysm
    OD = SD-SO
    px = xml['datanode']['pX'][0,prj_order,0]
    py = xml['datanode']['pY'][0,prj_order,0]
    dety = xml['datanode']['DetY'][0,prj_order,0]
    detz = xml['datanode']['DetZ'][0,prj_order,0]
    prj_pixel_size = xml['datanode']['prj_pixel_size']
    theta = xml['datanode']['angles'][0,prj_order,0]
    # vol_pixel_size = xml['datanode']['vol_pixel_size']
    
    # Source (x,y,z) unit: mm
    s = np.zeros(3, dtype=np.float32)
    s[0] = SO*np.sin(theta)
    s[1] = -SO*np.cos(theta)
    s[2] = (py-detz)/sysm*prj_pixel_size
    
    gamma = np.arccos((dety-px)*prj_pixel_size/SD)
    alpha = gamma-np.pi/2+theta
    
    # Detector Center
    d = np.zeros(3, dtype=np.float32)
    d[0] = -OD*np.sin(theta)
    d[1] = OD*np.cos(theta)
    d[2] = s[2]+(detz-py)*prj_pixel_size
    
    # Detector Top Left Corner
    o = np.zeros(3, dtype=np.float32)
    o[0] = d[0]-px*np.cos(alpha)*prj_pixel_size
    o[1] = d[1]-px*np.sin(alpha)*prj_pixel_size
    o[2] = d[2]+py*prj_pixel_size
    
    pc = np.zeros((len(marks.T),2),dtype=np.float32)
    # d,h,w = xml['datanode']['vol_height'],xml['datanode']['vol_size'],xml['datanode']['vol_size']
    z,y,x = marks
    pc[:,1] = ((x-s[0])*(o[1]-s[1])-(y-s[1])*(o[0]-s[0]))/(np.cos(alpha)*(y-s[1])-np.sin(alpha)*(x-s[0]))/prj_pixel_size
    t = (x-s[0])/(o[0]+np.cos(alpha)*prj_pixel_size*pc[:,1]-s[0])
    pc[:,0] = (o[2]-s[2]-(z-s[2])/t)/prj_pixel_size
        
    return pc.astype(np.int16), s, o, theta


def cal_dis(marks, nn, s, o, theta, pps):
    '''
    marks [[z1,y1,x1],[z1,y2,x2],...] shape=(n, 3) world coordinate
    nn [[y1,x1],[y2,x2],...] shape=(n, 2) projection image coordinate
    s [x, y, z] source world coordinate
    o [x, y, z] detector top left corner world coordinate
    theta rotate radius
    pps prj_pixel_size
    '''
    u = pps*np.array([[np.cos(theta),np.sin(theta),0]],dtype=np.float32)    # (1,3)
    v = pps*np.array([[0,0,1]],dtype=np.float32)                            # (1,3)
    
    wc = o[::-1]+np.dot(nn[:,1:],u)+np.dot(nn[:,:1],v)                      # (n,3)
    
    v1, v2 = wc - s[::-1], marks - s[::-1]                                  # (n,3)
    v1_l2 = np.sum(v1*v1, axis=1)                                           # (n,)
    v2_l2 = np.sum(v2*v2, axis=1)                                           # (n,)
    h2 = np.sum(v1*v2, axis=1)**2/v1_l2                                     # (n,)
    
    d = np.sqrt(v2_l2-h2)                                                   # (n,)
        
    return d


def cal_weight(g, marks, proj, nn, normal_v, theta):
    '''
    Parameters
    ----------
    g : [[[[z1,y1,x1],...],...],...] shape=(n,n,n,3)
        3D vol normal vector.
    marks : [[z1,z2,...],[y1,y2,...],[x1,x2,...]] shape=(3,n)
        3D points vol coordinate
    proj : shape=(h,w)
        projection image.
    nn : [[y1,x1],[y2,x2],...] shape=(n, 2)
        2D nearest edge point (projection image coordinate).
    normal_v : [x, y, z] shape=(3,)
        the normal vector of projection plane.
    theta : float
        rotate radius.
    '''
    normal_v = np.array(normal_v[::-1], dtype=np.float32).reshape((3,1))
    marks_g = g[tuple(marks)].astype(np.float32)                        # (n,3)
    proj_g = marks_g - np.dot(np.dot(marks_g, normal_v), normal_v.T)/np.sum(normal_v**2)    # (n,3)
    
    if theta == 0:
        theta = 1e-2
    dy_g = proj_g[:,0]                                                  # (n,)
    dx_g = (proj_g[:,1]/np.cos(theta) - proj_g[:,2]/np.sin(theta))/2    # (n,)
    
    [dy,dx] = np.gradient(proj)
    dy_nn = dy[tuple(nn.T)]     # (n,)
    dx_nn = dx[tuple(nn.T)]     # (n,)
    
    cos_v = (dy_g*dy_nn+dx_g*dx_nn)/(np.sqrt(dy_g**2+dx_g**2)*np.sqrt(dx_nn**2+dy_nn**2))
    cos_v[(cos_v<0) | (np.isnan(cos_v))] = 0
    
    return cos_v


def cal_mask(vol, proj, xml, proj_order, v, thre=0.006):
    # tmp_vol = vol.copy()
    # tmp_vol[tmp_vol<2e-3] = 0
    marks, _ = cal_marks(vol, v, thre)
    d, h, w = vol.shape
    marks = marks[:,np.where((marks[0,:]!=0)&(marks[0,:]!=d-1))]
    
    marks_wc = vol2world(marks, vol.shape, xml['datanode']['vol_pixel_size'])
    marks_pc, _, _, _ = world2proj(marks_wc.T, xml, proj_order)
    
    h, w = proj.shape
    marks_pc[np.where(marks_pc<0)[0]] = 0
    marks_pc[np.where((marks_pc[:,0]>=h) | (marks_pc[:,1]>=w))] = 0
    
    mask = np.zeros(proj.shape, dtype=np.bool_)
    mask[tuple(marks_pc.T)] = 1
    # mask = dilation(mask, square(3))
    
    return mask    


def fit(g, proj, kdt, xml, proj_order, v, nu=10, t=[0]*3, rotate_m=np.eye(3), upper_b=50):
    marks = cal_marks(g[0], g[1], g[2], g[3], v, 0.15)
    marks_wc = vol2world(marks, g[0].shape, xml['datanode']['vol_pixel_size'], t, rotate_m)
    marks_pc, s, o, theta = world2proj(marks_wc.T, xml, proj_order)
    
    # edge = canny(proj, 2, 0.01, 0.02)
    # nn, u = cal_nearest_point(marks_pc, edge)
    dd, ii = kdt.query(marks_pc, k=1, distance_upper_bound=upper_b)
    
    new_ii = OrderedDict() # 去除重复的点
    for d,i in zip(dd,ii):
        new_ii[i] = min(new_ii.get(i, np.inf), d)
    new_dd = np.array(list(new_ii.values()))
    u = new_dd!=np.inf
    # u = dd != np.inf
    # ii[ii==len(kdt.data)]=0
    # dd[dd==np.inf]=upper_b
    # d = np.exp(-dd)
    # e_fit = (1-u*d).mean()
    
    # ii = np.unique(ii)
    # nn = kdt.data[ii].astype(np.int)
    # dis = cal_dis(marks_wc, nn, s, o, theta, xml['datanode']['prj_pixel_size'])
    
    w = 1
    # w = cal_weight(g, marks, proj, kdt.data.astype(np.int)[ii], v, theta)
    
    #dis = dis/dis.mean()*dd.mean()
    d = (np.exp(-new_dd/nu))#+np.exp(-dd/nu))/2
    
    e_fit = (1 - u*w*d).sum()/sum(u)
    
    return e_fit


def draw_edge(g, proj, xml, t, v, proj_order, rotate_m=np.eye(3)):
    marks = cal_marks(g[0], g[1], g[2], g[3], v, 0.5)
    marks_wc = vol2world(marks, g[0].shape, xml['datanode']['vol_pixel_size'], t, rotate_m)
    marks_pc, s, o, theta = world2proj(marks_wc.T, xml, proj_order)
    t1 = proj.copy()
    h,w = proj.shape
    for p in marks_pc:
        if 0<p[0]<h and 0<p[1]<w:
            t1[p[0],p[1]]=1.5
    return t1
