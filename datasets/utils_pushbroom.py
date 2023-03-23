"""
This script contains functions that are useful to handle satellite images and georeferenced data
"""
import numpy as np
import rasterio
import datetime
import os, sys, re
import shutil
import json
import glob
import cv2
import torch
from torchvision import transforms as T
'''
row: y: line
col: x: samp
'''

def GET_PLH_COEF(P, L, H):
    '''
    Args:
        PLH: torch Tensor [B, 3]
    '''
    #P, L, H = torch.hsplit(PLH, 3)
    B = P.shape[0]
    coef = torch.zeros((B, 20))
    coef[:, 0] = 1.0
    coef[:, 1] = L
    coef[:, 2] = P
    coef[:, 3] = H
    coef[:, 4] = L * P
    coef[:, 5] = L * H
    coef[:, 6] = P * H
    coef[:, 7] = L * L
    coef[:, 8] = P * P
    coef[:, 9] = H * H
    coef[:, 10] = P * L * H
    coef[:, 11] = L * L * L
    coef[:, 12] = L * P * P
    coef[:, 13] = L * H * H
    coef[:, 14] = L * L * P
    coef[:, 15] = P * P * P
    coef[:, 16] = P * H * H
    coef[:, 17] = L * L * H
    coef[:, 18] = P * P * H
    coef[:, 19] = H * H * H

    return coef

def PHOTO2OBJ_std(samp, line, H, rpc):
    '''
    Args:
        slH: torch Tensor: meshgrid of samp(y), line(x), Height of [B, 3]
        rpc: torch Tensor: [170,]
    return:
        PLH torch Tensor: meshgrid of P(lat), l(lon), Height of [B, 3]
    '''
    #samp, line, H = torch.hsplit(slH, 3)
    samp -= rpc[1]#SAMP_OFF
    samp /= rpc[6]#SAMP_SCALE

    line -= rpc[0]#LINE_OFF
    line /= rpc[5]#LINE_SCALE

    H -= rpc[4]#HEIGHT_OFF
    H /= rpc[9]#HEIGHT_SCALE

    coef = GET_PLH_COEF(samp, line, H) # [B, 20]

    # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
    P_std = torch.sum(coef * rpc[90:110], axis=-1) / torch.sum(coef * rpc[110:130], axis=-1) # LATNUM LATDEM
    L_std = torch.sum(coef * rpc[130:150], axis=-1) / torch.sum(coef * rpc[150:170], axis=-1) # LONNUM LONDEM
    #PLH_std = torch.cat((P_std, L_std, H), -1)

    return P_std, L_std, H

def PHOTO2OBJ(samp, line, H, rpc):
    '''
    Args:
        slH: torch Tensor: meshgrid of samp(x), line(y), Height of [B, 3]
        rpc: torch Tensor: [170,]
    return:
        PLH torch Tensor: meshgrid of P(lat), l(lon), Height of [B, 3]
    '''
    #samp, line, H = torch.hsplit(slH, 3)
    samp -= rpc[1]#SAMP_OFF
    samp /= rpc[6]#SAMP_SCALE

    line -= rpc[0]#LINE_OFF
    line /= rpc[5]#LINE_SCALE

    H_std = H - rpc[4]#HEIGHT_OFF
    H_std = H_std / rpc[9]#HEIGHT_SCALE

    coef = GET_PLH_COEF(samp, line, H_std) # [B, 20]

    # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
    lat_std = torch.sum(coef * rpc[90:110], axis=-1) / torch.sum(coef * rpc[110:130], axis=-1) # LATNUM LATDEM
    lon_std = torch.sum(coef * rpc[130:150], axis=-1) / torch.sum(coef * rpc[150:170], axis=-1) # LONNUM LONDEM

    lat = lat_std * rpc[7]  # LAT_SCALE
    lat += rpc[2]  # LAT_OFF

    lon = lon_std * rpc[8]  # LONG_SCALE
    lon += rpc[3]  # LONG_OFF

    #latlonH = torch.cat((lat, lon, H), dim=-1)
    return lat, lon, H

def OBJ_std2PHOTO(P, L, H, rpc):
    '''
    Args:
        PLH: torch Tensor [B: 3]
        rpc: torch Tensor [170,]
    '''
    #P, L, H = torch.hsplit(PLH, 3)
    coef = GET_PLH_COEF(P, L, H)  # [B: 3]

    # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
    samp = torch.sum(coef * rpc[50:70], axis=-1) / torch.sum(coef * rpc[70:90], axis=-1)  # SNUM SDEM [B, 1]
    line = torch.sum(coef * rpc[10:30], axis=-1) / torch.sum(coef * rpc[30:50], axis=-1)  # LNUM LDEM [B, 1]

    samp *= rpc[6]  #SAMP_SCALE
    samp += rpc[1]  #SAMP_OFF

    line *= rpc[5]  #LINE_SCALE
    line += rpc[0]  #LINE_OFF

    return samp, line

def OBJ2PHOTO(lat, lon, Height, rpc):
    '''
    Args:
        LLH: torch Tensor [B: 3]
        rpc: torch Tensor [170,]
    '''
    #lat, lon, Height = torch.hsplit(LLH, 3)
    P = lat - rpc[2]  # LAT_OFF
    P /= rpc[7] # LAT_SCALE
    L = lon - rpc[3]  # LON_OFF
    L /= rpc[8]  # LON_SCALE
    H = Height - rpc[4]  # height_off
    H /= rpc[9]  # height_scale
    coef = GET_PLH_COEF(P, L, H)  # [B: 3]

    # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
    samp = torch.sum(coef * rpc[50:70], axis=-1) / torch.sum(coef * rpc[70:90], axis=-1)  # SNUM SDEM [B, 1]
    line = torch.sum(coef * rpc[10:30], axis=-1) / torch.sum(coef * rpc[30:50], axis=-1)  # LNUM LDEM [B, 1]

    samp *= rpc[6]  #SAMP_SCALE
    samp += rpc[1]  #SAMP_OFF

    line *= rpc[5]  #LINE_SCALE
    line += rpc[0]  #LINE_OFF

    return samp, line

def latlon_to_ecef_custom(lat, lon, alt):
    """
    convert from geodetic (lat, lon, alt) to geocentric coordinates (x, y, z)
    """
    rad_lat = lat * (torch.pi / 180.0)
    rad_lon = lon * (torch.pi / 180.0)
    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / torch.sqrt(1 - e2 * torch.sin(rad_lat) * torch.sin(rad_lat))

    x = (v + alt) * torch.cos(rad_lat) * torch.cos(rad_lon)
    y = (v + alt) * torch.cos(rad_lat) * torch.sin(rad_lon)
    z = (v * (1 - e2) + alt) * np.sin(rad_lat)
    return x, y, z

def ecef_to_latlon_custom(x, y, z):
    """
    convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)
    """
    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a ** 2
    esq = e ** 2
    b = torch.sqrt(asq * (1 - esq))
    bsq = b ** 2
    ep = torch.sqrt((asq - bsq) / bsq)
    p = torch.sqrt((x ** 2) + (y ** 2))
    th = torch.arctan2(a * z, b * p)
    lon = torch.arctan2(y, x)
    lat = torch.arctan2((z + (ep ** 2) * b * (torch.sin(th) ** 3)), (p - esq * a * (torch.cos(th) ** 3)))
    N = a / (torch.sqrt(1 - esq * (torch.sin(lat) ** 2)))
    alt = p / torch.cos(lat) - N
    lon = lon * 180 / torch.pi
    lat = lat * 180 / torch.pi
    return lat, lon, alt

def get_rays(samps, lines, rpc, max_H, min_H):
    """
    Draw a set of rays from a satellite image for the standardized PLH space.
    Each ray: [rays_o: origin 3d point, rays_d:a direction vector]
    steps:
    1. draw the ray bounds on the altitude direction, by localizing pixels at min_alt and max_alt
    2. direction vector is given bounded in the [-1, 1] cube
    Then the corresponding direction vector is found by the difference between such bounds
    Args:
        samps: 1d array with image column coordinates
        lines: 1d array with image row coordinates
        rpc: torch Tensor of the 170 inverse RPC parameters
    Returns:
        rays: (h*w, 8) tensor of floats encoding h*w rays
              columns 0,1,2 correspond to the rays origin
              columns 3,4,5 correspond to the direction vector
              columns 6,7 correspond to the distance of the ray bounds with respect to the camera
        """
    assert samps.shape == lines.shape
    min_alts = min_H * torch.ones(samps.shape)  # h*w
    max_alts = max_H * torch.ones(samps.shape)  # h*w

    # assume the points of maximum altitude are those closest to the camera
    P_min, L_min, H_min = PHOTO2OBJ(samps, lines, min_alts, rpc)
    x_far, y_far, z_far = latlon_to_ecef_custom(P_min, L_min, H_min)
    xyz_far = torch.vstack([x_far, y_far, z_far]).T

    # similarly, the points of minimum altitude are the furthest away from the camera
    P_max, L_max, H_max = PHOTO2OBJ(samps, lines, max_alts, rpc) # check H
    x_near, y_near, z_near = latlon_to_ecef_custom(P_max, L_max, H_max)
    xyz_near = torch.vstack([x_near, y_near, z_near]).T # [ncols(nrows), 3]
    # todo: plot to see whether to use ecef
    # define the rays origin as the nearest point coordinates
    rays_o = xyz_near

    # define the unit direction vector
    d = xyz_far - xyz_near
    rays_d = d / torch.linalg.norm(d, axis=1)[:, None]

    # assume the nearest points are at distance 0 from the camera
    # the furthest points are at distance Euclidean distance(far - near)
    fars = torch.linalg.norm(d, axis=1)
    nears = float(0) * torch.ones(fars.shape)

    # create a stack with the rays origin, direction vector and near-far bounds
    rays = torch.hstack([rays_o, rays_d, nears[:, None], fars[:, None]])
    #rays = torch.hstack([rays_o, rays_d, nears, fars])
    #rays = torch.hstack([rays_o, rays_d])
    rays = rays.type(torch.FloatTensor)
    return rays

def get_rays_ecef(samps, lines, rpc, max_H, min_H):
    """
    Draw a set of rays from a satellite image for the standardized PLH space.
    Each ray: [rays_o: origin 3d point, rays_d:a direction vector]
    steps:
    1. draw the ray bounds on the altitude direction, by localizing pixels at min_alt and max_alt
    2. direction vector is given bounded in the [-1, 1] cube
    Then the corresponding direction vector is found by the difference between such bounds
    Args:
        samps: 1d array with image column coordinates
        lines: 1d array with image row coordinates
        rpc: torch Tensor of the 170 inverse RPC parameters
    Returns:
        rays: (h*w, 8) tensor of floats encoding h*w rays
              columns 0,1,2 correspond to the rays origin
              columns 3,4,5 correspond to the direction vector
              columns 6,7 correspond to the distance of the ray bounds with respect to the camera
        """
    assert samps.shape == lines.shape
    min_alts = min_H * torch.ones(samps.shape)  # h*w
    max_alts = max_H * torch.ones(samps.shape)  # h*w

    # assume the points of maximum altitude are those closest to the camera
    # the bounds at bottom (xyz_far)
    P_min, L_min, H_min = PHOTO2OBJ_std(samps, lines, min_alts, rpc)

    x_far, y_far, z_far = latlon_to_ecef_custom(P_min, L_min, H_min)
    #xyz_far = torch.vstack([P_min, L_min, H_min]).T
    xyz_far = torch.vstack([x_far, y_far, z_far]).T

    # similarly, the points of minimum altitude are the furthest away from the camera
    P_max, L_max, H_max = PHOTO2OBJ_std(samps, lines, max_alts, rpc) # check H
    x_near, y_near, z_near = latlon_to_ecef_custom(P_max, L_max, H_max)
    #xyz_near = torch.vstack([P_max, L_max, H_max]).T # [ncols(nrows), 3]
    xyz_near = torch.vstack([x_near, y_near, z_near]).T  # [ncols(nrows), 3]

    # define the rays origin as the nearest point coordinates
    rays_o = xyz_near

    # define the unit direction vector
    d = xyz_far - xyz_near
    rays_d = d / torch.linalg.norm(d, axis=1)[:, None]

    # assume the nearest points are at distance 0 from the camera
    # the furthest points are at distance Euclidean distance(far - near)
    fars = torch.linalg.norm(d, axis=1)
    nears = float(0) * torch.ones(fars.shape)

    # create a stack with the rays origin, direction vector and near-far bounds
    rays = torch.hstack([rays_o, rays_d, nears[:, None], fars[:, None]])
    #rays = torch.hstack([rays_o, rays_d, nears, fars])
    #rays = torch.hstack([rays_o, rays_d])
    rays = rays.type(torch.FloatTensor)
    return rays

def load_gray_tensor_from_rgb_geotiff(img_path):
    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0)) / 255.
        #img.dtype=np.float32()# (3, h, w)  ->  (h, w, 3)
        img_gray = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY)
    #h, w = img.shape[:2]
    img_gray = T.ToTensor()(img_gray)
    gray = img_gray.view(1, -1).permute(1, 0)  # (h*w, 3)
    gray = gray.type(torch.FloatTensor)
    return gray

def load_gray_tensor_from_geotiff(img_path):
    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0)) / 255.
    #h, w = img.shape[:2]
    img_gray = T.ToTensor()(img)
    gray = img_gray
    gray = gray.type(torch.FloatTensor)
    return gray

def load_tensor_from_rgb_geotiff(img_path):
    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0)) / 255.
    h, w = img.shape[:2]
    #h, w = h_original // 3, w_original // 3
    img = T.ToTensor()(img)  # (3, h, w)
    #rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
    rgbs = img.permute(1, 2, 0)
    rgbs = rgbs.type(torch.FloatTensor)
    # res = {
    #     'rgbs':rgbs,
    #     'hw': [h, w]
    # }
    return rgbs

def load_aug_rpc_tensor_from_txt(filepath):
    """
    Read the direct and inverse rpc from a file
    :param filepath:
    :return:
    """
    if os.path.exists(filepath) is False:
        print("Error#001: cann't find " + filepath + " in the file system!")
        return

    try:
        with open(filepath, 'r',encoding='utf-8') as f:
            all_the_text = f.read().splitlines()
    except:
        with open(filepath, 'r',encoding='gbk') as f:
            all_the_text = f.read().splitlines()

    data = [text.split(' ')[1] for text in all_the_text]
    # print(data)
    data = np.array(data, dtype=np.float64)
    data = torch.from_numpy(data)
    return data

def GetH_MAX_MIN(rpc):
    """
    Get the max and min value of height based on rpc
    :return: hmax, hmin
    """
    hmax = rpc[4] + rpc[9]  # HEIGHT_OFF + HEIGHT_SCALE
    hmin = rpc[4] - rpc[9]  # HEIGHT_OFF - HEIGHT_SCALE

    return hmax, hmin

def utm_from_latlon(lats, lons):
    """
    convert lat-lon to utm
    """
    import pyproj
    import utm
    from pyproj import Transformer

    n = utm.latlon_to_zone_number(lats[0], lons[0])
    l = utm.latitude_to_zone_letter(lats[0])
    proj_src = pyproj.Proj("+proj=latlong")
    proj_dst = pyproj.Proj("+proj=utm +zone={}".format(n))
    transformer = Transformer.from_proj(proj_src, proj_dst)
    easts, norths = pyproj.transform(proj_src, proj_dst, lons, lats)
    #transformer = Transformer.from_crs(n, l)
    #easts, norths = transformer.transform(lons, lats)
    #easts, norths =
    return easts, norths

def rescale_rpc(rpc, alpha):
    """
    Scale a rpc model following an image resize
    Args:
        rpc: rpc tensor of 170
        alpha: resize factor
               e.g. 2 if the image is upsampled by a factor of 2
                    1/2 if the image is downsampled by a factor of 2
    Returns:
        rpc_scaled: the scaled version of P by a factor alpha
    """
    # import copy
    #
    # rpc_scaled = copy.copy(rpc)
    rpc[5] *= float(alpha)  # LINE_SCALE
    rpc[6] *= float(alpha)  # SAMP_SCALE
    rpc[0] *= float(alpha)  # LINE_OFF
    rpc[1] *= float(alpha)  # SAMP_OFF
    # rpc_scaled.LINE_SCALE *= float(alpha)
    # rpc_scaled.SAMP_SCALE *= float(alpha)
    # rpc_scaled.LINE_OFF *= float(alpha)
    # rpc_scaled.SAMP_OFF *= float(alpha)
    return rpc

def save_pfm(file, image, scale=1):
    file = open(file, mode='wb')

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(bytes('PF\n' if color else 'Pf\n', encoding='utf8'))
    file.write(bytes('%d %d\n' % (image.shape[1], image.shape[0]), encoding='utf8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(bytes('%f\n' % scale, encoding='utf8'))

    image_string = image.tostring()
    file.write(image_string)

    file.close()


def load_pfm(fname):
    file = open(fname, 'rb')
    header = str(file.readline().decode('latin-1')).rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('latin-1'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float((file.readline().decode('latin-1')).rstrip())
    if scale < 0:  # little-endian
        data_type = '<f'
    else:
        data_type = '>f'  # big-endian

    data = np.fromfile(file, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flip(data, 0)

    return data

