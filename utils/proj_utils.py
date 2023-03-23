import numpy as np
import numpy_groupies as npg
import torch
from osgeo import gdal, gdal_array, osr
import utm
import cv2
import os
from utils.render import LOCALIZATION, PROJECTION

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import pyproj


def get_driver(file):
    f_ext = os.path.splitext(file)[1]
    for i in range(gdal.GetDriverCount()):
        driver = gdal.GetDriver(i)
        if driver.GetMetadataItem(gdal.DCAP_RASTER):
            d_ext_str = driver.GetMetadataItem(gdal.DMD_EXTENSIONS)
            if d_ext_str is not None:
                for d_ext in d_ext_str.split(' '):
                    if f_ext == '.' + d_ext:
                        return driver
    return None

def points2dsm(bbx, points, tif_to_write):
    # write dsm to tif
    ul_e = bbx['ul_easting']
    ul_n = bbx['ul_northing']

    e_resolution = 0.5  # 0.5 meters per pixel
    n_resolution = 0.5
    e_size = int(bbx['width'] / e_resolution) + 1
    n_size = int(bbx['height'] / n_resolution) + 1
    dsm = proj_to_grid(points, ul_e, ul_n, e_resolution, n_resolution, e_size, n_size)
    # median filter
    dsm = cv2.medianBlur(dsm.astype(np.float32), 3)
    write_dsm_tif(dsm, tif_to_write,
                  (ul_e, ul_n, e_resolution, n_resolution),
                  (bbx['zone_number'], bbx['hemisphere']), nodata_val=-9999)

    # create a preview file
    # if jpg_to_write is not None:
    #     dsm = np.clip(dsm, bbx['alt_min'], bbx['alt_max'])
    #     plot_height_map(dsm, jpg_to_write, save_cbar=True)

    return (ul_e, ul_n, e_size, n_size, e_resolution, n_resolution)

def hei2dsm(heimap, rpc, bbx, tif_to_write):
    '''
    :param heimap: np array
    :param rpc:
    :param bbx:
    :param tif_to_write:
    :return:
    '''
    H, W = heimap.shape
    device = heimap.device
    xx, yy = torch.meshgrid([torch.arange(0, W, dtype=torch.float32, device=device),
                               torch.arange(0, H, dtype=torch.float32, device=device)])


def write_dsm_tif(image, out_file, geo, utm_zone='S', nodata_val=None):
    assert (len(image.shape) == 2)  # image should only be 2D

    ul_e, ul_n, e_resolution, n_resolution = geo
    zone_number, hemisphere = utm_zone

    # replace nan with no_data
    if nodata_val is not None:
        image = image.copy()  # avoid modify source data
        image[np.isnan(image)] = nodata_val
    else:
        nodata_val = np.nan

    driver = get_driver(out_file)
    out = driver.Create(out_file, image.shape[1], image.shape[0], 1,
                        gdal_array.NumericTypeCodeToGDALTypeCode(np.float32))
    band = out.GetRasterBand(1)  # one-based index
    band.WriteArray(image.astype(np.float32), 0, 0)
    band.SetNoDataValue(nodata_val)
    band.FlushCache()

    # syntax for geotransform
    # geotransform[0] = top left x
    # geotransform[1] = w-e pixel resolution
    # geotransform[2] = 0
    # geotransform[3] = top left y
    # geotransform[4] = 0
    # geotransform[5] = n-s pixel resolution (negative value)
    out.SetGeoTransform((ul_e, e_resolution, 0, ul_n, 0, -n_resolution))

    srs = osr.SpatialReference();
    srs.SetProjCS('WGS84 / UTM zone {}{}'.format(zone_number, hemisphere));
    srs.SetWellKnownGeogCS('WGS84');
    srs.SetUTM(zone_number, hemisphere == 'N');
    out.SetProjection(srs.ExportToWkt())
    out.SetMetadata({'AREA_OR_POINT': 'Area'})

    del out


def GetH_MAX_MIN(rpc):
    """
    Get the max and min value of height based on rpc
    :return: hmax, hmin
    """
    hmax = rpc[4] + rpc[9]  # HEIGHT_OFF + HEIGHT_SCALE
    hmin = rpc[4] - rpc[9]  # HEIGHT_OFF - HEIGHT_SCALE

    return hmax, hmin

# all the points should lie on the same hemisphere and UTM zone
def latlon_to_eastnorh(lat, lon):
    # assume all the points are either on north or south hemisphere
    assert(np.all(lat >= 0) or np.all(lat < 0))

    if lat[0, 0] >= 0: # north hemisphere
        south = False
    else:
        south = True

    _, _, zone_number, _ = utm.from_latlon(lat[0, 0], lon[0, 0])

    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone_number, south=south)
    east, north = proj(lon, lat)
    return east, north

def eastnorth_to_latlon(east, north, zone_number, hemisphere):
    if hemisphere == 'N':
        south = False
    else:
        south = True

    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone_number, south=south)
    lon, lat = proj(east, north, inverse=True)
    return lat, lon

def GetSize(path):
    dataset = gdal.Open(path)
    if dataset == None:
        print("GDAL RasterIO Error: Opening" + path + " failed!")
        return

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    del dataset
    return width, height

def Read_Img(path, x_lu, y_lu, xsize, ysize):
    dataset = gdal.Open(path)
    if dataset == None:
        print("GDAL RasterIO Error: Opening" + path + " failed!")
        return

    data = dataset.ReadAsArray(x_lu, y_lu, xsize, ysize)

    del dataset

    return data

def init_dsm(path, invalid):
    width, height = GetSize(path)
    data = Read_Img(path, 0, 0, width, height)
    data += invalid

    gdal_write_to_tif(path, 0, 0, data)

def Create_DSM_File(out_path, border, xuint, yuint):
    width = int((border[2] - border[0]) / xuint)
    height = int((border[3] - border[1]) / yuint)

    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    driver.Create(out_path, width, height, 1, gdal.GDT_Float32)

    text_ = ""
    text_ += str(xuint) + "\n0\n0\n" + str(-yuint) + "\n" + str(border[0]) + "\n" + str(border[3])
    tfw_path = out_path.replace(".tif", ".tfw")
    with open(tfw_path, "w") as f:
        f.write(text_)

def gdal_write_to_tif(out_path, xlu, ylu, data):
    dataset = gdal.Open(out_path, gdal.GF_Write)
    if dataset == None:
        print("GDAL RasterIO Error: Opening" + out_path + " failed!")
        return

    # 判读数组维数
    if len(data.shape) == 3:
        im_bands = data.shape[0]
    else:
        im_bands = 1

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(data, xlu, ylu)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(data[i], xlu, ylu)
    del dataset

def proj_to_grid_np(points, xoff, yoff, xresolution, yresolution, xsize, ysize):
    row = np.floor((yoff - points[:, 1]) / xresolution).astype(dtype=np.int)
    col = np.floor((points[:, 0] - xoff) / yresolution).astype(dtype=np.int)

    points_group_idx = row * xsize + col
    points_val = points[:, 2]

    # remove points that lie out of the dsm boundary
    mask = ((row >= 0) * (col >= 0) * (row < ysize) * (col < xsize)) > 0
    points_group_idx = points_group_idx[mask]
    points_val = points_val[mask]

    # create a place holder for all pixels in the dsm
    group_idx = np.arange(xsize * ysize).astype(dtype=np.int)
    group_val = np.empty(xsize * ysize)
    group_val.fill(np.nan)

    # concatenate place holders with the real valuies, then aggregate
    group_idx = np.concatenate((group_idx, points_group_idx))
    group_val = np.concatenate((group_val, points_val))

    dsm = npg.aggregate(group_idx, group_val, func='nanmax', fill_value=np.nan)
    dsm = dsm.reshape((ysize, xsize))

    # try to fill very small holes
    dsm_new = dsm.copy()
    nan_places = np.argwhere(np.isnan(dsm_new))
    for i in range(nan_places.shape[0]):
        row = nan_places[i, 0]
        col = nan_places[i, 1]
        neighbors = []
        for j in range(row-1, row+2):
            for k in range(col-1, col+2):
                if j >= 0 and j < dsm_new.shape[0] and k >=0 and k < dsm_new.shape[1]:
                    val = dsm_new[j, k]
                    if not np.isnan(val):
                        neighbors.append(val)

        if neighbors:
            dsm[row, col] = np.median(neighbors)

    return dsm

def proj_to_grid(points, xoff, yoff, xresolution, yresolution, xsize, ysize):
    row = torch.floor((yoff - points[:, 1]) / xresolution).astype(dtype=torch.int)
    col = torch.floor((points[:, 0] - xoff) / yresolution).astype(dtype=torch.int)

    points_group_idx = row * xsize + col
    points_val = points[:, 2]

    # remove points that lie out of the dsm boundary
    mask = ((row >= 0) * (col >= 0) * (row < ysize) * (col < xsize)) > 0
    points_group_idx = points_group_idx[mask]
    points_val = points_val[mask]

    # create a place holder for all pixels in the dsm
    group_idx = torch.arange(xsize * ysize).astype(dtype=torch.int32)
    group_val = torch.empty(xsize * ysize)
    group_val.fill(torch.nan)

    # concatenate place holders with the real valuies, then aggregate
    group_idx = torch.concat((group_idx, points_group_idx))
    group_val = torch.concat((group_val, points_val))

    dsm = npg.aggregate(group_idx, group_val, func='nanmax', fill_value=torch.nan)
    dsm = dsm.reshape((ysize, xsize))

    # try to fill very small holes
    dsm_new = dsm.copy()
    nan_places = torch.argwhere(torch.isnan(dsm_new))
    for i in range(nan_places.shape[0]):
        row = nan_places[i, 0]
        col = nan_places[i, 1]
        neighbors = []
        for j in range(row-1, row+2):
            for k in range(col-1, col+2):
                if j >= 0 and j < dsm_new.shape[0] and k >=0 and k < dsm_new.shape[1]:
                    val = dsm_new[j, k]
                    if not torch.isnan(val):
                        neighbors.append(val)

        if neighbors:
            dsm[row, col] = torch.median(neighbors)

    return dsm

def produce_dsm_from_points(points, ul_e, ul_n, rd_e, rd_n, xunit, yunit):
    # write dsm to tif
    e_size = int((rd_e - ul_e) / xunit)
    n_size = int((ul_n - rd_n) / yunit)

    dsm = proj_to_grid(points, ul_e, ul_n, xunit, yunit, e_size, n_size)
    # median filter
    # dsm = np.zeros((n_size, e_size))
    dsm = cv2.medianBlur(dsm.astype(np.float32), 3)

    return dsm

def produce_dsm_from_points2(points, ul_e, ul_n, e_size, n_size, xunit, yunit):
    # write dsm to tif
    # e_size = int((rd_e - ul_e) / xunit)
    # n_size = int((ul_n - rd_n) / yunit)

    dsm = proj_to_grid(points, ul_e, ul_n, xunit, yunit, e_size, n_size)
    # median filter
    # dsm = np.zeros((n_size, e_size))
    dsm = cv2.medianBlur(dsm.astype(np.float32), 3)

    return dsm

class Ellipsoid:
    '''
    based on (Coordinate Conversions and Transformation including Formulas, pdf, page 15)

    For WGS84:
        --ellipsoid_a = 6378137.000
        --inverse_flattening = 298.257223563

    '''
    a = 0 # semi-major axis
    b = 0 # semi-minor axis
    inv_f = 0 # inverse flattening 1
    f = 0 # flattening
    e = 0 # eccentricity
    sec_e = 0 # second eccentricity

    def __init__(self, ellipsoid_a=6378137.000, inverse_flattening=298.257223563):
        self.a = ellipsoid_a
        self.inv_f = inverse_flattening
        self.f = 1.0/self.inv_f
        self.b = self.a * (1 - self.f)
        self.e = np.sqrt(2*self.f - self.f * self.f)
        self.sec_e = np.sqrt((self.e * self.e)/(1 - self.e * self.e))

    def show_All_Info(self):
        print("========Info. for the Ellipsoid Defined==========")
        print("--semi-major axis: ", self.a)
        print("--semi-minor axis: ", self.b)
        print("--inverse flattening: ", self.inv_f)
        print("--flattening: ", self.f)
        print("--eccentricity: ", self.e)
        print("--second eccentricity: ", self.sec_e)
        print("=================================================")

class Transverse_Mercator:
    """
    Transverse_Mercator Projection
    based on (Coordinate Conversions and Transformation including Formulas, pdf, page 43, USGS formula)
    """

    def __init__(self, dict, device, latitude_origin=0.0, longitude_origin=0.0,
                 scale_factor=1.0, False_Easting=500000.0, False_Northing=0.0):
        self.config = dict
        self.device = device
        self.M_PI = 3.14159265358979323846
        self.a = dict['ellipsoid_a']  # ellipsoid.a
        self.inv_f = dict['inv_f']  # ellipsoid.f
        self.f = 1.0 / self.inv_f
        self.b = self.a * (1 - self.f)
        self.e = np.sqrt(2 * self.f - self.f * self.f)
        self.sec_e = np.sqrt((self.e * self.e) / (1 - self.e * self.e))
        '''
            a = 0 # semi-major axis
            b = 0 # semi-minor axis
            inv_f = 0 # inverse flattening 1
            f = 0 # flattening
            e = 0 # eccentricity
            sec_e = 0 # second eccentricity
        '''
        self.xuint = dict['XUint']
        self.yuint = dict['YUint']
        self.xoverlap = 1 - dict['XOverlap']
        self.yoverlap = 1 - dict['YOverlap']
        self.invalid = dict['invalid']
        self.bxsize = dict["bxsize"]
        self.bysize = dict["bysize"]
        self.para = dict['para']
        self.lat0_org = latitude_origin
        self.lon0_org = longitude_origin
        self.lat0 = latitude_origin / 180 * self.M_PI
        self.lon0 = longitude_origin / 180 * self.M_PI
        self.k0 = scale_factor
        self.FE = False_Easting
        self.FN = False_Northing

        # self.dsm_x_size = int((self.border[2] - self.border[0]) / self.xuint)
        # self.dsm_y_size = int((self.border[3] - self.border[1]) / self.yuint)
        # self.dsm_size = [self.dsm_x_size, self.dsm_y_size]


    def Show_Info(self):
        print("========Info. for the Projection Defined==========")
        print("------------------- Ellipsoid: -------------------")
        print("---- semi-major axis: ", self.a)
        print("---- semi-minor axis: ", self.b)
        print("---- flattening: ", self.f)
        print("---- eccentricity: ", self.e)
        print("---- second eccentricity: ", self.sec_e)
        print("--------------------------------------------------")
        print("--------------- Projection Para.: ----------------")
        print("---- latitude origin: ", self.lat0_org)
        print("---- longitude origin: ", self.lon0_org)
        print("---- scale factor: ", self.k0)
        print("---- False Easting: ", self.FE)
        print("---- False Northing: ", self.FN)
        print("--------------------------------------------------")
        print("=================================================")


    def proj(self, pts, reverse=False):
        """
        :param pts:
        :param reverse: True, EastNorth2latlon; False, latlon2EastNorth
        :return:
        """
        shape = pts.shape
        reshaped_pts = pts.reshape(-1, 2)
        if reverse:
            output = self.EastNorth2latlon(reshaped_pts)
        else:
            output = self.latlon2EastNorth(reshaped_pts)
        return output.reshape(shape)

    def proj_cuda(self, pts, reverse=False):
        """
        :param pts:
        :param reverse: True, EastNorth2latlon; False, latlon2EastNorth
        :return:
        """
        shape = pts.shape
        reshaped_pts = pts.reshape(-1, 2)
        if reverse:
            output = self.EastNorth2latlon_cuda(reshaped_pts)
        else:
            output = self.latlon2EastNorth_cuda(reshaped_pts)
        return output.reshape(shape)

    def latlon2EastNorth(self, pts):
        """
        For the calculation of easting and northing from latitude and longitude
        pts:(N, 2)
        """
        lat = pts[:, 0]
        lon = pts[:, 1]

        lat = lat / 180 * self.M_PI
        lon = lon / 180 * self.M_PI

        # Calculate Then the meridional arc distance from equator to the projection origin (M0)
        e_2 = self.e * self.e
        e_4 = e_2 * e_2
        e_6 = e_2 * e_2 * e_2
        M0 = self.a * ((1 - e_2 / 4 - 3 * e_4 / 64 - 5 * e_6 / 256) * self.lat0 -
                       (3 * e_2 / 8 + 3 * e_4 / 32 + 45 * e_6 / 1024) * np.sin(2 * self.lat0) +
                       (15 * e_4 / 256 + 45*e_6/1024)*np.sin(4*self.lat0) -
                       (35*e_6/3072)*np.sin(6*self.lat0))

        # calculate T C A v M
        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)
        tan_lat = np.tan(lat)

        T = tan_lat * tan_lat
        C = e_2 * cos_lat * cos_lat / (1 - e_2)
        A = (lon - self.lon0) * cos_lat
        v = self.a / np.sqrt(1 - e_2*sin_lat*sin_lat)
        M = self.a * ((1 - e_2 / 4 - 3 * e_4 / 64 - 5 * e_6 / 256) * lat -
                       (3 * e_2 / 8 + 3 * e_4 / 32 + 45 * e_6 / 1024) * np.sin(2 * lat) +
                       (15 * e_4 / 256 + 45*e_6/1024)*np.sin(4*lat) -
                       (35*e_6/3072)*np.sin(6*lat))

        A2 = A * A
        A3 = A * A * A
        E = self.FE + self.k0 * v * (A + (1 - T + C) * A3 / 6 + (
                    5 - 18 * T + T * T + 72 * C - 58 * self.sec_e * self.sec_e) * A2 * A3 / 120)
        N = self.FN + self.k0 * (M - M0 + v * tan_lat * (
                A2 / 2 + (5 - T + 9 * C + 4 * C * C) * A2 * A2 / 24 + (
                61 - 58 * T + T * T + 600 * C - 330 * self.sec_e * self.sec_e) * A3 * A3 / 720))

        return np.stack((E, N), axis=-1)

    def latlon2EastNorth_cuda(self, pts):
        """
        For the calculation of easting and northing from latitude and longitude
        pts:(N, 2)
        """
        lat = pts[:, 0]
        lon = pts[:, 1]

        lat = lat / 180 * self.M_PI
        lon = lon / 180 * self.M_PI

        # Calculate Then the meridional arc distance from equator to the projection origin (M0)
        e_2 = self.e * self.e
        e_4 = e_2 * e_2
        e_6 = e_2 * e_2 * e_2
        M0 = self.a * ((1 - e_2 / 4 - 3 * e_4 / 64 - 5 * e_6 / 256) * self.lat0 -
                       (3 * e_2 / 8 + 3 * e_4 / 32 + 45 * e_6 / 1024) * torch.sin(2 * self.lat0) +
                       (15 * e_4 / 256 + 45 * e_6 / 1024) * torch.sin(4 * self.lat0) -
                       (35 * e_6 / 3072) * torch.sin(6 * self.lat0))

        # calculate T C A v M
        cos_lat = torch.cos(lat)
        sin_lat = torch.sin(lat)
        tan_lat = torch.tan(lat)

        T = tan_lat * tan_lat
        C = e_2 * cos_lat * cos_lat / (1 - e_2)
        A = (lon - self.lon0) * cos_lat
        v = self.a / torch.sqrt(1 - e_2*sin_lat*sin_lat)
        M = self.a * ((1 - e_2 / 4 - 3 * e_4 / 64 - 5 * e_6 / 256) * lat -
                       (3 * e_2 / 8 + 3 * e_4 / 32 + 45 * e_6 / 1024) * torch.sin(2 * lat) +
                       (15 * e_4 / 256 + 45 * e_6 / 1024) * torch.sin(4 * lat) -
                       (35 * e_6 / 3072) * torch.sin(6 * lat))

        A2 = A * A
        A3 = A * A * A
        E = self.FE + self.k0 * v * (A + (1 - T + C) * A3 / 6 + (
                    5 - 18 * T + T * T + 72 * C - 58 * self.sec_e * self.sec_e) * A2 * A3 / 120)
        N = self.FN + self.k0 * (M - M0 + v * tan_lat * (
                A2 / 2 + (5 - T + 9 * C + 4 * C * C) * A2 * A2 / 24 + (
                61 - 58 * T + T * T + 600 * C - 330 * self.sec_e * self.sec_e) * A3 * A3 / 720))

        return torch.stack((E, N), axis=-1)

    def EastNorth2latlon(self, pts):
        """
        The reverse formulas to convert Easting and Northing projected coordinates to latitude and longitude
        pts:(N, 2)
        """
        E = pts[:, 0]
        N = pts[:, 1]

        e_2 = self.e * self.e
        e_4 = e_2 * e_2
        e_6 = e_2 * e_2 * e_2
        M0 = self.a * ((1 - e_2 / 4 - 3 * e_4 / 64 - 5 * e_6 / 256) * self.lat0 -
                       (3 * e_2 / 8 + 3 * e_4 / 32 + 45 * e_6 / 1024) * np.sin(2 * self.lat0) +
                       (15 * e_4 / 256 + 45 * e_6 / 1024) * np.sin(4 * self.lat0) -
                       (35 * e_6 / 3072) * np.sin(6 * self.lat0))

        # calculate e1 u1 M1
        temp_e = np.sqrt(1 - self.e * self.e)
        e1 = (1 - temp_e) / (1 + temp_e)
        M1 = M0 + (N - self.FN) / self.k0
        u1 = M1 / (self.a * (1 - e_2 / 4 - 3 * e_4 / 64 - 5 * e_6 / 256))

        # calculate lat1
        e1_2 = e1*e1
        lat1 = u1 + (3 * e1 / 2 - 27 * e1_2 * e1 / 32) * np.sin(2 * u1) + (
                21 * e1_2 / 16 - 55 * e1_2 * e1_2 / 32) * np.sin(4 * u1) + (
                151 * e1_2 * e1 / 96) * np.sin(6 * u1) + (
                1097 * e1_2 * e1_2 / 512) * np.sin(8 * u1)

        temp = np.sqrt(1 - e_2 * np.sin(lat1) * np.sin(lat1))
        v1 = self.a / temp
        p1 = self.a * (1 - e_2) / (temp * temp * temp)
        T1 = np.tan(lat1) * np.tan(lat1)

        C1 = self.sec_e * np.cos(lat1)
        C1 = C1 * C1

        D = (E - self.FE) / (v1 * self.k0)

        # calculate lat, lon
        D2 = D * D
        D3 = D2 * D
        sece_2 = self.sec_e * self.sec_e

        lat = lat1 - (v1 * np.tan(lat1) / p1) * (
                    D2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * sece_2) * D2 * D2 / 24 + (
                        61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * sece_2 - 3 * C1 * C1) * D3 * D3 / 720)
        lon = self.lon0 + (D - (1 + 2 * T1 + C1) * D3 / 6 + (
                    5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * sece_2 + 24 * T1 * T1) * D2 * D3 / 120) / np.cos(lat1)

        lat = lat * 180 / self.M_PI
        lon = lon * 180 / self.M_PI

        return np.stack((lat, lon), axis=-1)

    def EastNorth2latlon_cuda(self, pts):
        """
        The reverse formulas to convert Easting and Northing projected coordinates to latitude and longitude
        pts:(N, 2)
        """
        E = pts[:, 0]
        N = pts[:, 1]

        e_2 = self.e * self.e
        e_4 = e_2 * e_2
        e_6 = e_2 * e_2 * e_2
        M0 = self.a * ((1 - e_2 / 4 - 3 * e_4 / 64 - 5 * e_6 / 256) * self.lat0 -
                       (3 * e_2 / 8 + 3 * e_4 / 32 + 45 * e_6 / 1024) * torch.sin(2 * self.lat0) +
                       (15 * e_4 / 256 + 45 * e_6 / 1024) * torch.sin(4 * self.lat0) -
                       (35 * e_6 / 3072) * torch.sin(6 * self.lat0))

        # calculate e1 u1 M1
        temp_e = torch.sqrt(1 - self.e * self.e)
        e1 = (1 - temp_e) / (1 + temp_e)
        M1 = M0 + (N - self.FN) / self.k0
        u1 = M1 / (self.a * (1 - e_2 / 4 - 3 * e_4 / 64 - 5 * e_6 / 256))

        # calculate lat1
        e1_2 = e1*e1
        lat1 = u1 + (3 * e1 / 2 - 27 * e1_2 * e1 / 32) * torch.sin(2 * u1) + (
                21 * e1_2 / 16 - 55 * e1_2 * e1_2 / 32) * torch.sin(4 * u1) + (
                151 * e1_2 * e1 / 96) * torch.sin(6 * u1) + (
                1097 * e1_2 * e1_2 / 512) * torch.sin(8 * u1)

        temp = torch.sqrt(1 - e_2 * torch.sin(lat1) * torch.sin(lat1))
        v1 = self.a / temp
        p1 = self.a * (1 - e_2) / (temp * temp * temp)
        T1 = torch.tan(lat1) * torch.tan(lat1)

        C1 = self.sec_e * torch.cos(lat1)
        C1 = C1 * C1

        D = (E - self.FE) / (v1 * self.k0)

        # calculate lat, lon
        D2 = D * D
        D3 = D2 * D
        sece_2 = self.sec_e * self.sec_e

        lat = lat1 - (v1 * torch.tan(lat1) / p1) * (
                    D2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * sece_2) * D2 * D2 / 24 + (
                        61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * sece_2 - 3 * C1 * C1) * D3 * D3 / 720)
        lon = self.lon0 + (D - (1 + 2 * T1 + C1) * D3 / 6 + (
                    5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * sece_2 + 24 * T1 * T1) * D2 * D3 / 120) / torch.cos(lat1)

        lat = lat * 180 / self.M_PI
        lon = lon * 180 / self.M_PI

        return torch.stack((lat, lon), axis=-1)

    def Get_Border_torch(self, rpc, h_max, h_min, imgH, imgW):

        if self.config['border']:
            x_min = self.config["ul_easting"]
            y_max = self.config["ul_northing"]
            width = self.config["dsm_easting_size"] * self.config['XUint']
            height = self.config["dsm_northing_size"] * self.config['YUint']
            x_max = x_min + width
            y_min = y_max - height

        else:

            X, Y, Hei = torch.meshgrid(torch.arange(imgW), torch.arange(imgH), torch.arange(int(h_min), int(h_max)))

            X = X.reshape(-1, 1).to(self.device).to(torch.float32)
            Y = Y.reshape(-1, 1).to(self.device).to(torch.float32)
            Hei = Hei.reshape(-1, 1).to(self.device).to(torch.float32)

            lat, lon = LOCALIZATION(rpc, X, Y, Hei)

            # print(lat, lon)
            geopts = torch.stack([lat, lon], axis=-1)
            Proj_pts = self.proj_cuda(geopts)
            Min = torch.min(Proj_pts, axis=0)
            Max = torch.max(Proj_pts, axis=0)


            x_min = Min[0]
            y_min = Min[1]
            x_max = Max[0]
            y_max = Max[1]
            # print([x_min, y_min, x_max, y_max])
        return [x_min, y_min, x_max, y_max]



    def getDSM(self, masked_height_map, src_rpc, outpath, x_size, y_size):
        '''
        :param masked_height_map: [H, W] on cuda
        :param src_rpc: [170,] on cuda
        :param outpath:
        :return:
        '''
        h_max, h_min = GetH_MAX_MIN(src_rpc)
        border = self.Get_Border(src_rpc, h_max, h_min, y_size, x_size)
        Create_DSM_File(outpath, border, self.xuint, self.yuint)
        init_dsm(outpath, self.invalid)

        height_map = masked_height_map
        # height_map = height_map[None, None, :, :]
        # src_rpc = src_rpc[None, :]
        H, W = height_map.shape
        heis = height_map.view(-1, 1)

        x, y = torch.meshgrid([torch.arange(0, W, dtype=torch.float32, device=height_map.device),
                               torch.arange(0, H, dtype=torch.float32, device=height_map.device)])

        y, x = y.contiguous(), x.contiguous()
        lines, samps = y.view(-1, 1), x.view(-1, 1)
        lat, lon = LOCALIZATION(src_rpc, samps, lines, heis)
        # RPC_Photo2Obj(x, y, height_map, ref_rpc, coef)
        lat = lat.view(H, W)
        lon = lon.view(H, W)

        geopts = torch.stack((lat, lon), axis=-1)
        projpts = self.proj_cuda(geopts, False)

        points = np.stack((projpts[:, 0], projpts[:, 1], height_map), axis=-1)

        dsm = produce_dsm_from_points(points, border[0], border[3], x_size,
                                      y_size, self.xuint, self.yuint)
        '''
            # write dsm to tif
            e_size = int((rd_e - ul_e) / xunit)
            n_size = int((ul_n - rd_n) / yunit)
        
            dsm = proj_to_grid(points, ul_e, ul_n, xunit, yunit, e_size, n_size)
            # median filter
            # dsm = np.zeros((n_size, e_size))
            dsm = cv2.medianBlur(dsm.astype(np.float32), 3)
        '''

        show_dsm = dsm.copy()
        dsm[np.isnan(dsm)] = self.invalid
        gdal_write_to_tif(outpath, 0, 0, dsm)



# def Test():
#     wgs84 = Ellipsoid()
#     proj = Transverse_Mercator(wgs84, 0, 123)
#     # proj.Show_Info()
#
#     pts = np.zeros((100, 192, 96, 2), dtype=np.float64)
#     pts[:, :, :, 0] = 29.267563
#     pts[:, :, :, 1] = 120.653181
#
#     import time
#
#     start = time.time()
#     out = proj.proj(pts)
#     back = proj.proj(out, reverse=True)
#     end = time.time()
#
#     print(out)
#     print(back)
#     print("finished in ", end - start, " s")
#
# if __name__ == "__main__":
#     wgs84 = Ellipsoid()
#     proj = Transverse_Mercator(wgs84, 0, 123)
#
#     pts = np.zeros((2, 2), dtype=np.float64)
#     pts[0, 0] = 29.267563
#     pts[0, 1] = 120.653181
#     pts[1, 0] = 29.26756264
#     pts[1, 1] = 120.65318143
#
#     proj_pts = proj.proj(pts, False)
#     distance = (proj_pts[0][0] - proj_pts[1][0]) * (proj_pts[0][0] - proj_pts[1][0]) + (
#                 proj_pts[0][1] - proj_pts[1][1]) * (proj_pts[0][1] - proj_pts[1][1])
#
#     print(np.sqrt(distance))
