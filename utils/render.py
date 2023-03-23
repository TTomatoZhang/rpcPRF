import torch
import torch.nn as nn
import numpy as np
from datasets.utils_pushbroom import *
# for test
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def find_clip(transparency_acc, weights, thres, DepthSample):
    '''
    :param transparency_acc: [B, nSample, C, H, W]
    :return:
    '''
    _, nSample, C, H, W = weights.shape
    T_a = transparency_acc.squeeze().permute(1, 2, 0).view(-1, 32).contiguous().flip(-1)
    vals = torch.ones((T_a.shape[0], 1)).to(T_a.device) * thres
    ids1 = torch.searchsorted(T_a, vals)
    ids1 = ids1.squeeze().view(H, W)
    Wts = weights.squeeze().permute(1, 2, 0).view(-1, nSample).contiguous().flip(-1)
    vals = torch.ones((Wts.shape[0], 1)).to(Wts.device) * thres
    ids2 = torch.searchsorted(Wts, vals)
    ids2 = ids2.squeeze().view(H, W)
    '''
    import matplotlib
    matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行
    import matplotlib.pyplot as plt
    
    plt.imshow(ids2.cpu().numpy())
    plt.show()
    '''
    ids1 = nSample - ids1
    ids2 = nSample - ids2
    # hei = DepthSample.squeeze().expand(H, W, nSample)
    hei1 = DepthSample.squeeze()[ids1]
    hei2 = DepthSample.squeeze()[ids2]
    # hei1[:, :] = DepthSample[ids1[:, :]]
    # hei2[:, :] = DepthSample[ids2[:, :]]
    return hei1, hei2

def alphaRenderingRPC(rgb_MPI, sigma_MPI, XYH_coor):
    """
    Rendering image, follow the equation of volume rendering process
    Args:
        rgb_MPI: rgb MPI representation, type:torch.Tensor, shape:[B, Nsample, 3, H, W]
        sigma_MPI: sigma MPI representation, type:torch.Tensor, shape:[B, Nsample, 1, H, W]
        XYH_coor: photo2obj coordinates in camera coordinate, shape:[B, Nsample, 3, H, W]

    Returns:
        rgb_syn: synthetic RGB image, type:torch.Tensor, shape:[B, 3, H, W]
        altitude_syn: synthetic height, type:torch.Tensor, shape:[B, 1, H, W]
        transparency_acc: accumulated transparency, type:torch.Tensor, shape:[B, Nsample, 1, H, W]
        weights: render weights in per plane and per pixel, type:torch.Tensor, shape:[B, Nsample, 1, H, W]

    """
    B, Nsample, _, H, W = sigma_MPI.shape
    XYH_coor_diff = XYH_coor[:, 1:, :, :, :] - XYH_coor[:, :-1, :, :, :]    # [B, Nsample-1, 1, H, W]
    XYH_coor_diff = torch.norm(XYH_coor_diff, dim=-1, keepdim=True)  # calculate distance, [B, Nsample-1, 3, H, W]
    XYH_coor_diff = torch.cat((XYH_coor_diff,
                               torch.full((B, 1, 1, H, W), fill_value=1e3, dtype=XYH_coor_diff.dtype, device=XYH_coor_diff.device)),
                              dim=1)    # [B, Nsample, H, W, 1]
    transparency = torch.exp(-sigma_MPI * XYH_coor_diff)    # [B, Nsample, 1, H, W]
    alpha = 1 - transparency    # [B, Nsample, 1, H, W]

    alpha_comp_cumprod = torch.cumprod(1 - alpha, dim=1)  # [B, Nsample, 1, H, W]
    preserve_ratio = torch.cat((torch.ones((B, 1, H, W, 1), dtype=alpha.dtype, device=alpha.device),
                                alpha_comp_cumprod[:, 0:Nsample-1, :, :, :]), dim=1)  # [B, Nsample, 1, H, W]
    weights = alpha * preserve_ratio  # [B, Nsample, 1, H, W]
    rgb_syn = torch.sum(weights * rgb_MPI, dim=1, keepdim=False)  # [B, 3, H, W]
    weights_sum = torch.sum(weights, dim=1, keepdim=False)  # [B, 1, H, W]
    altitude_syn = torch.sum(weights * XYH_coor[:, :, :, :, 2:], dim=1, keepdim=False) / (weights_sum + 1e-5)  # [B, 1, H, W]

    return rgb_syn, altitude_syn, transparency, weights


def planeVolumeRenderingRPC(rgb_MPI, sigma_MPI, xyz_coor):
    """
    Rendering image, follow the equation of volume rendering process
    Args:
        rgb_MPI: rgb MPI representation, type:torch.Tensor, shape:[B, ndepth, 3, H, W]
        sigma_MPI: sigma MPI representation, type:torch.Tensor, shape:[B, ndepth, 1, H, W]
        xyz_coor: pixel2camera coordinates in camera coordinate, shape:[B, ndepth, 3, H, W]

    Returns:
        rgb_syn: synthetic RGB image, type:torch.Tensor, shape:[B, 3, H, W]
        depth_syn: synthetic depth, type:torch.Tensor, shape:[B, 1, H, W]
        transparency_acc: accumulated transparency, type:torch.Tensor, shape:[B, ndepth, 1, height, width]
        weights: render weights in per plane and per pixel, type:torch.Tensor, shape:[B, ndepth, 1, height, width]

    """
    B, ndepth, _, height, width = sigma_MPI.shape
    xyz_coor_diff = xyz_coor[:, 1:, :, :, :] - xyz_coor[:, :-1, :, :, :]    # [B, ndepth-1, 3, height, width]
    xyz_coor_diff = torch.norm(xyz_coor_diff, dim=2, keepdim=True)  # calculate distance, [B, ndepth-1, 1, height, width]
    xyz_coor_diff = torch.cat((xyz_coor_diff,
                               torch.full((B, 1, 1, height, width), fill_value=1e3, dtype=xyz_coor_diff.dtype, device=xyz_coor_diff.device)),
                              dim=1)    # [B, ndepth, 1, height, width]
    transparency = torch.exp(-sigma_MPI * xyz_coor_diff)    # [B, ndepth, 1, height, width]
    alpha = 1 - transparency    # [B, ndepth, 1, height, width]

    transparency_acc = torch.cumprod(transparency + 1e-6, dim=1)    # [B, ndepth, 1, height, width]
    transparency_acc = torch.cat((torch.ones((B, 1, 1, height, width), dtype=transparency_acc.dtype, device=transparency_acc.device),
                                  transparency_acc[:, 0:-1, :, :, :]),
                                 dim=1) # [B, ndepth, 1, height, width]

    weights = transparency_acc * alpha  # [B, ndepth, 1, height, width]
    h = transparency * sigma_MPI
    # calculate rgb_syn, depth_syn
    rgb_syn = torch.sum(weights * rgb_MPI, dim=1, keepdim=False)    # [B, 3, height, width]
    weights_sum = torch.sum(weights, dim=1, keepdim=False)  # [B, 1, height, width]
    depth_syn = torch.sum(weights * xyz_coor[:, :, 2:, :, :], dim=1, keepdim=False) / (weights_sum + 1e-5)  # [B, 1, height, width]
    return rgb_syn, depth_syn, transparency_acc, weights


'''  sampling  '''
'''  uniform '''
def sampleAltitude(altitude_min, altitude_max, altitude_hypothesis_num):
    """
    Uniformly sample altitude from [inversed_altitude_max, inversed_altitude_max]
    Args:
        altitude_min: min altitude value, type:torch.Tensor, shape:[B,]
        altitude_max: max altitude value, type:torch.Tensor, shape:[B,]
        altitude_hypothesis_num: altitude hypothesis number, type:int

    Returns:
        altitude_sample: altitude sample, type:torch.Tensor, shape:[B, Nsample]
    """
    altitude_samples = []
    for i in range(altitude_min.shape[0]):
        altitude_samples.append(torch.linspace(start=altitude_min[i].item(), end=altitude_max[i].item(), steps=altitude_hypothesis_num, device=altitude_min.device))
    altitude_sample = torch.stack(altitude_samples, dim=0)    # [B, Nsample]
    return altitude_sample.flip(-1)


def sampleAltitudeInv(altitude_min, altitude_max, altitude_hypothesis_num):
    """
    Uniformly sample altitude from [inversed_altitude_max, inversed_altitude_max]
    Args:
        altitude_min: min altitude value, type:torch.Tensor, shape:[B,]
        altitude_max: max altitude value, type:torch.Tensor, shape:[B,]
        altitude_hypothesis_num: altitude hypothesis number, type:int

    Returns:
        altitude_sample: altitude sample, type:torch.Tensor, shape:[B, Nsample]
    """
    altitude_samples = []
    for i in range(altitude_min.shape[0]):
        altitude_samples.append(torch.linspace(start=1.0/altitude_min[i].item(), end=1.0/altitude_max[i].item(), steps=altitude_hypothesis_num, device=altitude_min.device))
    altitude_sample = torch.stack(altitude_samples, dim=0)    # [B, Nsample]
    # return 1.0 / altitude_sample.flip(-1)
    return 1.0 / altitude_sample

def sample_pdf(values, weights, Nsample):
    """
    draw samples from distribution approximated by values and weights.
    the probability distribution can be denoted as weights = p(values)
    :param values: Bx1xNxS
    :param weights: Bx1xNxS
    :param Nsample: number of sample to draw
    :return:
    """
    B, N, S = weights.size(0), weights.size(2), weights.size(3)
    assert values.size() == (B, 1, N, S)

    # convert values to bin edges
    bin_edges = (values[:, :, :, 1:] + values[:, :, :, :-1]) * 0.5  # Bx1xNxS-1
    bin_edges = torch.cat((values[:, :, :, 0:1],
                           bin_edges,
                           values[:, :, :, -1:]), dim=3)  # Bx1xNxS+1

    pdf = weights / (torch.sum(weights, dim=3, keepdim=True) + 1e-5)  # Bx1xNxS
    cdf = torch.cumsum(pdf, dim=3)  # Bx1xNxS
    cdf = torch.cat((torch.zeros((B, 1, N, 1), dtype=cdf.dtype, device=cdf.device),
                     cdf), dim=3)  # Bx1xNxS+1

    # uniform sample over the cdf values
    u = torch.rand((B, 1, N, Nsample), dtype=weights.dtype, device=weights.device)  # Bx1xNxNsample

    # get the index on the cdf array
    cdf_idx = torch.searchsorted(cdf, u, right=True)  # Bx1xNxNsample
    cdf_idx_lower = torch.clamp(cdf_idx-1, min=0)  # Bx1xNxNsample
    cdf_idx_upper = torch.clamp(cdf_idx, max=S)  # Bx1xNxNsample

    # linear approximation for each bin
    cdf_idx_lower_upper = torch.cat((cdf_idx_lower, cdf_idx_upper), dim=3)  # Bx1xNx(Nsamplex2)
    cdf_bounds_N2 = torch.gather(cdf, index=cdf_idx_lower_upper, dim=3)  # Bx1xNx(Nsamplex2)
    cdf_bounds = torch.stack((cdf_bounds_N2[..., 0:Nsample], cdf_bounds_N2[..., Nsample:]), dim=4)
    bin_bounds_N2 = torch.gather(bin_edges, index=cdf_idx_lower_upper, dim=3)  # Bx1xNx(Nsamplex2)
    bin_bounds = torch.stack((bin_bounds_N2[..., 0:Nsample], bin_bounds_N2[..., Nsample:]), dim=4)

    # avoid zero cdf_intervals
    cdf_intervals = cdf_bounds[:, :, :, :, 1] - cdf_bounds[:, :, :, :, 0] # Bx1xNxNsample
    bin_intervals = bin_bounds[:, :, :, :, 1] - bin_bounds[:, :, :, :, 0]  # Bx1xNxNsample
    u_cdf_lower = u - cdf_bounds[:, :, :, :, 0]  # Bx1xNxNsample
    # there is the case that cdf_interval = 0, caused by the cdf_idx_lower/upper clamp above, need special handling
    t = u_cdf_lower / torch.clamp(cdf_intervals, min=1e-5)
    t = torch.where(cdf_intervals <= 1e-4,
                    torch.full_like(u_cdf_lower, 0.5),
                    t)

    samples = bin_bounds[:, :, :, :, 0] + t*bin_intervals
    return samples


def GetXYHfromAltitudes(H, W, alt_sample, rpc):
    """
    Generate src-view planes 3D position XYH
    Args:
        H: rendered image height, type:int
        W: rendered image width, type:int
        alt_sample: altitude sample in src-view, corresponding to MPI planes' Z, type:torch.Tensor, shape:[B, Nsample]
        rpc: [170,]

    Returns:
        XYH_src: 3D position in src-RPC, type:torch.Tensor, shape:[B, Nsample, 3, H, W]

    """
    device = rpc.device
    # generate meshgrid for src-view.
    B, Nsample = alt_sample.shape
    alt_sample = alt_sample.view(B * Nsample).to(device)

    y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.double, device=device),
                           torch.arange(0, W, dtype=torch.double, device=device)])
    y, x = y.contiguous(), x.contiguous()
    z = torch.ones_like(x)  # cpu

    meshgrid_flat = torch.stack((x, y, z), dim=2).to(device)  # [H, W, 3]

    meshgrid = meshgrid_flat.unsqueeze(0).repeat(B*Nsample, 1, 1, 1).reshape(B*Nsample, H, W, 3).permute(1, 2, 3, 0)  # [H, W, 3, 32]
    meshgrid[:, :, 2, :] = meshgrid[:, :, 2, :] * alt_sample

    #meshgrid = meshgrid.contiguous().view(H*W*B*Nsample, 3)

    ''' 
    meshgrid0 = meshgrid.permute(3, 0, 1, 2)
    meshgrid1 = meshgrid0[5:10, :, :, :]
    meshgrid2 = meshgrid1.contiguous().view(-1, 3)
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    xyh = meshgrid2.detach().cpu().numpy()
    # xyh = xyh.reshape(-1, 3).detach().cpu().numpy()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xyh[:, 0], xyh[:, 1], xyh[:, 2], 'gray')
    fig.show()

    '''
    # line off [0], samp off [1], line scale[5], samp off [6]
    # todo: test whether need normalization
    # meshgrid[:, 0] -= rpc[1]  # sampoff
    # meshgrid[:, 1] -= rpc[0]  # lineoff
    # meshgrid[:, 0] /= rpc[6]
    # meshgrid[:, 1] /= rpc[5]

    # expand to Nsamples of depth level # [H, W, 3, 32]
    XYH_src = meshgrid.permute(3, 2, 0, 1).contiguous().view(B, Nsample, 3, H, W)  # [B, Nsample, 3, H, W] # right
                                                                  #[1, 32, 3, 262144], [1, 32, 1, 1]
    # for plot
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    ndepth = XYH_src.shape[1]
    XYZ = XYH_src.squeeze().view(ndepth, 3, -1).permute(0, 2, 1).detach().cpu().numpy()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(ndepth):
        xyh = XYZ[i, :, :]
        ax.plot3D(xyh[:, 0], xyh[:, 1], xyh[:, 2], 'gray')
    # xyh = meshgrid.reshape(-1, 3).detach().cpu().numpy()
    fig.show()
    '''
    return XYH_src


'''
obj and photo transformation
row: y: line
col: x: samp
'''
def GET_PLH_COEF(P, L, H):
    """
    :param PLH: standardized 3D position XYH for src_RPC, type:torch.Tensor, shape:[B * Nsample, 1, H * W]
    :return: coef: type:torch.Tensor, shape:[B * Nsample * H * W, 20]
    """
    device = P.device
    P, L, H = P.squeeze(), L.squeeze(), H.squeeze()
    n_num = P.shape[0]
    #coef = np.zeros((n_num, 20))
    with torch.no_grad():
        coef = torch.ones((n_num, 20), dtype=torch.double).to(device)
        # coef[:, 0] = 1.0
        #try:
        coef[:, 1] = L
        # except:
        #     print('a')
        coef[:, 2] = P
        coef[:, 3] = H
        coef[:, 4] = L * P
        coef[:, 5] = L * H
        coef[:, 6] = P * H
        coef[:, 7] = L * L
        coef[:, 8] = P * P
        coef[:, 9] = H * H
        coef[:, 10] = P * coef[:, 5]
        coef[:, 11] = L * coef[:, 7]
        coef[:, 12] = L * coef[:, 8]
        coef[:, 13] = L * coef[:, 9]
        coef[:, 14] = L * coef[:, 4]
        coef[:, 15] = P * coef[:, 8]
        coef[:, 16] = P * coef[:, 9]
        coef[:, 17] = L * coef[:, 5]
        coef[:, 18] = P * coef[:, 6]
        coef[:, 19] = H * coef[:, 9]
        # coef[:, 1] = L
        # coef[:, 2] = P
        # coef[:, 3] = H
        # coef[:, 4] = L * P
        # coef[:, 5] = L * H
        # coef[:, 6] = P * H
        # coef[:, 7] = L * L
        # coef[:, 8] = P * P
        # coef[:, 9] = H * H
        # coef[:, 10] = P * L * H
        # coef[:, 11] = L * L * L
        # coef[:, 12] = L * P * P
        # coef[:, 13] = L * H * H
        # coef[:, 14] = L * L * P
        # coef[:, 15] = P * P * P
        # coef[:, 16] = P * H * H
        # coef[:, 17] = L * L * H
        # coef[:, 18] = P * P * H
        # coef[:, 19] = H * H * H

    return coef

def PROJECTION(rpc, Lat, Lon, H):
    """
    From (lat: X, lon: Y, hei: Z) to (samp, line) using the direct rpc
    Args:
        rpc: tgt RPC
        XYH: 3D position for src_RPC, type:torch.Tensor, shape:[B, Nsample, 3,  H,  W]
    Returns:
        samplineheight_grid: shape:[B, Nsample, 3,  H, W]
    """
    device = rpc.device
    # B, Nsample, _, H_im, W_im = XYH.shape
    # XYH = XYH.permute(0, 1, 3, 4, 2)
    # XYH = XYH.reshape(-1, 3)
    # lat, lon, H = torch.hsplit(XYH, 3) # torch.split(XYH, [1,1,1], -1)


    with torch.no_grad():
        lat = Lat.clone()
        lon = Lon.clone()
        hei = H.clone()

        # lat -= rpc[2].view(-1, 1) # self.LAT_OFF
        # lat /= rpc[7].view(-1, 1) # self.LAT_SCALE
        #
        # lon -= rpc[3].view(-1, 1) # self.LONG_OFF
        # lon /= rpc[8].view(-1, 1) # self.LONG_SCALE
        #
        # hei -= rpc[4].view(-1, 1) # self.HEIGHT_OFF
        # hei /= rpc[9].view(-1, 1) # self.HEIGHT_SCALE

        lat -= rpc[2]  # self.LAT_OFF
        lat /= rpc[7]  # self.LAT_SCALE

        lon -= rpc[3]  # self.LONG_OFF
        lon /= rpc[8]  # self.LONG_SCALE

        hei -= rpc[4]  # self.HEIGHT_OFF
        hei /= rpc[9]  # self.HEIGHT_SCALE

        coef = GET_PLH_COEF(lat, lon, hei)

        # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
        samp = torch.sum(coef * rpc[50: 70].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[70:90].view(-1, 1, 20), dim=-1)
        line = torch.sum(coef * rpc[10: 30].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[30:50].view(-1, 1, 20), dim=-1)

        samp *= rpc[6]  # self.SAMP_SCALE
        samp += rpc[1]  # self.SAMP_OFF

        line *= rpc[5]  # self.LINE_SCALE
        line += rpc[0]  # self.LINE_OFF

        samp = samp.permute(1, 0)
        line = line.permute(1, 0)
    return samp, line  # col, row

def LOCALIZATION(rpc, S, L, H):
    """
    From (samp: S, line: L, hei: H) to (lat, lon) using the inverse rpc
    photo to object space
    Args:
        rpc: src RPC
        SLH: 3D position for src_RPC, type:torch.Tensor, shape: [B, Nsample, 3, H, W]
    Returns:
        XYH: 3D position for src_RPC in object space, type:torch.Tensor, shape: [B, Nsample, 3, H, W]
    """
    device = rpc.device
    with torch.no_grad():
        # torch.cuda.synchronize()
        # t0 = time.time()
        samp = S.clone()
        line = L.clone()
        hei = H.clone()

        samp -= rpc[1].view(-1, 1)  # self.SAMP_OFF

        samp /= rpc[6].view(-1, 1)  # self.SAMP_SCALE

        line -= rpc[0].view(-1, 1)  # self.LINE_OFF
        line /= rpc[5].view(-1, 1)  # self.LINE_SCALE

        hei -= rpc[4].view(-1, 1)  # self.HEIGHT_OFF
        hei /= rpc[9].view(-1, 1)  # self.HEIGHT_SCALE
        # t1 = time.time()
        coef = GET_PLH_COEF(samp, line, hei)
        # torch.cuda.synchronize()
        # t2 = time.time()

        # coef: (B, ndepth*H*W, 20) rpc[:, 90:110] (B, 20)
        lat = torch.sum(coef * rpc[90:110].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[110:130].view(-1, 1, 20), dim=-1)
        lon = torch.sum(coef * rpc[130:150].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[150:170].view(-1, 1, 20), dim=-1)

        # torch.cuda.synchronize()
        # t3 = time.time()
        lat *= rpc[7].view(-1, 1)
        lat += rpc[2].view(-1, 1)

        lon *= rpc[8].view(-1, 1)
        lon += rpc[3].view(-1, 1)
        lat = lat.permute(1, 0)
        lon = lon.permute(1, 0)
    return lat, lon

def project_src2tgt(src_rgb_syn, src_alt_syn, src_rpc, tgt_rpc):
    B, dim, H, W = src_rgb_syn.shape
    device = src_rpc.device
    #assert B == 1, "B must be 1"
    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.double, device=device),  # row: line
                               torch.arange(0, W, dtype=torch.double, device=device)])  # col: samp
        yv, xv = y.contiguous(), x.contiguous()  # [H, W, 1]
        xv = xv.view(-1, 1)  # [H * W, 1]
        yv = yv.view(-1, 1)  # [H * W, 1]
        h = src_alt_syn.squeeze().view(H*W, 1)  # [H, W, 1] penalize on the depth
        h = h.double()

        lat, lon = LOCALIZATION(src_rpc, xv, yv, h)  # [1, H*W]
        samp, line = PROJECTION(tgt_rpc, lat, lon, h)  # [1, H*W]

        samp = samp.float().permute(1,0)
        line = line.float().permute(1,0)

        tgt_x_normalized = samp / ((W - 1) / 2) - 1  # [1, H * W, 1]
        tgt_y_normalized = line / ((H - 1) / 2) - 1  # [1, H * W, 1]
        #tgt_x, tgt_y = tgt_x_normalized.view(H, W, 1), tgt_y_normalized.view(H, W, 1)
        tgt_grid = torch.stack((tgt_x_normalized, tgt_y_normalized), dim=-1)#.view(B, H, W, 2)   # [H * W, 2]
        # tgt_sampled_rgb = tgt_rgb_syn[:, tgt_y, tgt_x]  # [1, H, W]
    projected_tgt = torch.nn.functional.grid_sample(
        #tgt_rgb_syn,  # img.unsqueeze(0) # [B, C, H, W]
        src_rgb_syn.to(torch.float32),
        tgt_grid.unsqueeze(2),    # [1, H * W, 1, 2]
        align_corners=False,
        mode='bilinear', padding_mode='zeros').squeeze()
    projected_tgt = projected_tgt.view(dim, H, W).unsqueeze(0)  # [B, C, H, W]
    # mask
    tgt_grif = tgt_grid.view(H, W, 2)
    valid_mask_x = torch.logical_and(tgt_grid[:, :, 0] < W, tgt_grid[:, :, 0] > -1)  # [H * W, 2]
    valid_mask_y = torch.logical_and(tgt_grid[:, :, 1] < H, tgt_grid[:, :, 1] > -1)
    valid_mask = torch.logical_and(valid_mask_x, valid_mask_y)  # [B*Nsample, H_render, W_render]
    valid_mask = valid_mask.reshape(B, 1, H, W)  # [B, Nsample, H_render, W_render]

    return projected_tgt, valid_mask

def runCheck(pfmp1, pfmp2, img1, rpc1, rpc2):
    hei1 = torch.Tensor(load_pfm(pfmp1).copy())
    #hei2 = torch.Tensor(load_pfm(pfmp2).copy())
    img = img1.squeeze().permute(2,0,1).unsqueeze(0)
    H, W = hei1.shape
    # img1 = img1.reshape(H, W, 3)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32))
    yy, xx = xx.reshape(H*W), yy.reshape(H*W)
    hei_src = hei1.reshape(H*W)
    lat, lon = LOCALIZATION(rpc1.unsqueeze(-1), xx.unsqueeze(-1), yy.unsqueeze(-1), hei_src.unsqueeze(-1))
    samp, line = PROJECTION(rpc1, lat, lon, hei_src)

    samp = samp.float()
    line = line.float()

    proj_x_normalized = samp / ((W - 1) / 2) - 1
    proj_y_normalized = line / ((H - 1) / 2) - 1
    proj_x_normalized = proj_x_normalized.squeeze()
    proj_y_normalized = proj_y_normalized.squeeze()
    # proj_x_normalized = proj_x_normalized.view(B, Nsample, feat_dim, H_mpi, W_mpi)
    # proj_y_normalized = proj_y_normalized.view(B, Nsample, feat_dim, H_mpi, W_mpi)

    proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=-1)  # [B, Ndepth, H*W, 2]
    grid = proj_xy

    newim = torch.nn.functional.grid_sample(img, grid.view(1, H, W, 2), mode='bilinear',
                                   padding_mode='zeros')
    newim = newim.squeeze().permute(1,2,0)
    # newim2 = torch.zeros([H, W, 3], dtype=torch.float32)
    # mask = torch.zeros([H, W], dtype=torch.uint8)
    #
    # idx = 0
    # for i in range(len(new_x)):
    #     if new_x[i] < 0 or new_x[i] > W or new_y[i] < 0 or new_y[i] > H:
    #         idx += 1
    #         print('number{} no'.format(i))
    #         continue
    #     mask[int(new_y[i]), int(new_x[i])] = 1
    #     newim2[int(new_y[i]), int(new_x[i]), :] = img1[int(yy[i]), int(xx[i]), :]
    #     # print(img1[int(y[i]),int(x[i]),:])
    #     # print("完成第{}行".format(i))
    # print("废弃比例{}---{}".format(idx, len(new_x)))
    plt.imshow(newim)
    plt.show()
    return newim

def runCheckSelf(pfmp1, rpc1):
    hei1 = torch.Tensor(load_pfm(pfmp1).copy())
    H, W = hei1.shape
    xx, yy = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32))
    xx, yy = xx.reshape(H*W), yy.reshape(H*W)
    hei_src = hei1.reshape(H*W)
    lat, lon = LOCALIZATION(rpc1.unsqueeze(-1), xx.unsqueeze(-1), yy.unsqueeze(-1), hei_src.unsqueeze(-1))
    new_x, new_y = PROJECTION(rpc1, lat, lon, hei_src)
    e1 = torch.abs(xx-new_x.squeeze())
    e2 = torch.abs(yy-new_y.squeeze())
    return e1, e2


'''
view planes transformation, 3d but not in object space

SLH to SLH
'''

def warpMPIRPC(MPIxyz_src, altSample, src_RPC, tgt_RPC, H_render, W_render):
    """
    map MPI representation to tgt-RPC, sample planes along the rays
    Args:
        MPI_src: src-view MPI and XYH representation, type:torch.Tensor, shape:[B, Nsample, C：7, H, W]
        altSample: altitude sample [B, Nsample]
        H_render: rendered image/depth height, type:int
        W_render: rendered image/depth width, type:int

    Returns:
        MPI_XYH_tgt: tgt-view MPI and XYH representation, type:torch.Tensor, shape:[B, Nsample, C：7, H_render, W_render, ]
    """
    B, Nsample, feat_dim, H_mpi, W_mpi = MPIxyz_src.shape
    #Nsample = altSample.shape[1]

    device = MPIxyz_src.device

    # 0. generate the tgt feature grids
    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, H_mpi, dtype=torch.double, device=src_RPC.device),
                               torch.arange(0, W_mpi, dtype=torch.double, device=src_RPC.device)])
        y, x = y.contiguous(), x.contiguous()
        y = y.view(1, 1, H_mpi, W_mpi).repeat(B, Nsample, 1, 1, 1)  # (B, ndepth, 1, H, W)
        x = x.view(1, 1, H_mpi, W_mpi).repeat(B, Nsample, 1, 1, 1)
        h = altSample.view(B, Nsample, 1, 1, 1).repeat(1, 1, 1, H_render, W_render)  # [B, Nsample, 1, H_render, W_render]
        h = h.double().to(device)

        x = x.view(-1, 1)
        y = y.view(-1, 1)
        h = h.view(-1, 1)

        # 1. localization and projection
        # start = time.time()
        # lat, lon = RPC_Photo2Obj(x, y, h, ref_rpc, coef)  # [1, B]
        # samp, line = RPC_Obj2Photo(lat, lon, h, src_rpc, coef)  # (B, ndepth*H*W)
        lat, lon = LOCALIZATION(src_RPC, x, y, h)
        samp, line = PROJECTION(tgt_RPC, lat, lon, h)
        # samp, line = RPC_Obj2Photo(lat, lon, h, ref_rpc, coef)  # (B, ndepth*H*W)
        # end = time.time()

        # print(torch.mean(samp - x), torch.var(samp - x))
        # print(torch.mean(line - y), torch.var(line - y))

        samp = samp.float()
        line = line.float()

        proj_x_normalized = samp / ((W_mpi - 1) / 2) - 1
        proj_y_normalized = line / ((H_mpi - 1) / 2) - 1
        proj_x_normalized = proj_x_normalized.squeeze()
        proj_y_normalized = proj_y_normalized.squeeze()
        #proj_x_normalized = proj_x_normalized.view(B, Nsample, feat_dim, H_mpi, W_mpi)
        #proj_y_normalized = proj_y_normalized.view(B, Nsample, feat_dim, H_mpi, W_mpi)

        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=-1)  # [B, Ndepth, H*W, 2]
        tgt_grid = proj_xy.view(B * Nsample, H_mpi, W_mpi, 2)  # [B * Ndepth, H, W, 2]

    # src_grid = torch.cat((xv, yv, h), dim=2)  # [B, Nsample, 1, H_render, W_render]

    # latlonalt_grid = LOCALIZATION(src_RPC, src_grid)
    # proj_grid = PROJECTION(tgt_RPC, latlonalt_grid)
    """    
    import numpy as np
    import matplotlib.pyplot as plt
    # x_, y_, h_ = s0.squeeze().detach().cpu().numpy(), v0.squeeze().detach().cpu().numpy(), h.squeeze().detach().cpu().numpy()
    # x_, y_, h_ = lat.squeeze().detach().cpu().numpy(), lon.squeeze().detach().cpu().numpy(), h.squeeze().detach().cpu().numpy()
    # x_, y_, h_ = samp.squeeze().detach().cpu().numpy(), line.squeeze().detach().cpu().numpy(), h.squeeze().detach().cpu().numpy()
    # proj_x_normalized, proj_y_normalized = proj_x_normalized.view(B*Nsample*H_mpi*W_mpi), proj_y_normalized.view(B*Nsample*H_mpi*W_mpi)
    #x_, y_, h_ = proj_x_normalized.squeeze().detach().cpu().numpy(), proj_y_normalized.squeeze().detach().cpu().numpy(), h.squeeze().detach().cpu().numpy()
    # xyh = XYH_tgt.permute(0, 2, 3, 1)
    xyh = XYH_src_reshaped.permute(0, 2, 3, 1)
    xyh = xyh.reshape(-1, 3).detach().cpu().numpy()
    x_, y_, h_ = xyh[:, 0], xyh[:, 1], xyh[:, 2] 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_, y_, h_, 'gray')
    fig.show()
    """
    # extra: test the warping of tgt grid
    #XYH_src = MPIxyz_src[:, :, 4:, :, :].view(B * Nsample, 3, H_mpi, W_mpi)
    #XYH_tgt = torch.nn.functional.grid_sample(XYH_src, tgt_grid, padding_mode="border", align_corners=False)
    MPIxyz_src_reshaped = MPIxyz_src.view(B * Nsample, feat_dim, H_mpi, W_mpi)
    MPI_xyz_tgt = torch.nn.functional.grid_sample(MPIxyz_src_reshaped, tgt_grid, padding_mode="border", align_corners=False)
    # top focus: XYH_tgt[29:30, :, :, :], XYH_tgt[30:31, :, :, :],XYH_tgt[31:32, :, :, :]：a line
    MPI_xyz_tgt = MPI_xyz_tgt.view(B, Nsample, feat_dim, H_mpi, W_mpi)

    # 2. find tgt grid valid mask
    tgt_grid = tgt_grid.view(B, Nsample, 2, H_render, W_render)  # right in the coordinate
    valid_mask_x = torch.logical_and(tgt_grid[:, :, 0, :, :] < W_mpi, tgt_grid[:, :, 0, :, :] > -1)
    valid_mask_y = torch.logical_and(tgt_grid[:, :, 1, :, :] < H_mpi, tgt_grid[:, :, 1, :, :] > -1)
    valid_mask = torch.logical_and(valid_mask_x, valid_mask_y)  # [B*Nsample, H_render, W_render]
    valid_mask = valid_mask.reshape(B*Nsample, H_render, W_render)  # [B, Nsample, H_render, W_render]


    return MPI_xyz_tgt, valid_mask




''' from MPI to results: NVS & DSM'''
'''Renderer'''
def renderSrcViewRPC(rgb_MPI_src, sigma_MPI_src, XYH_src, use_alpha=False):
    """
    Render novel view using decoder output, rgb_MPI, sigma_MPI
    Args:
        rgb_MPI_src: decoder output, rgb MPI representation in src-view, type:torch.Tensor, shape:[B, Nsample, 3, H, W]
        sigma_MPI_src: decoder output, sigma MPI representation in src-view, type:torch.Tensor, shape:[B, Nsample, 1, H, W]
        XYH_src: src-view planes 3D position XYZ
    Returns:
        rgb_tgt_syn: rgb image rendered in tgt-view, type:torch.Tensor, shape:[B, 3, H, W]
        height_sample_tgt_syn: tgt height sample corresponding to height_hypothesis_src, type:torch.Tensor, shape:[B, 1, H, W]
        mask_tgt_syn: rendered mask in tgt-view, type:torch.Tensor, shape:[B, 1, H, W]
    """
    # device = rgb_MPI_src.device
    # batch_size, _, H_mpi, W_mpi, _ = rgb_MPI_src.shape
    #
    device = XYH_src.device
    torch.cuda.memory_summary()
    if not use_alpha:
        rgb_syn_src, altitude_syn_src, blended_weights, weights_src = planeVolumeRenderingRPC(rgb_MPI_src, sigma_MPI_src, XYH_src)
    else:
        rgb_syn_src, altitude_syn_src, transparency_acc_src, weights_src = alphaRenderingRPC(rgb_MPI_src, sigma_MPI_src, XYH_src)
        blended_weights = torch.zeros_like(rgb_MPI_src).to(device)
    return rgb_syn_src, altitude_syn_src, blended_weights, weights_src


def renderNovelViewRPC(rgb_MPI_src, sigma_MPI_src, XYH_src, altSample, src_RPC, tgt_RPC, H_render, W_render, use_alpha=False):
    """
    Render novel view using decoder output, rgb_MPI, sigma_MPI
    Args:
        rgb_MPI_src: decoder output, rgb MPI representation in src-view, type:torch.Tensor, shape:[B, Nsample, 3, H, W]
        sigma_MPI_src: decoder output, sigma MPI representation in src-view, type:torch.Tensor, shape:[B, Nsample, 1, H, W]
        XYH_src: the src
        altSample: the sampeld altitudes
        H_render: rendered image/height height, type:int
        W_render: rendered image/height width, type:int
        use_alpha: if alpha composition
    Returns:
        rgb_tgt_syn: rgb image rendered in tgt-view, type:torch.Tensor, shape:[B, 3, H, W]
        height_sample_tgt_syn: tgt height sample corresponding to height_hypothesis_src, type:torch.Tensor, shape:[B, 1, H, W]
        mask_tgt_syn: rendered mask in tgt-view, type:torch.Tensor, shape:[B, 1, H, W]
    """
    device = rgb_MPI_src.device
    B, Nsample, _, H_mpi, W_mpi = rgb_MPI_src.shape
    #assert rgb_MPI_src.shape[0] == sigma_MPI_src.shape[0] =

    # 0. transfer the rgba coordinates of src RPC camera to the tgt RPC camera
    #XYH_tgt = TieXYHSrc2Tgt(XYH_src, src_RPC, tgt_RPC)
    # 1. concat MPI features and the tgt MPI coordinate to be the MPI representations
    #MPI_XYH_src = torch.cat((rgb_MPI_src, sigma_MPI_src, XYH_tgt.float()), dim=2)   # [B, Nsample, 3+1+3, H_mpi, W_mpi]
    MPI_XYH_src = torch.cat((rgb_MPI_src, sigma_MPI_src, XYH_src.float()), dim=2)  # [B, Nsample, 3+1+3, H_mpi, W_mpi]
    # 2. sample in the tgt coordinates to get tgt MPI representations
    MPI_XYH_tgt, mask_tgt = warpMPIRPC(MPI_XYH_src, altSample, src_RPC, tgt_RPC, H_render, W_render) # [B, Nsample, 3+1+3, H_render, W_render], [B, Nsample, H_render, W_render]
    rgb_MPI_tgt = MPI_XYH_tgt[:, :, :3, :, :]   # [B, Nsample, 3, H_render, W_render]
    sigma_MPI_tgt = MPI_XYH_tgt[:, :, 3:4, :, :]    # [B, Nsample, 1, H_render, W_render]
    XYH_tgt_warped = MPI_XYH_tgt[:, :, 4:, :, :]    # [B, Nsample, 3, H_render, W_render]

    tgt_mask = torch.where(mask_tgt,
                           torch.ones((B, Nsample, H_render, W_render), dtype=torch.float32, device=device),
                           torch.zeros((B, Nsample, H_render, W_render), dtype=torch.float32, device=device))    # [B, Nsample, H, W]
    #tgt_warped_H = XYH_tgt_warped[:, :, -1:]    # [B, Nsample, 1, H_render, W_render]
    # check whether correct
    # tgt_MPI_sigma = torch.where(tgt_warped_H >= 0,
    #                             tgt_MPI_sigma,
    #                             torch.zeros_like(tgt_MPI_sigma, device=device)) # [B, Nsample, 1, H_render, W_render]
    if not use_alpha:
        tgt_rgb_syn, tgt_altitude_syn, tgt_blended_weights, tgt_weights = planeVolumeRenderingRPC(rgb_MPI_tgt, sigma_MPI_tgt, XYH_tgt_warped)
    else:
        tgt_rgb_syn, tgt_altitude_syn, tgt_blended_weights, tgt_weights = alphaRenderingRPC(rgb_MPI_tgt, sigma_MPI_tgt, XYH_tgt_warped)

    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    #xyh = XYH_tgt_warped.squeeze().permute(0, 2, 3, 1)
    #xyh = xyh.contiguous().view(-1, 3).detach().cpu().numpy()
    # xyh = xyh.reshape(-1, 3).detach().cpu().numpy()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xyh[:, 0], xyh[:, 1], xyh[:, 2], 'gray')
    fig.show()
    '''

    # tgt_mask = torch.sum(tgt_mask, dim=1, keepdim=True)  # [B, 1, H_render, W_render], when all plane is not visible, mask value equal to zero
    tgt_mask = torch.sum(tgt_mask, dim=1,
                         keepdim=True) / Nsample  # [B, 1, H_render, W_render], when all plane is not visible, mask value equal to zero
    return tgt_rgb_syn, tgt_altitude_syn, tgt_mask, tgt_weights


###############  for test ################
import matplotlib.pyplot as plt
def testRenderSrcView():
    pass

def testRenderNovelView():
    pass

def testWarpMPI(rpcpath1='', rpcpath2='', H=384, W=768):
    pass
    #MPI_xyz_tgt, valid_mask = warpMPIRPC(MPI_src=MPI_src, altSample=depthHyps, tgt_RP)


if __name__ == "__main__":
    # rpc1 = load_aug_rpc_tensor_from_txt('/home/pc/Desktop/TLC/test/rpc_aug/1/0000_0016_aug.rpc')
    # rpc2 = load_aug_rpc_tensor_from_txt('/home/pc/Desktop/TLC/test/rpc_aug/0/0000_0016_aug.rpc')
    #
    # img1 = torch.Tensor(cv2.imread('/home/pc/Desktop/TLC/test/image/1/base0000block0016.png'))


    #----------------------
    # rpc1 = load_aug_rpc_tensor_from_txt('/home/pc/Desktop/TLC/train/rpc_aug/2/0000_0025_aug.rpc')
    # rpc2 = load_aug_rpc_tensor_from_txt('/home/pc/Desktop/TLC/train/rpc_aug/0/0000_0025_aug.rpc')
    #
    # img1 = torch.Tensor(cv2.imread('/home/pc/Desktop/TLC/train/image/2/base0000block0025.png'))



    rpc1 = load_aug_rpc_tensor_from_txt('/home/pc/Documents/ztt/satdata/iarpa_mvs/Train_ex/MasterProvisional3/augrpc/60/2.txt')
    rpc2 = load_aug_rpc_tensor_from_txt('/home/pc/Documents/ztt/satdata/iarpa_mvs/Train_ex/MasterProvisional3/augrpc/60/23.txt')

    img1 = load_tensor_from_rgb_geotiff('/home/pc/Documents/ztt/satdata/iarpa_mvs/Train_ex/MasterProvisional3/img/60/2.tif')

    # hpath1 = '/home/pc/Desktop/TLC/test/newhei/1/base0000block0016.pfm'
    # hpath2 = '/home/pc/Desktop/TLC/test/newhei/0/base0000block0016.pfm'
    #runCheckSelf('/home/pc/Desktop/TLC/test/newhei/1/base0000block0016.pfm', rpc1)


    # hpath1 = '/home/pc/Desktop/TLC/train/newhei/2/base0000block0025.pfm'
    # hpath2 = '/home/pc/Desktop/TLC/train/newhei/0/base0000block0025.pfm'

    hpath1 = '/home/pc/Documents/ztt/satdata/iarpa_mvs/Train_ex/MasterProvisional3/alt/60/2.pfm'
    hpath2 = '/home/pc/Documents/ztt/satdata/iarpa_mvs/Train_ex/MasterProvisional3/alt/60/23.pfm'
    runCheck(hpath1, hpath2, img1, rpc1, rpc2)