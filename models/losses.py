"""
Loss Functions
"""
import torch
import torch.nn.functional as F
from math import exp
from utils.render import LOCALIZATION, PROJECTION
'''
supervision
'''
    

def compute_normal_by_depth(alt_syn, nei=1):
    '''
    :param alt_syn:  [B, 1, H, W]
    :param nei:
    :return:
    '''
    B, C, H, W = alt_syn.shape
    assert B == C == 1, "batchsize should be 1"
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=alt_syn.device), torch.arange(W, dtype=torch.float32, device=alt_syn.device))
    pts_3d_map = torch.stack([xx, yy, alt_syn.squeeze()], dim=-1)  # 1*128*160*3
    pts_3d_map = pts_3d_map.contiguous()

    ## shift the 3d pts map by nei along 8 directions
    pts_3d_map_ctr = pts_3d_map[nei:-nei, nei:-nei, :]
    pts_3d_map_x0 = pts_3d_map[nei:-nei, 0:-(2 * nei), :]
    pts_3d_map_y0 = pts_3d_map[0:-(2 * nei), nei:-nei, :]
    pts_3d_map_x1 = pts_3d_map[nei:-nei, 2 * nei:, :]
    pts_3d_map_y1 = pts_3d_map[2 * nei:, nei:-nei, :]
    pts_3d_map_x0y0 = pts_3d_map[0:-(2 * nei), 0:-(2 * nei), :]
    pts_3d_map_x0y1 = pts_3d_map[2 * nei:, 0:-(2 * nei), :]
    pts_3d_map_x1y0 = pts_3d_map[0:-(2 * nei), 2 * nei:, :]
    pts_3d_map_x1y1 = pts_3d_map[2 * nei:, 2 * nei:, :]

    ## generate difference between the central pixel and one of 8 neighboring pixels
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0  # 因为是求向量，所以不用除以相邻两点之间的距离
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

    ## flatten the diff to a #pixle by 3 matrix
    # pix_num = kitti_shape[0] * (kitti_shape[1]-2*nei) * (kitti_shape[2]-2*nei)
    pix_num = B * (H - 2 * nei) * (W - 2 * nei)
    # print(pix_num)
    # print(diff_x0.shape)
    diff_x0 = diff_x0.view(pix_num, 3)
    diff_y0 = diff_y0.view(pix_num, 3)
    diff_x1 = diff_x1.view(pix_num, 3)
    diff_y1 = diff_y1.view(pix_num, 3)
    diff_x0y0 = diff_x0y0.view(pix_num, 3)
    diff_x0y1 = diff_x0y1.view(pix_num, 3)
    diff_x1y0 = diff_x1y0.view(pix_num, 3)
    diff_x1y1 = diff_x1y1.view(pix_num, 3)

    ## calculate normal by cross product of two vectors
    normals0 = F.normalize(torch.cross(diff_x1, diff_y1))  # * tf.tile(normals0_mask[:, None], [1,3]) tf.tile=.repeat
    normals1 = F.normalize(torch.cross(diff_x0, diff_y0))  # * tf.tile(normals1_mask[:, None], [1,3])
    normals2 = F.normalize(torch.cross(diff_x0y1, diff_x0y0))  # * tf.tile(normals2_mask[:, None], [1,3])
    normals3 = F.normalize(torch.cross(diff_x1y0, diff_x1y1))  # * tf.tile(normals3_mask[:, None], [1,3])

    normal_vector = normals0 + normals1 + normals2 + normals3

    normal_vector = F.normalize(normal_vector)
    # normal_map = tf.reshape(tf.squeeze(normal_vector), [kitti_shape[0]]+[kitti_shape[1]-2*nei]+[kitti_shape[2]-2*nei]+[3])
    normal_map = normal_vector.view(B, H - 2 * nei, W - 2 * nei, 3)

    normal_map = F.pad(normal_map, (0, 0, nei, nei, nei, nei), "constant", 0)

    return normal_map


def compute_depth_by_normal(alt_syn, normal_map, tgt_image, nei=1):
    depth_init = alt_syn.clone()

    d2n_nei = 1  # normal_depth转化的时候的空边
    alt_syn = alt_syn[:, d2n_nei:-(d2n_nei), d2n_nei:-(d2n_nei)]
    normal_map = normal_map[:, d2n_nei:-(d2n_nei), d2n_nei:-(d2n_nei), :]

    # depth_dims = alt_syn.get_shape().as_list()
    B, C, H, W = alt_syn.shape


    y_ctr, x_ctr = torch.meshgrid(
        [torch.arange(d2n_nei, H + d2n_nei, dtype=torch.float32, device=normal_map.device),
         torch.arange(d2n_nei, W + d2n_nei, dtype=torch.float32, device=normal_map.device)])
    y_ctr, x_ctr = y_ctr.contiguous(), x_ctr.contiguous()

    x_ctr_tile = x_ctr.unsqueeze(0).repeat(B, 1, 1)  # B*H*width
    y_ctr_tile = y_ctr.unsqueeze(0).repeat(B, 1, 1)

    x0 = x_ctr_tile - d2n_nei
    y0 = y_ctr_tile - d2n_nei
    x1 = x_ctr_tile + d2n_nei
    y1 = y_ctr_tile + d2n_nei
    normal_x = normal_map[:, :, :, 0]
    normal_y = normal_map[:, :, :, 1]
    normal_z = normal_map[:, :, :, 2]


    # cx_tile = cx.unsqueeze(-1).unsqueeze(-1).repeat(1, H, W)
    # cy_tile = cy.unsqueeze(-1).unsqueeze(-1).repeat(1, H, W)
    # fx_tile = fx.unsqueeze(-1).unsqueeze(-1).repeat(1, H, W)
    # fy_tile = fy.unsqueeze(-1).unsqueeze(-1).repeat(1, H, W)

    numerator = (x_ctr_tile - cx_tile) / fx_tile * normal_x + (y_ctr_tile - cy_tile) / fy_tile * normal_y + normal_z
    denominator_x0 = (x0 - cx_tile) / fx_tile * normal_x + (y_ctr_tile - cy_tile) / fy_tile * normal_y + normal_z
    denominator_y0 = (x_ctr_tile - cx_tile) / fx_tile * normal_x + (y0 - cy_tile) / fy_tile * normal_y + normal_z
    denominator_x1 = (x1 - cx_tile) / fx_tile * normal_x + (y_ctr_tile - cy_tile) / fy_tile * normal_y + normal_z
    denominator_y1 = (x_ctr_tile - cx_tile) / fx_tile * normal_x + (y1 - cy_tile) / fy_tile * normal_y + normal_z
    denominator_x0y0 = (x0 - cx_tile) / fx_tile * normal_x + (y0 - cy_tile) / fy_tile * normal_y + normal_z
    denominator_x0y1 = (x0 - cx_tile) / fx_tile * normal_x + (y1 - cy_tile) / fy_tile * normal_y + normal_z
    denominator_x1y0 = (x1 - cx_tile) / fx_tile * normal_x + (y0 - cy_tile) / fy_tile * normal_y + normal_z
    denominator_x1y1 = (x1 - cx_tile) / fx_tile * normal_x + (y1 - cy_tile) / fy_tile * normal_y + normal_z

    mask_x0 = denominator_x0 == 0
    denominator_x0 = denominator_x0 + 1e-3 * mask_x0.float()
    mask_y0 = denominator_y0 == 0
    denominator_y0 = denominator_y0 + 1e-3 * mask_y0.float()
    mask_x1 = denominator_x1 == 0
    denominator_x1 = denominator_x1 + 1e-3 * mask_x1.float()
    mask_y1 = denominator_y1 == 0
    denominator_y1 = denominator_y1 + 1e-3 * mask_y1.float()
    mask_x0y0 = denominator_x0y0 == 0
    denominator_x0y0 = denominator_x0y0 + 1e-3 * mask_x0y0.float()
    mask_x0y1 = denominator_x0y1 == 0
    denominator_x0y1 = denominator_x0y1 + 1e-3 * mask_x0y1.float()
    mask_x1y0 = denominator_x1y0 == 0
    denominator_x1y0 = denominator_x1y0 + 1e-3 * mask_x1y0.float()
    mask_x1y1 = denominator_x1y1 == 0
    denominator_x1y1 = denominator_x1y1 + 1e-3 * mask_x1y1.float()

    # alt_syn_x0 = (F.sigmoid(numerator / denominator_x0 - 1.0) * 2.0 + 4.0) * alt_syn
    # alt_syn_y0 = (F.sigmoid(numerator / denominator_y0 - 1.0) * 2.0 + 4.0) * alt_syn
    # alt_syn_x1 = (F.sigmoid(numerator / denominator_x1 - 1.0) * 2.0 + 4.0) * alt_syn
    # alt_syn_y1 = (F.sigmoid(numerator / denominator_y1 - 1.0) * 2.0 + 4.0) * alt_syn

    alt_syn_x0 = numerator / denominator_x0 * alt_syn
    alt_syn_y0 = numerator / denominator_y0 * alt_syn
    alt_syn_x1 = numerator / denominator_y0 * alt_syn
    alt_syn_y1 = numerator / denominator_y0 * alt_syn
    alt_syn_x0y0 = numerator / denominator_x0y0 * alt_syn
    alt_syn_x0y1 = numerator / denominator_x0y1 * alt_syn
    alt_syn_x1y0 = numerator / denominator_x1y0 * alt_syn
    alt_syn_x1y1 = numerator / denominator_x1y1 * alt_syn

    # print(alt_syn_x0.shape) #4*126*158

    depth_x0 = depth_init
    depth_x0[:, d2n_nei:-(d2n_nei), :-(2 * d2n_nei)] = alt_syn_x0
    depth_y0 = depth_init
    depth_y0[:, 0:-(2 * d2n_nei), d2n_nei:-(d2n_nei)] = alt_syn_y0
    depth_x1 = depth_init
    depth_x1[:, d2n_nei:-(d2n_nei), 2 * d2n_nei:] = alt_syn_x1
    depth_y1 = depth_init
    depth_y1[:, 2 * d2n_nei:, d2n_nei:-(d2n_nei)] = alt_syn_y1
    depth_x0y0 = depth_init
    depth_x0y0[:, 0:-(2 * d2n_nei), 0:-(2 * d2n_nei)] = alt_syn_x0y0
    depth_x1y0 = depth_init
    depth_x1y0[:, 0:-(2 * d2n_nei), 2 * d2n_nei:] = alt_syn_x1y0
    depth_x0y1 = depth_init
    depth_x0y1[:, 2 * d2n_nei:, 0:-(2 * d2n_nei)] = alt_syn_x0y1
    depth_x1y1 = depth_init
    depth_x1y1[:, 2 * d2n_nei:, 2 * d2n_nei:] = alt_syn_x1y1

    # --------------------计算权重--------------------------
    tgt_image = tgt_image.permute(0, 2, 3, 1)
    tgt_image = tgt_image.contiguous()  # 4*128*160*3

    # print(alt_syn_x0.shape)  #4*124*156
    # normal_map = F.pad(normal_map,(0,0,nei,nei,nei,nei),"constant", 0)

    img_grad_x0 = tgt_image[:, d2n_nei:-d2n_nei, :-2 * d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    # print(img_grad_x0.shape) #4*126*158*3
    img_grad_x0 = F.pad(img_grad_x0, (0, 0, 0, 2 * d2n_nei, d2n_nei, d2n_nei), "constant", 1e-3)
    img_grad_y0 = tgt_image[:, :-2 * d2n_nei, d2n_nei:-d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_y0 = F.pad(img_grad_y0, (0, 0, d2n_nei, d2n_nei, 0, 2 * d2n_nei), "constant", 1e-3)
    img_grad_x1 = tgt_image[:, d2n_nei:-d2n_nei, 2 * d2n_nei:, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_x1 = F.pad(img_grad_x1, (0, 0, 2 * d2n_nei, 0, d2n_nei, d2n_nei), "constant", 1e-3)
    img_grad_y1 = tgt_image[:, 2 * d2n_nei:, d2n_nei:-d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_y1 = F.pad(img_grad_y1, (0, 0, d2n_nei, d2n_nei, 2 * d2n_nei, 0), "constant", 1e-3)

    img_grad_x0y0 = tgt_image[:, :-2 * d2n_nei, :-2 * d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_x0y0 = F.pad(img_grad_x0y0, (0, 0, 0, 2 * d2n_nei, 0, 2 * d2n_nei), "constant", 1e-3)
    img_grad_x1y0 = tgt_image[:, :-2 * d2n_nei, 2 * d2n_nei:, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_x1y0 = F.pad(img_grad_x1y0, (0, 0, 2 * d2n_nei, 0, 0, 2 * d2n_nei), "constant", 1e-3)
    img_grad_x0y1 = tgt_image[:, 2 * d2n_nei:, :-2 * d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_x0y1 = F.pad(img_grad_x0y1, (0, 0, 0, 2 * d2n_nei, 2 * d2n_nei, 0), "constant", 1e-3)
    img_grad_x1y1 = tgt_image[:, 2 * d2n_nei:, 2 * d2n_nei:, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_x1y1 = F.pad(img_grad_x1y1, (0, 0, 2 * d2n_nei, 0, 2 * d2n_nei, 0), "constant", 1e-3)

    # print(img_grad_x0.shape) #4*128*160*3

    alpha = 0.1
    weights_x0 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x0), 3))
    weights_y0 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_y0), 3))
    weights_x1 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x1), 3))
    weights_y1 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_y1), 3))

    weights_x0y0 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x0y0), 3))
    weights_x1y0 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x1y0), 3))
    weights_x0y1 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x0y1), 3))
    weights_x1y1 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x1y1), 3))

    # print(weights_x0.shape)    #4*128*160
    weights_sum = torch.sum(torch.stack(
        (weights_x0, weights_y0, weights_x1, weights_y1, weights_x0y0, weights_x1y0, weights_x0y1, weights_x1y1), 0), 0)

    # print(weights.shape) 4*128*160
    weights = torch.stack(
        (weights_x0, weights_y0, weights_x1, weights_y1, weights_x0y0, weights_x1y0, weights_x0y1, weights_x1y1),
        0) / weights_sum
    alt_syn_avg = torch.sum(
        torch.stack((depth_x0, depth_y0, depth_x1, depth_y1, depth_x0y0, depth_x1y0, depth_x0y1, depth_x1y1),
                    0) * weights, 0)

    return alt_syn_avg


def loss_normal_depth(alt_syn, src_img, nei=1):
    normal_by_depth = compute_normal_by_depth(alt_syn, nei)

    #normal_to_depth
    depth_by_normal = compute_depth_by_normal(alt_syn, normal_by_depth, src_img)
    loss_normal = 0

    if F.smooth_l1_loss(depth_by_normal, alt_syn).nelement() == 0:
        loss_normal += torch.tensor(0.)
    else:
        loss_normal += F.smooth_l1_loss(depth_by_normal, alt_syn)
    return loss_normal


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        # mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        try:
            mu2 = F.conv2d(img2, window.to(torch.float32), padding=window_size // 2, groups=channel)
        except:
            img2 = img2.to(torch.float32)
            mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d((img2 * img2).to(torch.float32), window.to(torch.float32), padding=window_size // 2, groups=channel) - mu2_sq.to(torch.float32)
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


def loss_fcn_rgb_SSIM(ssim_calculator, tgt_rgb_syn, tgt_mask, image_tgt):
    """
    Calculate SSIM loss between rgb syn and rgb groundtruth, only use neighbor-view rgb image to calculate loss
    Args:
        ssim_calculator: SSIM object instance, type:object-SSIM
        tgt_rgb_syn: tgt synthetic rgb images, type:torch.Tensor, shape:[B, 3, H, W]
        tgt_mask: tgt synthetic masks, type:torch.Tensor, shape:[B, 1, H, W], value:0or1
        image_tgt: tgt groundtruth rgb images, type:torch.Tensor, shape:[B, 3, H, W]

    Returns:
        loss_ssim: ssim loss between rgb_syn and rgb_gts

    """
    #assert tgt_mask.dtype == tgt_rgb_syn.dtype == image_tgt.dtype, "dtype different"

    loss_ssim = ssim_calculator(tgt_rgb_syn * tgt_mask, image_tgt * tgt_mask)

    return 1 - loss_ssim


def loss_fcn_rgb_lpips(lpips_calculator, tgt_rgb_syn, tgt_mask, image_tgt):
    """
    Calculate LPIPS loss between rgb syn and rgb groundtruth, only use neighbor-view rgb image to calculate loss
    Args:
        lpips_calculator: lpips.LPIPS object instance, type:object-LPIPS
        tgt_rgb_syn: tgt synthetic rgb images, type:torch.Tensor, shape:[B, 3, H, W]
        tgt_mask: tgt synthetic masks, type:torch.Tensor, shape:[B, 1, H, W], value:0or1
        image_tgt: tgt groundtruth rgb image, type:torch.Tensor, shape:[B, 3, H, W]

    Returns:
        loss_lpips: loss between rgb_syn and rgb_gts
    """
    loss_lpips = lpips_calculator((tgt_rgb_syn * tgt_mask).to(torch.float32), (image_tgt * tgt_mask).to(torch.float32)).mean()
    return loss_lpips


def loss_fcn_rgb_L1(tgt_rgb_syn, tgt_mask, image_tgt):
    """
    Calculate smooth-L1 loss between rgb syn and rgb groundtruth, only use neighbor-view rgb image to calculate loss
    Args:
        tgt_rgb_syn: tgt synthetic rgb images, type:torch.Tensor, shape:[B, 3, H, W]
        tgt_mask: tgt synthetic masks, type:torch.Tensor, shape:[B, 1, H, W], value:0or1
        image_tgt: tgt groundtruth rgb image, type:torch.Tensor, shape:[B, 3, H, W]

    Returns:
        loss_rgb: loss between rgb_syn and rgb_gts

    """
    # calculate tgt-view and ref-view L1 rgb loss, with mask
    if torch.nonzero(tgt_mask).shape[0] == 0:
        tgt_mask = torch.ones_like(image_tgt)
    loss_rgb = torch.sum(torch.abs(tgt_rgb_syn * tgt_mask - image_tgt * tgt_mask)) / torch.sum(tgt_mask)
    # loss_rgb = F.l1_loss(tgt_rgb_syn * tgt_mask, image_tgt * tgt_mask, reduction="mean")
    return loss_rgb


def loss_fcn_rgb_L2(tgt_rgb_syn, tgt_mask, image_tgt):
    """
    Calculate smooth-L1 loss between rgb syn and rgb groundtruth, only use neighbor-view rgb image to calculate loss
    Args:
        tgt_rgb_syn: tgt synthetic rgb images, type:torch.Tensor, shape:[B, 3, H, W]
        tgt_mask: tgt synthetic masks, type:torch.Tensor, shape:[B, 1, H, W], value:0or1
        image_tgt: tgt groundtruth rgb image, type:torch.Tensor, shape:[B, 3, H, W]

    Returns:
        loss_rgb: loss between rgb_syn and rgb_gts

    """
    # calculate tgt-view and ref-view L1 rgb loss, with mask
    loss_rgb = torch.sum((tgt_rgb_syn * tgt_mask - image_tgt * tgt_mask) ** 2) / torch.sum(tgt_mask)
    # loss_rgb = F.l1_loss(tgt_rgb_syn * tgt_mask, image_tgt * tgt_mask, reduction="mean")
    return loss_rgb

def mse2(image_pred, image_gt, valid_mask=None, reduction='mean'):
    if valid_mask is not None:
        image_pred = image_pred * valid_mask
    value = (image_pred-image_gt)**2

    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr2(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse2(image_pred, image_gt, valid_mask, reduction))


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value * valid_mask
        #value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def loss_fcn_edge_aware(ref_depth_syn, image_ref, depth_min_ref, depth_max_ref):
    """
    Calculate edge-aware loss between depth syn and rgb groundtruth, only use neighbor-view rgb image to calculate loss
    Args:
        ref_depth_syn: ref synthetic depth, type:torch.Tensor, shape:[B, 1, H, W]
        image_ref: ref-view groundtruth rgb image, type:torch.Tensor, shape:[B, 3, H, W]
        depth_min_ref: depth min value, type:torch.Tensor, shape:[B,]
        depth_max_ref: depth max value, type:torch.Tensor, shape:[B,]

    Returns:
        loss_edge: loss between depth syn and rgb groundtruth
    """
    device = ref_depth_syn.device
    depth_min_ref = depth_min_ref.unsqueeze(1).unsqueeze(2).repeat(1, 1, ref_depth_syn.shape[0],
                                                                                ref_depth_syn.shape[1]).to(device)
    depth_max_ref = depth_max_ref.unsqueeze(1).unsqueeze(2).repeat(1, 1, ref_depth_syn.shape[0],
                                                                                ref_depth_syn.shape[1]).to(device)
    #try:
    ref_depth_syn = (ref_depth_syn - depth_min_ref) / (depth_max_ref - depth_min_ref)
    # except:
    #     print('wrong')
    ref_depth_syn = ref_depth_syn.squeeze(0)
    # calculate depth gradient
    grad_depth_x = torch.abs(ref_depth_syn[:, :, :-1] - ref_depth_syn[:, :, 1:])  # [B, 1, H, W-1]
    grad_depth_y = torch.abs(ref_depth_syn[:, :-1, :] - ref_depth_syn[:, 1:, :])  # [B, 1, H, W-1]
    # calculate image gradient
    grad_image_x = torch.mean(torch.abs(image_ref[:, :, :-1] - image_ref[:, :, 1:]), 1, keepdim=True)  # [B, 1, H, W-1]
    grad_image_y = torch.mean(torch.abs(image_ref[:, :-1, :] - image_ref[:, 1:, :]), 1, keepdim=True)  # [B, 1, H, W-1]

    loss_edge = torch.mean(grad_depth_x * torch.exp(-grad_image_x)) + torch.mean(
        grad_depth_y * torch.exp(-grad_image_y))
    return loss_edge


def edge_aware_loss_v2(img, depth):
    """Computes the smoothness loss for a deptharity image
    The color image is used for edge-aware smoothness
    """
    mean_depth = depth.mean(2, True).mean(3, True)
    depth = depth / (mean_depth + 1e-7)

    grad_depth_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    grad_depth_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)

    return grad_depth_x.mean() + grad_depth_y.mean()

def gather_pixel_by_pxpy(img, pxpy):
    """
    :param img: Bx3xHxW
    :param pxpy: Bx2xN
    :return:
    """
    with torch.no_grad():
        B, C, H, W = img.size()
        if pxpy.dtype == torch.float32:
            pxpy_int = torch.round(pxpy).to(torch.int64)
        else:
            pxpy_int = pxpy.to(torch.int64)
        pxpy_int[:, 0, :] = torch.clamp(pxpy_int[:, 0, :], min=0, max=W-1)
        pxpy_int[:, 1, :] = torch.clamp(pxpy_int[:, 1, :], min=0, max=H-1)
        pxpy_idx = pxpy_int[:, 0:1, :] + W * pxpy_int[:, 1:2, :]  # Bx1xN_pt
    rgb = torch.gather(img.view(B, C, H * W), dim=2,
                       index=pxpy_idx.repeat(1, C, 1).to(img.device))  # BxCxN_pt
    return rgb



def loss_pts(pts_syn, pts_gt):
    torch.mean(torch.abs(
        pts_syn - pts_gt
    ))

def loss_pts_log(alt_syn, alt_gt):
    loss = torch.mean(torch.abs(
        torch.log(1/alt_syn) - torch.log(1/alt_gt)
    ))
    return loss

def loss_reproject_pts2img(rgb_syn, rgb_gt, alt_selected, src_rpc, tgt_rpc, pts_id, H, W):
    '''

    :param rgb_syn:
    :param rgb_gt:
    :param pts_id: Bx2xN
    :return:
    '''
    B,_,N = pts_id.shape
    ids = pts_id.permute(0,2,1).view(-1, 2)  # BxN,2
    xv, yv = torch.hsplit(ids, 2)
    h = alt_selected.squeze().unsqueeze(-1)
    lat, lon = LOCALIZATION(src_rpc, xv, yv, h)  # [1, H*W]
    samp, line = PROJECTION(tgt_rpc, lat, lon, h)  # [1, H*W]

    samp = samp.float()
    line = line.float()

    tgt_x = samp / ((W - 1) / 2) - 1  # [1, H * W, 1]
    tgt_y = line / ((H - 1) / 2) - 1  # [1, H * W, 1]
    tgt_grid = torch.stack((tgt_x, tgt_y), dim=-1)

    warped = F.grid_sample()



    pass


'''
depth loss
'''
def loss_reproject_src2tgt(src_rgb_syn, src_alt_syn, tgt_image, src_rpc, tgt_rpc):
    '''
    :param src_rgb_syn: [B, 3, H, W]
    src_alt_syn: [B, 1, H, W]
    :param tgt_rgb_syn: [B, 3, H, W]
    :param src_rpc: [B, 170]
    :param tgt_rpc: [B, 170]
    :return:
    '''

    B, dim, H, W = src_rgb_syn.shape
    device = src_rpc.device
    #assert B == 1, "B must be 1"
    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.double, device=device),
                               torch.arange(0, W, dtype=torch.double, device=device)])
        yv, xv = y.contiguous(), x.contiguous()  # [H, W, 1]
        xv = xv.view(-1, 1)  # [H * W, 1]
        yv = yv.view(-1, 1)  # [H * W, 1]
        h = src_alt_syn.squeeze(0).permute(1, 2, 0).view(-1, 1)  # [H, W, 1] penalize on the depth
        h = h.double()

        lat, lon = LOCALIZATION(src_rpc, xv, yv, h)  # [1, H*W]
        samp, line = PROJECTION(tgt_rpc, lat, lon, h)  # [1, H*W]

        samp = samp.float()
        line = line.float()

        tgt_x_normalized = samp / ((W - 1) / 2) - 1  # [1, H * W, 1]
        tgt_y_normalized = line / ((H - 1) / 2) - 1  # [1, H * W, 1]

        tgt_x, tgt_y = tgt_x_normalized.view(H, W), tgt_y_normalized.view(H, W)
        tgt_grid = torch.stack((tgt_x, tgt_y), dim=-1)    # [H, W, 2]
        '''
        tgt_h = h.view(H * W, 1)
        tgtxyz = torch.cat((tgt_grid.view(H * W, 2), tgt_h), dim=-1)
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['agg.path.chunksize'] = 10000
        xyh = tgtxyz.detach().cpu().numpy()
        # xyh = xyh.reshape(-1, 3).detach().cpu().numpy()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(xyh[:, 0], xyh[:, 1], xyh[:, 2], 'gray')
        fig.show()
        '''
        projected_src2tgt = F.grid_sample(
            src_rgb_syn.to(torch.float32),  # img.unsqueeze(0) # [B, C, H, W] = [1, 3, H, W]
            tgt_grid.unsqueeze(0).to(torch.float32),    # [1, H, W, 2]
            align_corners=False,
            mode='bilinear', padding_mode='zeros').squeeze()
    projected_src2tgt = projected_src2tgt.view(dim, H, W)
    '''
        tgtim = projected_src2tgt.squeeze().permute(1,2,0).detach().cpu().numpy()
        maskim = masked_tgt_rgb_syn.squeeze().permute(1,2,0).detach().cpu().numpy()
    '''
    loss_pixels = F.smooth_l1_loss(projected_src2tgt, tgt_image.squeeze().to(torch.float32))
    return loss_pixels


def compute_depth_errors(gt, pred, min_depth=1e-3, max_depth=80):
    """Computation of error metrics between predicted and ground truth depths
    """
    pred[pred < min_depth] = min_depth
    pred[pred > max_depth] = max_depth

    thresh = torch.maximum((gt / pred), (pred / gt))
    # a1 = (thresh < 1.25     ).mean()
    # a2 = (thresh < 1.25 ** 2).mean()
    # a3 = (thresh < 1.25 ** 3).mean()
    a1 = (thresh < 1     ).mean()
    a2 = (thresh < 2.5).mean()
    a3 = (thresh < 7.5).mean()


    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def loss_reproject_tgt2src(tgt_rgb_syn, tgt_alt_syn, src_image, tgt_rpc, src_rpc):
    '''
    :param src_rgb_syn: [B, 3, H, W]
    src_alt_syn: [B, 1, H, W]
    :param tgt_rgb_syn: [B, 3, H, W]
    :param src_rpc: [B, 170]
    :param tgt_rpc: [B, 170]
    :return:
    '''

    B, dim, H, W = tgt_rgb_syn.shape
    device = src_rpc.device
    #assert B == 1, "B must be 1"
    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.double, device=device),
                               torch.arange(0, W, dtype=torch.double, device=device)])
        yv, xv = y.contiguous(), x.contiguous()  # [H, W, 1]
        xv = xv.view(-1, 1)  # [H * W, 1]
        yv = yv.view(-1, 1)  # [H * W, 1]
        h = tgt_alt_syn.squeeze(0).permute(1, 2, 0).view(-1, 1)  # [H, W, 1] penalize on the depth
        h = h.double()

        lat, lon = LOCALIZATION(src_rpc, xv, yv, h)  # [1, H*W]
        samp, line = PROJECTION(tgt_rpc, lat, lon, h)  # [1, H*W]

        samp = samp.float()
        line = line.float()

        src_x_normalized = samp / ((W - 1) / 2) - 1  # [1, H * W, 1]
        src_y_normalized = line / ((H - 1) / 2) - 1  # [1, H * W, 1]

        src_x, src_y = src_x_normalized.view(H, W), src_y_normalized.view(H, W)
        src_grid = torch.stack((src_x, src_y), dim=-1)    # [H, W, 2]
        '''
        tgt_h = h.view(H * W, 1)
        tgtxyz = torch.cat((tgt_grid.view(H * W, 2), tgt_h), dim=-1)
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['agg.path.chunksize'] = 10000
        xyh = tgtxyz.detach().cpu().numpy()
        # xyh = xyh.reshape(-1, 3).detach().cpu().numpy()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(xyh[:, 0], xyh[:, 1], xyh[:, 2], 'gray')
        fig.show()
        '''
        projected_tgt2src = F.grid_sample(
            tgt_rgb_syn,  # img.unsqueeze(0) # [B, C, H, W] = [1, 3, H, W]
            src_grid.unsqueeze(0).float(),    # [1, H, W, 2]
            align_corners=False,
            mode='bilinear', padding_mode='zeros').squeeze()
        projected_tgt2src = projected_tgt2src.view(dim, H, W)
    '''
        tgtim = projected_tgt2src.squeeze().permute(1,2,0).detach().cpu().numpy()
        maskim = masked_tgt_rgb_syn.squeeze().permute(1,2,0).detach().cpu().numpy()
    '''
    loss_pixels = F.smooth_l1_loss(projected_tgt2src, src_image.squeeze())
    return loss_pixels


def loss_reproject_src2src(src_rgb_syn, src_alt_syn, src_image, src_rpc):
    '''
    :param src_rgb_syn: [B, 3, H, W]
    src_alt_syn: [B, 1, H, W]
    :param tgt_rgb_syn: [B, 3, H, W]
    :param src_rpc: [B, 170]
    :param tgt_rpc: [B, 170]
    :return:
    '''

    B, dim, H, W = src_rgb_syn.shape
    device = src_rpc.device
    img1 = src_image.squeeze().permute(1, 2, 0)
    #assert B == 1, "B must be 1"
    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.double, device=device),
                               torch.arange(0, W, dtype=torch.double, device=device)])
        yv, xv = y.contiguous(), x.contiguous()  # [H, W, 1]
        xv = xv.view(-1, 1)  # [H * W, 1]
        yv = yv.view(-1, 1)  # [H * W, 1]
        h = src_alt_syn.squeeze(0).permute(1, 2, 0).view(-1, 1)  # [H, W, 1] penalize on the depth
        h = h.double()

        lat, lon = LOCALIZATION(src_rpc, xv, yv, h)  # [1, H*W]
        samp, line = PROJECTION(src_rpc, lat, lon, h)  # [1, H*W]

        new_x = samp.float().squeeze()
        new_y = line.float().squeeze()

        newim = torch.zeros([H, W, 3], dtype=torch.float32).to(device)
        mask = torch.zeros([H, W], dtype=torch.uint8)

        idx = 0
        for i in range(len(new_x)):
            if new_x[i] < 0 or new_x[i] > W or new_y[i] < 0 or new_y[i] > H:
                idx += 1
                #print('number{} no'.format(i))
                continue
            mask[int(new_y[i]), int(new_x[i])] = 1
            newim[int(new_y[i]), int(new_x[i]), :] = img1[int(yv[i]), int(xv[i]), :]
        '''
        tgt_h = h.view(H * W, 1)
        tgtxyz = torch.cat((tgt_grid.view(H * W, 2), tgt_h), dim=-1)
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['agg.path.chunksize'] = 10000
        xyh = tgtxyz.detach().cpu().numpy()
        # xyh = xyh.reshape(-1, 3).detach().cpu().numpy()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(xyh[:, 0], xyh[:, 1], xyh[:, 2], 'gray')
        fig.show()
        '''
    '''
        tgtim = projected_src2tgt.squeeze().permute(1,2,0).detach().cpu().numpy()
        maskim = masked_tgt_rgb_syn.squeeze().permute(1,2,0).detach().cpu().numpy()
    '''
    loss_pixels = F.smooth_l1_loss(src_rgb_syn.squeeze().permute(1, 2, 0), newim.squeeze())
    return loss_pixels


def sample_grid_features0(grid, img):
    """
    grid: [B*H*W, 2] # the 2 columns store x(samp), y(line) coords normalized
    img: [B, C, H, W]
    -------------
    return
    color_bilinear: 3, B
    """
    grid = grid.float()
    grid_t = torch.ones_like(grid)  # [B*H*W, 2]
    grid_t[:, 0] = (grid[:, 0] / (img.shape[2] - 1) - 0.5) * 2  # [B*H*W, 2]
    grid_t[:, 1] = (grid[:, 1] / (img.shape[1] - 1) - 0.5) * 2  # [B*H*W, 2]

    color_bilinear = F.grid_sample(
        img,  # img.unsqueeze(0) # [B, C, H, W]
        grid_t.unsqueeze(0).unsqueeze(2).float(),
        align_corners=False,
        mode='bilinear', padding_mode='zeros').squeeze()

    return color_bilinear


def Thres_metrics(depth_syn, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_syn, depth_gt = depth_syn * mask, depth_gt * mask
    errors = torch.abs(depth_syn - depth_gt)
    err_mask = errors < thres
    return torch.mean(err_mask.float())


def Inter_metrics(depth_syn, depth_gt, interval, mask, thres):
    assert isinstance(thres, (int, float))
    depth_syn, depth_gt = depth_syn * mask, depth_gt * mask
    errors = torch.abs((depth_syn - depth_gt))
    errors = errors / interval
    err_mask = errors < thres
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
def AbsDepthError_metrics(depth_syn, depth_gt, mask, thres=10.0):
    depth_syn, depth_gt = depth_syn * mask, depth_gt * mask
    diff = (depth_syn - depth_gt).abs()
    mask2 = (diff < thres)
    result = diff[mask2]
    #result = torch.sum(result)/torch.sum(mask2*mask1)
    #return torch.mean((depth_syn - depth_gt).abs())
    return torch.mean(result)


if __name__ == '__main__':
    batch_size, height, width = 2, 512, 512
    neighbor_view_num = 4
    tgt_rgb_syn = torch.ones((batch_size, 3, height, width), dtype=torch.float32) * 0.2
    tgt_rgb_syns = [tgt_rgb_syn for i in range(neighbor_view_num)]
    tgt_mask = torch.ones((batch_size, 3, height, width), dtype=torch.float32)
    tgt_masks = [tgt_mask for i in range(neighbor_view_num)]
    images_tgt = torch.ones((batch_size, neighbor_view_num, 3, height, width), dtype=torch.float32)

    # For 4neighbor training
    loss_rgb_l1 = loss_fcn_rgb_L1(tgt_rgb_syn, tgt_mask, images_tgt[:, 1, :, :, :])
    print(loss_rgb_l1.item())

    ssim_calculator = SSIM()
    loss_ssim = loss_fcn_rgb_SSIM(ssim_calculator, tgt_rgb_syn, tgt_mask, images_tgt[:, 1, :, :, :])
    print(loss_ssim.item())

    # lpips_calculator = lpips.LPIPS(net="vgg")
    # lpips_calculator.requires_grad = False
    # loss_lpips = loss_fcn_rgb_lpips(lpips_calculator, tgt_rgb_syns, tgt_masks, images_tgt)
    # print(loss_lpips.item())
    # loss_lpips = loss_fcn_rgb_lpips(lpips_calculator, tgt_rgb_syns, tgt_masks, images_tgt)
    # print(loss_lpips.item())

    # For general use
    # loss_rgb_l1 = loss_fcn_rgb_L1(tgt_rgb_syn, tgt_mask, images_tgt[:, 0, :, :, :])
    # loss_rgb_smooth_l1 = loss_fcn_rgb_Smooth_L1(tgt_rgb_syn, tgt_mask, images_tgt[:, 0, :, :, :])
    # print(loss_rgb_l1.item(), loss_rgb_smooth_l1.item())
    #
    # ssim_calculator = SSIM()
    # loss_ssim = loss_fcn_rgb_SSIM(ssim_calculator, tgt_rgb_syn, tgt_mask, images_tgt[:, 0, :, :, :])
    # print(loss_ssim.item())
    #
    # lpips_calculator = lpips.LPIPS(net="vgg")
    # lpips_calculator.requires_grad = False
    # loss_lpips = loss_fcn_rgb_lpips(lpips_calculator, tgt_rgb_syn, tgt_mask, images_tgt[:, 0, :, :, :])
    # print(loss_lpips.item())

    # import cv2
    # depth_min_ref = torch.tensor([485.703,], dtype=torch.float32)
    # depth_max_ref = torch.tensor([536.844,], dtype=torch.float32)
    # ref_depth_syn = torch.from_numpy(cv2.imread("../testdata/depth_007.png", cv2.IMREAD_ANYDEPTH) / 64.0).unsqueeze(0).unsqueeze(1)
    # ref_depth_syn = ref_depth_syn + torch.randn(ref_depth_syn.shape) * 0.

    # image_ref = torch.from_numpy(cv2.imread("../testdata/image_007.png") / 255.0).unsqueeze(0).permute(0, 3, 1, 2)
    # loss_edge = loss_fcn_edge_aware(ref_depth_syn, image_ref, depth_min_ref, depth_max_ref)
    # print(loss_edge)
