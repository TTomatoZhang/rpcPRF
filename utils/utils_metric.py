import torch
import torch.nn.functional as F
from math import exp

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value * valid_mask
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


# def psnr_(img1, img2):
#     mse = ((img1 - img2) ** 2).mean((1, 2, 3))
#     psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
#     return psnr.mean()

def psnr_(img1, img2):
   mse = torch.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


class SSIM_(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_, self).__init__()
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
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


# def ssim(image_pred, image_gt):
#     """
#     image_pred and image_gt: (1, 3, H, W)
#     important: kornia==0.5.3
#     """
#     # kornia_ssim = ssim_()
#     ssimor = SSIM()
#     out = ssimor(image_pred, image_gt)
#     return torch.mean(out)
def ssim(image_pred, image_gt):

    ssimor = SSIM_()
    out = ssimor(image_pred, image_gt)
    return torch.mean(out)


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

def ME(depth_syn, depth_gt, mask, valid=False):
    depth_syn, depth_gt = depth_syn * mask, depth_gt * mask
    diff = (depth_syn - depth_gt).abs()
    if valid:
        diff = diff[diff!=0]
    me = torch.median(diff)
    return me