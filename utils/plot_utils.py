import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2
import pyvista as pv
from datasets.utils_pushbroom import load_pfm
from matplotlib.colors import ListedColormap
import os
import torch
import torch.nn.functional as F


def plot_grid(gt_img, imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [gt_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

def savepvplot(depth, pvpath):
    h, w = depth.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    #pts = np.stack([xx, yy, depth], -1)
    pts = np.stack([xx, yy, depth], -1)
    pots = pts.reshape(h * w, 3)
    pots = pv.PolyData(pots)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pots)
    plotter.show(screenshot=pvpath)

def pvplotpfm(pfmpath):
    depth = load_pfm(pfmpath)
    print(depth.min())
    h, w = depth.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    #pts = np.stack([xx, yy, depth], -1)
    pts = np.stack([xx, yy, depth], -1)
    #pots = pts.reshape(-1, 3)
    pots = pts.reshape(h * w, 3)
    pots = pv.PolyData(pots)
    pots.plot(eye_dome_lighting=True)

def save_png(matrix, save_file, maskout=None, cmap='winter', norm=None, save_cbar=False, save_mask=False, plot=True):
    if not plot:
        im = matrix
        # im values should be inside [0, 1]

        nan_mask = np.any(np.isnan(matrix), axis=2)
        # for visualization
        matrix[nan_mask] = 0.0

        if maskout is not None:
            nan_mask = np.logical_or(nan_mask, maskout)
        nan_mask = np.tile(nan_mask[:, :, np.newaxis], (1, 1, 3))

        eps = 1e-7
        assert (im.min() > -eps and im.max() < (1.0 + eps))
        im = np.uint8(im * 255.0)
    else:
        # for visualization, set nan value to nanmin
        nan_mask = np.isnan(matrix)
        matrix[nan_mask] = np.nanmin(matrix)

        if maskout is not None:
            nan_mask = np.logical_or(nan_mask, maskout)
        # add third channel
        nan_mask = np.tile(nan_mask[:, :, np.newaxis], (1, 1, 3))

        fig = plt.figure()
        dpi = fig.get_dpi()
        fig.set_size_inches(matrix.shape[1] / dpi, matrix.shape[0] / dpi)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)

        if norm is None:
            mpb = ax.imshow(matrix, cmap=cmap)
        else:
            mpb = ax.imshow(matrix, cmap=cmap, norm=norm)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((h, w, 3)).astype(dtype=np.uint8)

        # resize (im size might mismatch matrix size by 1)
        im = cv2.resize(im, (matrix.shape[1], matrix.shape[0]), interpolation=cv2.INTER_NEAREST)

        if save_cbar:
            fig_new, ax = plt.subplots()
            cbar = plt.colorbar(mpb, ax=ax, orientation='horizontal')
            ax.remove()

            # adjust color bar texts
            cbar.ax.tick_params(labelsize=10, rotation=-30)
            # save the same figure with some approximate autocropping
            idx = save_file.rfind('.')
            fig_new.savefig(save_file[:idx] + '.cbar.jpg', bbox_inches='tight')
            plt.close(fig_new)

        plt.close(fig)

    im[nan_mask] = 0
    imageio.imwrite(save_file, im)

    if save_mask:
        idx = save_file.rfind('.')
        valid_mask = 1.0 - np.float32(nan_mask[:, :, 0])
        imageio.imwrite(save_file[:idx] + '.mask.jpg', np.uint8(valid_mask * 255.0))

def plot_height_map(height_map, save=True, out_file=None, maskout=None, save_cbar=False, force_range=None):

    if force_range is None:
        min_val, max_val = np.nanpercentile(height_map, [1, 99])
        force_range = (min_val, max_val)

    min_val, max_val = force_range
    height_map = np.clip(height_map, min_val, max_val)
    # make sure the color map spans exactly [min_val, max_val]
    height_map[0, 0] = min_val
    height_map[0, 1] = max_val

    # cmap_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'colormap_height.txt')
    # colors = np.loadtxt(cmap_file)
    # colors = np.flip(colors, 0)
    # colors = (np.float32(colors) / 255.0).tolist()
    # cmap = ListedColormap(colors)
    # save image and mask
    out = save_image_only(height_map, out_file, save=save, cmap='viridis', save_cbar=save_cbar, maskout=maskout, plot=True)
    cv2.imwrite(out_file, out)
    return out

def plot_height_map_triple(height_gt, height_syn, outpath):
    height_syn = plot_height_map(height_syn, save=False)
    height_gt = plot_height_map(height_gt, save=False)
    height_diff = plot_height_map(abs(height_syn - height_gt), save=False)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(height_gt)

    plt.subplot(1, 3, 2)
    plt.imshow(height_syn)

    plt.subplot(1, 3, 3)
    plt.imshow(height_diff)

    plt.show()
    plt.savefig(outpath)
    print('{} saved'.format(outpath))

def save_image_only(matrix, save_file, save=True, maskout=None, cmap='viridis', norm=None, save_cbar=False, save_mask=False, plot=True):
    if not plot:
        im = matrix
        # im values should be inside [0, 1]

        nan_mask = np.any(np.isnan(matrix))
        # for visualization
        matrix[nan_mask] = 0.0

        # if maskout is not None:
        #     nan_mask = np.logical_or(nan_mask, maskout)
        # nan_mask = np.tile(nan_mask[:, :, np.newaxis], (1, 1, 3))

        eps = 1e-7
        assert (im.min() > -eps and im.max() < (1.0 + eps))
        im = np.uint8(im * 255.0)
    else:
        # for visualization, set nan value to nanmin
        nan_mask = np.isnan(matrix)
        matrix[nan_mask] = np.nanmin(matrix)

        if maskout is not None:
            nan_mask = np.logical_or(nan_mask, maskout)
        # add third channel
        nan_mask = np.tile(nan_mask[:, :, np.newaxis], (1, 1, 3))

        fig = plt.figure()
        dpi = fig.get_dpi()
        fig.set_size_inches(matrix.shape[1] / dpi, matrix.shape[0] / dpi)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)

        if norm is None:
            mpb = ax.imshow(matrix, cmap=cmap)
        else:
            mpb = ax.imshow(matrix, cmap=cmap, norm=norm)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((h, w, 3)).astype(dtype=np.uint8)

        # resize (im size might mismatch matrix size by 1)
        im = cv2.resize(im, (matrix.shape[1], matrix.shape[0]), interpolation=cv2.INTER_NEAREST)

        if save_cbar:
            fig_new, ax = plt.subplots()
            cbar = plt.colorbar(mpb, ax=ax, orientation='horizontal')
            ax.remove()

            # adjust color bar texts
            cbar.ax.tick_params(labelsize=10, rotation=-30)
            # save the same figure with some approximate autocropping
            idx = save_file.rfind('.')
            fig_new.savefig(save_file[:idx] + '.cbar.jpg', bbox_inches='tight')
            plt.close(fig_new)

        plt.close(fig)
        return im

    im[nan_mask] = 0
    if save:
        imageio.imwrite(save_file, im)

    if save_mask:
        idx = save_file.rfind('.')
        valid_mask = 1.0 - np.float32(nan_mask[:, :, 0])
        imageio.imwrite(save_file[:idx] + '.mask.jpg', np.uint8(valid_mask * 255.0))
        return im, valid_mask

def getKL(weights, depth_tgt):
    """
    weights: [n_depth, h, W]
    depth_gt: [h, w]
    """
    weights = weights.squeeze()
    n_depth, h, w = weights.shape
    weights = weights.reshape(n_depth, -1).transpose(0, 1)
    dmax = torch.max(depth_tgt)
    dmin = torch.min(depth_tgt)
    d_tgt = (depth_tgt.squeeze() - dmin * torch.ones(depth_tgt.squeeze().shape).cuda()) /(dmax - dmin) * (n_depth - 1)
    d_tgt = d_tgt.type(torch.LongTensor)
    weights_gt = F.one_hot(d_tgt, num_classes=n_depth)
    weights_gt = weights_gt.reshape(-1, n_depth).cuda()
    klloss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    lkl = klloss(weights_gt, weights)
    return lkl, weights_gt

# def plot_save_weights(weights_gt_tgt, tgt_weights, weights_gt_ref, src_weights, savefigpath):
#     fig = plt.figure()
#     fig.subtitle('tgt_weights')
#     plt.subplot(121)
#     average_weights_tgt = tgt_weights.squeeze().mean(-1).mean(-1).detach().cpu().numpy()
#     # weight_point_tgt = tgt_weights.squeeze()[:, 40, 40].detach().cpu().numpy()
#     weights_gt_tgt = weights_gt_tgt.squeeze().detach().cpu().numpy()[40, :]
#     plt.plot(average_weights_tgt, color='g')
#     plt.plot(weights_gt_tgt, color='b')
#     # plt.plot(weights_gt_tgt)
#     plt.xlabel('alt_samples')
#     plt.ylabel('probability')
#     plt.title('')
#     plt.subplot(122)
#     average_weights_ref = src_weights.squeeze().mean(-1).mean(-1).detach().cpu().numpy()
#     # weight_point_ref = src_weights.squeeze()[:, 40, 40].detach().cpu().numpy()
#     weights_gt_ref = weights_gt_ref.squeeze().detach().cpu().numpy()[40, :]
#     plt.plot(average_weights_ref, color='g')
#     plt.plot(weights_gt_ref, color='b')
#     plt.xlabel('alt_samples')
#     plt.ylabel('probability')
#     # plt.title('src_weights')
#     plt.savefig(savefigpath)
#     print('distributions saved to %s', savefigpath)
#     plt.close("all")
#     return


def plot_save_weights(weights_gt_ref, src_weights, savefigpath):
    #fig = plt.figure()
    # fig.subtitle('tgt_weights')
    # plt.subplot(121)
    # average_weights_tgt = tgt_weights.squeeze().mean(-1).mean(-1).detach().cpu().numpy()
    # # weight_point_tgt = tgt_weights.squeeze()[:, 40, 40].detach().cpu().numpy()
    # weights_gt_tgt = weights_gt_tgt.squeeze().detach().cpu().numpy()[40, :]
    # plt.plot(average_weights_tgt, color='g')
    # plt.plot(weights_gt_tgt, color='b')
    # plt.plot(weights_gt_tgt)
    # plt.xlabel('alt_samples')
    # plt.ylabel('probability')
    # plt.title('')
    # plt.subplot(122)
    average_weights_ref = src_weights.squeeze().mean(-1).mean(-1).detach().cpu().numpy()
    # weight_point_ref = src_weights.squeeze()[:, 40, 40].detach().cpu().numpy()
    weights_gt_ref = weights_gt_ref.squeeze().detach().cpu().numpy()[40, :]
    plt.plot(average_weights_ref, color='g')
    plt.plot(weights_gt_ref, color='b')
    plt.xlabel('alt_samples')
    plt.ylabel('probability')
    # plt.title('src_weights')
    plt.savefig(savefigpath)
    print('distributions saved to %s', savefigpath)
    return

def plot_weights(src_weights, T_src):
    src_weights = src_weights.squeeze().permute(1, 2, 0).view(-1, 32)
    weights_np = src_weights.cpu().numpy()
    T_src = T_src.squeeze().permute(1, 2, 0).view(-1, 32)
    T_np = T_src.cpu().numpy()
    weights_part = weights_np[:10, :]


def plotheifolders():
    outroot = '/home/pc/Documents/ztt/newgitrepos/results/SatMINE_results/IARPA/heiplot/'
    folder_path = '/home/pc/Documents/ztt/newgitrepos/results/SatMINE_results/IARPA/Hei/'
    sub = 'bf/'
    os.makedirs(outroot + sub, exist_ok=True)
    pfmglob = glob.glob(folder_path + sub + '*.pfm')
    for pfm in pfmglob:
        hei = load_pfm(pfm)


def changenames():
    from tqdm import tqdm
    root = '/home/pc/Documents/ztt/newgitrepos/results/SatMINE_results/IARPA/RGB/'
    subs = os.listdir(root)
    for sub in tqdm(subs):
        subroot = os.path.join(root, sub)
        pathglob = glob.glob(subroot + '/*.png')
        for path in pathglob:
            name = os.path.basename(path)
            #sufix = name.split('.')[1]
            #prefix = name.split('.')[0]

            newname = name.replace('(', '').replace(')', '').replace(',', '').replace('\'', '')
            os.rename(path, os.path.join(subroot, newname))
            print('name changed to' + newname)
        print('Done on '+ sub)

if __name__ == "__main__":
    plotheifolders()