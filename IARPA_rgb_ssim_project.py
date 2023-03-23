import os
import configargparse
import random
import time
import lpips
import matplotlib.pyplot as plt
import torch.cuda
import torch.nn as nn
import torch.autograd.profiler as prof
from utils.miscs import *
from models.mpi_generator_alt import MPIGenerator
from models.feature_generator import FeatureGenerator
from datasets.IARPA import MVSDataset_IARPA
from datasets.utils_pushbroom import *
from torch.utils.data import DataLoader
from utils.render import *
from torch.utils.tensorboard import SummaryWriter
from models.losses import *
import shutil
from pyhocon import ConfigFactory
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Task_IARPA():
    def __init__(self, conf, device, resume=True):
        super(Task_IARPA, self).__init__()
        self.conf = conf
        self.dname = conf['dataset.dname']
        self.upsample = True

        # output config
        self.ssim_calculator = SSIM().to(device)
        self.loss_rgb_weight = conf['loss.loss_rgb_weight']
        self.loss_ssim_weight = conf['loss.loss_ssim_weight']
        self.loss_lpips_weight = conf['loss.loss_lpips_weight']
        self.lpips_calculator = lpips.LPIPS(net="vgg").to(device)
        expname = 'rgb{}_ssim{}_warp_{}'.format(self.loss_rgb_weight, self.loss_ssim_weight, self.dname)
        self.logdir = os.path.join(conf['general.exp_dir'], 'logs',  expname)

        self.logger = SummaryWriter(self.logdir)

        self.pretraineddir = conf['general.exp_dir'] + '/pretrained/' + expname + '/'
        if not os.path.exists(self.pretraineddir):
            os.mkdir(self.pretraineddir)
        # self.device_ids = device_ids
        self.cache_dir = conf['general.exp_dir'] + conf['general.arcname'] + '/cas/'
        self.out_img_path = conf['general.exp_dir'] + conf['general.arcname'] + '/images/'
        self.device = device

        # train config
        self.start_epoch = 0
        self.epochs = conf['train.epochs']

        self.current_epoch = 0
        self.neighbor_view_num = conf['general.neighbor_view_num']
        self.add_edgeloss = conf['train.add_edge_loss']

        self.feature_generator, self.mpi_generator = self.model_definition()

        self.feature_generator = self.feature_generator.to(device)
        self.mpi_generator = self.mpi_generator.to(device)
        self.optimizer, self.lr_scheduler = self.optimizer_definition()
        self.train_dataloader, self.validate_dataloader = self.dataloader_definition()
        if resume:
            self.resume_training()

    def model_definition(self):
        """
        model definition
        Returns: models
        """
        feature_generator = FeatureGenerator(model_type=self.conf['model.feature_generator_model_type'], device=self.device, pretrained=False)
        mpi_generator = MPIGenerator(device=self.device, feature_out_chs=feature_generator.encoder_channels)

        train_params = sum(params.numel() for params in feature_generator.parameters() if params.requires_grad) + \
                       sum(params.numel() for params in mpi_generator.parameters() if params.requires_grad)
        print("Total_paramteters: {}".format(train_params))
        return feature_generator.to(device), mpi_generator.to(device)

    def optimizer_definition(self):
        """
        optimizer definition
        Returns:
        """
        params = [
            {"params": self.feature_generator.parameters(), "lr": self.conf['train.lr_encoder']},
            {"params": self.mpi_generator.parameters(), "lr": self.conf['train.lr_decoder']}
        ]
        optimizer = torch.optim.Adam(params, weight_decay=self.conf['train.lr_weightdecay'])


        milestones = self.conf['train.lr_ds_epoch_idx']
        lr_gamma = 0.5 # [1 / float(lr_g) for lr_g in self.conf['train.lr_ds_epoch_idx']]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                            last_epoch=self.start_epoch - 1)
        return optimizer, lr_scheduler

    def dataloader_definition(self):
        # dataset, dataloader definition

        train_dataset = MVSDataset_IARPA(self.conf, 'train')
        validate_dataset = MVSDataset_IARPA(self.conf, 'val')
        train_dataloader = DataLoader(train_dataset, self.conf['train.batch_size'], shuffle=True, num_workers=self.conf['train.num_workers'], drop_last=True)
        validate_dataloader = DataLoader(validate_dataset, self.conf['train.batch_size'], shuffle=True,num_workers=self.conf['train.num_workers'], drop_last=False)
        return train_dataloader, validate_dataloader

    def resume_training(self):
        """
        training process resume, load model and optimizer ckpt
        """
        if self.conf['general.ckptpath'] == 'None':
            saved_models = [fn for fn in os.listdir(self.pretraineddir) if fn.endswith(".ckpt")]
            saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            # use the latest checkpoint file
            ckptname = os.path.join(self.pretraineddir, saved_models[-1])
        else:
            ckptname = self.conf['general.ckptpath']
        print("resuming from checkpoints ", ckptname)
        state_dict = torch.load(ckptname)
        self.start_epoch = state_dict["epoch"]
        self.feature_generator.load_state_dict(state_dict["feature_generator"])
        self.mpi_generator.load_state_dict(state_dict["mpi_generator"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        # self.start_epoch = state_dict["epoch"] + 1
        self.start_epoch = 0  # fine tune from whu_view_syn_small model:799

        # redefine lr_schedular
        milestones = [int(epoch_idx) for epoch_idx in self.conf['train.lr_ds_epoch_idx']]
        lr_gamma = [1 / float(lr_g) for lr_g in self.conf['train.lr_ds_epoch_idx']]
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones, gamma=lr_gamma,
                                                                 last_epoch=self.start_epoch - 1)

    def set_data(self, sample):
        """
        set batch_sample data
        Args:
            sample:
        Returns:
        """
        self.image_src = sample["image_src"].to(device).squeeze(0)
        self.alt_min_src, self.alt_max_src = sample["alt_min_src"], sample["alt_max_src"]
        self.rpc_src = sample["rpc_src"].to(device).squeeze(0)
        self.rpcs_tgt = sample["rpcs_tgt"].to(device).squeeze(0)
        self.alt_src = sample["alt_src"].to(device).squeeze(0)
        self.images_tgt = sample["images_tgt"].to(device)
        self.alts_tgt = sample["alts_tgt"].to(device)

        self.height, self.width = self.image_src.shape[-2], self.image_src.shape[-1]

    def train(self):
        for epoch_idx in range(self.start_epoch, self.epochs):
            print("Training at Epoch: {}/{}".format(epoch_idx, self.epochs))
            for batch_idx, sample in enumerate(self.train_dataloader):
                start_time = time.time()
                self.global_step = len(self.train_dataloader) * epoch_idx + batch_idx
                self.set_data(sample)
                summary_scalars, summary_images = self.train_triple(self.conf['model.alt_sample_num'])
                print("Epoch:{}/{}, Iteration:{}/{}, train loss={:.4f}, time={:.4f}".format(epoch_idx, self.epochs,
                                                                                            batch_idx,
                                                                                            len(self.train_dataloader),
                                                                                            summary_scalars["loss_all"],
                                                                                            time.time() - start_time))
                print('lr:{}'.format(self.lr_scheduler.get_last_lr()))

                if self.global_step % self.conf['train.summary_scalars_freq'] == 0:
                    save_scalars(self.logger, "Train", summary_scalars,
                                 self.global_step)  # scalars for random sampled tgt-view image
                if self.global_step % self.conf['train.summary_images_freq'] == 0:
                    for scale in range(4):
                        save_images(self.logger, "Train_scale_left_{}".format(scale),
                                    summary_images["scale_{}_left".format(scale)],
                                    self.global_step)  # summary images for random sampled tgt-image
                        save_images(self.logger, "Train_scale_right_{}".format(scale),
                                    summary_images["scale_{}_right".format(scale)],
                                    self.global_step)  # summary images for random sampled tgt-image
            if (epoch_idx + 1) % self.conf['train.save_ckpt_freq'] == 0:
                torch.save({
                    "epoch": epoch_idx,
                    "feature_generator": self.feature_generator.state_dict(),
                    "mpi_generator": self.mpi_generator.state_dict(),
                    "optimizer": self.optimizer.state_dict(), },
                    "{}/rpcmpimodel_{:0>4}.ckpt".format(self.pretraineddir, epoch_idx)
                )

            if (epoch_idx + 1) % self.conf['train.validate_freq'] == 0:
                self.validate(epoch_idx, self.conf['model.alt_sample_num'])
            self.lr_scheduler.step()

    def train_triple(self, alt_sample_num):
        """
        calculate 4 scale loss, loss backward per tgt image
        Returns: summary_scalars, summary_images
        """
        self.feature_generator.train()
        self.mpi_generator.train()

        # network forward, generate mpi representations
        conv1_out, block1_out, block2_out, block3_out, block4_out = self.feature_generator(self.image_src)
        mpi_outputs = self.mpi_generator(input_features=[conv1_out, block1_out, block2_out, block3_out, block4_out],
                                         alt_sample_num=alt_sample_num)

        rgb_mpi_src_dict = {
            "scale_0": mpi_outputs["MPI_{}".format(0)][:, :, :3, :, :], #[B, Nsample, C=3, Hrender, Wrender]
            "scale_1": mpi_outputs["MPI_{}".format(1)][:, :, :3, :, :],
            "scale_2": mpi_outputs["MPI_{}".format(2)][:, :, :3, :, :],
            "scale_3": mpi_outputs["MPI_{}".format(3)][:, :, :3, :, :],
        }
        sigma_mpi_src_dict = {
            "scale_0": mpi_outputs["MPI_{}".format(0)][:, :, 3:, :, :],#[B, Nsample, C=1, Hrender, Wrender]
            "scale_1": mpi_outputs["MPI_{}".format(1)][:, :, 3:, :, :],
            "scale_2": mpi_outputs["MPI_{}".format(2)][:, :, 3:, :, :],
            "scale_3": mpi_outputs["MPI_{}".format(3)][:, :, 3:, :, :],
        }

        summary_scalars, summary_images = {}, {}
        assert self.neighbor_view_num - 1 == 2
        neighbor_image_idx = 0
        summary_scalars_0, summary_images_0 = self.train_per_triple_0(rgb_mpi_src_dict, sigma_mpi_src_dict,
                                                               neighbor_image_idx, alt_sample_num)

        neighbor_image_idx = 1
        summary_scalars_1, summary_images_1 = self.train_per_triple_1(rgb_mpi_src_dict, sigma_mpi_src_dict,
                                                                                  neighbor_image_idx, alt_sample_num)


        summary_scalars.update(summary_scalars_0)
        summary_scalars.update(summary_scalars_1)
        summary_images.update(summary_images_0)
        summary_images.update(summary_images_1)

        loss = summary_scalars_0['loss_left'] + summary_scalars_1['loss_right']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        summary_scalars.update({'loss_all': loss})
        return summary_scalars, summary_images

    def train_per_triple_0(self, rgb_mpi_src_dict, sigma_mpi_src_dict, neighbor_image_idx, alt_sample_num):

        summary_scalars, summary_images = {}, {}
        loss_left, loss_rgb_left_image, loss_ssim_left_image = 0.0, 0.0, 0.0
        # the losses are exerted on target images
        if self.add_edgeloss:
            loss_depth_edge_left_image = 0.0

        for scale in range(4):
            with torch.no_grad():
                # rescale intrinsics for ref-view, tgt-views
                src_rpc_scaled = rescale_rpc(self.rpc_src, 1. / (2 ** scale))
                tgt_rpc_scaled = rescale_rpc(self.rpcs_tgt[0], 1. / (2 ** scale))

                H_render, W_render = self.height // 2 ** scale, self.width // 2 ** scale
                # rescale image_src, alt_src, images_tgt
                downsize = (H_render, W_render)
                image_src = F.interpolate(self.image_src, size=downsize, mode="bilinear")  # [B, 3, H//scale, W//scale]
                # alt_src = F.interpolate(self.alt_src.unsqueeze(1), size=downsize, mode="nearest")  # Not for loss, for monitor alt MAE
                image_tgt = F.interpolate(self.images_tgt[:, neighbor_image_idx, :, :, :], size=downsize,mode="bilinear")  # [B, H//scale, W//scale, 3]
                alt_neighborview = self.alts_tgt[:, neighbor_image_idx, :, :].unsqueeze(1)
                alt_tgt = F.interpolate(alt_neighborview, size=downsize, mode="bilinear")
                # alt_tgt = F.interpolate(self.alts_tgt[:, neighbor_image_idx, :, :].squeeze(), size=(H_render, W_render), mode="bilinear")  # [B, H//scale, W//scale]

            rgb_mpi_src = rgb_mpi_src_dict["scale_{}".format(scale)]
            sigma_mpi_src = sigma_mpi_src_dict["scale_{}".format(scale)]  # [b, n_alt, 1, H, W]

            srcDepthSample = sampleAltitudeInv(self.alt_min_src.unsqueeze(0), self.alt_max_src.unsqueeze(0), alt_sample_num)

            XYH_src = GetXYHfromAltitudes(H_render, W_render, srcDepthSample, src_rpc_scaled)  # [B, Nsample, 3, H, W]
            src_rgb_syn, src_alt_syn, blended_weights, src_weights =renderSrcViewRPC(
                rgb_MPI_src=rgb_mpi_src,
                sigma_MPI_src=sigma_mpi_src,
                XYH_src=XYH_src, # double
                use_alpha=False)

            if self.conf['model.src_rgb_blending']:
                # Bx3xHxW
                rgb_MPI_src = blended_weights * image_src.unsqueeze(1) + (1 - blended_weights) * rgb_mpi_src
                # render with this MPI
                src_rgb_syn = torch.sum(src_weights * rgb_MPI_src, dim=1, keepdim=False)  # Bx3xHxW
                src_alt_syn = torch.sum(src_weights * XYH_src[:, :, 2:, :, :], dim=1, keepdim=False) \
                    / (torch.sum(src_weights, dim=1, keepdim=False) + 1e-5)  # Bx1xHxW


            ################## render tgt-view syn image and depth ##############
            tgt_rgb_syn, tgt_mask = project_src2tgt(src_rgb_syn, src_alt_syn, src_rpc_scaled, tgt_rpc_scaled)
            # tgt_rgb_syn, tgt_alt_syn, tgt_mask, tgt_weights = renderNovelViewRPC(
            #     rgb_MPI_src=rgb_mpi_src,
            #     sigma_MPI_src=sigma_mpi_src,
            #     XYH_src=XYH_src,
            #     altSample=srcDepthSample,
            #     src_RPC=src_rpc_scaled,
            #     tgt_RPC=tgt_rpc_scaled,
            #     H_render=H_render,
            #     W_render=W_render,
            #     use_alpha=False
            # )
            ################## start to calculate multi-loss ###############
            src_mask = torch.ones_like(tgt_mask)
            loss_rgb = (loss_fcn_rgb_L1(tgt_rgb_syn, tgt_mask, image_tgt)
                        + loss_fcn_rgb_L1(src_rgb_syn, src_mask, image_src)) * self.loss_rgb_weight
            loss_ssim = (loss_fcn_rgb_SSIM(self.ssim_calculator, tgt_rgb_syn, tgt_mask, image_tgt)
                         + loss_fcn_rgb_SSIM(self.ssim_calculator, src_rgb_syn.to(torch.float32), src_mask, image_src)) * self.loss_ssim_weight
            # loss_lpips = (loss_fcn_rgb_lpips(self.lpips_calculator, tgt_rgb_syn.to(torch.float32), tgt_mask, image_tgt)
            #                + loss_fcn_rgb_lpips(self.lpips_calculator, src_rgb_syn.to(torch.float32), src_mask, image_src)) * self.loss_lpips_weight
            loss_rgb_left_image = loss_rgb_left_image + loss_rgb
            loss_ssim_left_image = loss_ssim_left_image + loss_ssim
            #loss_lpips_left_image = loss_lpips_left_image + loss_lpips


            if self.add_edgeloss:
                loss_depth_edge = loss_fcn_edge_aware(src_alt_syn.squeeze(), image_src.squeeze(),
                                                      self.alt_min_src.to(device),
                                                      self.alt_max_src.to(device)) * self.loss_edge_weight  # 80
                loss_depth_edge_left_image = loss_depth_edge_left_image + loss_depth_edge
                loss_per_image = loss_rgb_left_image + loss_depth_edge_left_image + loss_ssim_left_image
                loss_left = loss_left + loss_per_image
            else:
                loss_per_image = loss_rgb_left_image + loss_ssim_left_image
                loss_left = loss_left + loss_per_image


            with torch.no_grad():
                summary_images["scale_{}_left".format(scale)] = {
                    "src_rgb_syn_left": src_rgb_syn,
                    "src_image_left": image_src,
                    "src_alt_syn_left": src_alt_syn,
                    "tgt_rgb_syn_left": tgt_rgb_syn,
                    #"tgt_alt_syn_left": tgt_alt_syn,
                    "tgt_image_left": image_tgt,
                    #"tgt_alt_diff_left": torch.abs(alt_tgt - tgt_alt_syn)
                }

        with torch.no_grad():
            if self.add_edgeloss:
                    summary_scalars = {
                        "loss_ssim_left": loss_ssim_left_image.item(),
                        #"loss_lpips_left": loss_lpips_left_image.item(),
                        "loss_rgb_left": loss_rgb_left_image.item(),
                        "loss_depth_edge_tgt_left": loss_depth_edge_left_image,
                        "loss_left": loss_left
                    }
            else:

                    summary_scalars = {
                        "loss_ssim_left": loss_ssim_left_image.item(),
                        #"loss_lpips_left": loss_lpips_left_image.item(),
                        "loss_rgb_left": loss_rgb_left_image.item(),
                        "loss_left": loss_left
                    }
        return summary_scalars, summary_images

    def train_per_triple_1(self, rgb_mpi_src_dict, sigma_mpi_src_dict, neighbor_image_idx, alt_sample_num):

        summary_scalars, summary_images = {}, {}
        loss_right, loss_rgb_right_image, loss_ssim_right_image = 0.0, 0.0, 0.0
        # the losses are exerted on target images
        if self.add_edgeloss:
            loss_depth_edge_right_image = 0.0

        for scale in range(4):
            with torch.no_grad():
                # rescale intrinsics for ref-view, tgt-views
                src_rpc_scaled = rescale_rpc(self.rpc_src, 1. / (2 ** scale))
                tgt_rpc_scaled = rescale_rpc(self.rpcs_tgt[0], 1. / (2 ** scale))

                H_render, W_render = self.height // 2 ** scale, self.width // 2 ** scale
                # rescale image_src, alt_src, images_tgt
                downsize = (H_render, W_render)
                image_src = F.interpolate(self.image_src, size=downsize, mode="bilinear")  # [B, 3, H//scale, W//scale]
                # alt_src = F.interpolate(self.alt_src.unsqueeze(1), size=downsize, mode="nearest")  # Not for loss, for monitor alt MAE
                image_tgt = F.interpolate(self.images_tgt[:, neighbor_image_idx, :, :, :], size=downsize,
                                          mode="bilinear")  # [B, H//scale, W//scale, 3]
                alt_neighborview = self.alts_tgt[:, neighbor_image_idx, :, :].unsqueeze(1)
                alt_tgt = F.interpolate(alt_neighborview, size=downsize, mode="bilinear")
                # alt_tgt = F.interpolate(self.alts_tgt[:, neighbor_image_idx, :, :].squeeze(), size=(H_render, W_render), mode="bilinear")  # [B, H//scale, W//scale]

            rgb_mpi_src = rgb_mpi_src_dict["scale_{}".format(scale)]
            sigma_mpi_src = sigma_mpi_src_dict["scale_{}".format(scale)]  # [b, n_alt, 1, H, W]

            srcDepthSample = sampleAltitudeInv(self.alt_min_src.unsqueeze(0), self.alt_max_src.unsqueeze(0),
                                               alt_sample_num)

            XYH_src = GetXYHfromAltitudes(H_render, W_render, srcDepthSample, src_rpc_scaled)  # [B, Nsample, 3, H, W]
            src_rgb_syn, src_alt_syn, blended_weights, src_weights = renderSrcViewRPC(
                rgb_MPI_src=rgb_mpi_src,
                sigma_MPI_src=sigma_mpi_src,
                XYH_src=XYH_src,
                use_alpha=False)

            if self.conf['model.src_rgb_blending']:
                # Bx3xHxW
                rgb_MPI_src = blended_weights * image_src.unsqueeze(1) + (1 - blended_weights) * rgb_mpi_src
                # render with this MPI
                src_rgb_syn = torch.sum(src_weights * rgb_MPI_src, dim=1, keepdim=False)  # Bx3xHxW
                src_alt_syn = torch.sum(src_weights * XYH_src[:, :, 2:, :, :], dim=1, keepdim=False) \
                              / (torch.sum(src_weights, dim=1, keepdim=False) + 1e-5)  # Bx1xHxW

            ################## render tgt-view syn image and depth ##############
            tgt_rgb_syn, tgt_mask = project_src2tgt(src_rgb_syn, src_alt_syn, src_rpc_scaled, tgt_rpc_scaled)
            # tgt_rgb_syn, tgt_alt_syn, tgt_mask, tgt_weights = renderNovelViewRPC(
            #     rgb_MPI_src=rgb_mpi_src,
            #     sigma_MPI_src=sigma_mpi_src,
            #     XYH_src=XYH_src,
            #     altSample=srcDepthSample,
            #     src_RPC=src_rpc_scaled,
            #     tgt_RPC=tgt_rpc_scaled,
            #     H_render=H_render,
            #     W_render=W_render,
            #     use_alpha=False
            # )
            ################## start to calculate multi-loss ###############
            src_mask = torch.ones_like(tgt_mask)
            loss_rgb = (loss_fcn_rgb_L1(tgt_rgb_syn, tgt_mask, image_tgt)
                        + loss_fcn_rgb_L1(src_rgb_syn, src_mask, image_src)) * self.loss_rgb_weight
            loss_ssim = (loss_fcn_rgb_SSIM(self.ssim_calculator, tgt_rgb_syn, tgt_mask, image_tgt)
                         + loss_fcn_rgb_SSIM(self.ssim_calculator, src_rgb_syn.to(torch.float32), src_mask, image_src)) * self.loss_ssim_weight
            # loss_lpips = (loss_fcn_rgb_lpips(self.lpips_calculator, tgt_rgb_syn, tgt_mask, image_tgt)
            #               + loss_fcn_rgb_lpips(self.lpips_calculator, src_rgb_syn.to(torch.float32), src_mask, image_src)) * self.loss_lpips_weight

            loss_rgb_right_image = loss_rgb_right_image + loss_rgb
            loss_ssim_right_image = loss_ssim_right_image + loss_ssim
            #loss_lpips_right_image = loss_lpips_right_image + loss_lpips

            if self.add_edgeloss:
                loss_depth_edge = loss_fcn_edge_aware(src_alt_syn.squeeze(), image_src.squeeze(),
                                                      self.alt_min_src.to(device),
                                                      self.alt_max_src.to(device)) * self.loss_edge_weight  # 80
                loss_depth_edge_right_image = loss_depth_edge_right_image + loss_depth_edge
                loss_per_image = loss_rgb_right_image + loss_depth_edge_right_image + loss_ssim_right_image
                loss_right = loss_right + loss_per_image
            else:
                loss_per_image = loss_rgb_right_image + loss_ssim_right_image
                loss_right = loss_right + loss_per_image

            with torch.no_grad():
                summary_images["scale_{}_right".format(scale)] = {
                    "src_rgb_syn_right": src_rgb_syn,
                    "src_image_right": image_src,
                    "src_alt_syn_right": src_alt_syn,
                    "tgt_rgb_syn_right": tgt_rgb_syn,
                    #"tgt_alt_syn_right": tgt_alt_syn,
                    "tgt_image_right": image_tgt,
                    #"tgt_alt_diff_right": torch.abs(alt_tgt - tgt_alt_syn)
                }

        with torch.no_grad():

            if self.add_edgeloss:
                summary_scalars = {
                    "loss_ssim_right": loss_ssim_right_image.item(),
                    "loss_rgb_right": loss_rgb_right_image.item(),
                    "loss_depth_edge_tgt_right": loss_depth_edge_right_image,
                    "loss_right": loss_right
                }
            else:
                summary_scalars = {
                    "loss_ssim_right": loss_ssim_right_image.item(),
                    "loss_rgb_right": loss_rgb_right_image.item(),
                    "loss_right": loss_right
                }
        return summary_scalars, summary_images

    def validate(self, epoch_idx, alt_sample_num):
        print("Validating process, Epoch: {}/{}".format(epoch_idx, self.epochs))
        average_validate_scalars = ScalarDictMerge()
        for batch_idx, sample in enumerate(self.validate_dataloader):
            self.set_data(sample)
            self.val_id = batch_idx
            summary_scalars, summary_images = self.validate_sample(alt_sample_num)
            average_validate_scalars.update(summary_scalars)
        save_scalars(self.logger, "Validate", average_validate_scalars.mean(), epoch_idx)
        save_images(self.logger, "Validate", summary_images["scale_0"], epoch_idx)


    def validate_sample(self, alt_sample_num):
        self.feature_generator.eval()
        self.mpi_generator.eval()
        with torch.no_grad():
            # get mpi representations from network output
            conv1_out, block1_out, block2_out, block3_out, block4_out = self.feature_generator(self.image_src)
            mpi_outputs = self.mpi_generator(input_features=[conv1_out, block1_out, block2_out, block3_out, block4_out],
                                             alt_sample_num=alt_sample_num)

            summary_scalars, summary_images = {}, {}  # 0-idx tgt-view summary, scale_0
            neighbor_image_idx = 0  # left
                    #self.neighbor_view_num-1):  # loss backward and optimizer step neighbor_view_num times
            loss_per_image, loss_lpips_per_tgt_image, loss_rgb_per_image, loss_ssim_per_image = 0.0, 0.0, 0.0, 0.0
            #loss_pdf_per_image, loss_alt_per_image = 0.0, 0.0
            for scale in range(4):
                with torch.no_grad():
                    # rescale intrinsics for ref-view, tgt-views
                    src_rpc_scaled = rescale_rpc(self.rpc_src, 1. / (2 ** scale))
                    tgt_rpc_scaled = rescale_rpc(self.rpc_src, 1. / (2 ** scale))

                    H_render, W_render = self.height // 2 ** scale, self.width // 2 ** scale
                    # rescale image_ref, alt_ref, images_tgt
                    downsize = (H_render, W_render)
                    image_src = F.interpolate(self.image_src, size=downsize,
                                              mode="bilinear")  # [B, 3, H//scale, W//scale]
                    alt_s = self.alt_src.unsqueeze(1)
                    alt_src = F.interpolate(alt_s, size=downsize, mode="nearest")  # Not for loss, for monitor alt MAE
                    image_tgt = F.interpolate(self.images_tgt[:, neighbor_image_idx, :, :, :], size=downsize,
                                              mode="bilinear")  # [B, 3, H//scale, W//scale]
                    alt_neighborview = self.alts_tgt[:, neighbor_image_idx, :, :].unsqueeze(1)
                    alt_tgt = F.interpolate(alt_neighborview, size=downsize, mode="bilinear")
                    # alt_tgt = F.interpolate(self.alts_tgt[:, neighbor_image_idx, :, :].squeeze(), size=(height_render, width_render), mode="bilinear")  # [B, H//scale, W//scale]
                # prepare the input for rendering
                rgb_mpi_src = mpi_outputs["MPI_{}".format(scale)][:, :, :3, :, :]
                sigma_mpi_src = mpi_outputs["MPI_{}".format(scale)][:, :, 3:, :, :]  # [b, n_alt, 1, H, W]

                #torch.cuda.empty_cache()
                # render ref-view syn image
                # src_weights [n_alt, H, W]
                ################# render src-view syn image and depth ##############
                # src_weights [n_alt, H, W]
                # 0. height sample
                srcDepthSample = sampleAltitudeInv(self.alt_min_src.unsqueeze(0), self.alt_max_src.unsqueeze(0),
                                                   alt_sample_num)
                XYH_src = GetXYHfromAltitudes(H_render, W_render, srcDepthSample, src_rpc_scaled)  # [B, Nsample, 3, H, W]
                print('validate on rendering src image')
                src_rgb_syn, src_alt_syn, blended_weights, src_weights = renderSrcViewRPC(
                    rgb_MPI_src=rgb_mpi_src,
                    sigma_MPI_src=sigma_mpi_src,
                    XYH_src=XYH_src,
                    use_alpha=False)

                if self.conf['model.src_rgb_blending']:
                    # Bx3xHxW
                    rgb_MPI_src = blended_weights * image_src.unsqueeze(1) + (1 - blended_weights) * \
                                  rgb_mpi_src
                    # render with this MPI
                    src_rgb_syn = torch.sum(src_weights * rgb_MPI_src, dim=1, keepdim=False)  # Bx3xHxW
                    src_alt_syn = torch.sum(src_weights * XYH_src[:, :, 2:, :, :], dim=1, keepdim=False) \
                                  / (torch.sum(src_weights, dim=1, keepdim=False) + 1e-5)  # Bx1xHxW

                ################## render tgt-view syn image and depth ##############
                print('validate on rendering tgt image')
                tgt_rgb_syn, tgt_mask = project_src2tgt(src_rgb_syn, src_alt_syn, src_rpc_scaled, tgt_rpc_scaled)
                # tgt_rgb_syn, tgt_alt_syn, tgt_mask, tgt_weights = renderNovelViewRPC(
                #     rgb_MPI_src=rgb_mpi_src,
                #     sigma_MPI_src=sigma_mpi_src,
                #     XYH_src=XYH_src,
                #     altSample=srcDepthSample,
                #     src_RPC=src_rpc_scaled,
                #     tgt_RPC=tgt_rpc_scaled,
                #     H_render=H_render,
                #     W_render=W_render,
                #     use_alpha=False
                # )
                ################## start to calculate multi-loss ###############
                loss_rgb = loss_fcn_rgb_L1(tgt_rgb_syn, tgt_mask, image_tgt) * self.loss_rgb_weight
                loss_ssim = loss_fcn_rgb_SSIM(self.ssim_calculator, tgt_rgb_syn, tgt_mask,
                                              image_tgt) * self.loss_ssim_weight
                loss_lpips = loss_fcn_rgb_lpips(self.lpips_calculator, tgt_rgb_syn, tgt_mask, image_tgt) * self.loss_lpips_weight
                loss_sum = loss_lpips + loss_rgb + loss_ssim

                # loss_pdf_ref, weights_gt_ref = loss_pdf_old(src_weights, alt_src)
                # loss_pdf_tgt, weights_gt_tgt = loss_pdf_old(tgt_weights, alt_tgt)
                # loss_pdf_ = loss_pdf_ref + loss_pdf_tgt
                # loss_per_image = loss_rgb + loss_ssim + loss_depth_edge + loss_pdf_

                loss_rgb_per_image = loss_rgb_per_image + loss_rgb
                #loss_pdf_per_image = loss_pdf_per_image + loss_pdf_
                loss_ssim_per_image = loss_ssim_per_image + loss_ssim
                loss_lpips_per_tgt_image = loss_lpips_per_tgt_image + loss_lpips

                loss_per_image = loss_per_image + loss_sum

                # plot and save
                #savefigpath = self.outpath + '/weight_avg_400_400/' + 'valbatch_%d_neighbor_%d_pdf.png' % (self.val_id, neighbor_image_idx)

                #plot_save_weights(weights_gt_tgt, tgt_weights, weights_gt_ref, src_weights, savefigpath)


                with torch.no_grad():
                    summary_images["scale_{}".format(scale)] = {
                        "src_image": image_src,
                        "src_rgb_syn": src_rgb_syn,
                        "tgt_image": image_tgt,
                        "tgt_rgb_syn": tgt_rgb_syn,
                        "src_alt_syn": src_alt_syn,
                        #"tgt_alt_syn": tgt_alt_syn,
                        # "src_alt": alt_src,
                        "src_alt_diff": torch.abs(alt_src - src_alt_syn),
                    }

                # self.optimizer.zero_grad()
                # loss_per_image.backward()

                with torch.no_grad():
                    if neighbor_image_idx == 0:
                        summary_scalars.update( {
                            "loss_per_left": loss_per_image.item(),
                            "loss_rgb_left": loss_rgb_per_image.item(),
                            "loss_ssim_left": loss_ssim_per_image.item(),
                            "loss_lpips_left": loss_lpips_per_tgt_image.item(),
                        }
                        )
                    else:
                        summary_scalars.update( {
                            "loss_per_right": loss_per_image.item(),
                            "loss_rgb_right": loss_rgb_per_image.item(),
                            "loss_ssim_right": loss_ssim_per_image.item(),
                            "loss_lpips_right": loss_lpips_per_tgt_image.item()
                            #"loss_depth_edge_right": loss_depth_edge_per_tgt_image.item(),
                        }
                        )

        return summary_scalars, summary_images




if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--conf_path',
                        default='configs/IARPA/mp3.conf', help='config file path')
    parser.add_argument("--resume", default=False, help="continue to train the model")
    # log writer and random seed parameters
    args = parser.parse_args()
    print_args(args)


    # load conf file
    conf_path = args.conf_path
    conf = ConfigFactory.parse_file(conf_path)

    # fix random seed
    torch.manual_seed(conf['train.seed'])
    torch.cuda.manual_seed(conf['train.seed'])

    # training process
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device_ids = [0,1]
    Task = Task_IARPA(conf, device, resume=args.resume)
    Task.train()
    #Task.validate(0, 32)


