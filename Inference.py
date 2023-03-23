import json
import os
import csv
import configargparse
import rasterio
from rasterio import Affine, CRS
from rich.progress import track
from rich.table import Table
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import pandas as pd
import prettytable as PT
from torch.profiler import profile, record_function, ProfilerActivity
from rich.console import Console
import time
import lpips
import matplotlib.pyplot as plt
import torch.cuda
import torch.nn as nn
import torch.autograd.profiler as prof
from utils.miscs import *
from models.mpi_generator_alt import MPIGenerator
from models.feature_generator import FeatureGenerator
from utils.proj_utils import *
from datasets.utils_pushbroom import *
from datasets import *
from torch.utils.data import DataLoader
from utils.render import *
from utils.plot_utils import *
from torch.utils.tensorboard import SummaryWriter
from utils.utils_metric import *
import shutil
from pyhocon import ConfigFactory
#from utils.dsm_utils import *
from collections import defaultdict
from fvcore.nn import FlopCountAnalysis, parameter_count_table

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Recon():
    def __init__(self, conf, device):
        super(Recon, self).__init__()
        self.conf = conf
        self.device = device
        self.console = Console()

        self.lpips = lpips.LPIPS(net="vgg").to(device)
        self.lpips.requires_grad = False
        self.ckptname = conf['test.ckptname']
        self.output_root = conf['test.output_root']
        self.ckpt_root = conf['test.ckpt_root']
        self.dataname = conf['test.dataset']

        # ellipsoid_path = '/home/pc/Documents/ztt/satdata/iarpa_mvs/Train_ex/MasterProvisional3/ellipsoid.json'
        # f = open(ellipsoid_path, encoding='utf-8')
        # cfg = json.load(f)
        # f.close()
        # self.proj = Transverse_Mercator(cfg)

        self.zone_number = 21
        self.hemisphere = "S"


        self.test_dataset = eval('MVSDataset_{}'.format(self.dataname))(self.conf, 'test')
        self.test_dataloader = DataLoader(self.test_dataset, self.conf['train.batch_size'], shuffle=True,
                                          num_workers=self.conf['train.num_workers'], drop_last=True)

        self.ckptpath = conf['test.ckptpath']
        if self.ckptpath == None:
            #self.findckpt()
            ptlist = glob.glob(self.ckpt_root + '/*.ckpt')
            self.ckptpath = ptlist[-1]

        self.feature_generator = FeatureGenerator(model_type=self.conf['model.feature_generator_model_type'], device=self.device, pretrained=False).to(device)
        self.mpi_generator = MPIGenerator(device=self.device, feature_out_chs=self.feature_generator.encoder_channels).to(device)
        state_dict = torch.load(self.ckptpath, map_location=self.device)
        print('ckpt from {} loaded.'.format(self.ckptpath))
        try:
            self.feature_generator.load_state_dict(state_dict["feature_generator"])
        except:
            del state_dict["feature_generator"]['encoder.fc.weight']
            del state_dict["feature_generator"]['encoder.fc.bias']
        self.mpi_generator.load_state_dict(state_dict["mpi_generator"])

        # train_params = sum(params.numel() for params in self.feature_generator.parameters() if params.requires_grad) + \
        #                sum(params.numel() for params in self.mpi_generator.parameters() if params.requires_grad)


        self.modelstatname = os.path.join(self.output_root,  '{}_stats.json'.format(self.ckptname))
        self.outcsvpath = os.path.join(self.output_root,  '{}_result.csv'.format(self.ckptname))
        self.outaltpath = os.path.join(self.output_root, 'Hei', self.ckptname)
        self.outrgbpath = os.path.join(self.output_root, 'RGB', self.ckptname)
        self.outDSMpath = os.path.join(self.output_root, 'DSM', self.ckptname)
        self.outHEIpath = os.path.join(self.output_root, 'HEI', self.ckptname)
        self.outMPIpath = os.path.join(self.output_root, 'MPI', self.ckptname)
        os.makedirs(self.outaltpath, exist_ok=True)
        os.makedirs(self.outrgbpath, exist_ok=True)
        os.makedirs(self.outDSMpath, exist_ok=True)
        os.makedirs(self.outHEIpath, exist_ok=True)
        os.makedirs(self.outMPIpath, exist_ok=True)

        # self.table = Table(title=self.ckptname)
        # self.table.add_column("Names", style="blue", no_wrap=True)
        # self.table.add_column("PSNR_src", style="green")
        # self.table.add_column("LPIPS_src", style="green")
        # self.table.add_column("SSIM_src", style="green")
        # self.table.add_column("PSNR_tgt", style="magenta")
        # self.table.add_column("LPIPS_tgt", style="magenta")
        # self.table.add_column("SSIM_tgt", style="magenta")
        #
        # self.table.add_column("abs", style="cyan")
        # self.table.add_column("<2.5m", style="cyan")
        # self.table.add_column("<5m", style="cyan")
        # self.table.add_column("<7.5m", style="cyan")

        self.table = PT.PrettyTable()
        self.table.field_names = ["Names", "PSNR_src", "LPIPS_src", "SSIM_src", "PSNR_tgt", "LPIPS_tgt", "SSIM_tgt", "abs", "me", "<1.0m",  "<2.5m", "<5m", "<7.5m"]

    def evaluate(self, alt_sample_num=24, inter=True):
        print("Testing process begins")
        # model definition and load ckpt
        with torch.no_grad():
            results = defaultdict(list)
            for batch_idx, sample in enumerate(self.test_dataloader):
                scenename = sample['names'][0][0].split('_')[0]
                nadir = sample['names'][0][0].split('_')[1]
                self.console.rule('batch_idx of scene {} with nadir {}'.format(scenename, nadir))
                metrics = self.render_novel_view_metrics(sample, alt_sample_num, inter)
                #self.console.log(metrics, log_locals=True)

                for k, v in metrics.items():
                    results[k] += [v]
                    self.table.add_row(
                        [sample['names'][0][0],
                        metrics['src_PSNR'],
                        metrics['src_LPIPs'],
                        metrics['src_SSIM'],
                        metrics['tgt_PSNR'],
                        metrics['tgt_LPIPs'],
                        metrics['tgt_SSIM'],
                        metrics['abs'],
                        metrics['Median error'],
                        metrics['acc_1.0m'],
                        metrics['acc_2.5m'],
                        metrics['acc_5.0m'],
                        metrics['acc_7.5m']]
                    )
                    # self.table.add_row(
                    #     sample['names'][0][0],
                    #     "{:.4f}".format(metrics['src_PSNR']),
                    #     "{:.4f}".format(metrics['src_LPIPs']),
                    #     "{:.4f}".format(metrics['src_SSIM']),
                    #     "{:.4f}".format(metrics['tgt_PSNR']),
                    #     "{:.4f}".format(metrics['tgt_LPIPs']),
                    #     "{:.4f}".format(metrics['tgt_SSIM']),
                    #     "{:.4f}".format(metrics['abs']),
                    #     "{:.4f}".format(metrics['acc_2.5m']),
                    #     "{:.4f}".format(metrics['acc_5.0m']),
                    #     "{:.4f}".format(metrics['acc_7.5m'])
                    # )
                self.console.log(f"[green]Finish inference on [/green] {sample['names'][0][0]}")
            self.console.log(f'[bold][red]Done!')
            #self.console.print(self.table)
            print(self.table)

            if os.path.exists(self.outcsvpath):
                os.remove(self.outcsvpath)

            data = pd.DataFrame(results)
            # mode='a'表示追加, index=True表示给每行数据加索引序号, header=False表示不加标题
            data.to_csv(self.outcsvpath, mode='a', index=False, header=True)

    def evaluate_time(self, alt_sample_num=32, inter=False):
        print("Testing process begins")
        # model definition and load ckpt
        with torch.no_grad():
            results = defaultdict(list)
            src_ls, tgt_ls = [], []
            for batch_idx, sample in enumerate(self.test_dataloader):
                scenename = sample['names'][0][0].split('_')[0]
                nadir = sample['names'][0][0].split('_')[1]
                self.console.rule('batch_idx of scene {} with nadir {}'.format(scenename, nadir))
                src_cost, tgt_cost_ls = self.render_novel_view_time(sample, alt_sample_num)
                src_ls.append(src_cost)
                tgt_cost = sum(tgt_cost_ls) / len(tgt_cost_ls)
                tgt_ls.append(tgt_cost)
                print('src time:{} tgt time {}'.format(src_cost, tgt_cost))
                self.console.log(f"[green]Finish inference on [/green] {sample['names'][0][0] }")
            self.console.log(f'[bold][red]Done!')
            #self.console.print(self.table)
            #print(self.table)
            srcmean = sum(src_ls) / len(src_ls)
            tgtmean = sum(tgt_ls) / len(tgt_ls)
            print(srcmean, tgtmean)

    def profile_sample(self, alt_sample_num=32, inter=True):
        print("Testing process begins")
        # model definition and load ckpt
        with torch.no_grad():
            results = defaultdict(list)
            for batch_idx, sample in enumerate(self.test_dataloader):
                scenename = sample['names'][0][0].split('_')[0]
                nadir = sample['names'][0][0].split('_')[1]
                self.console.rule('batch_idx of scene {} with nadir {}'.format(scenename, nadir))
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("model_inference"):
                        src_cost_ls, tgt_cost_ls = self.render_novel_view_time(sample, alt_sample_num, inter)

                print('cost time ')
                self.console.log(f"[green]Finish inference on [/green] {sample['names'][0][0] }")
            self.console.log(f'[bold][red]Done!')
            #self.console.print(self.table)
            print(self.table)

            if os.path.exists(self.outcsvpath):
                os.remove(self.outcsvpath)

            data = pd.DataFrame(results)
            # mode='a'表示追加, index=True表示给每行数据加索引序号, header=False表示不加标题
            data.to_csv(self.outcsvpath, mode='a', index=False, header=True)

    def render_novel_view_time(self, sample, alt_sample_num=24, countmeta=False):
        # model definition and load ckpt
        pairnames = sample['names']
        image_src, rpc_src, alt_src = sample["image_src"].squeeze().to(self.device), sample["rpc_src"].squeeze().to(self.device), \
            sample["alt_src"].unsqueeze(1).to(self.device)
        images_tgt, rpcs_tgt, alts_tgt = sample["images_tgt"].squeeze().to(self.device), sample["rpcs_tgt"].squeeze().to(self.device), sample["alts_tgt"].to(self.device)
        H_render, W_render = image_src.shape[-2], image_src.shape[-1]

        if countmeta:
            flops_encoder = FlopCountAnalysis(self.feature_generator, image_src) # 6707681280
            conv1_out, block1_out, block2_out, block3_out, block4_out = self.feature_generator(image_src)
            input_features = [conv1_out, block1_out, block2_out, block3_out, block4_out]
            flops_decoder = FlopCountAnalysis(self.mpi_generator, input_features)  # 32: 195522001920 ======202229683200
            # flops_decoder_40 = FlopCountAnalysis(self.mpi_generator, (input_features, 40)) # 244163312640===== 250870993920
            # flops_decoder_24 = FlopCountAnalysis(self.mpi_generator, (input_features, 24)) # 146880691200=====153588372480
            # flops_decoder_16 = FlopCountAnalysis(self.mpi_generator, (input_features, 16)) # 98239380480=======104947061760
            print('16-40\n')

            print(flops_decoder)

            ''''
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
            '''
        else:

            print('begin to count time')
            global_start = time.time()
            conv1_out, block1_out, block2_out, block3_out, block4_out = self.feature_generator(image_src)
            mpi_outputs = self.mpi_generator(input_features=[conv1_out, block1_out, block2_out, block3_out, block4_out],
                                        alt_sample_num=alt_sample_num)
            ''' memory
            16:
            torch.cuda.memory_allocated: 0.318607GB
            torch.cuda.memory_reserved: 2.521484GB
            torch.cuda.max_memory_reserved: 2.521484GB
            24:

            32:
            torch.cuda.memory_allocated: 0.318607GB
            torch.cuda.memory_reserved: 2.521484GB
            torch.cuda.max_memory_reserved: 2.521484GB

            40:


            '''
            '''
            torch.cuda.memory_allocated: 0.214922GB
            torch.cuda.memory_reserved: 1.642578GB
            torch.cuda.max_memory_reserved: 1.642578GB
            '''
            rgb_mpi_src = mpi_outputs["MPI_0"][:, :, :3, :, :]
            sigma_mpi_src = mpi_outputs["MPI_0"][:, :, 3:, :, :]

            altmax, altmin = alt_src.max(), alt_src.min()  # GetH_MAX_MIN(rpc_src)
            srcDepthSample = sampleAltitudeInv(altmin.unsqueeze(0), altmax.unsqueeze(0), alt_sample_num)
            src_start = time.time()

            XYH_src = GetXYHfromAltitudes(H_render, W_render, srcDepthSample, rpc_src)  # [B, Nsample, 3, H, W]
            _, src_alt_syn, _, _ = renderSrcViewRPC(
                rgb_MPI_src=rgb_mpi_src,
                sigma_MPI_src=sigma_mpi_src,
                XYH_src=XYH_src,  # double
                use_alpha=False)
            src_end = time.time()
            src_time = src_end - src_start
            print('rendering srd view cost {} seconds'.format(src_time))

            timecosts = []

            for i in range(2):
                tgt_start = time.time()
                tgt_rgb_syn, tgt_alt_syn, tgt_mask, tgt_weights = renderNovelViewRPC(
                        rgb_MPI_src=rgb_mpi_src,
                        sigma_MPI_src=sigma_mpi_src,
                        XYH_src=XYH_src,
                        altSample=srcDepthSample,
                        src_RPC=rpc_src,
                        tgt_RPC=rpcs_tgt[i, :],
                        H_render=H_render,
                        W_render=W_render,
                        use_alpha=False
                    )
                tgt_end = time.time()
                print('render on {}th image cost {} seconds'.format(i, tgt_end-tgt_start))
                timecosts.append(tgt_end - tgt_start)

            global_end = time.time()
            total_time = global_end - global_start
            print('finish io and triple rendering in {} seconds'.format(total_time))
            return src_time, timecosts
        ''' memory
        16:
        torch.cuda.memory_allocated: 0.318607GB
        torch.cuda.memory_reserved: 2.521484GB
        torch.cuda.max_memory_reserved: 2.521484GB
        24:
        
        32:
        torch.cuda.memory_allocated: 0.318607GB
        torch.cuda.memory_reserved: 2.521484GB
        torch.cuda.max_memory_reserved: 2.521484GB
        
        40:
        
        
        '''
        ''' time
        16:

        24:

        32:

        40:


        '''


    def render_novel_view_metrics(self, sample, alt_sample_num=32, inter=True):
        # model definition and load ckpt
        pairnames = sample['names']
        scenename = pairnames[0][0].split('_')[0]
        #valid_mask = sample['masks'][1]
        mpipath = os.path.join(self.outMPIpath, pairnames[0][0])
        if not os.path.exists(mpipath):
            os.mkdir(mpipath)
        image_src, rpc_src, alt_src = sample["image_src"].squeeze().to(self.device), sample["rpc_src"].squeeze().to(self.device), \
            sample["alt_src"].unsqueeze(1).to(self.device)
        images_tgt, rpcs_tgt, alts_tgt = sample["images_tgt"].squeeze().to(self.device), sample["rpcs_tgt"].squeeze().to(self.device), sample["alts_tgt"].to(self.device)
        H_render, W_render = image_src.shape[-2], image_src.shape[-1]

        conv1_out, block1_out, block2_out, block3_out, block4_out = self.feature_generator(image_src.to(self.device))
        mpi_outputs = self.mpi_generator(input_features=[conv1_out, block1_out, block2_out, block3_out, block4_out],
                                         alt_sample_num=alt_sample_num)
        rgb_mpi_src = mpi_outputs["MPI_0"][:, :, :3, :, :]  # [B, Nsample, C=3, Hrender, Wrender]
        sigma_mpi_src = mpi_outputs["MPI_0"][:, :, 3:, :, :]  # [B, Nsample, C=1, Hrender, Wrender]
        if inter:
            rgbmpis = rgb_mpi_src.squeeze(0).permute(0, 2, 3, 1)
            sigmpis = sigma_mpi_src.squeeze(0).permute(0, 2, 3, 1)
            for i in range(0, alt_sample_num, 5):
                rgbmpi = rgbmpis[i, :, :, :].squeeze().detach().cpu().numpy()
                sigmpi = rgbmpis[i, :, :, :].squeeze().detach().cpu().numpy()
                rgbname = 'MPI_RGB_{}.png'.format(i)
                signame = 'MPI_sigma_{}.png'.format(i)
                rgbpath = os.path.join(mpipath, rgbname)
                sigpath = os.path.join(mpipath, signame)
                plt.imsave(rgbpath, rgbmpi)
                plt.imsave(sigpath, sigmpi)
            print('MPIs saved!')

        altmax, altmin = alt_src.max(), alt_src.min()
        srcDepthSample = sampleAltitudeInv(altmin.unsqueeze(0), altmax.unsqueeze(0), alt_sample_num)
        XYH_src = GetXYHfromAltitudes(H_render, W_render, srcDepthSample, rpc_src).to(self.device)  # [B, Nsample, 3, H, W]
        # src_PSNRs, src_LPIPs, src_SSIMs = [], [], []
        tgt_PSNRs, tgt_LPIPs, tgt_SSIMs = [], [], []
        src_abs, mes,src_acc_1ms, src_acc_25ms, src_acc_5ms, src_acc_75ms = [], [], [], [], [], []
        # tgt_abs, tgt_acc_1ms, tgt_acc_25ms, tgt_acc_75ms = [], [], [], []
        src_rgb_syn, src_alt_syn, _, _ = renderSrcViewRPC(
            rgb_MPI_src=rgb_mpi_src,
            sigma_MPI_src=sigma_mpi_src,
            XYH_src=XYH_src,  # double
            use_alpha=False)
        print(torch.mean(src_rgb_syn), torch.max(src_rgb_syn), torch.min(src_rgb_syn))

        src_mask = torch.ones_like(src_rgb_syn.to(torch.float32))
        loss_ssim_src = 1 - ssim(src_rgb_syn.to(torch.float32)*src_mask.to(torch.float32), image_src.to(torch.float32)).mean()
        loss_lpips_src = self.lpips(src_rgb_syn.to(torch.float32)*src_mask.to(torch.float32), image_src.to(torch.float32))
        loss_psnr_src = psnr(src_rgb_syn, image_src, src_mask)
        # src_SSIMs.append(loss_ssim_src)
        # src_LPIPs.append(loss_lpips_src)
        # src_PSNRs.append(loss_psnr_src)

        valid_mask = torch.ones_like(alt_src)

        abs_depth_acc = AbsDepthError_metrics(src_alt_syn, alt_src, valid_mask, 250.0)
        me = ME(src_alt_syn, alt_src, valid_mask)

        acc_1m = Thres_metrics(src_alt_syn, alt_src, valid_mask, 1.0)
        acc_2p5m = Thres_metrics(src_alt_syn, alt_src, valid_mask, 2.5)
        acc_5m = Thres_metrics(src_alt_syn, alt_src, valid_mask, 5.0)  # 0.6
        acc_7p5m = Thres_metrics(src_alt_syn, alt_src, valid_mask, 7.5)
        src_abs.append(abs_depth_acc.item())

        mes.append(me.item())
        src_acc_1ms.append(acc_1m.item())
        src_acc_25ms.append(acc_2p5m.item())
        src_acc_5ms.append(acc_5m.item())
        src_acc_75ms.append(acc_7p5m.item())

        masked_height_map = src_alt_syn.squeeze()



        dsmpath = os.path.join(self.outDSMpath, '{}.tif'.format(scenename))
        #self.proj.getDSM(masked_height_map, rpc_src, dsmpath)

        height_map = masked_height_map.squeeze().detach().cpu().numpy()
        heipath = os.path.join(self.outHEIpath, '{}.png'.format(scenename))

        # plot_height_map(height_map, heipath)
        plot_height_map_triple(alt_src.squeeze().detach().cpu().numpy(), height_map.squeeze(), heipath)
        # heipath2 = os.path.join(self.outHEIpath, '{}_mesh.png'.format(scenename))
        # savepvplot(height_map, heipath2)

        for i in range(2):
            tgt_rgb_syn, tgt_mask = project_src2tgt(src_rgb_syn, src_alt_syn, rpc_src, rpcs_tgt[i, :])
            # tgt_rgb_syn, tgt_alt_syn, tgt_mask, tgt_weights = renderNovelViewRPC(
            #     rgb_MPI_src=rgb_mpi_src,
            #     sigma_MPI_src=sigma_mpi_src,
            #     XYH_src=XYH_src,
            #     altSample=srcDepthSample,
            #     src_RPC=rpc_src,
            #     tgt_RPC=rpcs_tgt[i, :],
            #     H_render=H_render,
            #     W_render=W_render,
            #     use_alpha=False
            # )


            loss_ssim_tgt = 1 - ssim(tgt_rgb_syn * tgt_mask, images_tgt[i, ...]).mean()
            loss_lpips_tgt = self.lpips(tgt_rgb_syn * tgt_mask.detach(), images_tgt[i, ...].detach()).mean()
            loss_psnr_tgt1 = psnr(tgt_rgb_syn * tgt_mask, images_tgt[i, ...]*tgt_mask)
            loss_psnr_tgt = psnr(tgt_rgb_syn, images_tgt[i, ...])
            loss_psnr_tgt = torch.max(loss_psnr_tgt1, loss_psnr_tgt)
            tgt_SSIMs.append(loss_ssim_tgt.item())
            tgt_LPIPs.append(loss_lpips_tgt.item())
            tgt_PSNRs.append(loss_psnr_tgt.item())

            # depth errors:
            #src_mask = torch.ones_like(tgt_mask)


            src_alt_syn_np = src_alt_syn.squeeze().to(torch.float32).to("cpu").numpy()
            # subname = pairnames[i][0].replace('(', '').replace(')', '').replace(',', '').replace('\'', '')
            save_pfm(os.path.join(self.outaltpath, "{}.pfm".format(pairnames[i][0])), src_alt_syn_np)
            image_rendered = tgt_rgb_syn.squeeze().to("cpu").numpy()
            # image_rendered = (image_rendered * 255.).astype(np.uint8)
            # cv2.imwrite(os.path.join(self.outrgbpath, "{}.png".format(pairnames[i])), image_rendered)
            ras_meta = {'driver': 'GTiff',
                        'dtype': 'uint8',
                        'nodata': None,
                        'width': image_rendered.shape[1],
                        'height': image_rendered.shape[0],
                        'count': 3,
                        #'crs': CRS.from_epsg(32736), 'transform': Affine(10.0, 0.0, 653847.1979372115,0.0, -10.0, 7807064.5603836905), 'tiled': False, 'interleave': 'band'
                        #'interleaving': 'interleaving.pixel'
                        }
            image_rendered = (255 * image_rendered).astype(rasterio.uint8)
            with rasterio.open(os.path.join(self.outrgbpath, "{}.tif".format(pairnames[i][0])), 'w', **ras_meta) as dst:
                for i in range(3):
                    dst.write(image_rendered[i, :, :], i+1)
            # with rasterio.open(os.path.join(self.outrgbpath, "{}.tif".format(pairnames[i][0])), 'w') as dst:
            #     dst.write(image_rendered)

        redict = {
            # 'blended_weights': blended_weights,
            # 'src_weights':  src_weights,
            'src_PSNR': loss_psnr_src.item(),  # src_PSNRs,
            'src_SSIM': loss_ssim_src.item(),  # src_SSIMs,
            'src_LPIPs': loss_lpips_src.item(),  # src_LPIPs,
            'tgt_PSNR': np.mean(tgt_PSNRs),
            'tgt_SSIM': np.mean(tgt_SSIMs),
            'tgt_LPIPs': np.mean(tgt_LPIPs),
            'abs': np.mean(src_abs),
            'Median error': np.mean(mes),
            'acc_1.0m': np.mean(src_acc_1ms),
            'acc_2.5m': np.mean(src_acc_25ms),
            'acc_5.0m': np.mean(src_acc_5ms),
            'acc_7.5m': np.mean(src_acc_75ms)
        }
        # redict = {
        #     # 'blended_weights': blended_weights,
        #     # 'src_weights':  src_weights,
        #     'src_PSNR': loss_psnr_src,  # src_PSNRs,
        #     'src_SSIM': loss_ssim_src,  # src_SSIMs,
        #     'src_LPIPs': loss_lpips_src,  # src_LPIPs,
        #     'tgt_PSNR': torch.mean(tgt_PSNRs),
        #     'tgt_SSIM': torch.mean(tgt_SSIMs),
        #     'tgt_LPIPs': torch.mean(tgt_LPIPs),
        #     'abs': torch.mean(src_abs),
        #     'acc_2.5m': torch.mean(src_acc_25ms),
        #     'acc_5.0m': torch.mean(src_acc_5ms),
        #     'acc_7.5m': torch.mean(src_acc_75ms)
        # }
        return redict


if __name__ == "__main__":
    #confpath = 'configs/TLC/all.conf'
    confpath = 'configs/IARPA/mp3.conf'
    conf = ConfigFactory.parse_file(confpath)
    recon = Recon(conf, device)
    recon.evaluate()
    #recon.evaluate_time(alt_sample_num=40)
    #recon.profile_sample()
    '''
    16:
    torch.cuda.memory_allocated: 0.214922GB
    torch.cuda.memory_reserved: 1.642578GB
    torch.cuda.max_memory_reserved: 1.642578GB
    
    torch.cuda.memory_allocated: 0.318607GB
    torch.cuda.memory_reserved: 2.521484GB
    torch.cuda.max_memory_reserved: 2.521484GB
    24:
    
    torch.cuda.memory_allocated: 0.244268GB
    torch.cuda.memory_reserved: 2.619141GB
    torch.cuda.max_memory_reserved: 2.619141GB
    
    
    torch.cuda.memory_allocated: 0.397392GB
    torch.cuda.memory_reserved: 3.939453GB
    torch.cuda.max_memory_reserved: 3.939453GB
    
    32:
    
torch.cuda.memory_allocated: 0.273748GB
torch.cuda.memory_reserved: 3.261719GB
torch.cuda.max_memory_reserved: 3.261719GB

torch.cuda.memory_allocated: 0.476997GB
torch.cuda.memory_reserved: 5.019531GB
torch.cuda.max_memory_reserved: 5.019531GB

40:
torch.cuda.memory_allocated: 0.302351GB
torch.cuda.memory_reserved: 4.548828GB
torch.cuda.max_memory_reserved: 4.548828GB

torch.cuda.memory_allocated: 0.554351GB
torch.cuda.memory_reserved: 6.748047GB
torch.cuda.max_memory_reserved: 6.748047GB

    
    '''