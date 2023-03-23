import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.feature_generator
from models.submodules import *


# Embedder definition
class Embedder(object):
    # Positional encoding
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        try:
            assert type(self.kwargs["max_freq_log2"]) == int
        except:
            print('ddd')
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]

        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            try:
                freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            except:
                print('aaa')
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """

        Args:
            inputs: type:torch.Tensor, shape:[B, nalt]

        Returns:

        """
        output = torch.cat([fn(inputs) for fn in self.embed_fns], -1).reshape(inputs.shape[0], inputs.shape[1], -1)
        return output


class MPIGenerator(nn.Module):
    def __init__(self, device, feature_out_chs, output_channel_num=4, depth_embedding_multires=10, sigma_dropout_rate=0.0, use_skips=True):
        """
        Args:
            feature_out_chs: feature generator output feature channels, for resnet18 np.array([64, 64, 128, 256, 512])
            output_channel_num: MPI generator output channel number, 4 means [R,G,B,sigma]
            sigma_dropout_rate: dropout rate when training sigma branch
        """
        super(MPIGenerator, self).__init__()
        self.device = device
        self.output_channel_num = output_channel_num
        self.depth_embedding_multires = depth_embedding_multires
        self.sigma_dropout_rate = sigma_dropout_rate
        self.use_skips = use_skips

        # depth hypothesis embedder, for depth hypothesis use
        self.depth_embedder, self.embedding_dim = self.depth_embedding(self.depth_embedding_multires)

        # feature extractor, for input features use
        self.encoder_out_ch = [ch + self.embedding_dim for ch in feature_out_chs]
        self.decoder_out_ch = np.array([16, 32, 64, 128, 256])

        # conv layers definition
        final_enc_out_channels = feature_out_chs[-1]
        # self.downsample = nn.MaxPool2d(3, stride=2, padding=1)
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        '''
        for odd size, stride shall be odd
        '''
        self.conv_down1 = conv(final_enc_out_channels, 512, 1, False)
        self.conv_down2 = conv(512, 256, 5, False)
        self.conv_up1 = conv(256, 256, 5, True)
        self.conv_up2 = conv(256, final_enc_out_channels, 1, True)

        # decoder
        self.convs = nn.ModuleDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.encoder_out_ch[-1] if i == 4 else self.decoder_out_ch[i + 1]
            num_ch_out = self.decoder_out_ch[i]
            self.convs[self.tuple_to_str(("upconv", i, 0))] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.decoder_out_ch[i]
            if self.use_skips and i > 0:
                num_ch_in += self.encoder_out_ch[i - 1]
            num_ch_out = self.decoder_out_ch[i]
            self.convs[self.tuple_to_str(("upconv", i, 1))] = ConvBlock(num_ch_in, num_ch_out)

        for s in range(4):
            self.convs[self.tuple_to_str(("dispconv", s))] = Conv3x3(self.decoder_out_ch[s], self.output_channel_num)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, alt_sample_num):
        """
        Args:
            input_features: encoder outputs, 5 scale feature map: [conv1_out, block1_out, block2_out, block3_out, block4_out]
            alt_sample_num: depth sample_number, type:int
        Returns:
            4 scale mpi representations
        """
        batch_size = input_features[0].shape[0]
        # generate depth hypothesis embedding
        depth_hypothesis = self.generate_depth_hypothesis(alt_sample_num=alt_sample_num, batch_size=batch_size)  # [B, ndepth]
        depth_hypothesis_embedding = self.depth_embedder(depth_hypothesis).view(batch_size * alt_sample_num, -1).unsqueeze(2).unsqueeze(3)  # [B*ndepth, embed_dim, 1, 1]

        # extension of encoder to increase receptive field
        encoder_out = input_features[-1]
        # conv_down1 = self.conv_down1(self.downsample(encoder_out))
        # conv_down2 = self.conv_down2(self.downsample(conv_down1))
        # conv_up1 = self.conv_up1(self.upsample(conv_down2))
        # conv_up2 = self.conv_up2(self.upsample(conv_up1))
        conv_down1 = self.conv_down1(encoder_out)
        conv_down2 = self.conv_down2(conv_down1)
        conv_up1 = self.conv_up1(conv_down2)
        conv_up2 = self.conv_up2(conv_up1)

        batchsize, C_feat, H_feat, W_feat = conv_up2.shape
        # if W_feat != encoder_out.shape[-1]:
        #     W_feat = encoder_out.shape[-1]
        # if H_feat != encoder_out.shape[-2]:
        #     H_feat = encoder_out.shape[-2]
        feat_tmp = conv_up2.unsqueeze(1).resize(batchsize, C_feat, H_feat, W_feat)
        feat_tmp = feat_tmp.expand(batch_size, alt_sample_num, C_feat, H_feat, W_feat).contiguous().view(batch_size * alt_sample_num, C_feat, H_feat, W_feat)
        depth_hypothesis_embedding_tmp = depth_hypothesis_embedding.repeat(1, 1, H_feat, W_feat).to(self.device)  # [B*ndepth, embed_dim, H_feat, W_feat]
        x = torch.cat((feat_tmp, depth_hypothesis_embedding_tmp), dim=1)  # [32, 533, 12, 15]

        # input features processing, concatenate depth hypothesis embedding for each feature
        for i, feature_map in enumerate(input_features):
            _, C_feat, H_feat, W_feat = feature_map.shape
            feat_tmp = feature_map.unsqueeze(1).expand(batch_size, alt_sample_num, C_feat, H_feat, W_feat).contiguous().view(batch_size*alt_sample_num, C_feat, H_feat, W_feat)
            depth_hypothesis_embedding_tmp = depth_hypothesis_embedding.repeat(1, 1, H_feat, W_feat).to(self.device)  # [B*ndepth, embed_dim, H_feat, W_feat]
            input_features[i] = torch.cat((feat_tmp, depth_hypothesis_embedding_tmp), dim=1)    # concatenate depth embedding at each input feature scale

        # generate 4-scale mpi representation
        outputs = {}
        outputs['depth_hypothesis'] = depth_hypothesis.squeeze()
        for i in range(4, -1, -1):
        #for i in range(5):
            x = self.convs[self.tuple_to_str(("upconv", i, 0))](x)  # [32, 256, 12, 16]
            x = [upsample(x)] # [32, 256, 24, 32]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            '''
            IARPA
            x:[
            0: [32 256, 24 32]
            1: [32, 277, 24, 30]
            ]
            '''
            x = torch.cat(x, 1)  # wrong
            x = self.convs[self.tuple_to_str(("upconv", i, 1))](x)
            if i in range(4):
                output = self.convs[self.tuple_to_str(("dispconv", i))](x)
                H_mpi, W_mpi = output.size(2), output.size(3)
                mpi = output.view(batch_size, alt_sample_num, 4, H_mpi, W_mpi)
                mpi_rgb = self.sigmoid(mpi[:, :, 0:3, :, :])
                mpi_sigma = torch.abs(mpi[:, :, 3:, :, :]) + 1e-4
                if self.sigma_dropout_rate > 0.0 and self.training:
                    mpi_sigma = F.dropout2d(mpi_sigma, p=self.sigma_dropout_rate)

                outputs["MPI_{}".format(i)] = torch.cat((mpi_rgb, mpi_sigma), dim=2)

        return outputs

    def tuple_to_str(self, key_tuple):
        key_str = '-'.join(str(key_tuple))
        return key_str

    def generate_depth_hypothesis(self, alt_sample_num, batch_size, depth_min=0.01, depth_max=1.0):
        """
        To generate depth hypothesis uniformly sample in range (depth_min, depth_max]
        Args:
            alt_sample_num: depth sample number, type:int
            batch_size: batch size, type: int

        Returns:
            depth_hypothesis: depth hypothesis, type:torch.Tensor, shape:[B, ndepth]
        """
        depth_hypothesis = torch.linspace(start=depth_min, end=depth_max, steps=alt_sample_num)
        depth_hypothesis = depth_hypothesis.unsqueeze(0).repeat(batch_size, 1)
        return depth_hypothesis

    def upsample_depth_hypothesis(self, depth_upsample_num, batch_size, depth_min=0.01, depth_max=1.0):
        """
        To update depth hypothesis according to sample in range (depth_min, depth_max]
        todo
        """
        pass

    def depth_embedding(self, multires=10):
        embed_kwargs = {
            "include_input": True,
            "input_dims": 1,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }
        embedder = Embedder(**embed_kwargs)

        def embed(x, eo=embedder):
            return eo.embed(x)

        return embed, embedder.out_dim


if __name__ == '__main__':
    print("MPI generator")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import cv2
    from models.feature_generator import FeatureGenerator
    from PIL import Image
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureGenerator(model_type="resnet18", pretrained=True).to(device)

    #batch_size, height, width = 1, 256, 256
    #ref_images = 256 * torch.rand((batch_size, 3, height, width), device=device)
    refimg0 = torch.from_numpy(cv2.imread('/home/pc/Desktop/WHU-TLC/open_dataset_rpc/train/image/0/base0000block0008.png')).unsqueeze(0)
    refimg1 = torch.from_numpy(cv2.imread('/home/pc/Desktop/WHU-TLC/open_dataset_rpc/train/image/1/base0000block0008.png')).unsqueeze(0)
    #refimg2 = torch.from_numpy(cv2.imread('/home/pc/Desktop/WHU-TLC/open_dataset_rpc/train/image/2/base0000block0008.png')).unsqueeze(0)
    #ref_images = torch.cat((refimg0, refimg1, refimg2), dim=0).permute(0,3,1,2).cuda()
    ref_images = torch.cat((refimg0, refimg1), dim=0).permute(0, 3, 1, 2).cuda()
    conv1_out, block1_out, block2_out, block3_out, block4_out = model(ref_images)

    mpi_generator = MPIGenerator(np.array([64, 64, 128, 256, 512])).to(device)

    total_params = sum(params.numel() for params in mpi_generator.parameters())
    train_params = sum(params.numel() for params in mpi_generator.parameters() if params.requires_grad)
    print("total_paramteters: {}, train_parameters: {}".format(total_params, train_params))
    input_features = [conv1_out, block1_out, block2_out, block3_out, block4_out]
    #input_features2 = input_features[::-1]
    mpis = mpi_generator(input_features, alt_sample_num=32)
    np.savez('/home/pc/Documents/ztt/gitrepos/SatMINE/test_asset/mpis',
             mpi3 = mpis['MPI_3'].detach().cpu().numpy(),
             mpi2 = mpis['MPI_2'].detach().cpu().numpy(),
             mpi1 = mpis['MPI_1'].detach().cpu().numpy(),
             mpi0 = mpis['MPI_0'].detach().cpu().numpy(),
             depthHyps = mpis['depth_hypothesis'].detach().cpu().numpy())

    for key, value in mpis.items():
        print(key, value.shape)

