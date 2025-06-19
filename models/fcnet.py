import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from torchvision import utils as vutils

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def preprocess_lab(lab): # standrad LAB -> [-1, 1]
		L_chan, a_chan, b_chan =torch.unbind(lab,dim=2)
		# L_chan: black and white with input range [0, 100]
		# a_chan/b_chan: color channels with input range ~[-110, 110], not exact
		# [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
		return [L_chan / 50.0 - 1.0, a_chan / 110.0, b_chan / 110.0]

def deprocess_lab(L_chan, a_chan, b_chan): # [-1, 1] -> standrad LAB
		#TODO This is axis=3 instead of axis=2 when deprocessing batch of images 
			   # ( we process individual images but deprocess batches)
		#return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
		return torch.stack([(L_chan + 1) / 2.0 * 100.0, a_chan * 110.0, b_chan * 110.0], dim=2)

def lab_to_rgb(lab):
    assert lab.dim() == 4
    b, c, h, w = lab.size()
    assert c == 3
    lab_pixels = torch.reshape(lab.permute(0, 2, 3, 1), [-1, 3])
    # convert to fxfyfz
    lab_to_fxfyfz = torch.tensor([
        #   fx      fy        fz
        [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
        [1 / 500.0, 0.0, 0.0],  # a
        [0.0, 0.0, -1 / 200.0],  # b
    ]).to(lab.device)
    fxfyfz_pixels = torch.mm(
        lab_pixels + torch.tensor([16.0, 0.0, 0.0]).to(lab.device),
        lab_to_fxfyfz,
    )

    # convert to xyz
    epsilon = 6.0 / 29.0
    linear_mask = ((fxfyfz_pixels <= epsilon).type(torch.FloatTensor).to(lab.device))
    exponential_mask = ((fxfyfz_pixels > epsilon).type(torch.FloatTensor).to(lab.device))

    xyz_pixels = (3 * epsilon**2 *
                  (fxfyfz_pixels - 4 / 29.0)) * linear_mask + ((fxfyfz_pixels + 0.000001)**3) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).to(lab.device))

    xyz_to_rgb = torch.tensor([
        #     r           g          b
        [3.2404542, -0.9692660, 0.0556434],  # x
        [-1.5371385, 1.8760108, -0.2040259],  # y
        [-0.4985314, 0.0415560, 1.0572252],  # z
    ]).to(lab.device)

    rgb_pixels = torch.mm(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    # clip
    rgb_pixels[rgb_pixels > 1] = 1
    rgb_pixels[rgb_pixels < 0] = 0

    linear_mask = ((rgb_pixels <= 0.0031308).type(torch.FloatTensor).to(lab.device))
    exponential_mask = ((rgb_pixels > 0.0031308).type(torch.FloatTensor).to(lab.device))
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((
        (rgb_pixels + 0.000001)**(1 / 2.4) * 1.055) - 0.055) * exponential_mask

    return torch.reshape(srgb_pixels, [b, h, w, c]).permute(0, 3, 1, 2)

def rgb_to_lab(srgb):
    assert srgb.dim() == 4
    b, c, h, w = srgb.size()
    assert c == 3
    srgb_pixels = torch.reshape(srgb.permute(0, 2, 3, 1), [-1, 3])

    linear_mask = ((srgb_pixels <= 0.04045).type(torch.FloatTensor).to(srgb.device))
    exponential_mask = ((srgb_pixels > 0.04045).type(torch.FloatTensor).to(srgb.device))
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055)**2.4) * exponential_mask

    rgb_to_xyz = torch.tensor([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ]).to(srgb.device)

    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)

    # XYZ to Lab
    xyz_normalized_pixels = torch.mul(
        xyz_pixels,
        torch.tensor([1 / 0.950456, 1.0, 1 / 1.088754]).to(srgb.device),
    )

    epsilon = 6.0 / 29.0

    linear_mask = ((xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).to(srgb.device))

    exponential_mask = ((xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).to(srgb.device))

    fxfyfz_pixels = ((xyz_normalized_pixels / (3 * epsilon**2) + 4.0 / 29.0) * linear_mask +
                     ((xyz_normalized_pixels + 0.000001)**(1.0 / 3.0)) * exponential_mask)
    # convert to lab
    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [0.0, 500.0, 0.0],  # fx
        [116.0, -500.0, 200.0],  # fy
        [0.0, 0.0, -200.0],  # fz
    ]).to(srgb.device)
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).to(srgb.device)
    # return tf.reshape(lab_pixels, tf.shape(srgb))
    return torch.reshape(lab_pixels, [b, h, w, c]).permute(0, 3, 1, 2)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])  # share same NC
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def save_image_tensor(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)

def remake_masks(masks):
    masks[:, 2:3] = masks[:, 2:4].sum(dim=1, keepdim=True)
    masks[:, 2:3][masks[:, 2:3] > 1] = 1                              # attention this!!!
    masks = torch.concat((masks[:, 0:3], masks[:, 4:]), dim=1) 
    masks[masks > 1] = 1                                              # ensure no overlap

    mask_background = masks.sum(dim=1, keepdim=True)  # torch.zeros_like(masks[:, 0:1, :, :]) # N 1 256 256
    mask_background[mask_background != 0] = -1
    mask_background += 1

    masks = torch.cat([masks, mask_background], dim=1)
    return masks

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel_size, kernel_size)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class RSIM_SPADE_Block(nn.Module):
    def __init__(self, input_channels, output_channels, num_domains, ks=3):
        '''
        :param input_channels:      input_channels
        :param output_channels:     output_channels
        :param num_domains:         face edit domains
        '''
        super().__init__()
        self.num_domains = num_domains

        self.conv1x1s = nn.ModuleList()
        for _ in range(num_domains):
            self.conv1x1s += [nn.Conv2d(input_channels, output_channels, 1, 1, 0)]
        
        nhidden = 512
        self.param_free_norm = nn.InstanceNorm2d(input_channels, affine=False)

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=ks, padding=pw),
            nn.LeakyReLU(0.2) #nn.ReLU()
        )
        self.mlp_shared_2 = nn.Sequential(
            nn.Conv2d(output_channels, nhidden, kernel_size=ks, padding=pw),
            nn.LeakyReLU(0.2) #nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, input_channels, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, input_channels, kernel_size=ks, padding=pw)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, y, masks):
        '''
        :param x:           Source feature map, NCHW
        :param y:           reference feature map, NCHW
        :param masks:       0/1 masks of every part, N domain_num H W
        :return:            NCHW
        '''
        RATs = y
        MTs = []

        for i in range(self.num_domains):
            RAT = self.conv1x1s[i](RATs * masks[:, i:i + 1])
            MTs.append(RAT)  # list of NCHW, len is num_domain

        feat_y = sum(MTs)
        normalized = self.param_free_norm(x)
        
        actv = self.mlp_shared_2(self.mlp_shared(feat_y))
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        
        out = normalized * (1 + gamma) + beta

        return out

    def forward_mul(self, x, y, masks):
        '''
        :param x:           Source feature map, NCHW
        :param y:           reference feature map, [NCHW]  list, len==domain_num
        :param masks:       0/1 masks of every part, N domain_num H W
        :return:            NCHW
        '''
        RATs = y
        MTs = []

        for i in range(self.num_domains):
            RAT = self.conv1x1s[i](RATs[i] * masks[:, i:i + 1])
            MTs.append(RAT)  # list of NCHW, len is num_domain
        MT = torch.cat(MTs, dim=1)  # N filters * num_domain H W

        feat_y = self.conv1x1_last(MT)
        normalized = self.param_free_norm(x)
        
        actv = self.mlp_shared(feat_y)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        
        out = normalized * (1 + gamma) + beta

        return out

class SPADE(nn.Module):
    # Creates SPADE normalization layer based on the given configuration
    # SPADE consists of two steps. First, it normalizes the activations using
    # your favorite normalization method, such as Batch Norm or Instance Norm.
    # Second, it applies scale and bias to the normalized output, conditioned on
    # the segmentation map.
    # The format of |config_text| is spade(norm)(ks), where
    # (norm) specifies the type of parameter-free normalization.
    #       (e.g. syncbatch, batch, instance)
    # (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
    # Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
    # Also, the other arguments are
    # |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
    # |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
    def __init__(self, norm_nc, label_nc, norm_type='instance', ks=3): # x channel; mask channel
        super().__init__()
        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % norm_type)
        
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.LeakyReLU(0.2) #nn.ReLU() 
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        #if x.size()[2:] != segmap.size()[2:]:
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class AdainResBlk_dummy(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        out = self._residual(x)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class HighPass(nn.Module):
    def __init__(self, w_hpf):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

class ResBlk_Grouped(nn.Module):
    def __init__(self, dim_in, dim_out, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False, groups=1):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, groups)

    def _build_weights(self, dim_in, dim_out, groups):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1, groups=groups)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1, groups=groups)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False, groups=groups)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s=None):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class SPADE_Grouped(nn.Module):
    # (norm) specifies the type of parameter-free normalization.
    #       (e.g. syncbatch, batch, instance)
    # (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
    # Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
    # |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
    # |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
    def __init__(self, norm_nc, label_nc, norm_type='instance', ks=3, groups=1):  # x channel; mask channel
        super().__init__()
        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % norm_type)
        nhidden = 128 * groups

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw, groups=groups),
            nn.LeakyReLU(0.2)  # nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, groups=groups)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw, groups=groups)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # if x.size()[2:] != segmap.size()[2:]:
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class RSIM_SPADE_Block_Grouped(nn.Module):
    def __init__(self, input_channels, output_channels, num_domains, ks=3):
        '''
        :param input_channels:      input_channels
        :param output_channels:     output_channels
        :param num_domains:         face edit domains
        '''
        super().__init__()
        self.num_domains = num_domains
        self.conv1x1s = nn.ModuleList()
        for _ in range(num_domains):
            self.conv1x1s += [nn.Conv2d(input_channels, output_channels, 1, 1, 0)]

        nhidden = 512
        self.param_free_norm = nn.InstanceNorm2d(input_channels, affine=False)

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=ks, padding=pw),
            nn.LeakyReLU(0.2)  # nn.ReLU()
        )
        self.mlp_shared_2 = nn.Sequential(
            nn.Conv2d(output_channels, nhidden, kernel_size=ks, padding=pw),
            nn.LeakyReLU(0.2)  # nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, input_channels, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, input_channels, kernel_size=ks, padding=pw)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, y, masks):
        '''
        :param x:           Source feature map, NCHW
        :param y:           reference feature map, N C*domain_num HW
        :param masks:       0/1 masks of every part, N domain_num H W
        :return:            NCHW
        '''
        N, C, H, W = x.shape
        RATs = y
        MTs = []

        for i in range(self.num_domains):
            RAT = self.conv1x1s[i](
                RATs[:, i * C:(i + 1) * C] * masks[:, i:i + 1])
            MTs.append(RAT)  # list of NCHW, len is num_domain

        feat_y = sum(MTs)  # here should be N, output_channels, H, W # feat_y torch.Size([4, 256, 64, 64])
        normalized = self.param_free_norm(x)

        actv = self.mlp_shared_2(self.mlp_shared(feat_y))
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out

    def forward_mul(self, x, y, masks):
        '''
        :param x:           Source feature map, NCHW
        :param y:           reference feature map, [N C*domain_num HW]  list, len==domain_num
        :param masks:       0/1 masks of every part, N domain_num H W
        :return:            NCHW
        '''
        N, C, H, W = x.shape
        RATs = y
        MTs = []

        for i in range(self.num_domains):
            RAT = self.conv1x1s[i](RATs[i][:, i * C:(i + 1) * C] * masks[:,
                                                                   i:i + 1])  # RAT = self.conv1x1s[i](RATs[i] * masks[:, i:i + 1])
            MTs.append(RAT)  # list of NCHW, len is num_domain
        MT = torch.cat(MTs, dim=1)  # N filters * num_domain H W

        feat_y = self.conv1x1_last(MT)
        normalized = self.param_free_norm(x)

        actv = self.mlp_shared(feat_y)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out

class RSIM_SPADE_Block_Grouped__alter(nn.Module):
    def __init__(self, input_channels, output_channels, num_domains, ks=3):
        '''
        :param input_channels:      input_channels
        :param output_channels:     output_channels
        :param num_domains:         face edit domains
        '''
        super().__init__()
        self.num_domains = num_domains

        self.conv1x1 = nn.Conv2d(input_channels * num_domains, output_channels * num_domains, kernel_size=1, padding=0, groups=num_domains)

        nhidden = 512
        self.param_free_norm = nn.InstanceNorm2d(input_channels, affine=False)

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=ks, padding=pw),
            nn.LeakyReLU(0.2)  # nn.ReLU()
        )
        self.mlp_shared_2 = nn.Sequential(
            nn.Conv2d(output_channels, nhidden, kernel_size=ks, padding=pw),
            nn.LeakyReLU(0.2)  # nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, input_channels, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, input_channels, kernel_size=ks, padding=pw)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, y, masks):
        '''
        :param x:           Source feature map, NCHW
        :param y:           reference feature map, N C*domain_num HW
        :param masks:       0/1 masks of every part, N domain_num H W
        :return:            NCHW
        '''
        N, C, H, W = x.shape
        MTs = []

        y_ = self.conv1x1(y)
        for i in range(self.num_domains):
            RAT = y_[:, i * C:(i + 1) * C] * masks[:, i:i + 1]
            MTs.append(RAT)
        feat_y = sum(MTs)
        normalized = self.param_free_norm(x)

        actv = self.mlp_shared_2(self.mlp_shared(feat_y))
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out

class ResBlk_Grouped_dsz(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 downsample=False, groups=1):
        super().__init__()
        self.actv = actv
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, groups)

    def _build_weights(self, dim_in, dim_out, groups):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1, groups=groups)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1, groups=groups)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False, groups=groups)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class ColorDomain_transformation(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.lab_to_fxfyfz = torch.tensor([
        #   fx      fy        fz
        [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
        [1 / 500.0, 0.0, 0.0],  # a
        [0.0, 0.0, -1 / 200.0],  # b
        ]).type(torch.cuda.FloatTensor)

        self.fxfyfz_pixels_mid = torch.tensor([16.0, 0.0, 0.0]).type(torch.cuda.FloatTensor)

        self.epsilon = 6.0 / 29.0

        self.xyz_pixels_mid = torch.tensor([0.950456, 1.0, 1.088754]).type(torch.cuda.FloatTensor)

        self.xyz_to_rgb = torch.tensor([
            #     r           g          b
            [3.2404542, -0.9692660, 0.0556434],  # x
            [-1.5371385, 1.8760108, -0.2040259],  # y
            [-0.4985314, 0.0415560, 1.0572252],  # z
        ]).type(torch.cuda.FloatTensor)

        self.rgb_to_xyz = torch.tensor([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
        ]).type(torch.cuda.FloatTensor)

        self.xyz_normalized_pixels_mid = torch.tensor([1 / 0.950456, 1.0, 1 / 1.088754]).type(torch.cuda.FloatTensor)

        self.fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [0.0, 500.0, 0.0],  # fx
        [116.0, -500.0, 200.0],  # fy
        [0.0, 0.0, -200.0],  # fz
        ]).type(torch.cuda.FloatTensor)

        self.lab_pixels_mid = torch.tensor([-16.0, 0.0, 0.0]).type(torch.cuda.FloatTensor)

    def forward(self, ):
        pass

    def rgb_to_lab(self, srgb):
        assert srgb.dim() == 4
        b, c, h, w = srgb.size()
        assert c == 3
        srgb_pixels = torch.reshape(srgb.permute(0, 2, 3, 1), [-1, 3])

        linear_mask = (srgb_pixels <= 0.04045)
        exponential_mask = (srgb_pixels > 0.04045)
        rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
        
        xyz_pixels = torch.mm(rgb_pixels, self.rgb_to_xyz)

        xyz_normalized_pixels = torch.mul(
            xyz_pixels,
            self.xyz_normalized_pixels_mid,
        )

        linear_mask = (xyz_normalized_pixels <= (self.epsilon ** 3))

        exponential_mask = (xyz_normalized_pixels > (self.epsilon ** 3))

        fxfyfz_pixels = ((xyz_normalized_pixels / (3 * self.epsilon ** 2) + 4.0 / 29.0) * linear_mask +
                         ((xyz_normalized_pixels + 0.000001) ** (1.0 / 3.0)) * exponential_mask)

        # convert to lab
        lab_pixels = torch.mm(fxfyfz_pixels, self.fxfyfz_to_lab) + self.lab_pixels_mid

        return torch.reshape(lab_pixels, [b, h, w, c]).permute(0, 3, 1, 2)

    def lab_to_rgb(self, lab):
        assert lab.dim() == 4
        b, c, h, w = lab.size()
        assert c == 3
        lab_pixels = torch.reshape(lab.permute(0, 2, 3, 1), [-1, 3])
        # convert to fxfyfz
        fxfyfz_pixels = torch.mm(
            lab_pixels + self.fxfyfz_pixels_mid,
            self.lab_to_fxfyfz,
        )
        # convert to xyz
        linear_mask = (fxfyfz_pixels <= self.epsilon)
        exponential_mask = (fxfyfz_pixels > self.epsilon)

        xyz_pixels = (3 * self.epsilon ** 2 *
                      (fxfyfz_pixels - 4 / 29.0)) * linear_mask + ((fxfyfz_pixels + 0.000001) ** 3) * exponential_mask

        # denormalize for D65 white point
        xyz_pixels = torch.mul(xyz_pixels, self.xyz_pixels_mid)

        rgb_pixels = torch.mm(xyz_pixels, self.xyz_to_rgb)
        # avoid a slightly negative number messing up the conversion
        rgb_pixels = torch.clip(rgb_pixels, min=0, max=1)

        linear_mask = (rgb_pixels <= 0.0031308)
        exponential_mask = (rgb_pixels > 0.0031308)
        srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (( (rgb_pixels + 0.000001) ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

        return torch.reshape(srgb_pixels, [b, h, w, c]).permute(0, 3, 1, 2)

class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=0, num_domains=4, RSIM_size=128,
                 use_mask=True):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(1, dim_in, 3, 1, 1) # lab luma as input
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.style_dim = style_dim

        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 2, 1, 1, 0))
        self.w_hpf = w_hpf
        
        self.actv = nn.LeakyReLU(0.2)

        self.use_mask = use_mask
        self.RSIM_size = RSIM_size

        self.from_rgb_ref = nn.Conv2d(2, dim_in, 3, 1, 1)  # only lab's ab channel
        encode_ref = []
        decode_ref = nn.ModuleList()
        decode_ref_spades = nn.ModuleList()
        
        ref_num = int(np.log2(img_size))
        ref_dim_in = dim_in
        for _ in range(ref_num):
            ref_dim_out = min(ref_dim_in * 2, max_conv_dim)
            encode_ref += [ResBlk(int(ref_dim_in), int(ref_dim_out), downsample=True)]
            ref_dim_in = ref_dim_out
        # 256 ->128->64->32->16->8->4->2->1
        self.encode_ref = nn.Sequential(*encode_ref)

        self.encode_ref_tail = nn.Conv2d(ref_dim_out, style_dim * (num_domains + 1), 1, 1, 0)
        self.decode_ref_head = nn.Conv2d(style_dim * (num_domains + 1), ref_dim_out, 1, 1, 0)
        
        self.SPADE_head = SPADE(norm_nc=dim_in, label_nc=num_domains + 1, norm_type='instance')

        max_decoder_conv_dim = 128

        self.decode_ref_head = nn.Conv2d(style_dim * (num_domains + 1), max_decoder_conv_dim * (num_domains + 1), 1, 1,
                                         0, groups=(num_domains + 1))  # changed

        cut_num = ref_num - int(np.log2(img_size / RSIM_size))  # 8-2 = 6
        dec_dim_in = max_decoder_conv_dim * (num_domains + 1)  # dec_dim_in = ref_dim_out  # 128 = 2^7
        for idx in range(cut_num):
            if idx < cut_num - 4:
                dec_dim = max_decoder_conv_dim * (num_domains + 1)
                dec_dim_in = max_decoder_conv_dim * (num_domains + 1)

            decode_ref.append(ResBlk_Grouped(int(dec_dim_in), int(dec_dim), w_hpf=0, upsample=True, groups=(num_domains + 1)))
            decode_ref_spades.append(SPADE_Grouped(norm_nc=int(dec_dim), label_nc=num_domains + 1, norm_type='instance',
                                        groups=(num_domains + 1)))

            dec_dim_in = dec_dim
            dec_dim = max(dec_dim / 2, 32 * (num_domains + 1))
        dec_dim = dec_dim_in
        self.decode_ref = decode_ref
        self.decode_ref_spades = decode_ref_spades 

        repeat_num = int(np.log2(img_size / RSIM_size))
        
        self.SPADEs = nn.ModuleList()
        
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))

            self.decode.insert(
                0, AdainResBlk_dummy(dim_out, dim_in, style_dim,
                                     w_hpf=w_hpf, upsample=True))  # stack-like

            self.SPADEs.insert(0, SPADE(norm_nc=dim_out, label_nc= 1, norm_type='instance'))
            
            dim_in = dim_out

        self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
        self.decode.insert(0, AdainResBlk_dummy(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)
 
        self.conv1x1_self = nn.Conv2d(int(dec_dim), int(dim_out * (num_domains + 1)), 1, 1, 0, groups=(num_domains + 1))
        self.conv1x1_gamma = nn.Conv2d(int(dec_dim), int(dim_out * (num_domains + 1)), 1, 1, 0,
                                       groups=(num_domains + 1))
        self.conv1x1_beta = nn.Conv2d(int(dec_dim), int(dim_out * (num_domains + 1)), 1, 1, 0, groups=(num_domains + 1))

        self.RSIM = RSIM_SPADE_Block_Grouped__alter(input_channels=dim_out, output_channels=dim_out, num_domains=num_domains + 1)

        self.convert_color = ColorDomain_transformation()
        self.lock_net_param(self.convert_color)

    def Get_y_sty(self, y, y_mask):
        '''
        y: ref images   N 3 H W
        return:
        y_sty           N 512 1 1
        '''
        y_in = self.convert_color.rgb_to_lab(de_norm(y))        # -1,1 -> 0,1 -> lab
        y_in = y_in[:,1:3] / 110.0                              # a,b channel -110,110 -> -1,1
        y = self.from_rgb_ref(y_in)
        
        y = self.SPADE_head(y, y_mask)
        y_sty = self.encode_ref(y)  # y_sty
        y_sty = self.encode_ref_tail(y_sty)
        return y_sty
    
    def lock_net_param(self, module):
        for parameter in module.parameters():
            parameter.requires_grad = False

    def forward(self, x, y, masks_x, masks_y, is_training=False):
        '''
        x: input images N 3 H W
        y: ref   images N 3 H W
        masks_x:        N num_domains+1 H W
        # lip skin left_eye right_eye hair
        '''
        _,C,_,_ = masks_x.shape
        masks_x_org = masks_x
        masks_x_re = F.interpolate(masks_x_org, size=self.RSIM_size, mode='nearest')
        
        x_in = self.convert_color.rgb_to_lab(de_norm(x)) # -1,1 -> 0,1 -> lab
        x_in = x_in[:,0:1] / 50.0 - 1.0            # luma channel 0-100 -> -1,1
        y_in = self.convert_color.rgb_to_lab(de_norm(y)) # -1,1 -> 0,1 -> lab
        y_in = y_in[:,1:3] / 110.0           # a,b channel -110,110 -> -1,1
        
        x_gray = x_in # N1HW -1,1

        x = self.from_rgb(x_in)
        y = self.from_rgb_ref(y_in)

        for block in self.encode:
            x = block(x)
        
        y = self.SPADE_head(y, masks_y)
        y = self.encode_ref(y) # y_sty
        y = self.encode_ref_tail(y)
            
        if is_training:
            y_sty = y

        y = self.decode_ref_head(y)

        for idxx in range(len(self.decode_ref)):
            y = self.decode_ref[idxx](y)
            y = self.decode_ref_spades[idxx](y, masks_x_org)
        
        y = self.conv1x1_self(y) * self.conv1x1_gamma(y) + self.conv1x1_beta(y)

        x = self.RSIM.forward(x, y, masks_x_re)
        
        for idx in range(len(self.decode)):
            if idx>= 1:
                x = self.SPADEs[idx-1](x, x_gray)
            x = self.decode[idx](x, s=None)
                    
        x_ab = self.to_rgb(x)
        fake_rgb = self.convert_color.lab_to_rgb(torch.cat((x_in * 50.0 + 50.0, x_ab * 110.0), dim=1))   # 0,1
        fake_rgb = (fake_rgb - 0.5)/0.5  # 0,1 -> -1,1
        
        if is_training:
            return fake_rgb, y_sty
        else:
            return fake_rgb

    def forward_with_colorcode(self, x, y_sty, masks_x):
        '''
        x: input images   N 3 H W
        y_sty: colorcode  N dim
        masks_x:          N num_domains H W
        # lip skin left_eye right_eye hair
        '''
        _,C,_,_ = masks_x.shape
        masks_x_org = masks_x
        masks_x_re = F.interpolate(masks_x_org, size=self.RSIM_size, mode='nearest')

        x_in = self.convert_color.rgb_to_lab(de_norm(x)) # -1,1 -> 0,1 -> lab
        x_in = x_in[:,0:1] / 50.0 - 1.0            # luma channel 0-100 -> -1,1
        
        x_gray = x_in # N1HW -1,1
        
        x = self.from_rgb(x_in)

        for block in self.encode:
            x = block(x)

        y = y_sty

        y = self.decode_ref_head(y)
        for idx in range(len(self.decode_ref)):
            y = self.decode_ref[idx](y)
            y = self.decode_ref_spades[idx](y, masks_x_org)

        y = self.conv1x1_self(y) * self.conv1x1_gamma(y) + self.conv1x1_beta(y)

        x = self.RSIM.forward(x, y, masks_x_re)

        for idx in range(len(self.decode)):
            if idx>= 1:
                #x = self.SPADEs[idx-1](x, masks_x_org)
                x = self.SPADEs[idx-1](x, x_gray)
            x = self.decode[idx](x, s=None)
        
        x_ab = self.to_rgb(x)
        fake_rgb = self.convert_color.lab_to_rgb(torch.cat((x_in * 50.0 + 50.0, x_ab * 110.0), dim=1))   #0,1
        fake_rgb = (fake_rgb - 0.5)/0.5  # 0,1 -> -1,1
        
        return fake_rgb

    def merge_sty_codes(self, sty_codes, STYLECODE_DIM):
        '''
        sty_codes:  [N, STYLECODE_DIM * (num_domains+1), 1, 1]
        '''
        _,C,_,_ = sty_codes[0].shape
        assert C == 5 * STYLECODE_DIM
        output = sty_codes[0].clone()
        output[:,STYLECODE_DIM * 1:STYLECODE_DIM * 2] = sty_codes[1][:,STYLECODE_DIM * 1:STYLECODE_DIM * 2]
        output[:,STYLECODE_DIM * 2:STYLECODE_DIM * 3] = sty_codes[2][:,STYLECODE_DIM * 2:STYLECODE_DIM * 3]
        output[:,STYLECODE_DIM * 3:STYLECODE_DIM * 4] = sty_codes[3][:,STYLECODE_DIM * 3:STYLECODE_DIM * 4]
        output[:,STYLECODE_DIM * 4:STYLECODE_DIM * 5] = sty_codes[4][:,STYLECODE_DIM * 4:STYLECODE_DIM * 5]
        return output

    def forward_with_mul_colorcode(self, x, masks_x, y_sty_1, y_sty_2=None, y_sty_3=None, y_sty_4=None, y_sty_5=None):
        '''
        x: input images   N 3 H W
        y_sty: colorcode  N dim
        masks_x:          N num_domains H W
        # lip skin eyes hair background
        '''
        y_stys = [y_sty_1, y_sty_2, y_sty_3, y_sty_4, y_sty_5]
        fin_sty = self.merge_sty_codes(y_stys, self.style_dim)
        
        fake_rgb = self.forward_with_colorcode(x, fin_sty, masks_x)
        
        return fake_rgb

class Flow_Generator(nn.Module):
    def __init__(self, flow_dim=320, flow_block_num=8, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=0,
                 num_domains=4, RSIM_size=128,
                 use_mask=True,
                 Gen_path=''):
        super().__init__()
        self.flow_dim = flow_dim
        self.style_dim = style_dim
        self.inn = Ff.SequenceINN(self.flow_dim)
        for k in range(flow_block_num):
            self.inn.append(Fm.AllInOneBlock, subnet_constructor=self.subnet_fc, permute_soft=True)

        self.Generator = Generator(img_size=img_size, style_dim=style_dim, max_conv_dim=max_conv_dim, w_hpf=w_hpf,
                                   num_domains=num_domains, RSIM_size=RSIM_size,
                                   use_mask=use_mask)
        self.Generator.load_state_dict(torch.load(Gen_path,
                                                  map_location='cpu'))
        self.lock_net_param(self.Generator)

    def lock_net_param(self, module):
        for parameter in module.parameters():
            parameter.requires_grad = False

    def subnet_fc(self, dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, 512), nn.ReLU(),
                             nn.Linear(512, dims_out))

    def forward(self, x, masks_x, y=None, y_mask=None, z=None, is_Training=False):
        '''
        x: input images   N 3 H W
        y_sty: colorcode  N dim 1 1
        z: latent code    N dim
        masks_x:          N num_domains H W
        # lip skin left_eye right_eye hair
        '''
        if is_Training:
            assert z is None
            assert y is not None
            y_sty = self.Generator.Get_y_sty(y=y, y_mask=y_mask).squeeze(-1).squeeze(-1)  # N dim
            z, log_jac_det = self.inn(y_sty)

            fake_colored = self.Generator.forward_with_colorcode(x=x, y_sty=y_sty.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=masks_x)
            return z, log_jac_det, fake_colored
        else:
            assert z is not None
            assert y is None
            y_sty, _ = self.inn(z, rev=True)

            fake_colored = self.Generator.forward_with_colorcode(x=x, y_sty=y_sty.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=masks_x)
            return fake_colored

    def forward_w_diff_img(self, x, masks_x, fake_y_sty):
        y_sty = fake_y_sty
        fake_colored = self.Generator.forward_with_colorcode(x=x, y_sty=y_sty.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=masks_x)
        return fake_colored    
    
    def forward_w_diff_mul_img(self, x, masks_x, fake_y_sty1, fake_y_sty2, fake_y_sty3, fake_y_sty4, fake_y_sty5):
        fake_colored = self.Generator.forward_with_mul_colorcode(x=x,
                                                                 y_sty_1=fake_y_sty1.unsqueeze(-1).unsqueeze(-1),
                                                                 y_sty_2=fake_y_sty2.unsqueeze(-1).unsqueeze(-1),
                                                                 y_sty_3=fake_y_sty3.unsqueeze(-1).unsqueeze(-1),
                                                                 y_sty_4=fake_y_sty4.unsqueeze(-1).unsqueeze(-1),
                                                                 y_sty_5=fake_y_sty5.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=masks_x)
        return fake_colored                                          
    
    def test_w_single_z(self, x, masks_x, z1):
        '''
        x: input images   N 3 H W
        zs: latent code   N dim
        masks_x:          N num_domains H W
        # lip skin eyes hair background
        '''
        y_sty_1, _ = self.inn(z1, rev=True)
        fake_colored = self.Generator.forward_with_colorcode(x=x, y_sty=y_sty_1.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=masks_x)
        return fake_colored
    
    def test_w_single_latent(self, x, masks_x, latent):
        '''
        x: input images   N 3 H W
        zs: latent code   N dim
        masks_x:          N num_domains H W
        # lip skin eyes hair background
        '''
        y_sty_1 = latent
        fake_colored = self.Generator.forward_with_colorcode(x=x, y_sty=y_sty_1.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=masks_x)
        return fake_colored
    
    def test(self, x, masks_x, z1, z2=None, z3=None, z4=None, z5=None):
        '''
        x: input images   N 3 H W
        zs: latent code   N dim
        masks_x:          N num_domains H W
        # lip skin eyes hair background
        '''
        y_sty_1, _ = self.inn(z1, rev=True)
        out1 = self.Generator.forward_with_colorcode(x=x, y_sty=y_sty_1.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=masks_x)
        if z2 is not None:
            y_sty_2, _ = self.inn(z2, rev=True)
            out2 = self.Generator.forward_with_colorcode(x=x, y_sty=y_sty_2.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=masks_x)
        else:
            y_sty_2 = y_sty_1

        if z3 is not None:
            y_sty_3, _ = self.inn(z3, rev=True)
            out3 = self.Generator.forward_with_colorcode(x=x, y_sty=y_sty_3.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=masks_x)
        else:
            y_sty_3 = y_sty_1

        if z4 is not None:
            y_sty_4, _ = self.inn(z4, rev=True)
            out4 = self.Generator.forward_with_colorcode(x=x, y_sty=y_sty_4.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=masks_x)
        else:
            y_sty_4 = y_sty_1

        if z5 is not None:
            y_sty_5, _ = self.inn(z5, rev=True)
            out5 = self.Generator.forward_with_colorcode(x=x, y_sty=y_sty_5.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=masks_x)
        else:
            y_sty_5 = y_sty_1

        fake_colored = self.Generator.forward_with_mul_colorcode(x=x,
                                                                 y_sty_1=y_sty_1.unsqueeze(-1).unsqueeze(-1),
                                                                 y_sty_2=y_sty_2.unsqueeze(-1).unsqueeze(-1),
                                                                 y_sty_3=y_sty_3.unsqueeze(-1).unsqueeze(-1),
                                                                 y_sty_4=y_sty_4.unsqueeze(-1).unsqueeze(-1),
                                                                 y_sty_5=y_sty_5.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=masks_x)
        return fake_colored,out1,out2,out3,out4,out5

class auto_Z_Encoder_merged(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=4,
                 max_single_z_channel=128):
        super().__init__()
        repeat_num = int(np.log2(img_size))
        self.repeat_num = repeat_num
        self.auto_Z_Gen_SPADEs = nn.ModuleList()
        self.auto_Z_Gen_RESBLKs = nn.ModuleList()
        dim_in = style_dim * (num_domains + 1)  # 32 * 5
        self.auto_Z_Gen_HEAD = nn.Conv2d(3, dim_in, 3, 1, 1)  # grey img as input, so the input channel is 1.
        ref_dim_in = dim_in
        for _ in range(repeat_num):
            ref_dim_out = min(ref_dim_in * 2, max_single_z_channel * (num_domains + 1))
            self.auto_Z_Gen_SPADEs.append(SPADE(norm_nc=ref_dim_in, label_nc=num_domains + 1, norm_type='instance'))
            self.auto_Z_Gen_RESBLKs.append(ResBlk(int(ref_dim_in), int(ref_dim_out), normalize=False, downsample=True)) # normalize=True
            ref_dim_in = ref_dim_out
        # 256 ->128->64->32->16->8->4->2->1
        self.auto_Z_Gen_TAIL = nn.Conv2d(ref_dim_in, style_dim * (num_domains + 1), 1, 1, 0)

    def forward(self, x, mask):
        '''
        x: input images   N 3 H W
        mask:             N num_domains+1 H W
        # lip skin eyes hair background
        output:
        z:                N style_dim*(num_domains+1)
        '''
        x = self.auto_Z_Gen_HEAD(x)
        for i in range(self.repeat_num):
            x = self.auto_Z_Gen_SPADEs[i](x, mask)
            x = self.auto_Z_Gen_RESBLKs[i](x)
        z = self.auto_Z_Gen_TAIL(x)
        return z
        
class auto_Z_Generator_wo_flow(nn.Module):
    def __init__(self, flow_dim=320, flow_block_num=8, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=0,
                 num_domains=4, RSIM_size=128,
                 use_mask=True,
                 Gen_path='',
                 max_single_z_channel=128):
        super().__init__()
        
        self.Generator = Generator(img_size=img_size, style_dim=style_dim, max_conv_dim=max_conv_dim, w_hpf=w_hpf,
                                   num_domains=num_domains, RSIM_size=RSIM_size,
                                   use_mask=use_mask)
        self.Generator.load_state_dict(torch.load(Gen_path,
                                                  map_location='cpu'))
        self.lock_net_param(self.Generator)

        self.auto_Z_Encoder = auto_Z_Encoder_merged(img_size=img_size, style_dim=style_dim, num_domains=num_domains, max_single_z_channel=max_single_z_channel)

    def lock_net_param(self, module):
        for parameter in module.parameters():
            parameter.requires_grad = False
    
    def forward(self, x, mask):
        '''
        x: input images   N 3 H W
        mask:             N num_domains H W
        # lip skin eyes hair background
        output:
        out:              N 3 H W
        '''
        z = self.auto_Z_Encoder(x, mask).squeeze(-1).squeeze(-1)
        y_sty_1 = z
        fake_colored = self.Generator.forward_with_colorcode(x=x, y_sty=y_sty_1.unsqueeze(-1).unsqueeze(-1),
                                                                 masks_x=mask)
        return fake_colored

        
        
    
    
