import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 #用来保存热力图

from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
from .utils import crop_op, crop_to_shape

####
class HoVerNet(Net):
    """Initialise HoVer-Net."""

    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4
        self.image_count = 0 
        

        assert mode == 'original' or mode == 'fast', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast': # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)

        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            module_list = [ 
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder

        ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(ksize=ksize,out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize,out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    def forward(self, imgs):

        imgs = imgs / 255.0  # to 0-1 range to match XY

        if self.training:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0, self.freeze)
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(d0)
                d2 = self.d2(d1)
                d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]
        else:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0)
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
            d3 = self.conv_bot(d3)
            d = [d0, d1, d2, d3]

        # TODO: switch to `crop_to_shape` ?
        if self.mode == 'original':
            d[0] = crop_op(d[0], [184, 184])
            d[1] = crop_op(d[1], [72, 72])
        else:
            d[0] = crop_op(d[0], [92, 92])
            d[1] = crop_op(d[1], [36, 36])

        out_dict = OrderedDict()

        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

            # # 只为 "np" 分支生成热力图  
            # if branch_name == "np":  
            #     # 保存热力图  
            #     heatmap = u0[0].detach().cpu().numpy()  
            #     heatmap = heatmap[0]  # Assuming the heatmap has shape (1, height, width)  
            #     print("Original Heatmap Type:", heatmap.dtype)  
            #     print("Original Heatmap Shape:", heatmap.shape)  

            #     # Normalize and convert to uint8  
            #     heatmap_min = heatmap.min()  
            #     heatmap_max = heatmap.max()  
            #     heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min) * 255  
            #     heatmap = heatmap.astype(np.uint8)  

            #     # Apply color map  
            #     heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  

            #     # 在保存热力图时使用计数  
            #     save_path_color = f"/media/imris/guanbo/CLIP_GCN/hover_net-master/paper/png/MethodOverview/HeatMap/HeatMapMask/heatmap_np_color_{self.image_count}.png"  
            #     cv2.imwrite(save_path_color, heatmap_colored)  
            #     print(f"Heatmap (color) saved to {save_path_color}")  

            #     # 将热力图与原始图像融合  
            #     alpha = 0.5  # 调整融合的权重  
            #     # 获取原始图像  
            #     original_image = imgs[0].detach().cpu().numpy().transpose(1, 2, 0)  
            #     original_image = (original_image * 255).astype(np.uint8)  

            #     # 调整原始图像的尺寸与热图相匹配  
            #     resized_original_image = cv2.resize(original_image, (heatmap.shape[1], heatmap.shape[0]))  

            #     # 将热力图与调整尺寸后的原始图像融合  
            #     blended_image = cv2.addWeighted(resized_original_image, alpha, heatmap_colored, 1 - alpha, 0) 
                
                 

            #     # 在保存融合图像时使用计数  
            #     save_path_blend = f"/media/imris/guanbo/CLIP_GCN/hover_net-master/paper/png/MethodOverview/HeatMap/OverlayHeatMap/blend_np_{self.image_count}.png"  
            #     cv2.imwrite(save_path_blend, blended_image)  
            #     print(f"Blended Image saved to {save_path_blend}")  

            #     # 保存原图 
            #     save_path_raw = f"/media/imris/guanbo/CLIP_GCN/hover_net-master/paper/png/MethodOverview/HeatMap/HeatMapRawImage/heatmap_np_color_{self.image_count}.png"  
            #     cv2.imwrite(save_path_raw, resized_original_image)  

            #     # 在每次保存后增加计数  
            #     self.image_count += 1  



        return out_dict


####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return HoVerNet(mode=mode, **kwargs)

