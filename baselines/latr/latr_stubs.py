"""讓 LATR（ICCV 2023）在沒有 mmcv-full / mmdet3d 編譯運算子的環境跑推論。

環境：專案的 torch（CUDA）＋ pure-Python 的 mmcv 1.7.2（lite，D:/models/latr/wheels 自建 wheel）
＋ mmdet 2.28.2 ＋ fvcore。要在 import 任何 mmcv / mmdet / LATR 模組**之前**呼叫 install()。

替身：
- ``mmcv.ops`` 與其子模組：lite 版沒有編譯擴充，一 import 就失敗。這裡換成「要什麼名字就給一個
  空類別」的寬鬆模組，讓 mmdet 在 import 時用到的 RoIAlign / nms 之類能過（LATR 推論不會呼叫）。
  LATR 真正會用到的兩個給實作：
  * ``MultiScaleDeformableAttnFunction.apply`` → mmcv 自己的純 PyTorch 版
    ``multi_scale_deformable_attn_pytorch``（照抄，Apache 2.0；CUDA 版與它數值等價，
    mmcv 在 CPU 上就用它）
  * ``DCNv2``（ResNet stage 3、4 的可變形卷積 v2，mmcv ``ModulatedDeformConv2dPack``）→
    ``torchvision.ops.deform_conv2d``。參數名稱（weight / bias / conv_offset.*）與 mmcv 相同，
    權重可直接載入；offset 的通道排法兩邊都是原始 DCN 的「每個核點 (dy, dx) 交錯」，
    mmcv 先 cat(o1, o2) 再交給 kernel，這裡照做
- ``mmdet3d.models.build_backbone / build_neck``：LATR 只用來建 ResNet 與 FPN，兩者都在
  mmdet 的註冊表裡，直接轉給 mmdet
- ``mmseg.ops.resize``：F.interpolate 的包裝
- ``geffnet``：只有 EfficientNet 抽特徵器 import，ResNet 設定用不到
- NumPy 2 拿掉的別名（np.float / np.int / np.bool / np.object / np.asscalar / np.RankWarning）：LATR 寫於 NumPy 1
"""
import sys, types, importlib.abc, importlib.machinery, math

import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # mmcv 1.7.2 mmcv/ops/multi_scale_deform_attn.py，逐行照抄
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear',
                                          padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttnFunction:
    @staticmethod
    def apply(value, value_spatial_shapes, value_level_start_index, sampling_locations,
              attention_weights, im2col_step):
        shapes = [(int(h), int(w)) for h, w in value_spatial_shapes]
        return multi_scale_deformable_attn_pytorch(value, shapes, sampling_locations, attention_weights)


class ModulatedDeformConv2dPack(nn.Module):
    """mmcv ModulatedDeformConv2dPack 的等價實作（推論用）。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deform_groups=1, bias=True, **kw):
        super().__init__()
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        pair = lambda v: (v, v) if isinstance(v, int) else tuple(v)
        self.stride, self.padding, self.dilation = pair(stride), pair(padding), pair(dilation)
        self.groups, self.deform_groups = groups, deform_groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *k))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.conv_offset = nn.Conv2d(in_channels, deform_groups * 3 * k[0] * k[1], k,
                                     stride=self.stride, padding=self.padding,
                                     dilation=self.dilation, bias=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv_offset.weight); nn.init.zeros_(self.conv_offset.bias)

    def forward(self, x):
        from torchvision.ops import deform_conv2d
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(x, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)


class _Loose(types.ModuleType):
    """任何屬性都回傳一個空類別（mmdet import 時的 mmcv.ops 名稱）。"""

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        cls = type(name, (nn.Module,), {})
        setattr(self, name, cls)
        return cls


class _LooseFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('mmcv.ops.'):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _Loose(spec.name)
        m.__path__ = []
        if spec.name == 'mmcv.ops.multi_scale_deform_attn':
            m.MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction
            m.multi_scale_deformable_attn_pytorch = multi_scale_deformable_attn_pytorch
        if spec.name == 'mmcv.ops.modulated_deform_conv':
            m.ModulatedDeformConv2dPack = ModulatedDeformConv2dPack
        return m

    def exec_module(self, module):
        pass


def install():
    import numpy as np
    for k, v in dict(float=float, int=int, bool=bool, object=object, long=int).items():
        if k not in np.__dict__:
            setattr(np, k, v)
    if 'RankWarning' not in np.__dict__:
        np.RankWarning = np.exceptions.RankWarning
    if 'asscalar' not in np.__dict__:
        np.asscalar = lambda a: a.item()
    ops = _Loose('mmcv.ops')
    ops.__path__ = []
    ops.MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction
    ops.ModulatedDeformConv2dPack = ModulatedDeformConv2dPack
    sys.modules['mmcv.ops'] = ops
    sys.meta_path.insert(0, _LooseFinder())

    sys.modules['geffnet'] = types.ModuleType('geffnet')
    mmseg, mmseg_ops = types.ModuleType('mmseg'), types.ModuleType('mmseg.ops')
    mmseg_ops.resize = lambda input, size=None, scale_factor=None, mode='nearest', \
        align_corners=None, warning=True: F.interpolate(input, size, scale_factor, mode, align_corners)
    mmseg.ops = mmseg_ops
    sys.modules.update({'mmseg': mmseg, 'mmseg.ops': mmseg_ops})

    import mmcv
    mmcv.ops = ops
    from mmcv.cnn import CONV_LAYERS
    CONV_LAYERS.register_module('DCNv2', module=ModulatedDeformConv2dPack, force=True)

    from mmdet.models import builder as mb
    m3, m3m = types.ModuleType('mmdet3d'), types.ModuleType('mmdet3d.models')
    m3m.build_backbone, m3m.build_neck = mb.build_backbone, mb.build_neck
    m3.models = m3m
    sys.modules.update({'mmdet3d': m3, 'mmdet3d.models': m3m})
