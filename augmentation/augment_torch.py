import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import random
import collections

# Random Shift
# 尺寸为 A × B 的图像每条边填充 pad 个像素（通过重复边界像素），然后随机裁剪回原始 A × B 尺寸
class RandomShiftsAug(nn.Module):
    '''
        https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
    '''
    def __init__(self, pad=4,aug=True):
        # pad: padding 填充
        super().__init__()
        self.pad = pad
        self.aug = aug

    def forward(self, x):
        if self.aug:
            n, _, h, w = x.size()
            # n,c,h,w分别表示batch数，通道数，高，宽
            padding = tuple([self.pad] * 4)
            # 分别对前，后做多少位的padding操作
            # 例如tuple([4] * 4)=> (4, 4, 4, 4)，上下左右四个方位都做pad的填充
            x = F.pad(x, padding, 'replicate')
            # 对高维tensor的形状补齐操作
            # replicate​​​：使用tensor自身边界值补齐指定的维度。对于数据​​012​​​，结果可以为​​0001222​
            eps = 1.0 / (w + 2 * self.pad)
            # 若w=84, pad=4, 则w + 2 * self.pad=92, eps=0.010869565217391304
            arange = torch.linspace(-1.0 + eps, # 区间左侧
                                    1.0 - eps, # 区间右侧
                                    w + 2 * self.pad, # 92个点
                                    device=x.device,
                                    dtype=x.dtype)[:w]
            # 线性间距向量
            # torch.linspace(start, end, steps=100, out=None) → Tensor
            # 返回一个1维张量，包含在区间start和end上均匀间隔的step个点
            # 输出张量的长度由steps决定
            # 例如：生成0到10的5个数构成的等差数列
            # b = torch.linspace(0,10,steps=5)
            # tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])
            eps_h = 1.0 / (h + 2 * self.pad) # 若h=84, pad=4, 则eps_h=0.010869565217391304
            arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
            # 扩充第一个维度
            # 重复h行
            # 扩充第三个维度
            arange_w = torch.linspace(-1.0 + eps_h,
                                    1.0 - eps_h,
                                    h + 2 * self.pad,
                                    device=x.device,
                                    dtype=x.dtype)[:h]
            arange_w = arange_w.unsqueeze(1).repeat(1, w).unsqueeze(2)
            # 扩充第二个维度
            # 重复w列
            # 扩充第三个维度
            # arange_w = arange_w.unsqueeze(0).repeat(w, 1).unsqueeze(2)
            base_grid = torch.cat([arange, arange_w], dim=2)
            # 将两个张量按第三个维度拼接在一起
            base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
            # n个数据，重复n次操作

            shift = torch.randint(0,
                                2 * self.pad + 1, # pad=4, 值为9
                                size=(n, 1, 1, 2),
                                device=x.device,
                                dtype=x.dtype)
            shift[:,:,:,0] *= 2.0 / (w + 2 * self.pad)
            shift[:,:,:,1] *= 2.0 / (h + 2 * self.pad)
            grid = base_grid + shift
            # 随机平移
            # 在 random shift 之后还使用了 bilinear interpolation
            return F.grid_sample(x, # x.shape: torch.Size([1, 3, 420, 720])
                                grid, # grid.shape: torch.Size([1, 300, 600, 2])
                                padding_mode='zeros',
                                align_corners=False)
            # 输出： torch.Size([1, 3, 300, 600])
        # 应用双线性插值，把输入的tensor转换为指定大小
        # 参考：https://betheme.net/qianduan/43027.html?action=onClick
        # 给定维度为(N,C,Hin,Win) 的input，维度为(N,Hout,Wout,2) 的grid
        # 则该函数output的维度为(N,C,Hout,Wout)
        # padding_mode表示当grid中的坐标位置超出边界时像素值的填充方式
        # 如果为zeros，则表示一旦grid坐标超出边界，则用0去填充输出特征图的相应位置元素
        else:
            return x

class RandomShiftsAug_v1(nn.Module):
    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Size([128, 9, 84, 84])):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
