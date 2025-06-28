# import warnings
# from collections import OrderedDict
# from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
# from functools import partial

# from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
# from mmcv.cnn.bricks.transformer import FFN, build_dropout
# from mmcv.cnn.utils.weight_init import trunc_normal_
# from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint,
#                          load_state_dict)
# from mmcv.utils import to_2tuple

# import math

# from ...utils import get_root_logger
# from ..builder import BACKBONES
# from ..utils import nchw_to_nlc, nlc_to_nchw, smt_convert


class Mlp(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.,
                 init_cfg=None):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Regularization(nn.Module):
    def __init__(self, 
                 dim, 
                 ca_num_heads=4, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.,
                 ca_attention=1,
                 expand_ratio=2,
                 init_cfg=None,
                 alpha=0.6):
        super(Regularization, self).__init__()
        
        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.alpha = alpha
      

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
       

        self.act = nn.GELU()
        self.actm = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.split_groups = self.dim//ca_num_heads
        
    
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)
        self.s_m = nn.Linear(dim//2, dim, bias=qkv_bias)
        for j in range(self.ca_num_heads):
            for i in range(self.ca_num_heads):
                local_conv = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1,groups=dim//self.ca_num_heads)
                setattr(self, f"local_conv_{j+1}_{i + 1}", local_conv)
                bn=nn.BatchNorm2d(dim//self.ca_num_heads)
                setattr(self,f'local_bn_{j+1}_{i + 1}',bn)
                mlocal_conv = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1,groups=dim//self.ca_num_heads)
                setattr(self, f"mlocal_conv_{j+1}_{i + 1}", mlocal_conv)
                mbn=nn.BatchNorm2d(dim//self.ca_num_heads)
                setattr(self,f'mlocal_bn_{j+1}_{i + 1}',mbn)

        self.proj0 = nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups)
        self.bn = nn.BatchNorm2d(dim*expand_ratio)
        self.proj1 = nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1)
        self.proj0m = nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups)
        self.bnm = nn.BatchNorm2d(dim*expand_ratio)
        self.proj1m = nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1)

      
        

    def forward(self, x, mi_x,H, W):
        B, N, C = x.shape
      
        v = self.v(x)
        s = self.s(x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2)
        for i in range(self.ca_num_heads):
            s_i= s[i]
            for r in range (self.ca_num_heads):
                local_conv=getattr(self, f"local_conv_{i+1}_{r + 1}")
                bn = getattr(self, f"local_bn_{i+1}_{r + 1}")
                if r==0:
                   out=bn(local_conv(s_i))
                else:
                   out = out + bn(local_conv(s_i))
            out =out.reshape(B, self.split_groups, -1, H, W)
            if i == 0:
                s_out = out
            else:
                s_out = torch.cat([s_out,out],2)
    
        s_out = s_out.reshape(B, C, H, W)
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out)))).reshape(B, C, N).permute(0, 2, 1)
        x_s = s_out * v

        m_s = self.s_m(mi_x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2)
        for i in range(self.ca_num_heads):
            ms_i= m_s[i]
            for r in range (self.ca_num_heads):
                local_conv=getattr(self, f"mlocal_conv_{i+1}_{r + 1}")
                bn = getattr(self, f"mlocal_bn_{i+1}_{r + 1}")
                if r==0:
                   mout=bn(local_conv(ms_i))
                else:
                   mout = mout + bn(local_conv(ms_i))
            mout =mout.reshape(B, self.split_groups, -1, H, W)
            if i == 0:
                ms_out = mout
            else:
                ms_out = torch.cat([ms_out,mout],2)

        ms_out = ms_out.reshape(B, C, H, W)
        ms_out = self.proj1m(self.actm(self.bnm(self.proj0m(ms_out)))).reshape(B, C, N).permute(0, 2, 1)

        x_m=ms_out*v
        x=self.alpha*x_s+(1-self.alpha)*x_m

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ADRBlock(nn.Module):

    def __init__(self, 
                 dim, 
                 ca_num_heads=4, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0.3, 
                 attn_drop=0.,
                 drop_path=0.3, 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 ca_attention=1,
                 expand_ratio=2,
                 init_cfg=None,
                 alpha=0.6):
        super(ADRBlock, self).__init__()

        self.init_cfg = init_cfg

        self.norm1 = norm_layer(dim)
        self.norm1m = norm_layer(dim//2)
        self.regu = Regularization(
            dim,
            ca_num_heads=ca_num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, ca_attention=ca_attention, 
            expand_ratio=expand_ratio,init_cfg=None,alpha=alpha)
            
      
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop,
            init_cfg=None)


    def forward(self, x,m_x):
        B,C,H,W=x.shape
        x = x.flatten(2).transpose(1, 2)
        m_x = m_x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.regu(self.norm1(x), self.norm1m(m_x),H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x