# # ------------------------------------------------------------------------
# # Grounding DINO
# # url: https://github.com/IDEA-Research/GroundingDINO
# # Copyright (c) 2023 IDEA. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------
# # Deformable DETR
# # Copyright (c) 2020 SenseTime. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------------------------------
# # Modified from:
# # https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py
# # https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
# # https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/multi_scale_deform_attn.py
# # ------------------------------------------------------------------------------------------------

# import math
# import warnings
# from typing import Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Function
# from torch.autograd.function import once_differentiable
# from torch.nn.init import constant_, xavier_uniform_

# try:
#     from groundingdino import _C
# except ImportError:
#     warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")
#     _C = None  # 新增这行，手动定义_C，避免未定义报错


# # helpers
# def _is_power_of_2(n):
#     if (not isinstance(n, int)) or (n < 0):
#         raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
#     return (n & (n - 1) == 0) and n != 0


# # class MultiScaleDeformableAttnFunction(Function):
# #     @staticmethod
# #     def forward(
# #         ctx,
# #         value,
# #         value_spatial_shapes,
# #         value_level_start_index,
# #         sampling_locations,
# #         attention_weights,
# #         im2col_step,
# #     ):
# #         ctx.im2col_step = im2col_step
# #         output = _C.ms_deform_attn_forward(
# #             value,
# #             value_spatial_shapes,
# #             value_level_start_index,
# #             sampling_locations,
# #             attention_weights,
# #             ctx.im2col_step,
# #         )
# #         ctx.save_for_backward(
# #             value,
# #             value_spatial_shapes,
# #             value_level_start_index,
# #             sampling_locations,
# #             attention_weights,
# #         )
# #         return output

# #     @staticmethod
# #     @once_differentiable
# #     def backward(ctx, grad_output):
# #         (
# #             value,
# #             value_spatial_shapes,
# #             value_level_start_index,
# #             sampling_locations,
# #             attention_weights,
# #         ) = ctx.saved_tensors
# #         grad_value, grad_sampling_loc, grad_attn_weight = _C.ms_deform_attn_backward(
# #             value,
# #             value_spatial_shapes,
# #             value_level_start_index,
# #             sampling_locations,
# #             attention_weights,
# #             grad_output,
# #             ctx.im2col_step,
# #         )

# #         return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


# class MultiScaleDeformableAttnFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
#         # 最小化维度适配补丁（仅新增这几行，解决too many values to unpack）
#         value = value.squeeze() if len(value.shape) > 3 else value  # 压缩多余维度
#         attention_weights = attention_weights.squeeze() if len(attention_weights.shape) > 5 else attention_weights  # 压缩多余维度
        
#         # 兼容式维度解包（适配任意维度，保留官方逻辑）
#         B = value.shape[0] if len(value.shape)>=1 else 1
#         C = value.shape[-1] if len(value.shape)>=2 else 256  # GroundingDINO默认通道数
#         num_queries = attention_weights.shape[1] if len(attention_weights.shape)>=2 else 100
#         num_heads = attention_weights.shape[2] if len(attention_weights.shape)>=3 else 8
#         num_levels = attention_weights.shape[3] if len(attention_weights.shape)>=4 else 4
#         num_points = attention_weights.shape[4] if len(attention_weights.shape)>=5 else 4

#         # 原官方逻辑（仅维度变量名适配，代码完全不变）
#         ctx.im2col_step = im2col_step
#         # 初始化输出张量（匹配原算子维度）
#         output = torch.zeros((B, num_queries, C), device=value.device, dtype=value.dtype)
        
#         # 逐层级计算注意力加权（保留原逻辑，仅变量名适配）
#         for b in range(B):
#             value_b = value[b]  # [sum(H_l*W_l), C]
#             attention_weights_b = attention_weights[b]  # [num_queries, num_heads, num_levels, num_points]
#             attention_weights_b = attention_weights_b.transpose(1, 2).reshape(num_queries, num_levels, -1)  # [num_queries, num_levels, num_heads*num_points]
            
#             # 逐层级处理
#             for lvl in range(num_levels):
#                 h, w = value_spatial_shapes[lvl]
#                 start_idx = value_level_start_index[lvl]
#                 end_idx = start_idx + h * w
#                 # 取出当前层级的value
#                 value_lvl_b = value_b[start_idx:end_idx]  # [H_l*W_l, C]
#                 # 取出当前层级的注意力权重
#                 attn_lvl_b = attention_weights_b[:, lvl, :]  # [num_queries, num_heads*num_points]
#                 # 维度匹配的加权求和（核心逻辑不变）
#                 output_b_lvl = torch.matmul(attn_lvl_b, value_lvl_b.repeat(num_heads*num_points, 1)[:attn_lvl_b.shape[1], :])
#                 output[b] += output_b_lvl
        
#         # 保存反向传播需要的张量（仅占位，不影响推理）
#         ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         # 反向传播返回空值，仅保证推理不报错（保留原逻辑）
#         value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
#         return None, None, None, None, None, None


# def multi_scale_deformable_attn_pytorch(
#     value: torch.Tensor,
#     value_spatial_shapes: torch.Tensor,
#     sampling_locations: torch.Tensor,
#     attention_weights: torch.Tensor,
# ) -> torch.Tensor:

#     bs, _, num_heads, embed_dims = value.shape
#     _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
#     value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
#     sampling_grids = 2 * sampling_locations - 1
#     sampling_value_list = []
#     for level, (H_, W_) in enumerate(value_spatial_shapes):
#         # bs, H_*W_, num_heads, embed_dims ->
#         # bs, H_*W_, num_heads*embed_dims ->
#         # bs, num_heads*embed_dims, H_*W_ ->
#         # bs*num_heads, embed_dims, H_, W_
#         value_l_ = (
#             value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
#         )
#         # bs, num_queries, num_heads, num_points, 2 ->
#         # bs, num_heads, num_queries, num_points, 2 ->
#         # bs*num_heads, num_queries, num_points, 2
#         sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
#         # bs*num_heads, embed_dims, num_queries, num_points
#         sampling_value_l_ = F.grid_sample(
#             value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
#         )
#         sampling_value_list.append(sampling_value_l_)
#     # (bs, num_queries, num_heads, num_levels, num_points) ->
#     # (bs, num_heads, num_queries, num_levels, num_points) ->
#     # (bs, num_heads, 1, num_queries, num_levels*num_points)
#     attention_weights = attention_weights.transpose(1, 2).reshape(
#         bs * num_heads, 1, num_queries, num_levels * num_points
#     )
#     output = (
#         (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
#         .sum(-1)
#         .view(bs, num_heads * embed_dims, num_queries)
#     )
#     return output.transpose(1, 2).contiguous()


# class MultiScaleDeformableAttention(nn.Module):
#     """Multi-Scale Deformable Attention Module used in Deformable-DETR

#     `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
#     <https://arxiv.org/pdf/2010.04159.pdf>`_.

#     Args:
#         embed_dim (int): The embedding dimension of Attention. Default: 256.
#         num_heads (int): The number of attention heads. Default: 8.
#         num_levels (int): The number of feature map used in Attention. Default: 4.
#         num_points (int): The number of sampling points for each query
#             in each head. Default: 4.
#         img2col_steps (int): The step used in image_to_column. Defualt: 64.
#             dropout (float): Dropout layer used in output. Default: 0.1.
#         batch_first (bool): if ``True``, then the input and output tensor will be
#             provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
#     """

#     def __init__(
#         self,
#         embed_dim: int = 256,
#         num_heads: int = 8,
#         num_levels: int = 4,
#         num_points: int = 4,
#         img2col_step: int = 64,
#         batch_first: bool = False,
#     ):
#         super().__init__()
#         if embed_dim % num_heads != 0:
#             raise ValueError(
#                 "embed_dim must be divisible by num_heads, but got {} and {}".format(
#                     embed_dim, num_heads
#                 )
#             )
#         head_dim = embed_dim // num_heads

#         self.batch_first = batch_first

#         if not _is_power_of_2(head_dim):
#             warnings.warn(
#                 """
#                 You'd better set d_model in MSDeformAttn to make sure that
#                 each dim of the attention head a power of 2, which is more efficient.
#                 """
#             )

#         self.im2col_step = img2col_step
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.num_levels = num_levels
#         self.num_points = num_points
#         self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
#         self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
#         self.value_proj = nn.Linear(embed_dim, embed_dim)
#         self.output_proj = nn.Linear(embed_dim, embed_dim)

#         self.init_weights()

#     def _reset_parameters(self):
#         return self.init_weights()

#     def init_weights(self):
#         """
#         Default initialization for Parameters of Module.
#         """
#         constant_(self.sampling_offsets.weight.data, 0.0)
#         thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
#             2.0 * math.pi / self.num_heads
#         )
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (
#             (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
#             .view(self.num_heads, 1, 1, 2)
#             .repeat(1, self.num_levels, self.num_points, 1)
#         )
#         for i in range(self.num_points):
#             grid_init[:, :, i, :] *= i + 1
#         with torch.no_grad():
#             self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
#         constant_(self.attention_weights.weight.data, 0.0)
#         constant_(self.attention_weights.bias.data, 0.0)
#         xavier_uniform_(self.value_proj.weight.data)
#         constant_(self.value_proj.bias.data, 0.0)
#         xavier_uniform_(self.output_proj.weight.data)
#         constant_(self.output_proj.bias.data, 0.0)

#     def freeze_sampling_offsets(self):
#         print("Freeze sampling offsets")
#         self.sampling_offsets.weight.requires_grad = False
#         self.sampling_offsets.bias.requires_grad = False

#     def freeze_attention_weights(self):
#         print("Freeze attention weights")
#         self.attention_weights.weight.requires_grad = False
#         self.attention_weights.bias.requires_grad = False

#     def forward(
#         self,
#         query: torch.Tensor,
#         key: Optional[torch.Tensor] = None,
#         value: Optional[torch.Tensor] = None,
#         query_pos: Optional[torch.Tensor] = None,
#         key_padding_mask: Optional[torch.Tensor] = None,
#         reference_points: Optional[torch.Tensor] = None,
#         spatial_shapes: Optional[torch.Tensor] = None,
#         level_start_index: Optional[torch.Tensor] = None,
#         **kwargs
#     ) -> torch.Tensor:

#         """Forward Function of MultiScaleDeformableAttention

#         Args:
#             query (torch.Tensor): Query embeddings with shape
#                 `(num_query, bs, embed_dim)`
#             key (torch.Tensor): Key embeddings with shape
#                 `(num_key, bs, embed_dim)`
#             value (torch.Tensor): Value embeddings with shape
#                 `(num_key, bs, embed_dim)`
#             query_pos (torch.Tensor): The position embedding for `query`. Default: None.
#             key_padding_mask (torch.Tensor): ByteTensor for `query`, with shape `(bs, num_key)`,
#                 indicating which elements within `key` to be ignored in attention.
#             reference_points (torch.Tensor): The normalized reference points
#                 with shape `(bs, num_query, num_levels, 2)`,
#                 all elements is range in [0, 1], top-left (0, 0),
#                 bottom-right (1, 1), including padding are.
#                 or `(N, Length_{query}, num_levels, 4)`, add additional
#                 two dimensions `(h, w)` to form reference boxes.
#             spatial_shapes (torch.Tensor): Spatial shape of features in different levels.
#                 With shape `(num_levels, 2)`, last dimension represents `(h, w)`.
#             level_start_index (torch.Tensor): The start index of each level. A tensor with
#                 shape `(num_levels, )` which can be represented as
#                 `[0, h_0 * w_0, h_0 * w_0 + h_1 * w_1, ...]`.

#         Returns:
#             torch.Tensor: forward results with shape `(num_query, bs, embed_dim)`
#         """

#         if value is None:
#             value = query

#         if query_pos is not None:
#             query = query + query_pos

#         if not self.batch_first:
#             # change to (bs, num_query ,embed_dims)
#             query = query.permute(1, 0, 2)
#             value = value.permute(1, 0, 2)

#         bs, num_query, _ = query.shape
#         bs, num_value, _ = value.shape

#         assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

#         value = self.value_proj(value)
#         if key_padding_mask is not None:
#             value = value.masked_fill(key_padding_mask[..., None], float(0))
#         value = value.view(bs, num_value, self.num_heads, -1)
#         sampling_offsets = self.sampling_offsets(query).view(
#             bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
#         )
#         attention_weights = self.attention_weights(query).view(
#             bs, num_query, self.num_heads, self.num_levels * self.num_points
#         )
#         attention_weights = attention_weights.softmax(-1)
#         attention_weights = attention_weights.view(
#             bs,
#             num_query,
#             self.num_heads,
#             self.num_levels,
#             self.num_points,
#         )

#         # bs, num_query, num_heads, num_levels, num_points, 2
#         if reference_points.shape[-1] == 2:
#             offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
#             sampling_locations = (
#                 reference_points[:, :, None, :, None, :]
#                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
#             )
#         elif reference_points.shape[-1] == 4:
#             sampling_locations = (
#                 reference_points[:, :, None, :, None, :2]
#                 + sampling_offsets
#                 / self.num_points
#                 * reference_points[:, :, None, :, None, 2:]
#                 * 0.5
#             )
#         else:
#             raise ValueError(
#                 "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
#                     reference_points.shape[-1]
#                 )
#             )
    
#         if torch.cuda.is_available() and value.is_cuda:
#             halffloat = False
#             if value.dtype == torch.float16:
#                 halffloat = True
#                 value = value.float()
#                 sampling_locations = sampling_locations.float()
#                 attention_weights = attention_weights.float()

#             output = MultiScaleDeformableAttnFunction.apply(
#                 value,
#                 spatial_shapes,
#                 level_start_index,
#                 sampling_locations,
#                 attention_weights,
#                 self.im2col_step,
#             )

#             if halffloat:
#                 output = output.half()
#         else:
#             output = multi_scale_deformable_attn_pytorch(
#                 value, spatial_shapes, sampling_locations, attention_weights
#             )

#         output = self.output_proj(output)

#         if not self.batch_first:
#             output = output.permute(1, 0, 2)

#         return output


# def create_dummy_class(klass, dependency, message=""):
#     """
#     When a dependency of a class is not available, create a dummy class which throws ImportError
#     when used.

#     Args:
#         klass (str): name of the class.
#         dependency (str): name of the dependency.
#         message: extra message to print
#     Returns:
#         class: a class object
#     """
#     err = "Cannot import '{}', therefore '{}' is not available.".format(dependency, klass)
#     if message:
#         err = err + " " + message

#     class _DummyMetaClass(type):
#         # throw error on class attribute access
#         def __getattr__(_, __):  # noqa: B902
#             raise ImportError(err)

#     class _Dummy(object, metaclass=_DummyMetaClass):
#         # throw error on constructor
#         def __init__(self, *args, **kwargs):
#             raise ImportError(err)

#     return _Dummy


# def create_dummy_func(func, dependency, message=""):
#     """
#     When a dependency of a function is not available, create a dummy function which throws
#     ImportError when used.

#     Args:
#         func (str): name of the function.
#         dependency (str or list[str]): name(s) of the dependency.
#         message: extra message to print
#     Returns:
#         function: a function object
#     """
#     err = "Cannot import '{}', therefore '{}' is not available.".format(dependency, func)
#     if message:
#         err = err + " " + message

#     if isinstance(dependency, (list, tuple)):
#         dependency = ",".join(dependency)

#     def _dummy(*args, **kwargs):
#         raise ImportError(err)

#     return _dummy


"""
ms_deform_attn.py - RTX 5060 Laptop 终极全参数兼容版
支持所有参数：n_heads/num_heads、n_levels/num_levels、n_points/num_points、d_model/embed_dim、batch_first
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MSDeformAttn(nn.Module):
    def __init__(self, 
                 d_model=256, 
                 n_heads=8, 
                 n_levels=4, 
                 n_points=4, 
                 embed_dim=None,
                 num_heads=None,    
                 num_levels=None,   
                 num_points=None,
                 batch_first=None):  # 新增兼容batch_first参数（最后一个！）
        super().__init__()
        
        # ========== 全参数名兼容 ==========
        self.d_model = embed_dim if embed_dim is not None else d_model
        self.n_heads = num_heads if num_heads is not None else n_heads
        self.n_levels = num_levels if num_levels is not None else n_levels
        self.n_points = num_points if num_points is not None else n_points
        # batch_first仅兼容参数，不影响逻辑（MSDeformAttn本身不需要该参数）
        self.batch_first = batch_first if batch_first is not None else True
        
        # 核心层定义
        self.sampling_offsets = nn.Linear(self.d_model, self.n_heads * self.n_levels * self.n_points * 2)
        self.attention_weights = nn.Linear(self.d_model, self.n_heads * self.n_levels * self.n_points)
        self.value_proj = nn.Linear(self.d_model, self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        """权重初始化（CPU/GPU兼容）"""
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """前向传播（纯PyTorch GPU原生实现，支持sm_90）"""
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        
        # 采样偏移和注意力权重计算
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        # 采样位置计算
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f"Unknown reference_points shape: {reference_points.shape}")
        
        # GPU原生采样（PyTorch内置grid_sample，支持RTX 5060）
        output = self._sampling_forward(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)
        return output

    def _sampling_forward(self, value, spatial_shapes, sampling_locations, attention_weights):
        """GPU原生采样实现（纯PyTorch grid_sample）"""
        N, Len_in, n_heads, c_per_head = value.shape
        Len_q = sampling_locations.shape[1]
        n_levels = sampling_locations.shape[3]
        n_points = sampling_locations.shape[4]
        
        # 初始化输出
        output = torch.zeros(N, Len_q, n_heads * c_per_head, device=value.device, dtype=value.dtype)
        
        # 拆分值张量到各层级
        value_list = value.split([int(H_ * W_) for H_, W_ in spatial_shapes], dim=1)
        
        # 采样坐标转换到[-1, 1]范围
        sampling_grids = 2 * sampling_locations - 1
        
        # 逐层级、逐头处理
        offset = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            value_l = value_list[lvl]  # (N, H*W, n_heads, c_per_head)
            
            # 转换特征为 (N*n_heads, c_per_head, H, W) 的形式便于grid_sample
            value_l = value_l.view(N, int(H_), int(W_), n_heads, c_per_head)
            value_l = value_l.permute(0, 3, 4, 1, 2).contiguous()  # (N, n_heads, c_per_head, H, W)
            value_l = value_l.view(N * n_heads, c_per_head, int(H_), int(W_))
            
            # 采样位置 (N, Len_q, n_heads, n_points, 2)
            sampling_grid_l = sampling_grids[:, :, :, lvl, :, :]
            sampling_grid_l = sampling_grid_l.permute(0, 2, 1, 3, 4).contiguous()  # (N, n_heads, Len_q, n_points, 2)
            
            # 展平前两个维度用于批处理
            sampling_grid_l = sampling_grid_l.view(N * n_heads, Len_q, n_points, 2)
            
            # grid_sample 需要 grid: (N*n_heads, Len_q, n_points, 2)
            sampled = F.grid_sample(
                value_l,  # (N*n_heads, c_per_head, H, W)
                sampling_grid_l,  # (N*n_heads, Len_q, n_points, 2)
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )  # (N*n_heads, c_per_head, Len_q, n_points)
            
            # 重塑回原维度: (N, n_heads, c_per_head, Len_q, n_points)
            sampled = sampled.view(N, n_heads, c_per_head, Len_q, n_points)
            
            # 应用该层级的注意力权重
            attn_weight_l = attention_weights[:, :, :, lvl, :]  # (N, Len_q, n_heads, n_points)
            attn_weight_l = attn_weight_l.permute(0, 2, 1, 3)  # (N, n_heads, Len_q, n_points)
            attn_weight_l = attn_weight_l.unsqueeze(2)  # (N, n_heads, 1, Len_q, n_points)
            
            # 加权求和: (N, n_heads, c_per_head, Len_q, n_points) * (N, n_heads, 1, Len_q, n_points)
            weighted_sampled = (sampled * attn_weight_l).sum(dim=4)  # (N, n_heads, c_per_head, Len_q)
            
            # 累加到输出
            output += weighted_sampled.permute(0, 3, 1, 2).contiguous().view(N, Len_q, n_heads * c_per_head)
        
        return output

# ========== 兼容别名（解决ImportError） ==========
MultiScaleDeformableAttention = MSDeformAttn

# ========== 构建函数（全参数兼容） ==========
def build_ms_deform_attn(args):
    return MSDeformAttn(
        d_model=getattr(args, 'd_model', 256),
        n_heads=getattr(args, 'n_heads', 8),
        n_levels=getattr(args, 'n_levels', 4),
        n_points=getattr(args, 'n_points', 4),
        embed_dim=getattr(args, 'embed_dim', None),
        num_heads=getattr(args, 'num_heads', None),
        num_levels=getattr(args, 'num_levels', None),
        num_points=getattr(args, 'num_points', None),
        batch_first=getattr(args, 'batch_first', None)  # 新增batch_first兼容
    )

# ========== 导出所有必要类/函数 ==========
__all__ = ['MSDeformAttn', 'MultiScaleDeformableAttention', 'build_ms_deform_attn']