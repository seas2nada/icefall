# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Tuple, Optional, Dict
import logging
import math

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter
from encoder_interface import EncoderInterface

from icefall.utils import add_sos, add_eos

# class MultiheadAttention(nn.Module):
#     """Multi-headed attention.

#     See "Attention Is All You Need" for more details.
#     """

#     def __init__(
#         self,
#         embed_dim,
#         num_heads,
#         kdim=None,
#         vdim=None,
#         dropout=0.0,
#         bias=True,
#         add_bias_kv=False,
#         add_zero_attn=False,
#         self_attention=False,
#         encoder_decoder_attention=False,
#         q_noise=0.0,
#         qn_block_size=8,
#         has_relative_attention_bias=False,
#         num_buckets=32,
#         max_distance=128,
#         gru_rel_pos=False,
#         rescale_init=False,
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

#         self.num_heads = num_heads
#         self.dropout_module = nn.Dropout(dropout)

#         self.has_relative_attention_bias = has_relative_attention_bias
#         self.num_buckets = num_buckets
#         self.max_distance = max_distance
#         if self.has_relative_attention_bias:
#             self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

#         self.head_dim = embed_dim // num_heads
#         self.q_head_dim = self.head_dim
#         self.k_head_dim = self.head_dim
#         assert (
#             self.head_dim * num_heads == self.embed_dim
#         ), "embed_dim must be divisible by num_heads"
#         self.scaling = self.head_dim**-0.5

#         self.self_attention = self_attention
#         self.encoder_decoder_attention = encoder_decoder_attention

#         assert not self.self_attention or self.qkv_same_dim, (
#             "Self-attention requires query, key and " "value to be of the same size"
#         )

#         k_bias = True
#         if rescale_init:
#             k_bias = False

#         k_embed_dim = embed_dim
#         q_embed_dim = embed_dim

#         self.k_proj = quant_noise(
#             nn.Linear(self.kdim, k_embed_dim, bias=k_bias), q_noise, qn_block_size
#         )
#         self.v_proj = quant_noise(
#             nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
#         )
#         self.q_proj = quant_noise(
#             nn.Linear(embed_dim, q_embed_dim, bias=bias), q_noise, qn_block_size
#         )

#         self.out_proj = quant_noise(
#             nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
#         )

#         if add_bias_kv:
#             self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
#             self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
#         else:
#             self.bias_k = self.bias_v = None

#         self.add_zero_attn = add_zero_attn

#         self.gru_rel_pos = gru_rel_pos
#         if self.gru_rel_pos:
#             self.grep_linear = nn.Linear(self.q_head_dim, 8)
#             self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.qkv_same_dim:
#             # Empirically observed the convergence to be much better with
#             # the scaled initialization
#             nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
#             nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
#             nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
#         else:
#             nn.init.xavier_uniform_(self.k_proj.weight)
#             nn.init.xavier_uniform_(self.v_proj.weight)
#             nn.init.xavier_uniform_(self.q_proj.weight)

#         nn.init.xavier_uniform_(self.out_proj.weight)
#         if self.out_proj.bias is not None:
#             nn.init.constant_(self.out_proj.bias, 0.0)
#         if self.bias_k is not None:
#             nn.init.xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             nn.init.xavier_normal_(self.bias_v)
#         if self.has_relative_attention_bias:
#             nn.init.xavier_normal_(self.relative_attention_bias.weight)

#     def _relative_positions_bucket(self, relative_positions, bidirectional=True):
#         num_buckets = self.num_buckets
#         max_distance = self.max_distance
#         relative_buckets = 0

#         if bidirectional:
#             num_buckets = num_buckets // 2
#             relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
#             relative_positions = torch.abs(relative_positions)
#         else:
#             relative_positions = -torch.min(
#                 relative_positions, torch.zeros_like(relative_positions)
#             )

#         max_exact = num_buckets // 2
#         is_small = relative_positions < max_exact

#         relative_postion_if_large = max_exact + (
#             torch.log(relative_positions.float() / max_exact)
#             / math.log(max_distance / max_exact)
#             * (num_buckets - max_exact)
#         ).to(torch.long)
#         relative_postion_if_large = torch.min(
#             relative_postion_if_large,
#             torch.full_like(relative_postion_if_large, num_buckets - 1),
#         )

#         relative_buckets += torch.where(
#             is_small, relative_positions, relative_postion_if_large
#         )
#         return relative_buckets

#     def compute_bias(self, query_length, key_length):
#         context_position = torch.arange(query_length, dtype=torch.long)[:, None]
#         memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
#         relative_position = memory_position - context_position
#         relative_position_bucket = self._relative_positions_bucket(
#             relative_position, bidirectional=True
#         )
#         relative_position_bucket = relative_position_bucket.to(
#             self.relative_attention_bias.weight.device
#         )
#         values = self.relative_attention_bias(relative_position_bucket)
#         values = values.permute([2, 0, 1])
#         return values

#     def forward(
#         self,
#         query,
#         key: Optional[Tensor],
#         value: Optional[Tensor],
#         key_padding_mask: Optional[Tensor] = None,
#         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
#         need_weights: bool = True,
#         static_kv: bool = False,
#         attn_mask: Optional[Tensor] = None,
#         before_softmax: bool = False,
#         need_head_weights: bool = False,
#         position_bias: Optional[Tensor] = None,
#     ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
#         """Input shape: Time x Batch x Channel

#         Args:
#             key_padding_mask (ByteTensor, optional): mask to exclude
#                 keys that are pads, of shape `(batch, src_len)`, where
#                 padding elements are indicated by 1s.
#             need_weights (bool, optional): return the attention weights,
#                 averaged over heads (default: False).
#             attn_mask (ByteTensor, optional): typically used to
#                 implement causal attention, where the mask prevents the
#                 attention from looking forward in time (default: None).
#             before_softmax (bool, optional): return the raw attention
#                 weights and values before the attention softmax.
#             need_head_weights (bool, optional): return the attention
#                 weights for each head. Implies *need_weights*. Default:
#                 return the average attention weights over all heads.
#         """
#         if need_head_weights:
#             need_weights = True

#         is_tpu = query.device.type == "xla"

#         tgt_len, bsz, embed_dim = query.size()
#         src_len = tgt_len
#         assert embed_dim == self.embed_dim
#         assert list(query.size()) == [tgt_len, bsz, embed_dim]
#         if key is not None:
#             src_len, key_bsz, _ = key.size()
#             if not torch.jit.is_scripting():
#                 assert key_bsz == bsz
#                 assert value is not None
#                 assert src_len, bsz == value.shape[:2]

#         if self.has_relative_attention_bias and position_bias is None:
#             position_bias = self.compute_bias(tgt_len, src_len)
#             position_bias = (
#                 position_bias.unsqueeze(0)
#                 .repeat(bsz, 1, 1, 1)
#                 .view(bsz * self.num_heads, tgt_len, src_len)
#             )

#         if (
#             not is_tpu  # don't use PyTorch version on TPUs
#             and incremental_state is None
#             and not static_kv
#             # A workaround for quantization to work. Otherwise JIT compilation
#             # treats bias in linear module as method.
#             and not torch.jit.is_scripting()
#             and self.q_head_dim == self.head_dim
#         ):
#             assert key is not None and value is not None
#             assert attn_mask is None

#             attn_mask_rel_pos = None
#             if position_bias is not None:
#                 attn_mask_rel_pos = position_bias
#                 if self.gru_rel_pos:
#                     query_layer = query.transpose(0, 1)
#                     new_x_shape = query_layer.size()[:-1] + (self.num_heads, -1)
#                     query_layer = query_layer.view(*new_x_shape)
#                     query_layer = query_layer.permute(0, 2, 1, 3)
#                     _B, _H, _L, __ = query_layer.size()

#                     gate_a, gate_b = torch.sigmoid(
#                         self.grep_linear(query_layer)
#                         .view(_B, _H, _L, 2, 4)
#                         .sum(-1, keepdim=False)
#                     ).chunk(2, dim=-1)
#                     gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
#                     attn_mask_rel_pos = (
#                         gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias
#                     )

#                 attn_mask_rel_pos = attn_mask_rel_pos.view((-1, tgt_len, tgt_len))
#             k_proj_bias = self.k_proj.bias
#             if k_proj_bias is None:
#                 k_proj_bias = torch.zeros_like(self.q_proj.bias)

#             x, attn = F.multi_head_attention_forward(
#                 query,
#                 key,
#                 value,
#                 self.embed_dim,
#                 self.num_heads,
#                 torch.empty([0]),
#                 torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
#                 self.bias_k,
#                 self.bias_v,
#                 self.add_zero_attn,
#                 self.dropout_module.p,
#                 self.out_proj.weight,
#                 self.out_proj.bias,
#                 self.training,
#                 # self.training or self.dropout_module.apply_during_inference,
#                 key_padding_mask,
#                 need_weights,
#                 attn_mask_rel_pos,
#                 use_separate_proj_weight=True,
#                 q_proj_weight=self.q_proj.weight,
#                 k_proj_weight=self.k_proj.weight,
#                 v_proj_weight=self.v_proj.weight,
#             )
#             return x, attn, position_bias

#         if incremental_state is not None:
#             saved_state = self._get_input_buffer(incremental_state)
#             if saved_state is not None and "prev_key" in saved_state:
#                 # previous time steps are cached - no need to recompute
#                 # key and value if they are static
#                 if static_kv:
#                     assert self.encoder_decoder_attention and not self.self_attention
#                     key = value = None
#         else:
#             saved_state = None

#         if self.self_attention:
#             q = self.q_proj(query)
#             k = self.k_proj(query)
#             v = self.v_proj(query)
#         elif self.encoder_decoder_attention:
#             # encoder-decoder attention
#             q = self.q_proj(query)
#             if key is None:
#                 assert value is None
#                 k = v = None
#             else:
#                 k = self.k_proj(key)
#                 v = self.v_proj(key)

#         else:
#             assert key is not None and value is not None
#             q = self.q_proj(query)
#             k = self.k_proj(key)
#             v = self.v_proj(value)
#         q *= self.scaling

#         if self.bias_k is not None:
#             assert self.bias_v is not None
#             k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
#             v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
#             if attn_mask is not None:
#                 attn_mask = torch.cat(
#                     [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
#                 )
#             if key_padding_mask is not None:
#                 key_padding_mask = torch.cat(
#                     [
#                         key_padding_mask,
#                         key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
#                     ],
#                     dim=1,
#                 )

#         q = (
#             q.contiguous()
#             .view(tgt_len, bsz * self.num_heads, self.q_head_dim)
#             .transpose(0, 1)
#         )
#         if k is not None:
#             k = (
#                 k.contiguous()
#                 .view(-1, bsz * self.num_heads, self.k_head_dim)
#                 .transpose(0, 1)
#             )
#         if v is not None:
#             v = (
#                 v.contiguous()
#                 .view(-1, bsz * self.num_heads, self.head_dim)
#                 .transpose(0, 1)
#             )

#         if saved_state is not None:
#             # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
#             if "prev_key" in saved_state:
#                 _prev_key = saved_state["prev_key"]
#                 assert _prev_key is not None
#                 prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
#                 if static_kv:
#                     k = prev_key
#                 else:
#                     assert k is not None
#                     k = torch.cat([prev_key, k], dim=1)
#                 src_len = k.size(1)
#             if "prev_value" in saved_state:
#                 _prev_value = saved_state["prev_value"]
#                 assert _prev_value is not None
#                 prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
#                 if static_kv:
#                     v = prev_value
#                 else:
#                     assert v is not None
#                     v = torch.cat([prev_value, v], dim=1)
#             prev_key_padding_mask: Optional[Tensor] = None
#             if "prev_key_padding_mask" in saved_state:
#                 prev_key_padding_mask = saved_state["prev_key_padding_mask"]
#             assert k is not None and v is not None
#             key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
#                 key_padding_mask=key_padding_mask,
#                 prev_key_padding_mask=prev_key_padding_mask,
#                 batch_size=bsz,
#                 src_len=k.size(1),
#                 static_kv=static_kv,
#             )

#             saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
#             saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
#             saved_state["prev_key_padding_mask"] = key_padding_mask
#             # In this branch incremental_state is never None
#             assert incremental_state is not None
#             incremental_state = self._set_input_buffer(incremental_state, saved_state)
#         assert k is not None
#         assert k.size(1) == src_len

#         # This is part of a workaround to get around fork/join parallelism
#         # not supporting Optional types.
#         if key_padding_mask is not None and key_padding_mask.dim() == 0:
#             key_padding_mask = None

#         if key_padding_mask is not None:
#             assert key_padding_mask.size(0) == bsz
#             assert key_padding_mask.size(1) == src_len

#         if self.add_zero_attn:
#             assert v is not None
#             src_len += 1
#             k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
#             v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
#             if attn_mask is not None:
#                 attn_mask = torch.cat(
#                     [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
#                 )
#             if key_padding_mask is not None:
#                 key_padding_mask = torch.cat(
#                     [
#                         key_padding_mask,
#                         torch.zeros(key_padding_mask.size(0), 1).type_as(
#                             key_padding_mask
#                         ),
#                     ],
#                     dim=1,
#                 )

#         attn_weights = torch.bmm(q, k.transpose(1, 2))
#         attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

#         assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

#         if attn_mask is not None:
#             attn_mask = attn_mask.unsqueeze(0)
#             attn_weights += attn_mask

#         if key_padding_mask is not None:
#             # don't attend to padding symbols
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
#             if not is_tpu:
#                 attn_weights = attn_weights.masked_fill(
#                     key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
#                     float("-inf"),
#                 )
#             else:
#                 attn_weights = attn_weights.transpose(0, 2)
#                 attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
#                 attn_weights = attn_weights.transpose(0, 2)
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

#         if before_softmax:
#             return attn_weights, v, position_bias

#         if position_bias is not None:
#             if self.gru_rel_pos == 1:
#                 query_layer = q.view(bsz, self.num_heads, tgt_len, self.q_head_dim)
#                 _B, _H, _L, __ = query_layer.size()
#                 gate_a, gate_b = torch.sigmoid(
#                     self.grep_linear(query_layer)
#                     .view(_B, _H, _L, 2, 4)
#                     .sum(-1, keepdim=False)
#                 ).chunk(2, dim=-1)
#                 gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
#                 position_bias = (
#                     gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias
#                 )

#             position_bias = position_bias.view(attn_weights.size())

#             attn_weights = attn_weights + position_bias

#         attn_weights_float = F.softmax(attn_weights, dim=-1)
#         attn_weights = attn_weights_float.type_as(attn_weights)
#         attn_probs = self.dropout_module(attn_weights)

#         assert v is not None
#         attn = torch.bmm(attn_probs, v)
#         assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
#         attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
#         attn = self.out_proj(attn)
#         attn_weights: Optional[Tensor] = None
#         if need_weights:
#             attn_weights = attn_weights_float.view(
#                 bsz, self.num_heads, tgt_len, src_len
#             ).transpose(1, 0)
#             if not need_head_weights:
#                 # average attention weights over heads
#                 attn_weights = attn_weights.mean(dim=0)

#         return attn, attn_weights, position_bias

#         @staticmethod
#         def _append_prev_key_padding_mask(
#             batch_size: int,
#             src_len: int,
#             static_kv: bool,
#             key_padding_mask: Optional[Tensor],
#             prev_key_padding_mask: Optional[Tensor],
#         ):
#             # saved key padding masks have shape (bsz, seq_len)
#             if prev_key_padding_mask is not None and static_kv:
#                 new_key_padding_mask = prev_key_padding_mask
#             elif prev_key_padding_mask is not None and key_padding_mask is not None:
#                 new_key_padding_mask = torch.cat(
#                     [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
#                 )
#             # During incremental decoding, as the padding token enters and
#             # leaves the frame, there will be a time when prev or current
#             # is None
#             elif prev_key_padding_mask is not None:
#                 if src_len > prev_key_padding_mask.size(1):
#                     filler = torch.zeros(
#                         (batch_size, src_len - prev_key_padding_mask.size(1)),
#                         device=prev_key_padding_mask.device,
#                     )
#                     new_key_padding_mask = torch.cat(
#                         [prev_key_padding_mask.float(), filler.float()], dim=1
#                     )
#                 else:
#                     new_key_padding_mask = prev_key_padding_mask.float()
#             elif key_padding_mask is not None:
#                 if src_len > key_padding_mask.size(1):
#                     filler = torch.zeros(
#                         (batch_size, src_len - key_padding_mask.size(1)),
#                         device=key_padding_mask.device,
#                     )
#                     new_key_padding_mask = torch.cat(
#                         [filler.float(), key_padding_mask.float()], dim=1
#                     )
#                 else:
#                     new_key_padding_mask = key_padding_mask.float()
#             else:
#                 new_key_padding_mask = prev_key_padding_mask
#             return new_key_padding_mask

# class TransformerSentenceEncoderLayer(nn.Module):
#     """
#     Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
#     models.
#     """

#     def __init__(
#         self,
#         embedding_dim: float = 768,
#         ffn_embedding_dim: float = 3072,
#         num_attention_heads: int = 8,
#         dropout: float = 0.1,
#         attention_dropout: float = 0.1,
#         activation_dropout: float = 0.1,
#         activation_fn: str = "relu",
#         layer_norm_first: bool = False,
#     ) -> None:

#         super().__init__()
#         # Initialize parameters
#         self.embedding_dim = embedding_dim
#         self.dropout = dropout
#         self.activation_dropout = activation_dropout

#         # Initialize blocks
#         self.self_attn = MultiheadAttention(
#             self.embedding_dim,
#             num_attention_heads,
#             dropout=attention_dropout,
#             self_attention=True,
#         )

#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(self.activation_dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.layer_norm_first = layer_norm_first

#         # layer norm associated with the self attention layer
#         self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
#         self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
#         self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

#         # layer norm associated with the position wise feed-forward NN
#         self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

#     def forward(
#         self,
#         query: torch.Tensor,
#         key: torch.Tensor,
#         value: torch.Tensor,
#         self_attn_mask: torch.Tensor = None,
#         self_attn_padding_mask: torch.Tensor = None,
#         need_weights: bool = False,
#         att_args=None,
#     ):
#         """
#         LayerNorm is applied either before or after the self-attention/ffn
#         modules similar to the original Transformer imlementation.
#         """
#         # TODO: positional encoding?
#         x = query
#         residual = x

#         if self.layer_norm_first:
#             x = self.self_attn_layer_norm(x)
#             x, attn, _ = self.self_attn(
#                 query=query,
#                 key=key,
#                 value=value,
#                 key_padding_mask=self_attn_padding_mask,
#                 attn_mask=self_attn_mask,
#                 need_weights=False,
#             )
#             x = self.dropout1(x)
#             x = residual + x

#             residual = x
#             x = self.final_layer_norm(x)
#             x = nn.functional.relu(self.fc1(x))
#             x = self.dropout2(x)
#             x = self.fc2(x)

#             layer_result = x

#             x = self.dropout3(x)
#             x = residual + x
#         else:
#             x, attn, _ = self.self_attn(
#                 query=query,
#                 key=key,
#                 value=value,
#                 key_padding_mask=self_attn_padding_mask,
#                 need_weights=False,
#             )

#             x = self.dropout1(x)
#             x = residual + x

#             x = self.self_attn_layer_norm(x)

#             residual = x
#             x = nn.functional.relu(self.fc1(x))
#             x = self.dropout2(x)
#             x = self.fc2(x)

#             layer_result = x

#             x = self.dropout3(x)
#             x = residual + x
#             x = self.final_layer_norm(x)

#         return x, (attn, layer_result)

# Model
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout, input_embed=True):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout), num_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim, dropout), num_layers)
        self.input_embed = input_embed
        if self.input_embed:
            self.src_embed = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
            self.tgt_embed = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.hidden_dim = hidden_dim
    
    def forward_encoder(self, src, device, src_mask=None):
        src = src.transpose(1, 0)
        tgt = tgt.transpose(1, 0)
        # TODO: maybe src & tgt masks not needed?
        src_mask_ = self._generate_square_subsequent_mask(src.size(0), device).to(device)
        tgt_mask_ = self._generate_square_subsequent_mask(tgt.size(0), device).to(device)
        mem_mask = self._generate_square_subsequent_mask(src.size(0), device, tz=tgt.size(0)).to(device)
        src_key_padding_mask = (src == 0).transpose(0,1) if src_mask is None else src_mask
        tgt_key_padding_mask = (tgt == 0).transpose(0,1) if tgt_mask is None else tgt_mask
        if self.input_embed:
            src_embed = self.dropout(self.src_embed(src))
            tgt_embed = self.dropout(self.tgt_embed(tgt))
        else:
            src_embed = src
            tgt_embed = tgt
        memory = self.encoder(src_embed, src_key_padding_mask=src_key_padding_mask, mask=src_mask_)
        
        outputs = self.linear(outputs)

        return outputs.permute(1, 0, 2)

    def forward_decoder(self, src, tgt, device, src_mask=None, tgt_mask=None, return_normalized=True):
        src = src.transpose(1, 0)
        tgt = tgt.transpose(1, 0)
        # TODO: maybe src & tgt masks not needed?
        src_mask_ = self._generate_square_subsequent_mask(src.size(0), device).to(device)
        tgt_mask_ = self._generate_square_subsequent_mask(tgt.size(0), device).to(device)
        mem_mask = self._generate_square_subsequent_mask(src.size(0), device, tz=tgt.size(0)).to(device)
        src_key_padding_mask = (src == 0).transpose(0,1) if src_mask is None else src_mask
        tgt_key_padding_mask = (tgt == 0).transpose(0,1) if tgt_mask is None else tgt_mask
        if self.input_embed:
            src_embed = self.dropout(self.src_embed(src))
            tgt_embed = self.dropout(self.tgt_embed(tgt))
        else:
            src_embed = src
            tgt_embed = tgt
        memory = self.encoder(src_embed, src_key_padding_mask=src_key_padding_mask, mask=src_mask_)
        outputs = self.decoder(
            tgt_embed, 
            memory, 
            tgt_mask=tgt_mask_, 
            memory_mask=mem_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=src_key_padding_mask,
            )
        outputs = self.linear(outputs)
        
        if return_normalized:
            outputs = self.softmax(outputs)

        return outputs.permute(1, 0, 2)

    # def forward_inference(self, src, projector, device, src_mask=None, return_normalized=True, max_len=300):
    #     src = src.transpose(1, 0)
    #     src_mask_ = self._generate_square_subsequent_mask(src.size(0), device).to(device)
    #     src_key_padding_mask = (src == 0).transpose(0,1) if src_mask is None else src_mask
    #     if self.input_embed:
    #         src_embed = self.src_embed(src)
    #     else:
    #         src_embed = src
    #     memory = self.encoder(src_embed, src_key_padding_mask=src_key_padding_mask, mask=src_mask_)

    #     # outputs = self.decoder(
    #     #         memory,
    #     #         memory,
    #     #         tgt_mask=src_mask_,
    #     #         memory_mask=src_mask_,
    #     #         tgt_key_padding_mask=src_key_padding_mask,
    #     #         memory_key_padding_mask=src_key_padding_mask,
    #     #         )

    #     # return self.linear(outputs).permute(1, 0, 2), None
        
    #     tgt = torch.zeros((src.size(1), 1), dtype=torch.long, device=device).transpose(0,1)
    #     outputs = torch.zeros((max_len, src.size(1), self.hidden_dim), dtype=torch.float, device=device)
    #     decoded = torch.zeros(src.size(1)).bool().to(device)
        
    #     regressive = 0
    #     while regressive < max_len:
    #         if self.input_embed:
    #             tgt_embed = self.tgt_embed(tgt)
    #         else:
    #             tgt_embed = tgt

    #         output = self.decoder(
    #             tgt_embed,
    #             memory,
    #             )

    #         output = self.linear(output[-1])    # B, C

    #         if return_normalized:
    #             output = self.softmax(output)
            
    #         hyp_tokens = projector(output).argmax(dim=-1)   # B
    #         for i, hyp_token_ in enumerate(hyp_tokens):
    #             if hyp_token_ == 0:
    #                 decoded[i] = True
    #         tgt = torch.cat((tgt, hyp_tokens.unsqueeze(0)), dim=0)  # T, B

    #         if False not in decoded:
    #             break
                
    #         outputs[regressive, :, :] = output

    #         regressive += 1

    #     return outputs[:regressive + 1,:,:].transpose(0,1), tgt.transpose(0,1)
    
    def _generate_square_subsequent_mask(self, sz, device, tz=None):
        if tz is None:
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        else:
            mask = (torch.triu(torch.ones(sz, tz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        
        self.internal_LM = Seq2Seq(
            input_dim=vocab_size, 
            output_dim=encoder_dim, 
            hidden_dim=decoder_dim, 
            num_layers=2, 
            num_heads=8, 
            dropout=0.1,
            )

        self.disrupter = Seq2Seq(
            input_dim=vocab_size, 
            output_dim=vocab_size, 
            hidden_dim=decoder_dim, 
            num_layers=2, 
            num_heads=8, 
            dropout=0.1,
            )

        self.sc_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

        self.modality_combiner = Seq2Seq(
            input_dim=encoder_dim, 
            output_dim=encoder_dim, 
            hidden_dim=encoder_dim, 
            num_layers=2, 
            num_heads=8, 
            dropout=0.1,
            input_embed=False,
            )

        # self.modality_combiner = nn.TransformerDecoder(
        #     nn.TransformerDecoderLayer(
        #         output_dim=encoder_dim,
        #         num_heads=8,
        #         hidden_dim=encoder_dim,
        #         dropout=0.1,
        #         ), 
        #     num_layers=2
        #     )

        self.IL_proj = nn.Linear(encoder_dim, vocab_size)
        self.encoder_out_proj = nn.Linear(encoder_dim, encoder_dim)

        self.simple_am_proj = nn.Linear(
            encoder_dim,
            vocab_size,
        )
        self.simple_lm_proj = nn.Linear(decoder_dim, vocab_size)

        self.ctc_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, vocab_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        ctc_only = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return a tuple containing simple loss, pruned loss, and ctc-output.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 2 or x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        use_lm_loss = True

        # compute ctc log-probs
        ctc_output = self.ctc_output(encoder_out)

        if ctc_only:
          return ctc_output

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        lm = self.simple_lm_proj(decoder_out)

        # Make y padding mask [B, S]
        y_padding_mask = decoder_out.new_ones((decoder_out.size(0), decoder_out.size(1)-1))
        for i, length in enumerate(y_lens):
          y_padding_mask[i, :length] = 0
        y_padding_mask = y_padding_mask.bool()

        # Make y_inp from ctc_output
        ctc_output_ = ctc_output.argmax(-1)
        hyp = []
        max_len = 0
        for encoder_out_ in ctc_output_:
          encoder_out_ = encoder_out_.unique_consecutive()
          encoder_out_tmp = encoder_out_[torch.where(encoder_out_ != 0)]

          # Prevent empty hypo
          if not len(encoder_out_tmp) < 1:
            encoder_out_ = encoder_out_tmp
          hyp.append(encoder_out_.detach().cpu().tolist())
          
          if max_len < len(encoder_out_):
            max_len = len(encoder_out_)
        hyp = k2.RaggedTensor(hyp).to(x.device)
        device = x.device

        # Make padding masks [B, S]
        row_splits = hyp.shape.row_splits(1)
        hyp_lens = row_splits[1:] - row_splits[:-1]
        hyp_padding_mask = decoder_out.new_ones((decoder_out.size(0), max_len))
        for i, length in enumerate(hyp_lens):
          hyp_padding_mask[i, :length] = 0
        hyp_padding_mask = hyp_padding_mask.bool()

        sos_hyp_padding_mask = decoder_out.new_ones((decoder_out.size(0), max_len + 1))
        sos_hyp_lens = hyp_lens + 1
        for i, length in enumerate(sos_hyp_lens):
          sos_hyp_padding_mask[i, :length] = 0
        sos_hyp_padding_mask = sos_hyp_padding_mask.bool()

        x_padding_mask = decoder_out.new_ones((encoder_out.size(0), encoder_out.size(1)))
        for i, length in enumerate(x_lens):
          x_padding_mask[i, :length] = 0
        x_padding_mask = x_padding_mask.bool()
        
        y_padding_mask = decoder_out.new_ones((len(y_lens), max(y_lens)))
        for i, length in enumerate(y_lens):
          y_padding_mask[i, :length] = 0
        y_padding_mask = y_padding_mask.bool()

        sos_y_padding_mask = decoder_out.new_ones((len(y_lens), max(y_lens) + 1))
        sos_y_lens = y_lens + 1
        for i, length in enumerate(sos_y_lens):
          sos_y_padding_mask[i, :length] = 0
        sos_y_padding_mask = sos_y_padding_mask.bool()

        hyp_padded = hyp.pad(mode="constant", padding_value=0)
        sos_hyp = add_sos(hyp, sos_id=blank_id)
        sos_hyp_padded = sos_hyp.pad(mode="constant", padding_value=blank_id)

        ### Do spell correcting ###
        # IL_out: [B, S + 1, encoder_dim]
        IL_out = self.internal_LM(
            hyp_padded, 
            sos_y_padded, 
            src_mask=hyp_padding_mask, 
            tgt_mask=sos_y_padding_mask, 
            device=device, 
            return_normalized=False
        )
        tar_y = add_eos(y, eos_id=blank_id)
        tar_y_padded = tar_y.pad(mode="constant", padding_value=blank_id)

        ### Do spell disrupting ###
        # dis_out: [B, S + 1, encoder_dim]
        # dis_out = self.disrupter(
        #     y_padded, 
        #     sos_hyp_padded, 
        #     src_mask=y_padding_mask, 
        #     tgt_mask=sos_hyp_padding_mask, 
        #     device=device, 
        #     return_normalized=False
        # )
        tar_hyp = add_eos(hyp, eos_id=blank_id)
        tar_hyp_padded = tar_hyp.pad(mode="constant", padding_value=blank_id)
        
        if use_lm_loss:
          hyp = self.IL_proj(IL_out)
          hyp_loss = self.sc_criterion(hyp.reshape(-1, hyp.size(-1)), tar_y_padded.view(-1).long().to(device)).sum()
          lm_loss = hyp_loss
        #   dis_loss = self.sc_criterion(dis_out.reshape(-1, dis_out.size(-1)), tar_hyp_padded.view(-1).long().to(device)).sum()
        #   lm_loss = hyp_loss + dis_loss

          # combine two modalities
          encoder_out = self.modality_combiner(
            IL_out, 
            encoder_out, 
            src_mask=sos_y_padding_mask,
            tgt_mask=x_padding_mask,
            device=device,
            return_normalized=False,
          )

        am = self.simple_am_proj(encoder_out)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        plm = self.joiner.decoder_proj(decoder_out)

        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=plm,
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return (simple_loss, pruned_loss, ctc_output), lm_loss

    def decode(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        sp,
    ):
        from beam_search import greedy_search_batch

        encoder_out, x_lens = self.encoder(x, x_lens)

        hyps = []
        hyp_tokens = greedy_search_batch(self, encoder_out, x_lens)

        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())

        return hyps

def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module