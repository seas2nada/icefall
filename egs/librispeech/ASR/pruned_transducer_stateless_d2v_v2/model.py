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
from transformer_scmodule import TransformerEncoder, TransformerDecoder

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
        self.sc_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        
        self.internal_LM_encoder = TransformerEncoder(
            input_size=vocab_size, 
            output_size=encoder_dim, 
            linear_units=2048, 
            num_blocks=2,
            attention_heads=4,
            dropout_rate=0.1,
            padding_idx=0,
            )
        self.internal_LM_decoder = TransformerDecoder(
            vocab_size=vocab_size, 
            encoder_output_size=encoder_dim,
            num_blocks=2,
            attention_heads=4,
            dropout_rate=0.1,
            use_output_layer=True,
            )

        self.modality_combiner = TransformerDecoder(
            vocab_size=encoder_dim, 
            encoder_output_size=vocab_size,
            num_blocks=2,
            attention_heads=4,
            dropout_rate=0.1,
            input_layer="linear",
            )
        self.combined_encoder_out_proj = nn.Linear(encoder_dim, encoder_dim)
        self.bias_dropout = torch.nn.Dropout(0.1)
        self.enc_layernorm = torch.nn.LayerNorm(encoder_dim, eps=1e-5, elementwise_affine=True)

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
        # IL_out: [B, S + 1, vocab_size]
        hs_padded, hs_lens = self.internal_LM_encoder(
            hyp_padded,
            hyp_lens,
        )
        IL_out, IL_lens = self.internal_LM_decoder(
            hs_padded,
            hs_lens,
            sos_y_padded,
            sos_y_lens,
        )

        # combine two modalities
        encoder_out_bias, encoder_out_lens = self.modality_combiner(
            IL_out,
            sos_y_lens,
            encoder_out,
            x_lens,
        )
        encoder_out_bias = self.combined_encoder_out_proj(encoder_out_bias)
        encoder_out_bias = self.bias_dropout(encoder_out_bias)
        encoder_out = encoder_out + encoder_out_bias
        encoder_out = self.enc_layernorm(encoder_out)

        tar_y = add_eos(y, eos_id=1)
        tar_y_padded = tar_y.pad(mode="constant", padding_value=-1)
        
        hyp = IL_out
        hyp_loss = self.sc_criterion(hyp.reshape(-1, hyp.size(-1)), tar_y_padded.view(-1).long().to(device)).sum()
        lm_loss = hyp_loss

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