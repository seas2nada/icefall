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


from typing import Tuple
import logging

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface

from icefall.utils import add_sos


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
        x: k2.RaggedTensor,
        y: k2.RaggedTensor,
        lm_scale: float = 0.0,
        return_hyp: bool = False,
        blank_id: int = 0,
        use_ctc: bool = False,
        input_is_gaussian: bool = True,
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
        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]
        row_splits = x.shape.row_splits(1)
        x_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id

        # make target
        with torch.no_grad():
          sos_y = add_sos(y, sos_id=blank_id)

          # sos_y_padded: [B, S + 1], start with SOS.
          sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

          # decoder_out: [B, S + 1, decoder_dim]
          decoder_out_y = self.decoder(sos_y_padded)

          lm_y = self.simple_lm_proj(decoder_out_y)
          prune_lm_y = self.joiner.decoder_proj(decoder_out_y)

        sos_x = add_sos(x, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_x_padded = sos_x.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_x_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)
        y_padded = y_padded.to(torch.int64)
        # boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
        # boundary[:, 2] = y_lens
        # boundary[:, 3] = x_lens

        lm = self.simple_lm_proj(decoder_out)
        prune_lm = self.joiner.decoder_proj(decoder_out)
        if return_hyp:
          return lm[:,1:,:].argmax(-1)
        # am = self.simple_am_proj(encoder_out)

        # if use_ctc:
        #   # TODO: CTC loss for word-replacement
        #   lprobs = torch.nn.functional.log_softmax(lm[:,1:,:].float(), dim=-1)

        #   loss = torch.nn.functional.ctc_loss(
        #     lprobs.transpose(0,1), # T,B,C
        #     torch.tensor(y_padded),  # B,T
        #     x_lens,
        #     y_lens,
        #     blank=blank_id,
        #     reduction="mean"
        #     )
        #   loss = loss * 0.00001

        # else:
        #   loss = torch.nn.functional.cross_entropy(lm[:,1:,:].reshape(-1, lm.size(-1)), y_padded.view(-1))

        # loss = torch.nn.functional.cross_entropy(lm[:,1:,:].reshape(-1, lm.size(-1)), y_padded.view(-1))

        loss = torch.nn.functional.mse_loss(lm, lm_y)
        loss += torch.nn.functional.mse_loss(prune_lm, prune_lm_y)

        # TODO: we need lm_pruned to be trained too
        # # ranges : [B, T, prune_range]
        # ranges = k2.get_rnnt_prune_ranges(
        #     px_grad=px_grad,
        #     py_grad=py_grad,
        #     boundary=boundary,
        #     s_range=prune_range,
        # )

        # # am_pruned : [B, T, prune_range, encoder_dim]
        # # lm_pruned : [B, T, prune_range, decoder_dim]
        # am_pruned, lm_pruned = k2.do_rnnt_pruning(
        #     am=self.joiner.encoder_proj(encoder_out),
        #     lm=self.joiner.decoder_proj(decoder_out),
        #     ranges=ranges,
        # )

        # # logits : [B, T, prune_range, vocab_size]

        # # project_input=False since we applied the decoder's input projections
        # # prior to do_rnnt_pruning (this is an optimization for speed).
        # logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        # with torch.cuda.amp.autocast(enabled=False):
        #     pruned_loss = k2.rnnt_loss_pruned(
        #         logits=logits.float(),
        #         symbols=y_padded,
        #         ranges=ranges,
        #         termination_symbol=blank_id,
        #         boundary=boundary,
        #         reduction="sum",
        #     )
        return loss

    def decode(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        sp,
        decoding_graph=None,
        method="greedy_search_batch",
    ):
        from beam_search import greedy_search_batch, ctc_greedy_search, modified_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG

        encoder_out, x_lens = self.encoder(x, x_lens)

        hyps = []
        if "fast_beam_search_nbest" in method:
          beam = 4
          max_contexts = 8
          max_states = 64
          num_paths = 200
          nbest_scale = 0.5

          if decoding_graph is None:
            raise NotImplementedError("need decoding graph for fast beam search")
          
          if method == "fast_beam_search_nbest_LG":
            hyp_tokens = fast_beam_search_nbest_LG(
                model=self,
                decoding_graph=decoding_graph,
                encoder_out=encoder_out,
                encoder_out_lens=x_lens,
                beam=beam,
                max_contexts=max_contexts,
                max_states=max_states,
                num_paths=num_paths,
                nbest_scale=nbest_scale,
            )

          else:
            method = fast_beam_search_nbest
            hyp_tokens = method(
              self, 
              decoding_graph=decoding_graph,
              encoder_out=encoder_out,
              encoder_out_lens=x_lens,
              beam=beam,
              max_contexts=max_contexts,
              max_states=max_states,
              num_paths=num_paths,
              nbest_scale=nbest_scale,
              )
        
        else:
          if method == "greedy_search_batch":
            method = greedy_search_batch
          elif method == "ctc_greedy_search":
            method = ctc_greedy_search
          else:
            method = modified_beam_search
          hyp_tokens = method(self, encoder_out, x_lens)

        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())

        return hyps