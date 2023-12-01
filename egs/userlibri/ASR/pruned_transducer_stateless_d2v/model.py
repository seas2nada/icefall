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

from icefall.utils import add_sos, add_eos


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
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        online_model = None,
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
        x_lens_org = x_lens

        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        # compute ctc log-probs
        ctc_output = self.ctc_output(encoder_out)

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

        # calculate lm ratio
        with torch.no_grad():
          if p13n_rnn_lm is not None:
            logits, _ = rnn_lm.module.predict_batch(y, y_lens, sos_id=1, eos_id=1, blank_id=0, return_logits=True)
            p13n_logits, _ = p13n_rnn_lm.module.predict_batch(y, y_lens, sos_id=1, eos_id=1, blank_id=0, return_logits=True)
            logits = logits[:,:-1,:]
            p13n_logits = p13n_logits[:,:-1,:]

            y = y_padded
            
            scores = torch.zeros_like(y, dtype=torch.float, device=x.device) # initialize scores tensor for general LM
            p13n_scores = torch.zeros_like(y, dtype=torch.float, device=x.device) # initialize scores tensor for p13n LM
            lm_ratio =  torch.zeros(logits.size(0), device=x.device)
            for batch in range(logits.size(0)):
              scores[batch] = torch.gather(logits[batch], 1, y[batch].unsqueeze(1)).squeeze(1) # index along 
              p13n_scores[batch] = torch.gather(p13n_logits[batch], 1, y[batch].unsqueeze(1)).squeeze(1) # index along 
            
              # temporal average score
              lm_ratio[batch] = torch.mean(p13n_scores[batch, :y_lens[batch]] / scores[batch, :y_lens[batch]])

            # normalize lm_ratio
            max_weight = 4
            eps = 1e-6
            min_value = torch.min(lm_ratio)
            max_value = torch.max(lm_ratio)
            lm_ratio = max_weight * (lm_ratio - min_value) / (max_value - min_value)
            lm_ratio += eps

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        reduction = "sum" if p13n_rnn_lm is None else "none"
        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction=reduction,
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
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
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
                reduction=reduction,
            )
        
        if online_model is not None:
          o_simple_loss, o_pruned_loss = None, None
          if online_model is not None:
            with torch.no_grad():
              o_encoder_out, x_lens = online_model.module.encoder(x, x_lens_org)
              o_ctc_output = online_model.module.ctc_output(o_encoder_out)

              # decoder_out: [B, S + 1, decoder_dim]
              o_decoder_out = online_model.module.decoder(sos_y_padded)

              with torch.cuda.amp.autocast(enabled=False):
                  o_simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
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
                
              o_am_pruned, o_lm_pruned = k2.do_rnnt_pruning(
                am=online_model.module.joiner.encoder_proj(o_encoder_out),
                lm=online_model.module.joiner.decoder_proj(o_decoder_out),
                ranges=ranges,
              )
              o_logits = self.joiner(o_am_pruned, o_lm_pruned, project_input=False)

              with torch.cuda.amp.autocast(enabled=False):
                o_pruned_loss = k2.rnnt_loss_pruned(
                    logits=o_logits.float(),
                    symbols=y_padded,
                    ranges=ranges,
                    termination_symbol=blank_id,
                    boundary=boundary,
                    reduction="sum",
                )

            return (simple_loss, pruned_loss, ctc_output, (o_simple_loss, o_pruned_loss))
        
        empty_return = None
        return (simple_loss, pruned_loss, ctc_output, empty_return)

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