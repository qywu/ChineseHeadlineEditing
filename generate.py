import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchfly.text.decode import top_filtering
from typing import List, Union

# pylint: disable=no-member

logger = logging.getLogger(__name__)


class DefaultDecodingConfig:
    max_length = 100


class DecodingHelper:
    def __init__(self, model, decode_config, device):
        self.config = decode_config
        self.config = {
            "max_length": 100,
            "do_sample": True,
            "num_beams": 1,
            "temperature": 0.9,
            "top_k": -1,
            "top_p": 0.9,
            "retition_penalty": None,
            "bos_token_id": None,
            "pad_token_id": None,
            "eos_token_ids": None,
            "length_penalty": None,
            "num_return_sequences": None,
        }
        self.device = device
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor = None,
        max_length: int = None,
        do_sample: bool = None,
        num_beams: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_ids=None,
        length_penalty=None,
        num_return_sequences=None,
    ):
        r""" Generates sequences for models with a LM head. The method currently supports greedy or penalized greedy decoding, nucleus sampling
        and beam-search.
        Parameters:
            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.
            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between 1 and infinity. Default to 20.
            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Default to greedy sampling.
            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.
            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictely positive. Default to 1.0.
            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.
            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.
            bos_token_id: (`optional`) int
                Beginning of sentence token if no prompt is provided. Default to 0.
            eos_token_ids: (`optional`) int or list of int
                End of sequence token or list of tokens to stop the generation. Default to 0.
            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.
            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.
        """
        # We cannot generate if the model does not have a LM head
        # if self.get_output_embeddings() is None:
        #     raise AttributeError(
        #         "You tried to generate sequences with a model that does not have a LM Head."
        #         "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`)"
        #     )

        # setup the configuration
        self.config.max_length = (
            max_length if max_length is not None else self.config.max_length
        )
        self.config.do_sample = (
            do_sample if do_sample is not None else self.config.do_sample
        )
        self.config.num_beams = (
            num_beams if num_beams is not None else self.config.num_beams
        )
        self.config.temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        self.config.top_k = top_k if top_k is not None else self.config.top_k
        self.config.top_p = top_p if top_p is not None else self.config.top_p
        self.config.repetition_penalty = (
            repetition_penalty
            if repetition_penalty is not None
            else self.config.repetition_penalty
        )
        self.config.bos_token_id = (
            bos_token_id if bos_token_id is not None else self.config.bos_token_id
        )
        self.config.pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        self.config.eos_token_ids = (
            eos_token_ids if eos_token_ids is not None else self.config.eos_token_ids
        )
        self.config.length_penalty = (
            length_penalty if length_penalty is not None else self.config.length_penalty
        )
        self.config.num_return_sequences = (
            num_return_sequences
            if num_return_sequences is not None
            else self.config.num_return_sequences
        )

        # setup batch size
        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        # make eos token into a list
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        # unconditional generation
        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=self.device,
            )

        # assertion
        self.assertion_check()

        # current position and vocab size
        cur_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size

        # calculate the effective batch size
        if num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = input_ids.unsqueeze(1).expand(
                batch_size, num_return_sequences, cur_len
            )
            input_ids = input_ids.contiguous().view(
                batch_size * num_return_sequences, cur_len
            )  # (batch_size * num_return_sequences, cur_len)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        # beam search or sampling
        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                length_penalty,
                num_beams,
                vocab_size,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
            )

        # if num_return_sequences != 1:
        #     output = output.view(batch_size, num_return_sequences, -1)
        # return output

        return None

    def assertion_check(self):
        assert (
            isinstance(self.config.max_length, int) and self.config.max_length > 0
        ), "`max_length` should be a strictely positive integer."
        assert isinstance(
            self.config.do_sample, bool
        ), "`do_sample` should be a boolean."
        assert (
            isinstance(self.config.num_beams, int) and self.config.num_beams > 0
        ), "`num_beams` should be a strictely positive integer."
        assert (
            self.config.temperature > 0
        ), "`temperature` should be strictely positive."
        assert (
            isinstance(self.config.top_k, int) and self.config.top_k >= 0
        ), "`top_k` should be a positive integer."
        assert 0 <= self.config.top_p <= 1, "`top_p` should be between 0 and 1."
        assert (
            self.config.repetition_penalty >= 1.0
        ), "`repetition_penalty` should be >= 1."
        assert (
            isinstance(self.config.bos_token_id, int) and self.config.bos_token_id >= 0
        ), "`bos_token_id` should be a positive integer."
        assert (
            isinstance(self.config.pad_token_id, int) and self.config.pad_token_id >= 0
        ), "`pad_token_id` should be a positive integer."
        assert isinstance(self.config.eos_token_ids, (list, tuple)) and (
            e >= 0 for e in self.config.eos_token_ids
        ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
        assert (
            self.config.length_penalty > 0
        ), "`length_penalty` should be strictely positive."
        assert (
            isinstance(self.config.num_return_sequences, int)
            and self.config.num_return_sequences > 0
        ), "`num_return_sequences` should be a strictely positive integer."

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        past = None

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation()
            next_token_logits, past = self.model(**model_inputs, past=past)

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to be multiplied to reduce the previous 
                        # token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            tokens_to_add = next_token * unfinished_sents + pad_token_id * (1 - unfinished_sents)
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            for eos_token_id in eos_token_ids:
                unfinished_sents.mul_(tokens_to_add.ne(eos_token_id).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add eos_token_ids to unfinished sentences
        if cur_len == max_length:
            input_ids[:, -1].masked_fill_(unfinished_sents.to(dtype=torch.bool), eos_token_ids[0])

        return input_ids


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.hyp)]
                )
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return (
                self.worst_score
                >= best_sum_logprobs / self.max_length ** self.length_penalty
            )

