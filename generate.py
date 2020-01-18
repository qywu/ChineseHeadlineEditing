import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torchfly
from torchfly.text.decode import top_filtering
from typing import List, Union, Dict

# pylint: disable=no-member

logger = logging.getLogger(__name__)


class DefaultDecodingConfig:
    num_return_sequences = 1
    max_length = 100
    do_sample = True
    num_beams = 1
    temperature = 0.9
    top_k = -1
    top_p = 0.9
    retition_penalty = 1.0
    length_penalty = 1.0
    eos_token_ids = []
    bos_token_id = None
    pad_token_id = None


class DecodingHelper:
    def __init__(self, model, device=None, decode_config=None):
        self.config = decode_config if decode_config else DefaultDecodingConfig
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
        self.config.max_length = (max_length if max_length is not None else self.config.max_length)
        self.config.do_sample = (do_sample if do_sample is not None else self.config.do_sample)
        self.config.num_beams = (num_beams if num_beams is not None else self.config.num_beams)
        self.config.temperature = (temperature if temperature is not None else self.config.temperature)
        self.config.top_k = top_k if top_k is not None else self.config.top_k
        self.config.top_p = top_p if top_p is not None else self.config.top_p
        self.config.repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        )
        self.config.bos_token_id = (bos_token_id if bos_token_id is not None else self.config.bos_token_id)
        self.config.pad_token_id = (pad_token_id if pad_token_id is not None else self.config.pad_token_id)
        self.config.eos_token_ids = (eos_token_ids if eos_token_ids is not None else self.config.eos_token_ids)
        self.config.length_penalty = (length_penalty if length_penalty is not None else self.config.length_penalty)
        self.config.num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
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
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=self.device,
            )

        # assertion
        self._assertion_check()

        # current position and vocab size
        cur_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size

        # calculate the effective batch size
        if num_return_sequences != 1 and do_sample:
            # Expand input to num return sequences
            input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
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

    def prepare_inputs_for_generation(self):
        return {}

    def _generate_no_beam_search(self, start_input_ids: torch.Tensor, states: Dict) -> Dict[str, List]:
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
            Efficient generation is implemented.
        """
        # record the index of each sequence for pop out
        sequence_indices = {i for i in range(self.config.batch_size)}
        token_sequences = {i: [self.config.bos_token_id] for i in range(self.config.batch_size)}
        log_prob_sequences = {i: [0.0] for i in range(self.config.batch_size)}

        # main generation loop
        for cur_len in range(self.config.max_length):
            # generate next token
            next_token_logits, past = self.model(**states)
            next_token_logits = next_token_logits[:, -1, :]
            next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if self.config.repetition_penalty != 1.0:
                for i, seq_idx in enumerate(sequence_indices):
                    for previous_token in set(token_sequences[seq_idx]):
                        # if score < 0 then repetition penalty has to be multiplied to reduce the previous
                        # token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= self.config.repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= self.config.repetition_penalty

            if self.config.do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                next_token_logits = next_token_logits / self.config.temperature
                # Top-p/top-k filtering
                next_token_logits = top_filtering(next_token_logits, top_k=self.config.top_k, top_p=self.config.top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

            next_token_log_probs = torch.gather(next_token_log_probs, dim=1, index=next_token)
            next_token_list = next_token.squeeze(1).tolist()
            next_token_log_probs_list = next_token_log_probs.squeeze(1).tolist()

            # collect next token
            # first add all the tokens to sequences
            for i, seq_idx in enumerate(sequence_indices):
                token_sequences[seq_idx].append(next_token_list[i])
                log_prob_sequences.append(next_token_log_probs_list[i])

            # then pop finished sequences
            pop_flag = False
            nonpop_indices = []
            for i, seq_idx in enumerate(sequence_indices):
                if len(token_sequences[seq_idx]) >= self.config.eos_token_ids:
                    # if match eos patterns
                    if (token_sequences[seq_idx][-len(self.config.eos_token_ids):] == self.config.eos_token_ids):
                        sequence_indices.remove(seq_idx)
                        pop_flag = True
                    else:
                        nonpop_indices.append(i)
                else:
                    nonpop_indices.append(i)

            # keeping the selected indices
            if pop_flag:
                past = [item[:, nonpop_indices] for item in past]
                next_token = next_token[nonpop_indices]
                next_position_id = next_position_id[nonpop_indices]
                mask = mask[nonpop_indices]

            # if every sequence is done
            if len(sequence_indices) == 0:
                break

        # add eos_token_ids to unfinished sentences
        if cur_len == (self.config.max_length - 1):
            for seq_idx in sequence_indices:
                token_sequences[seq_idx].extend(self.config.eos_token_ids)
                log_prob_sequences[seq_idx].extend([0.0 for _ in range(len(self.config.eos_token_ids))])

        return token_sequences, log_prob_sequences

    def _assertion_check(self):
        assert (
            isinstance(self.config.max_length, int) and self.config.max_length > 0
        ), "`max_length` should be a strictely positive integer."
        assert isinstance(self.config.do_sample, bool), "`do_sample` should be a boolean."
        assert (
            isinstance(self.config.num_beams, int) and self.config.num_beams > 0
        ), "`num_beams` should be a strictely positive integer."
        assert (self.config.temperature > 0), "`temperature` should be strictely positive."
        assert (isinstance(self.config.top_k, int) and self.config.top_k >= 0), "`top_k` should be a positive integer."
        assert 0 <= self.config.top_p <= 1, "`top_p` should be between 0 and 1."
        assert (self.config.repetition_penalty >= 1.0), "`repetition_penalty` should be >= 1."
        assert (
            isinstance(self.config.bos_token_id, int) and self.config.bos_token_id >= 0
        ), "`bos_token_id` should be a positive integer."
        assert (
            isinstance(self.config.pad_token_id, int) and self.config.pad_token_id >= 0
        ), "`pad_token_id` should be a positive integer."
        assert isinstance(self.config.eos_token_ids, (list, tuple)) and (
            e >= 0 for e in self.config.eos_token_ids
        ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
        assert (self.config.length_penalty > 0), "`length_penalty` should be strictely positive."
        assert (
            isinstance(self.config.num_return_sequences, int) and self.config.num_return_sequences > 0
        ), "`num_return_sequences` should be a strictely positive integer."

    def _generate_beam_search(
        self,
        start_input_ids,
    ):
        """ Generate sequences for each example with beam search.
        """
        # Expand input to num beams
        next_position_id = 0

        # (batch_size * num_beams, cur_len)

        input_ids = start_input_ids.unsqueeze(1).expand(
            start_input_ids.shape[0], self.config.num_beams, start_input_ids.shape[1]
        )
        input_ids = input_ids.contiguous().view(
            start_input_ids.shape[0] * self.config.num_beams, start_input_ids.shape[1]
        )

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(
                self.config.num_beams, self.config.max_length, self.config.length_penalty, early_stopping=False
            ) for _ in range(self.config.batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros(
            (self.config.batch_size, self.config.num_beams), dtype=torch.float, device=start_input_ids.device
        )
        beam_scores[:, 1:] = -1e5
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = None

        # done sentences
        done = [False for _ in range(self.config.batch_size)]

        for cur_len in range(self.config.max_length):
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            scores = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if self.config.repetition_penalty != 1.0:
                for i in range(self.config.batch_size * self.config.num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= self.config.repetition_penalty
                        else:
                            scores[i, previous_token] /= self.config.repetition_penalty

            if self.config.do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample 2 next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1), num_samples=2)  # (batch_size * num_beams, 2)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, 2)
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, 2)
                # Match shape of greedy beam search
                next_words = next_words.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
                next_scores = next_scores.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
            else:
                # do greedy beam search
                scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
                next_scores, next_words = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_ex in range(batch_size):

                # if we are done with this sentence
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    if word_id.item() in eos_token_ids or cur_len + 1 == max_length:
                        generated_hyps[batch_ex].add(
                            input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_length else num_beams
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            # re-order internal states
            if past:
                reordered_past = []
                for layer_past in past:
                    # get the correct batch idx from layer past batch dim
                    # batch dim of `past` and `mems` is at 2nd position
                    reordered_layer_past = [layer_past[:, i].unsqueeze(1).clone().detach() for i in beam_idx]
                    reordered_layer_past = torch.cat(reordered_layer_past, dim=1)
                    # check that shape matches
                    assert reordered_layer_past.shape == layer_past.shape
                    reordered_past.append(reordered_layer_past)
                past = tuple(reordered_past)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(batch_size):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = input_ids.new(batch_size)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = input_ids.new(batch_size, tgt_len.max().item()).fill_(pad_token_id)
        for i, hypo in enumerate(best):
            decoded[i, :tgt_len[i] - 1] = hypo
            decoded[i, tgt_len[i] - 1] = eos_token_ids[0]

        return decoded


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
        score = sum_logprobs / len(hyp)**self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
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
            return (self.worst_score >= best_sum_logprobs / self.max_length**self.length_penalty)
