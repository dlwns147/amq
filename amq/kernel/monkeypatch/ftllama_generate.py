import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.cache_utils import Cache
from transformers.utils import ModelOutput, logging
from transformers.generation.candidate_generator import (
    CandidateGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)

from transformers.generation.utils import (
    _speculative_sampling,
    GenerationMixin,
    _split_model_outputs
)
import types

import time

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)
    
@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    start_pos: Optional[int] = None # for fastertransformer
    acceptance_rate: Optional[float] = None # for assisted_decoding
    accept_length: Optional[float] = None # for assisted_decoding

# Equivalent classes (kept for retrocompatibility purposes)
GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
ContrastiveSearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput
GenerateNonBeamOutput = GenerateDecoderOnlyOutput

from transformers.generation.candidate_generator import (
    AssistedCandidateGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids
)

def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
    """
    Fetches the candidates to be tried for the current input.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

    Return:
        `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
        assessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,
        vocabulary_size)` containing the logits associated to each candidate.
    """
    input_ids = input_ids.to(self.assistant_model.device)

    # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
    new_cur_len = input_ids.shape[-1]
    max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - new_cur_len - 1)
    min_new_tokens = max(min(max_new_tokens, self.main_model_min_length - new_cur_len), 0)
    if max_new_tokens == 0:
        return input_ids, None

    # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
    # (which implicitly contains the number of accepted candidates from the previous round)
        
    start_pos = self.assistant_kwargs.get("start_pos", None)
    if start_pos:
        new_cache_size = new_cur_len - 1
        self.assistant_kwargs["start_pos"] = start_pos if new_cur_len - start_pos > 0 else new_cache_size
        self.assistant_kwargs = _prepare_attention_mask(
            self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
        )
        self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)
    # 2. Forecast next N tokens using the assistant model.
    assistant_generation_kwargs = {
        self.input_ids_key: input_ids,
        "min_new_tokens": min_new_tokens,
        "max_new_tokens": max_new_tokens,
        "generation_config": self.generation_config,
        "logits_processor": self.logits_processor,
    }

    assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

    # 3. Update variables for the next round of candidate generation
    self.assistant_kwargs["start_pos"] = assistant_output.start_pos

    # 4. Prepare variables for output
    candidate_logits = torch.stack(assistant_output.scores, dim=1)
    candidate_ids = assistant_output.sequences
    return candidate_ids, candidate_logits

def _update_model_kwargs_for_generation(
    self,
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any],
    is_encoder_decoder: bool = False,
    standardize_cache_format: bool = False,
    num_new_tokens: int = 1,
) -> Dict[str, Any]:
    if getattr(outputs, "start_pos", None) is not None: # for fastertransformer
        model_kwargs["start_pos"] = outputs.start_pos
        
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
    else:
        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

    if (
        model_kwargs.get("use_cache", True)
        and "cache_position" in model_kwargs
        and model_kwargs["cache_position"] is not None
    ):
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

    return model_kwargs

def _get_initial_cache_position(self, input_ids, model_kwargs):
    """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
    if not model_kwargs.get("use_cache", True):
        model_kwargs["cache_position"] = None
        return model_kwargs

    past_length = 0
    if "start_pos" in model_kwargs and model_kwargs["start_pos"] is not None:
        past_length = model_kwargs["start_pos"]
    if "inputs_embeds" in model_kwargs:
        cur_len = model_kwargs["inputs_embeds"].shape[1]
    else:
        cur_len = input_ids.shape[-1]
    model_kwargs["cache_position"] = torch.arange(past_length, cur_len, device=input_ids.device)
    return model_kwargs

def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    logits_warper: Optional[LogitsProcessorList] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
            `generation_config`)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    # pad_token_id = generation_config.pad_token_id.to(input_ids.device) # modify
    
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
        raise ValueError(
            "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
            f"{logits_warper})."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    past_key_values = model_kwargs.pop("past_key_values")
    
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        if do_sample:
            next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        return GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
            start_pos=model_kwargs.get("start_pos"), # for fastertransformer
        )
    else:
        return input_ids

def _assisted_decoding(
    self,
    input_ids: torch.LongTensor,
    candidate_generator: CandidateGenerator,
    logits_processor: LogitsProcessorList,
    logits_warper: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
    **sample** (depending on `do_sample`), assisted by candidate sequences. Assisted generation is an example of a
    candidate decoding strategy. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
    models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        candidate_generator (`CandidateGenerator`):
            A derived instance of [`CandidateGenerator`] that defines how candidate sequences are generated. For
            more information, the documentation of [`CandidateGenerator`] should be read.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        logits_warper (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step. Only used if sampling is active.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    do_sample = logits_warper is not None
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    input_ids = input_ids.to(self.device)
    batch_size = input_ids.shape[0]
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    past_key_values = model_kwargs.pop("past_key_values")

    this_peer_finished = False
    acceptance_rate_list = []
    accept_length_list = []
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        cur_len = input_ids.shape[-1]
        #  1. Fetch candidate sequences from a `CandidateGenerator`
        candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids)

        candidate_input_ids = candidate_input_ids.to(self.device)
        if candidate_logits is not None:
            candidate_logits = candidate_logits.to(self.device)

        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
        is_done_candidate = stopping_criteria(candidate_input_ids, None)

        # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
        # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
        # we use this forward pass to also pick the subsequent logits in the original model.

        # 2.1. Prepare the model inputs
        candidate_kwargs = copy.copy(model_kwargs)
        candidate_kwargs = _prepare_attention_mask(
            candidate_kwargs, candidate_input_ids.shape[1], self.config.is_encoder_decoder
        )
        candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
        if "cache_position" in candidate_kwargs:
            candidate_kwargs["cache_position"] = torch.cat(
                (
                    candidate_kwargs["cache_position"],
                    torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long),
                ),
                dim=0,
            )

        model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
        if "num_logits_to_keep" in model_inputs:
            model_inputs["num_logits_to_keep"] = candidate_length + 1

        # 2.2. Run a forward pass on the candidate sequence
        outputs = self(
            **model_inputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # 2.3. Process the new logits
        new_logits = outputs.logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
        next_token_logits = new_logits.clone()
        if len(logits_processor) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])
        if do_sample and len(logits_warper) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_warper(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

        # 3. Select the accepted tokens. There are two possible cases:
        # Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
        # 👉 Apply algorithm 1 from the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf).

        if do_sample and candidate_logits is not None:
            valid_tokens, n_matches = _speculative_sampling(
                candidate_input_ids,
                candidate_logits,
                candidate_length,
                new_logits,
                is_done_candidate,
            )

        # Case 2: all other cases (originally from assisted generation) 👉 Compare the tokens selected from the
        # original model logits with the candidate tokens. We can keep the candidate tokens until the first
        # mismatch, or until the max length is reached.
        else:
            if do_sample:
                probs = new_logits.softmax(dim=-1)
                selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
            else:
                selected_tokens = new_logits.argmax(dim=-1)

            candidate_new_tokens = candidate_input_ids[:, cur_len:]
            n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

            # Ensure we don't generate beyond max_len or an EOS token
            if is_done_candidate and n_matches == candidate_length:
                n_matches -= 1
            valid_tokens = selected_tokens[:, : n_matches + 1]

        # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
        # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
        # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
        # is no match.
        
        # for calculating acceptance ratio
        accept_length_list.append(n_matches.item() + 1)
        acceptance_rate_list.append((n_matches.item() + 1) / candidate_generator.num_assistant_tokens)
        
        # 4.1. Get the valid continuation, after the matching tokens
        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        if streamer is not None:
            streamer.put(valid_tokens.cpu())
        new_cur_len = input_ids.shape[-1]

        # 4.2. Discard past key values relative to unused assistant tokens
        new_cache_size = new_cur_len - 1
        outputs.start_pos = new_cache_size
        
        # 5. Update the candidate generation strategy if needed
        candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        # Store scores, attentions and hidden_states when required
        # Assistant: modified to append one tuple element per token, as in the other generation methods.
        if return_dict_in_generate:
            if output_scores:
                scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1))
            if output_logits:
                raw_logits += (next_token_logits,)

            if "start_pos" not in model_kwargs:
                added_len = new_cur_len
            else:
                added_len = n_matches + 1

            if output_attentions:
                if self.config.is_encoder_decoder:
                    cross_attentions = _split_model_outputs(
                        cross_attentions, outputs.cross_attentions, cur_len, added_len
                    )
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.decoder_attentions,
                        cur_len,
                        added_len,
                        is_decoder_attention=True,
                    )
                else:
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.attentions,
                        cur_len,
                        added_len,
                        is_decoder_attention=True,
                    )
            if output_hidden_states:
                if self.config.is_encoder_decoder:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states, outputs.decoder_hidden_states, cur_len, added_len
                    )
                else:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states, outputs.hidden_states, cur_len, added_len
                    )

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
            num_new_tokens=n_matches + 1,
        )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        
    if streamer is not None:
        streamer.end()

    if (
        hasattr(candidate_generator, "assistant_model")
        and candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic"
    ):
        candidate_generator.assistant_model.generation_config.num_assistant_tokens = (
            candidate_generator.num_assistant_tokens
        )
    if return_dict_in_generate:
        return GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
            start_pos=model_kwargs.get("start_pos"),
            acceptance_rate=sum(acceptance_rate_list) / len(acceptance_rate_list),
            accept_length=sum(accept_length_list) / len(accept_length_list),
        )
    else:
        return input_ids
    
def replace_generate_functions():
    # GenerationMixin._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation, GenerationMixin)
    # GenerationMixin._get_initial_cache_position = types.MethodType(_get_initial_cache_position, GenerationMixin)
    # GenerationMixin._sample = types.MethodType(_sample, GenerationMixin)
    # GenerationMixin._assisted_decoding = types.MethodType(_assisted_decoding, GenerationMixin)
    # AssistedCandidateGenerator.get_candidates = types.MethodType(get_candidates, AssistedCandidateGenerator)
    GenerationMixin._update_model_kwargs_for_generation = _update_model_kwargs_for_generation
    GenerationMixin._get_initial_cache_position = _get_initial_cache_position
    GenerationMixin._sample = _sample
    GenerationMixin._assisted_decoding = _assisted_decoding
    AssistedCandidateGenerator.get_candidates = get_candidates