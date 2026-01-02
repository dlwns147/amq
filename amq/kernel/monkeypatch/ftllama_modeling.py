import types
from weakref import proxy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import math
from transformers.cache_utils import Cache, StaticCache, DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaAttention, 
    LlamaDecoderLayer, 
    LlamaModel,
    LlamaForCausalLM, 
    apply_rotary_pos_emb
)
import faster_transformer
from transformers.utils import ModelOutput

@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    start_pos: Optional[int] = None # for fastertransformer

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    start_pos: Optional[int] = None # for fastertransformer
    
def replace_layernorm_forward_ft(self):
    def forward(self, x):
        output = torch.empty_like(x)
        faster_transformer.layernorm_forward_cuda(x, self.weight, output, self.variance_epsilon)

        return output
    
    self.forward = types.MethodType(forward, self)
        
def replace_attn_ft(self):
    class LlamaAttentionFT(LlamaAttention):
        def __init__(self, module):
            self.__dict__ = module.__dict__
            if hasattr(self.q_proj, 'weight'):
                dtype = self.q_proj.weight.dtype
                device = self.q_proj.weight.device
            elif hasattr(self.q_proj, 'scales'):
                dtype = self.q_proj.scales.dtype
                device = self.q_proj.scales.device
            else:
                raise ValueError
            assert dtype == torch.half, "dtype have to be torch.half to use attention kernel of FasterTransformer"
            self.register_buffer("value_cache",
                torch.zeros((1, self.num_key_value_heads, self.max_position_embeddings, self.head_dim), dtype=dtype, device=device),
                persistent=False
            )
            self.register_buffer("key_cache",
                torch.zeros((1, self.num_key_value_heads, self.head_dim // 8, self.max_position_embeddings, 8), dtype=dtype, device=device),
                persistent=False
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            start_pos: Optional[int] = None, # for fastertransformer
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            bsz, q_len, _ = hidden_states.size()
            
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            
            if q_len > 8:
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                cos, sin = self.rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                self.value_cache[:bsz, :, start_pos : start_pos + q_len, :] = value_states
                self.key_cache[:bsz, :, :, start_pos : start_pos + q_len, :] = key_states.reshape(bsz, self.num_key_value_heads, q_len, self.head_dim // 8, 8).permute(0, 1, 3, 2, 4).contiguous()
                
                if start_pos > 0:
                    value_states = self.value_cache[:bsz, :, : start_pos + q_len, :] 
                    key_states = self.key_cache[:bsz, :, :, : start_pos + q_len, :].permute(0,1,3,2,4).reshape(bsz, self.num_key_value_heads, start_pos + q_len, self.head_dim)
                
                value_states = torch.repeat_interleave(
                    value_states, dim=1, repeats=self.num_key_value_groups
                )
                key_states = torch.repeat_interleave(
                    key_states, dim=1, repeats=self.num_key_value_groups
                )

                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attention_mask is not None:  # no matter the length, we just slice it
                    causal_mask = attention_mask[:, :, :, -key_states.shape[-2] : ]
                    attn_weights = attn_weights + causal_mask

                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)

                if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                    raise ValueError(
                        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                        f" {attn_output.size()}"
                    )

                attn_output = attn_output.transpose(1, 2).contiguous()

                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            else:
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
                
                attn_output = []
                for i in range(q_len):
                    attn_output.append(
                        faster_transformer.single_query_attention(
                            query_states[:,i,...],
                            key_states[:,i,...],
                            value_states[:,i,...],
                            self.key_cache,
                            self.value_cache,
                            None,
                            # alibi position encodings
                            None,
                            start_pos + i,
                            self.head_dim,
                            self.rotary_emb.config.rope_theta,      # 10000.0
                            True,
                        )
                    )

                if q_len > 1:
                    attn_output = torch.stack(attn_output, dim=1)
                else:
                    attn_output = attn_output[0]
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            
            attn_output = self.o_proj(attn_output)
                
            if not output_attentions:
                attn_weights = None
            
            return attn_output, attn_weights, past_key_value

    return LlamaAttentionFT(self)

def replace_decoder_layer_forward(self):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        start_pos: Optional[int] = None, # for fastertransformer
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            start_pos=start_pos,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    self.forward = types.MethodType(forward, self)

def replace_llamamodel_forward(self):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        start_pos: Optional[int] = None, # for fastertransformer
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        
        if self.config._attn_implementation == 'ft':
            use_cache = False
            if start_pos is None:
                start_pos = 0
                
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = start_pos if self.config._attn_implementation == 'ft' else (past_key_values.get_seq_length() if past_key_values is not None else 0)
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, start_pos, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                start_pos=start_pos,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

            # if not getattr(decoder_layer, "skip", False) and output_hidden_states:
            #     all_hidden_states += (hidden_states,)
                
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
            
        if start_pos is not None:
            start_pos += hidden_states.shape[-2] # b, -s- ,d

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            start_pos=start_pos,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        start_pos: int,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = start_pos if self.config._attn_implementation == 'ft' else (past_key_values.get_seq_length() if past_key_values is not None else 0)
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
    self._update_causal_mask = types.MethodType(_update_causal_mask, self)
    self.forward = types.MethodType(forward, self)

def replace_llamaforcausallm_forward(self):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        start_pos: Optional[int] = None, # for fastertransformer
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            start_pos=start_pos,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            start_pos=outputs.start_pos,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        start_pos=None, # for fastertransformer
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)
        if cache_position is not None:
            cache_position = cache_position.to(input_ids.device)

        if start_pos is not None:
            cache_length = past_length = start_pos
            max_cache_length = None
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if use_cache:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]
        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "start_pos": start_pos,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    self.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, self)
    self.forward = types.MethodType(forward, self)

def convert_model_to_ft(model):
    model.config._attn_implementation = 'ft'
    for module in model.modules():
        if isinstance(module, LlamaRMSNorm):
            replace_layernorm_forward_ft(module)
        if isinstance(module, LlamaDecoderLayer):
            replace_decoder_layer_forward(module)
            module.self_attn = replace_attn_ft(module.self_attn)
        if isinstance(module, LlamaModel):
            replace_llamamodel_forward(module)
        if isinstance(module, LlamaForCausalLM):
            replace_llamaforcausallm_forward(module)
