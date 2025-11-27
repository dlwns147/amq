import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# class LlamaDecoderSkipLayer(LlamaDecoderLayer):
class LlamaDecoderSkipLayer(nn.Module):
    
    def __init__(self, original_decoder_layer):
        super().__init__()
        self.hidden_size = original_decoder_layer.hidden_size

        self.self_attn = original_decoder_layer.self_attn
        self.mlp = original_decoder_layer.mlp
        self.input_layernorm = original_decoder_layer.input_layernorm
        self.post_attention_layernorm = original_decoder_layer.post_attention_layernorm

        self.attn_skipped = False
        self.mlp_skipped = False

    def skip_attn(self, reuse=True):
        self.attn_skipped = True
        if not reuse:
            self.layer_idx = self.self_attn.layer_idx
            del self.self_attn, self.input_layernorm
    
    def use_attn(self):
        self.attn_skipped = False

    def skip_mlp(self, reuse=True):
        self.mlp_skipped = True
        if not reuse:
            del self.mlp, self.post_attention_layernorm
    
    def use_mlp(self):
        self.mlp_skipped = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
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
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        self_attn_weights = None
        present_key_value = past_key_value

        # device = next(self.parameters()).device
        # if hidden_states.device != device:
        #     hidden_states = hidden_states.to(device)

        residual = hidden_states
        

        if not self.attn_skipped:
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                # position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states
        else:
            if past_key_value is not None:
                q_len = hidden_states.size(1)
                # key_states = torch.empty(1, 1, q_len, 1, dtype=torch.bool)
                # value_states = torch.empty(1, 1, q_len, 1, dtype=torch.bool)
                key_states = value_states = torch.empty(1, 1, q_len, 1, dtype=torch.bool)
                cache_kwargs = {"cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.self_attn.layer_idx if hasattr(self, 'self_attn') else self.layer_idx, cache_kwargs)

        if not self.mlp_skipped:
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

def block_replace(model):
    # import pdb; pdb.set_trace()
    for i in range(len(model.model.layers)):
        model.model.layers[i] = LlamaDecoderSkipLayer(model.model.layers[i])
    
    # for i in range(len(model.model.layers)):
    #     model.model.layers[i].__class__ = LlamaDecoderSkipLayer
    #     model.model.layers[i].attn_skipped = False
    #     model.model.layers[i].mlp_skipped = False
    print("Replacement complete.")

    return model

def skip_attn(model, block_idx, reuse=True):
    model.model.layers[block_idx].skip_attn(reuse=reuse)
    
def skip_mlp(model, block_idx, reuse=True):
    model.model.layers[block_idx].skip_mlp(reuse=reuse)

def use_attn(model, block_idx):
    model.model.layers[block_idx].use_attn()

def use_mlp(model, block_idx):
    model.model.layers[block_idx].use_mlp()

def scan(model):

    attn_alive_list = []
    attn_skip_list = []
    mlp_alive_list = []
    mlp_skip_list = []

    for i, layer in enumerate(model.model.layers):
        if layer.attn_skipped == True:
            attn_skip_list.append(i)
        else :
            attn_alive_list.append(i)
            
        if layer.mlp_skipped == True:
            mlp_skip_list.append(i)
        else :
            mlp_alive_list.append(i)
            []
    print(
        f"attn alive layer: {attn_alive_list}\n"
        f"attn skip layer: {attn_skip_list}\n"
        f"mlp alive layer: {mlp_alive_list}\n"
        f"mlp skip layer: {mlp_skip_list}"
        )
    

    
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_value: Optional[Tuple[torch.Tensor]] = None,
    #     output_attentions: Optional[bool] = False,
    #     use_cache: Optional[bool] = False,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **kwargs,
    # ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    #     """
    #     Args:
    #         hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
    #         attention_mask (`torch.FloatTensor`, *optional*):
    #             attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
    #             query_sequence_length, key_sequence_length)` if default attention is used.
    #         output_attentions (`bool`, *optional*):
    #             Whether or not to return the attentions tensors of all attention layers. See `attentions` under
    #             returned tensors for more detail.
    #         use_cache (`bool`, *optional*):
    #             If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
    #             (see `past_key_values`).
    #         past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    #     """

    #     self_attn_weights = None
    #     present_key_value = past_key_value

    #     if not self.attn_skipped:
    #         residual = hidden_states
    #         hidden_states = self.input_layernorm(hidden_states)

    #         # Self Attention
    #         hidden_states, self_attn_weights, present_key_value = self.self_attn(
    #             hidden_states=hidden_states,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_value=past_key_value,
    #             output_attentions=output_attentions,
    #             use_cache=use_cache,
    #             cache_position=cache_position,
    #             **kwargs,
    #         )

    #         if residual.device != hidden_states.device:
    #             residual = residual.to(hidden_states.device)
    #         hidden_states = residual + hidden_states

    #     else:
    #         if use_cache: # Create dummy cache when attn is skipped.
    #             q_len = hidden_states.size(1)
    #             key_states = torch.empty(1, 1, q_len, 1, device=hidden_states.device, dtype=hidden_states.dtype)
    #             value_states = torch.empty(1, 1, q_len, 1, device=hidden_states.device, dtype=hidden_states.dtype)
    #             # key_states = value_states = hidden_states[0, :, 0].reshape(1, 1, -1, 1)
    #             cache_kwargs = {"cache_position": cache_position}
    #             present_key_value.update(key_states, value_states, (self.self_attn.layer_idx if hasattr(self, 'self_attn') else self.layer_idx), cache_kwargs=cache_kwargs)

    #     if not self.mlp_skipped:
    #         residual = hidden_states
    #         hidden_states = self.post_attention_layernorm(hidden_states)
    #         hidden_states = self.mlp(hidden_states)

    #         if residual.device != hidden_states.device:
    #             residual = residual.to(hidden_states.device)

    #         hidden_states = residual + hidden_states

    #     outputs = (hidden_states,)

    #     if output_attentions:
    #         outputs += (self_attn_weights,)

    #     if use_cache:
    #         outputs += (present_key_value,)

    #     return outputs