import torch
import torch.nn as nn
from tqdm import tqdm
import gc
import functools
from collections import defaultdict
from typing import List

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
# from tinychat.models import LlavaLlamaForCausalLM

# from .auto_scale import auto_scale_block, apply_scale
# from .auto_clip import auto_clip_block, apply_clip
from .auto_scale import *
from .auto_clip import *
from .module import append_str_prefix, get_op_name, is_owq
from amq.quantization.base import get_awq_calib_dataset

# __all__ = ["run_awq"]


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ in ["LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM"]:
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM)):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.to(device)
    # elif isinstance(model, LlavaLlamaForCausalLM):
    #     model.model.embed_tokens = model.model.embed_tokens.to(device)
    #     model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    else:
        raise NotImplementedError(type(model))


@torch.no_grad()
def run_awq(
    model,
    enc,
    arch,
    q_config,
    n_samples=512,
    seqlen=512,
    clip_asym=True,
    calib_data="pileval",
    do_owq=False,
    outlier=None
):

    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")


    assert arch is not None, "arch is not provided"

    # if samples is None:
        # samples = self.get_awq_calib_dataset(n_samples=n_samples, block_size=seqlen)
    samples = get_awq_calib_dataset(data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    samples = torch.cat(samples, dim=0)

    layers = get_blocks(model)
    move_embed(model, "cuda")

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].to("cuda")
    

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # import pdb; pdb.set_trace()
    # solve layer by layer
    for i in tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        if sum(1 for _ in layer.parameters()):
            inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # import pdb; pdb.set_trace()
        outlier_linear = {linear.replace(get_op_name(model, layer) + ".", ""): outlier[linear] for linear in outlier.keys() if get_op_name(model, layer) in linear} if do_owq else dict()
        scales_list = auto_scale_block(
            layer,
            layer_kwargs,
            input_feat=input_feat,
            module_bit={proj : int(arch['linear'][proj][i]) for proj in named_linears},
            q_config=q_config,
            do_owq=do_owq,
            outlier=outlier_linear,
        )
        apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
        awq_results["scale"] += append_str_prefix(
            scales_list, get_op_name(model, layer) + "."
        )

        # Clear GPU memory
        torch.cuda.empty_cache()

        if clip_asym:
            clip_list = auto_clip_block_asym(
                layer,
                input_feat=input_feat,
                module_bit={proj : int(arch['linear'][proj][i]) for proj in named_linears},
                q_config=q_config,
                outlier=outlier_linear,
            )
            apply_clip_asym(layer, clip_list, do_owq=do_owq, outlier=outlier_linear)
        else:
            clip_list = auto_clip_block_sym(
                layer,
                input_feat=input_feat,
                module_bit={proj : int(arch['linear'][proj][i]) for proj in named_linears},
                q_config=q_config,
            )
            apply_clip_sym(layer, clip_list)
        clip_list = append_str_prefix(
            clip_list, get_op_name(model, layer) + "."
        )
        awq_results["clip"] += clip_list

        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()
    
        layer = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results

@torch.no_grad()
def apply_awq(model, awq_results, q_config, arch, clip_asym, do_owq, outlier):
    
    apply_scale(model, awq_results["scale"])        
    if clip_asym:
        apply_clip_asym(model, awq_results["clip"], do_owq=do_owq, outlier=outlier)
    else:
        apply_clip_sym(model, awq_results["clip"])

    layers = get_blocks(model)
    # for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
    # import pdb; pdb.set_trace()
    for i, layer in tqdm(enumerate(layers), desc="pseudo weight quantization..."):
        named_linears = {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}
        for n, m in named_linears.items():
            key = append_str_prefix(n, get_op_name(model, layer) + ".")
            if do_owq and is_owq(arch['linear'][n][i]) and key in outlier:
                # assert outlier is not None, "outlier is not provided"
                # if key in outlier:
                orig = m.weight.data[:, outlier[key]].detach().clone()
                m.weight.data[:, outlier[key]] = 0

            target_path = f'model.layers.{i}.{n}'
            # if  target_path in ["model.layers.15.self_attn.v_proj", "model.layers.23.mlp.down_proj"]:
            target_path_list = ["model.layers.7.self_attn.q_proj", 
                                "model.layers.25.self_attn.k_proj", 
                                "model.layers.15.self_attn.v_proj", 
                                "model.layers.30.self_attn.o_proj", 
                                "model.layers.12.mlp.down_proj",
                                "model.layers.22.mlp.gate_proj",
                                "model.layers.6.mlp.up_proj"]
            if  target_path in target_path_list:
                weight, scale, zero_point = pseudo_quantize_tensor(
                    m.weight.data, n_bit=int(arch['linear'][n][i]), get_scale_zp=True,
                    **q_config
                )
                save_list = {'weight': weight.detach().cpu(), 'scale': scale.detach().cpu(), 'zero_point':zero_point.detach().cpu()}
                
                import os
                save_path = os.path.join('/NAS/SJ/nsgaquant/save', '_'.join(target_path.split('.')) + f'_{int(arch["linear"][n][i])}bit_weights.pth')
                torch.save(save_list, save_path)
            
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=int(arch['linear'][n][i]),
                **q_config
            )
            

            if do_owq and is_owq(arch['linear'][n][i]) and key in outlier:
                m.weight.data[:, outlier[key]] = orig
                del orig
                torch.cuda.empty_cache()
                gc.collect()
