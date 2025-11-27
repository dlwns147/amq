import torch
import torch.nn as nn
# from qeft.utils.misc import parsing_layers, find_layers
from .misc import parsing_layers, find_layers
from tqdm import tqdm
    
def sparse_to_dense_ids(sparse_ids, length):
    assert len(sparse_ids) < length
    
    temp_mask = torch.full([length], True, device=sparse_ids.device)
    temp_mask[sparse_ids] = False
    dense_ids = torch.cat([torch.arange(length, device=sparse_ids.device)[temp_mask], sparse_ids])
    return dense_ids

def reorder_embeds(l_pres, l_posts, out_ids):
    hidden_size = l_pres[0].weight.data.shape[1]
    dst_ids = sparse_to_dense_ids(out_ids, hidden_size)
    
    for l in l_pres:
        l.weight.data = torch.index_select(l.weight.data, 1, dst_ids.to(l.weight.device))
    
    for l in l_posts:
        l.weight.data = torch.index_select(l.weight.data, -1, dst_ids.to(l.weight.device))
        if hasattr(l, 'bias') and l.bias is not None: # for layernorm
            l.bias.data = torch.index_select(l.bias.data, -1, dst_ids.to(l.weight.device))
    
def reorder_qkv_ffn1_ln(l_qkv_ffn1, l_ln, out_ids):
    in_ch = l_qkv_ffn1[0].weight.shape[-1]
    dst_ids = sparse_to_dense_ids(out_ids, in_ch)

    for l in l_qkv_ffn1:
        l.weight.data = torch.index_select(l.weight.data, 1, dst_ids)

    for l in l_ln:
        l.weight.data = torch.index_select(l.weight.data, 0, dst_ids)
        if hasattr(l, 'bias') and l.bias is not None:
            l.bias.data = torch.index_select(l.bias.data, 0, dst_ids)

def reorder_out(l_out, l_out_quantizers, out_ids):
    out_ch, in_ch = l_out[0].weight.shape

    if l_out_quantizers[0].out_ids.numel() > 0:
        dst_ids = sparse_to_dense_ids(l_out_quantizers[0].out_ids, in_ch)
        
        for l in l_out:
            l.weight.data = torch.index_select(l.weight.data, 1, dst_ids)
            l.reorder_ids = dst_ids
    
    dst_ids = sparse_to_dense_ids(out_ids, out_ch)
    
    for l in l_out:
        l.weight.data = torch.index_select(l.weight.data, 0, dst_ids)
        if l.bias is not None:
            l.bias.data = torch.index_select(l.bias.data, 0, dst_ids)
    
    for l_quantizer in l_out_quantizers:
        if hasattr(l_quantizer,'scale_group'):
            l_quantizer.scale_group = torch.index_select(l_quantizer.scale_group, 0, dst_ids)
        else:
            l_quantizer.scale = torch.index_select(l_quantizer.scale, 0, dst_ids)
        if hasattr(l_quantizer,'zero_group'):
            l_quantizer.zero_group = torch.index_select(l_quantizer.zero_group, 0, dst_ids)
        else:
            l_quantizer.zero = torch.index_select(l_quantizer.zero, 0, dst_ids)

def reorder_qkv_out_perhead(l_qkv, l_out, l_qkv_quantizers, l_out_quantizers, out_ids, head_dim):
    in_ch = l_out[0].weight.shape[-1]

    # qkv out channel, out_proj in channel perhead reorder
    out_ids_perhead = l_out_quantizers[0].out_ids.reshape(in_ch // head_dim, -1).clone() % head_dim
    dst_ids = torch.concat([sparse_to_dense_ids(out_ids_perhead[i], head_dim) + i * head_dim for i in range(out_ids_perhead.shape[0])])
    
    for l in l_out:
        l.weight.data = torch.index_select(l.weight.data, 1, dst_ids)
        
    for l in l_qkv:
        l.weight.data = torch.index_select(l.weight.data, 0, dst_ids)
        if l.bias is not None:
            l.bias.data = torch.index_select(l.bias.data, 0, dst_ids)
    
    for l_quantizer in l_qkv_quantizers:
        if hasattr(l_quantizer,'scale_group'):
            l_quantizer.scale_group = torch.index_select(l_quantizer.scale_group, 0, dst_ids)
        else:
            l_quantizer.scale = torch.index_select(l_quantizer.scale, 0, dst_ids)
        if hasattr(l_quantizer,'zero_group'):
            l_quantizer.zero_group = torch.index_select(l_quantizer.zero_group, 0, dst_ids)
        else:
            l_quantizer.zero = torch.index_select(l_quantizer.zero, 0, dst_ids)
    
    # global out ids
    dst_ids = sparse_to_dense_ids(out_ids, in_ch)
    
    for l in l_out:
        l.weight.data = torch.index_select(l.weight.data, 0, dst_ids)
        if l.bias is not None:
            l.bias.data = torch.index_select(l.bias.data, 0, dst_ids)
    
    for l_quantizer in l_out_quantizers:
        if hasattr(l_quantizer,'scale_group'):
            l_quantizer.scale_group = torch.index_select(l_quantizer.scale_group, 0, dst_ids)
        else:
            l_quantizer.scale = torch.index_select(l_quantizer.scale, 0, dst_ids)
        if hasattr(l_quantizer,'zero_group'):
            l_quantizer.zero_group = torch.index_select(l_quantizer.zero_group, 0, dst_ids)
        else:
            l_quantizer.zero = torch.index_select(l_quantizer.zero, 0, dst_ids)

def reorder_in_mlp(l_ffn1, l_ffn2, l_ffn1_quantizers, l_ffn2_quantizers):
    ffn2_in_ch = l_ffn2[0].weight.shape[-1]
    dst_ids = sparse_to_dense_ids(l_ffn2_quantizers[0].out_ids, ffn2_in_ch)
    
    for l in l_ffn1:
        l.weight.data = torch.index_select(l.weight.data, 0, dst_ids)
        if l.bias is not None:
            l.bias.data = torch.index_select(l.bias.data, 0, dst_ids)
    
    for l_quantizer in l_ffn1_quantizers:
        if hasattr(l_quantizer,'scale_group'):
            l_quantizer.scale_group = torch.index_select(l_quantizer.scale_group, 0, dst_ids)
        else:
            l_quantizer.scale = torch.index_select(l_quantizer.scale, 0, dst_ids)
        if hasattr(l_quantizer,'zero_group'):
            l_quantizer.zero_group = torch.index_select(l_quantizer.zero_group, 0, dst_ids)
        else:
            l_quantizer.zero = torch.index_select(l_quantizer.zero, 0, dst_ids)
        
    for l in l_ffn2:
        l.weight.data = torch.index_select(l.weight.data, 1, dst_ids)
        
    ffn2_out_ch = l_ffn2[0].weight.shape[0]
    dst_ids = sparse_to_dense_ids(l_ffn1_quantizers[0].out_ids, ffn2_out_ch)
    
    for l in l_ffn2:
        l.weight.data = torch.index_select(l.weight.data, 0, dst_ids)
        if l.bias is not None:
            l.bias.data = torch.index_select(l.bias.data, 0, dst_ids)
    
    for l_quantizer in l_ffn2_quantizers:
        if hasattr(l_quantizer,'scale_group'):
            l_quantizer.scale_group = torch.index_select(l_quantizer.scale_group, 0, dst_ids)
        else:
            l_quantizer.scale = torch.index_select(l_quantizer.scale, 0, dst_ids)
        if hasattr(l_quantizer,'zero_group'):
            l_quantizer.zero_group = torch.index_select(l_quantizer.zero_group, 0, dst_ids)
        else:
            l_quantizer.zero = torch.index_select(l_quantizer.zero, 0, dst_ids)

# def make_reorder(model, quantizers, global_ids, meta):
def make_reorder(model, quantizers, global_ids, meta):
    from transformers import OPTForCausalLM
    # global_ids = args.outidx
    layers, pre_layers, post_layers = parsing_layers(model, meta)
    additional_post = [model.lm_head] if not isinstance(model, OPTForCausalLM) else []
    
    reorder_embeds(pre_layers, post_layers + additional_post, global_ids)
    
    for i in tqdm(range(len(layers)), "Reordering Blocks..."):
        layer = layers[i]
        block_layers = find_layers(layer)
        ln_attention = getattr(layer, meta['ln_layers'][0])
        ln_mlp = getattr(layer, meta['ln_layers'][1])
        l_qkv = [block_layers.get(key) for key in meta['sequential'][0]]
        l_out = [block_layers.get(key) for key in meta['sequential'][1]]
        l_ffn1 = [block_layers.get(key) for key in meta['sequential'][2]]
        l_ffn2 = [block_layers.get(key) for key in meta['sequential'][3]]
        
        l_qkv_quantizers = [quantizers[f"{meta['prefix']}.{i}.{name}"] for name in meta['sequential'][0]]
        l_out_quantizers = [quantizers[f"{meta['prefix']}.{i}.{name}"] for name in meta['sequential'][1]]
        l_ffn1_quantizers = [quantizers[f"{meta['prefix']}.{i}.{name}"] for name in meta['sequential'][2]]
        l_ffn2_quantizers = [quantizers[f"{meta['prefix']}.{i}.{name}"] for name in meta['sequential'][3]]
        
        reorder_qkv_ffn1_ln(l_qkv_ffn1=l_qkv+l_ffn1, l_ln=[ln_attention, ln_mlp], out_ids=global_ids)
        reorder_out(l_out=l_out, l_out_quantizers=l_out_quantizers, out_ids=global_ids)
        reorder_in_mlp(l_ffn1=l_ffn1, l_ffn2=l_ffn2, 
                       l_ffn1_quantizers=l_ffn1_quantizers, 
                       l_ffn2_quantizers=l_ffn2_quantizers)