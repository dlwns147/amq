import torch
import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation

from .base import BASE, get_awq_calib_dataset
from model.skip_llama import LlamaDecoderSkipLayer
from accelerate import dispatch_model

import gc
from tqdm import tqdm
import math
import functools
from copy import deepcopy

from collections import defaultdict

@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


def pseudo_quantize_tensor(w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False):
    assert n_bit == int(n_bit), "n_bit should be integer"
    assert q_group_size != 0, "q_group_size should not be 0"

    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    elif q_group_size == -1:
        w = w.reshape(-1, w.shape[-1])
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

# def get_op_name(module, op):
#     # get the name of the op relative to the module
#     for name, m in module.named_modules():
#         if m is op:
#             return name
#     raise ValueError(f"Cannot find op {op} in module {module}")


# def append_str_prefix(x, prefix):
#     if isinstance(x, str):
#         return prefix + x
#     elif isinstance(x, tuple):
#         return tuple([append_str_prefix(y, prefix) for y in x])
#     elif isinstance(x, list):
#         return [append_str_prefix(y, prefix) for y in x]
#     else:
#         return x


class AWQ(BASE):
    def __init__(self, model, tokenizer, method, arch, avg_bits, group_size=128, config=None, dev='cuda', **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, method=method, arch=arch, avg_bits=avg_bits, group_size=group_size, config=config, dev=dev)
        self.method = 'awq'

        self.clip_asym = kwargs.get('clip_asym', True)

    @torch.no_grad()
    def run_awq(
        self,
        samples=None,
        n_samples=512,
        seqlen=512,
    ):
        assert self.arch is not None, "arch is not provided"

        if samples is None:
            samples = get_awq_calib_dataset(tokenizer=self.tokenizer, n_samples=n_samples, block_size=seqlen)
            samples = torch.cat(samples, dim=0)

        layers = self.model.model.layers

        inps = []
        layer_kwargs = {}

        layers[0] = layers[0].to(self.dev)
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.dev)
        self.model.model.norm = self.model.model.norm.to(self.dev)
        self.model.model.rotary_emb = self.model.model.rotary_emb.to(self.dev)

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
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        del samples
        layers[0] = layers[0].module  # restore
        inps = inps[0]

        layers[0] = layers[0].cpu()
        self.model.model.embed_tokens = self.model.model.embed_tokens.to('cpu')
        self.model.model.norm = self.model.model.norm.to('cpu')
        self.model.model.rotary_emb = self.model.model.rotary_emb.to('cpu')

        gc.collect()
        torch.cuda.empty_cache()

        awq_results = {
            "scale": [],
            "clip": [],
        }

        # solve layer by layer
        for i in tqdm(range(len(layers)), desc="Running AWQ..."):
            layer = layers[i]
            layer = layer.to(self.dev)
            named_linears = self.get_named_linears(layer)

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

            scales_list = self.auto_scale_block_bit_adjust_per_linear_owq(
                layer,
                layer_kwargs,
                input_feat=input_feat,
                module_bit={proj : self.arch['linear'][proj][i] for proj in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']},
            )
            self.apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            awq_results["scale"] += self.append_str_prefix(
                scales_list, self.get_op_name(self.model, layer) + "."
            )

            # Clear GPU memory
            torch.cuda.empty_cache()

            if self.clip_asym:
                clip_list = self.auto_clip_block_asym(
                    layer,
                    input_feat=input_feat,
                    module_bit={proj : int(self.arch['linear'][proj][i]) for proj in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']},
                    # owq_layer=owq_layer,
                )
                self.apply_clip_asym(layer, clip_list)
            else:
                clip_list = self.auto_clip_block_sym(
                    layer,
                    input_feat=input_feat,
                    module_bit = {proj : int(self.arch['linear'][proj][i]) for proj in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']},
                )
                self.apply_clip_sym(layer, clip_list)
            clip_list = self.append_str_prefix(
                clip_list, self.get_op_name(self.model, layer) + "."
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
    def auto_scale_block_bit_adjust_per_linear(self, module, module_kwargs, input_feat, module_bit=None):

        def w_quantize_func(p, bit=None):
            assert not self.is_owq(bit), "bit should be integer"
            return pseudo_quantize_tensor(
                p,
                n_bit=bit,
                q_group_size=self.group_size,
            ).detach()

        if "use_cache" in module_kwargs:
            module_kwargs.pop("use_cache")

        def _search_module_scale_per_linear(block, linears2scale: dict, x, kwargs={}, module_bit=None):
            # w: co, ci
            # x: n, ci
            assert module_bit is not None
            assert isinstance(linears2scale, dict)

            x = x.to(next(block.parameters()).device)
            with torch.no_grad():
                org_out = block(x, **kwargs)
                if isinstance(org_out, tuple):
                    org_out = org_out[0]

            x_max = get_act_scale(x)

            best_error = float("inf")
            best_ratio = -1
            best_scales = None

            n_grid = 20
            history = []

            org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
            for ratio in range(n_grid):
                ratio = ratio * 1 / n_grid
                scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                for fc_name, fc in linears2scale.items():
                    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                    fc.weight.data = w_quantize_func(fc.weight.data, bit=int(module_bit[fc_name])) / (scales.view(1, -1))

                out = block(x, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]

                loss = (
                    (org_out - out).float().pow(2).mean().item()
                )  # float prevents overflow
                history.append(loss)
                is_best = loss < best_error
                if is_best:
                    best_error = loss
                    best_ratio = ratio
                    best_scales = scales
                block.load_state_dict(org_sd)
            if best_ratio == -1:
                print(history)
                raise Exception
            # print(best_ratio)
            best_scales = best_scales.view(-1)

            if self.do_owq:
                # for fc_name, fc in linears2scale.items():
                #     if fc_name != 'self_attn.o_proj':
                #         del original[fc_name]
                del original
                torch.cuda.empty_cache()
                gc.collect()

            assert torch.isnan(best_scales).sum() == 0, best_scales
            return best_scales.detach()


        def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}, module_bit=None, owq_layer=None):
            # module2inspect: if given, we will check the output diff of this module instead of layers
            if module2inspect is None:
                assert len(layers) == 1
                module2inspect = list(layers.values())[0]

            scales = _search_module_scale_per_linear(module2inspect, layers, inp, kwargs, module_bit=module_bit, owq_layer=owq_layer)
            scales = scales.detach().cpu()
            # prev_op_name, [layer_name], scale
            return (
                self.get_op_name(module, prev_op),
                tuple([self.get_op_name(module, m) for m in layers.values()]),
                scales,
            )

        scales_list = []  # return the searched scales

        if isinstance(module, LlamaDecoderLayer) or isinstance(module, LlamaDecoderSkipLayer):
            if isinstance(module, LlamaDecoderLayer) or module.attn_skipped is False:
                # attention input
                scales_list.append(
                    _auto_get_scale(
                        prev_op=module.input_layernorm, 
                        # layers=[
                        #     module.self_attn.q_proj,
                        #     module.self_attn.k_proj,
                        #     module.self_attn.v_proj,
                        # ],
                        layers={
                            'self_attn.q_proj': module.self_attn.q_proj,
                            'self_attn.k_proj': module.self_attn.k_proj,
                            'self_attn.v_proj': module.self_attn.v_proj,
                        },
                        inp=input_feat["self_attn.q_proj"],
                        module2inspect=module.self_attn,
                        kwargs=module_kwargs,
                        module_bit=module_bit,
                        owq_layer=owq_layer,
                    )
                )
                # attn out
                # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                    scales_list.append(
                        _auto_get_scale(
                            prev_op=module.self_attn.v_proj,
                            # layers=[module.self_attn.o_proj],
                            layers={
                                'self_attn.o_proj': module.self_attn.o_proj,
                            },
                            inp=input_feat["self_attn.o_proj"],
                            module_bit=module_bit,
                            owq_layer=owq_layer,
                        )
                    )
            if isinstance(module, LlamaDecoderLayer) or module.mlp_skipped is False:
                # fc1
                scales_list.append(
                    _auto_get_scale(
                        prev_op=module.post_attention_layernorm,
                        # layers=[module.mlp.gate_proj, module.mlp.up_proj],
                        layers={
                            'mlp.gate_proj': module.mlp.gate_proj,
                            'mlp.up_proj': module.mlp.up_proj,
                        },
                        inp=input_feat["mlp.gate_proj"],
                        module2inspect=module.mlp,
                        module_bit=module_bit,
                        owq_layer=owq_layer,
                    )
                )
                # fc2
                scales_list.append(
                    _auto_get_scale(
                        prev_op=module.mlp.up_proj,
                        # layers=[module.mlp.down_proj],
                        layers={
                            'mlp.down_proj': module.mlp.down_proj,
                        },
                        inp=input_feat["mlp.down_proj"],
                        module_bit=module_bit,
                        owq_layer=owq_layer,
                    )
                )
        else:
            raise NotImplementedError(f"{type(module)} not supported yet!")

        return scales_list
    

    def apply_scale(self, module, scales_list, input_feat_dict=None):
        for prev_op_name, layer_names, scales in scales_list:
            prev_op = self.get_op_by_name(module, prev_op_name)
            layers = [self.get_op_by_name(module, name) for name in layer_names]

            # prev_op.to(self.dev)
            # for layer in layers:
            #     layer.to(self.dev)
            # scales.to(self.dev)

            if isinstance(prev_op, nn.Linear):
                assert len(layers) == 1
                scale_fc_fc(prev_op, layers[0], scales)
            elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)):
                scale_ln_fcs(prev_op, layers, scales)
            elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
                new_module = self.ScaledActivation(prev_op, scales)
                self.set_op_by_name(module, prev_op_name, new_module)
                scale_gelu_fc(prev_op, layers[0], scales)
            else:
                raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

            # apply the scaling to input feat if given; prepare it for clipping
            if input_feat_dict is not None:
                for layer_name in layer_names:
                    inp = input_feat_dict[layer_name]
                    inp.div_(scales.view(1, -1).to(inp.device))

            # prev_op.cpu()
            # for layer in layers:
            #     layer.cpu()
            # scales.cpu()


    @torch.no_grad()
    def auto_clip_block_asym(self, module, input_feat, module_bit=None, owq_layer=None):
        named_linears = {
            name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
        }
        clip_list = []
        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
                continue      
            named_linears[name].to(self.dev)
            q_config = {}

            q_config['q_group_size'] = self.group_size
            max_val, min_val = self.auto_clip_layer_asym(
                named_linears[name].weight, input_feat[name], n_bit=module_bit[name], q_config=q_config,
                ## customizing
                owq_column = owq_layer[name] if owq_layer is not None and name in owq_layer else None
            )
            clip_list.append((name, max_val, min_val))
            
            named_linears[name].cpu()
        return clip_list


    @torch.no_grad()
    def auto_clip_layer_asym(
        self, w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512, owq_column=None
    ):
        assert n_bit == int(n_bit), "bit should be integer"
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = (
            q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
        )

        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
        input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
        input_feat = input_feat.to(w.device)
        w = w.reshape(w.shape[0], 1, -1, group_size)

        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
        assert w.shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []
        best_min_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group
            
            if owq_column is not None:
                w = w.reshape(oc_batch_size, -1)
                original = w[:, owq_column].detach().clone()
                w[:, owq_column] = float("-inf")
                w = w.reshape(oc_batch_size, 1, -1, group_size)
            org_max_val = w.amax(dim=-1, keepdim=True)

            if owq_column is not None:
                w = w.reshape(oc_batch_size, -1)
                w[:, owq_column] = float('inf')
                w = w.reshape(oc_batch_size, 1, -1, group_size)
            org_min_val = w.amin(dim=-1, keepdim=True)
            assert torch.isinf(org_max_val).sum() == 0, org_max_val
            assert torch.isinf(org_min_val).sum() == 0, org_min_val

            if owq_column is not None:
                w = w.reshape(oc_batch_size, -1)
                w[:, owq_column] = 0
                w = w.reshape(oc_batch_size, 1, -1, group_size)

            best_max_val = org_max_val.clone()
            best_min_val = org_min_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = org_min_val * (1 - i_s / n_grid)
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
                if owq_column is not None:
                    q_w = q_w.reshape(oc_batch_size, -1)
                    q_w[:, owq_column] = original
                    q_w = q_w.reshape(oc_batch_size, 1, -1, group_size)
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w, cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
                best_min_val[cur_best_idx] = min_val[cur_best_idx]
            best_max_val_all.append(best_max_val)
            best_min_val_all.append(best_min_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)
        best_min_val = torch.cat(best_min_val_all, dim=0)

        del input_feat, org_out
        gc.collect()
        torch.cuda.empty_cache()
        return best_max_val.squeeze(1), best_min_val.squeeze(1)


    @torch.no_grad()
    def apply_clip_asym(self, module, clip_list, owq_layer=None):
        for name, max_val, min_val in clip_list:
            if self.do_owq and owq_layer is None:
                i, proj = name.lstrip('model.layers').split('.', maxsplit=1)
                if proj in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']:
                    owq_layer = {proj : self.owq[name]}
            layer = self.get_op_by_name(module, name)
            # layer.to(self.dev)
            max_val = max_val.to(layer.weight.device)
            min_val = min_val.to(layer.weight.device)
            org_shape = layer.weight.shape
            if self.do_owq and owq_layer is not None and name in owq_layer:
                orig = layer.weight.data[:, owq_layer[name]].detach().clone()
                # layer.weight.data[:, owq_layer[name]] = 0
            layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)            
            layer.weight.data = torch.clamp(layer.weight.data, min_val, max_val)
            layer.weight.data = layer.weight.data.reshape(org_shape)
            if self.do_owq and owq_layer is not None and name in owq_layer:
                layer.weight.data[:, owq_layer[name]] = orig
                del orig
                torch.cuda.empty_cache()
                gc.collect()
            # layer.cpu()

    @torch.no_grad()
    def auto_clip_block_sym(self, module, input_feat, module_bit = None):
        named_linears = {
            name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
        }
        clip_list = []
        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
                continue
            # named_linears[name].cuda()
            q_config = {}

            q_config['q_group_size'] = self.group_size
            # print(f'name : {name}, named_linears[name].weight : {named_linears[name].weight.shape}, q_config : {q_config}')
            max_val = self.auto_clip_layer_sym(
                named_linears[name].weight, input_feat[name], n_bit=module_bit[name], q_config=q_config
            )
            clip_list.append((name, max_val))
            # named_linears[name].cpu()
        return clip_list


    @torch.no_grad()
    def auto_clip_layer_sym(
        self, w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512
    ):
        # import pdb; pdb.set_trace()
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = (
            q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
        )
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
        input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
        w = w.reshape(w.shape[0], 1, -1, group_size)

        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
        assert w.shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        del input_feat
        del org_out
        gc.collect()
        torch.cuda.empty_cache()
        return best_max_val.squeeze(1)

    @torch.no_grad()
    def apply_clip_sym(self, module, clip_list):
        for name, max_val in clip_list:
            layer = self.get_op_by_name(module, name)
            # layer.cuda()
            max_val = max_val.to(layer.weight.device)
            org_shape = layer.weight.shape
            layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
            layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
            layer.weight.data = layer.weight.data.reshape(org_shape)
            # layer.cpu()


    @torch.no_grad()
    def apply_awq(self, awq_results):
        
        self.apply_scale(self.model, awq_results["scale"])        
        if self.clip_asym:
            self.apply_clip_asym(self.model, awq_results["clip"])
        else:
            self.apply_clip_sym(self.model, awq_results["clip"])

        layers = self.model.model.layers
        for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
            named_linears = {name: m for name, m in layers[i].named_modules() if isinstance(m, nn.Linear)}
            for n, m in named_linears.items():
                if self.do_owq and self.is_owq(self.arch['linear'][n][i]):
                    assert self.owq is not None, "owq is not provided"
                    original = {}
                    key = f'model.layers.{i}.{n}'
                    if key in self.owq:
                        # original[key] = deepcopy(getattr(getattr(layers[i], module), proj).weight[:, self.owq[key]])
                        original[key] = m.weight[:, self.owq[key]].detach().clone()
                        m.weight[:, self.owq[key]] = 0

                    # m.to(self.dev)
                    m.weight.data = pseudo_quantize_tensor(
                        m.weight.data, n_bit = int(self.arch['linear'][n][i]),
                        q_group_size = self.group_size
                    )
                    # m.cpu()

                    if key in self.owq:
                        m.weight[:, self.owq[key]] = original[key]

                    # del original[key]
                    del original
                    torch.cuda.empty_cache()
                    gc.collect()

                else:
                    # m.to(self.dev)
                    m.weight.data = pseudo_quantize_tensor(
                        m.weight.data, n_bit=int(self.arch['linear'][n][i]),
                        q_group_size = self.group_size
                    )
                    # m.cpu()
    def run(self, nsamples=128, seqlen=512):
    # def run(self, nsamples=8, seqlen=32):
        awq_results = self.run_awq(n_samples=nsamples, seqlen=seqlen)
        self.load_model(device_map='cpu')
        # self.model = simple_dispatch_model(self.model, self.device_map)
        self.model = dispatch_model(self.model, self.device_map)
        self.apply_awq(awq_results)
        torch.cuda.empty_cache()
        gc.collect()