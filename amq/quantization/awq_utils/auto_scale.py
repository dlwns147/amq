import gc
import torch
import torch.nn as nn

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRMSNorm
from transformers.activations import GELUActivation
from amq.model.skip_llama import LlamaDecoderSkipLayer

from .qmodule import ScaledActivation
from .module import get_op_by_name, get_op_name, set_op_by_name, is_owq

# __all__ = ["auto_scale_block", "apply_scale"]


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


@torch.no_grad()
def auto_scale_block(module, module_kwargs, q_config, input_feat, do_owq, module_bit=None, outlier=None):
    from .quantizer import pseudo_quantize_tensor

    def w_quantize_func(p, bit=None):
        assert not is_owq(bit), "bit should be integer"
        return pseudo_quantize_tensor(
            p,
            n_bit=bit,
            **q_config,
        ).detach()

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    def _search_module_scale_per_linear(block, linears2scale: dict, x, kwargs={}, do_owq=False, module_bit=None, outlier=None):
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

        if do_owq:
            original = dict()
            for fc_name, fc in linears2scale.items():
                if fc_name in outlier:
                    original[fc_name] = fc.weight.data[:, outlier[fc_name]].detach().clone()

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc_name, fc in linears2scale.items():
                if do_owq and is_owq(module_bit[fc_name]) and fc_name in outlier:
                    fc.weight.data[:, outlier[fc_name]] = 0
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = w_quantize_func(fc.weight.data, bit=int(module_bit[fc_name])) / (scales.view(1, -1))
                if do_owq and is_owq(module_bit[fc_name]) and fc_name in outlier:
                    fc.weight[:, outlier[fc_name]] = original[fc_name]

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

        if do_owq:
            del original
            torch.cuda.empty_cache()
            gc.collect()

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()


    def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}, module_bit=None, do_owq=False, outlier=None):
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = list(layers.values())[0]

        scales = _search_module_scale_per_linear(module2inspect, layers, inp, kwargs, module_bit=module_bit, do_owq=do_owq, outlier=outlier)
        scales = scales.detach().cpu()
        # prev_op_name, [layer_name], scale
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers.values()]),
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
                    do_owq=do_owq,
                    outlier=outlier,
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
                        do_owq=False,
                        # outlier=outlier,
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
                    do_owq=do_owq,
                    outlier=outlier,
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
                    do_owq=do_owq,
                    outlier=outlier,
                )
            )
    
    elif isinstance(module, (Qwen2DecoderLayer, MistralDecoderLayer)):
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
                do_owq=do_owq,
                outlier=outlier,
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
                    do_owq=False,
                    # outlier=outlier,
                )
            )
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
                do_owq=do_owq,
                outlier=outlier,
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
                do_owq=do_owq,
                outlier=outlier,
            )
        )

    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list


def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm, Qwen2RMSNorm, MistralRMSNorm)):
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

