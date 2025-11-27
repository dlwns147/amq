import torch
import torch.nn as nn
from .quantizer import pseudo_quantize_tensor
from .module import get_op_by_name
import gc

# __all__ = ["auto_clip_block"]


@torch.no_grad()
def auto_clip_block_asym(module, input_feat, q_config, module_bit=None, outlier=None):
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }

    clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue      
        # named_linears[name].cuda()
        max_val, min_val = auto_clip_layer_asym(
            named_linears[name].weight, input_feat[name], n_bit=module_bit[name], q_config=q_config,
            ## customizing
            outlier = outlier[name] if outlier is not None and name in outlier else None,
            bias=named_linears[name].bias if hasattr(named_linears[name], 'bias') else None
        )
        clip_list.append((name, max_val, min_val))
        
        # named_linears[name].cpu()
    return clip_list


@torch.no_grad()
def auto_clip_layer_asym(
    w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512, outlier=None, bias=None
):
    assert n_bit == int(n_bit), "bit should be integer"
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]

    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    input_feat = input_feat.to(w.device)
    # input_feat = input_feat.flatten(-2).squeeze(0)
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []
    best_min_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        # if bias is not None:
        #     b = bias[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group
        
        if outlier is not None:
            w = w.reshape(oc_batch_size, -1)
            original = w[:, outlier].detach().clone()
            w[:, outlier] = float("-inf")
            w = w.reshape(oc_batch_size, 1, -1, group_size)
        org_max_val = w.amax(dim=-1, keepdim=True)

        if outlier is not None:
            w = w.reshape(oc_batch_size, -1)
            w[:, outlier] = float('inf')
            w = w.reshape(oc_batch_size, 1, -1, group_size)
        org_min_val = w.amin(dim=-1, keepdim=True)
        assert torch.isinf(org_max_val).sum() == 0, org_max_val
        assert torch.isinf(org_min_val).sum() == 0, org_min_val

        if outlier is not None:
            w = w.reshape(oc_batch_size, -1)
            w[:, outlier] = 0
            w = w.reshape(oc_batch_size, 1, -1, group_size)

        best_max_val = org_max_val.clone()
        best_min_val = org_min_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = org_min_val * (1 - i_s / n_grid)
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
            if outlier is not None:
                q_w = q_w.reshape(oc_batch_size, -1)
                q_w[:, outlier] = original
                q_w = q_w.reshape(oc_batch_size, 1, -1, group_size)
            # import pdb; pdb.set_trace()
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w, cur_out, q_w
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_min_val[cur_best_idx] = min_val[cur_best_idx]
        best_max_val_all.append(best_max_val)
        best_min_val_all.append(best_min_val)
        # print(f'min_errs.min() : {min_errs.min()}, min_errs.max() : {min_errs.max()}')

    best_max_val = torch.cat(best_max_val_all, dim=0)
    best_min_val = torch.cat(best_min_val_all, dim=0)

    del input_feat, org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1), best_min_val.squeeze(1)


@torch.no_grad()
def apply_clip_asym(module, clip_list, do_owq=False, outlier=None):
    for name, max_val, min_val in clip_list:
        layer = get_op_by_name(module, name)
        max_val = max_val.to(layer.weight.device)
        min_val = min_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        if do_owq and outlier is not None and name in outlier:
            orig = layer.weight.data[:, outlier[name]].detach().clone()
            layer.weight.data[:, outlier[name]] = 0
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)            
        layer.weight.data = torch.clamp(layer.weight.data, min_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        if do_owq and outlier is not None and name in outlier:
            layer.weight.data[:, outlier[name]] = orig
            del orig
            torch.cuda.empty_cache()
            gc.collect()

@torch.no_grad()
def auto_clip_block_sym(module, input_feat, q_config, module_bit = None):
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }

    clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue
        # named_linears[name].cuda()
        # print(f'name : {name}, named_linears[name].weight : {named_linears[name].weight.shape}, q_config : {q_config}')
        max_val = auto_clip_layer_sym(
            named_linears[name].weight, input_feat[name], n_bit=module_bit[name], q_config=q_config
        )
        clip_list.append((name, max_val))
        # named_linears[name].cpu()
    return clip_list


@torch.no_grad()
def auto_clip_layer_sym(
    w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512
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
def apply_clip_sym(module, clip_list):
    for name, max_val in clip_list:
        layer = get_op_by_name(module, name)
        # layer.cuda()
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        # layer.cpu()
