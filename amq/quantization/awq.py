import torch
import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation

# import sys
# sys.path.append('..')
from .base import BASE, get_awq_calib_dataset
# from utils.dispatch import simple_dispatch_model
from accelerate import dispatch_model

from .awq_utils.pre_quant import run_awq, apply_awq

import gc
from tqdm import tqdm
import math
import functools
from copy import deepcopy

from collections import defaultdict

class AWQ(BASE):
    def __init__(self, model_name, config, arch, device_map, dtype='auto', group_size=128, dev='cuda', prune=False, do_owq=False, outlier_path=None, **kwargs):
        super().__init__(model_name, config, arch, device_map=device_map, dtype=dtype, group_size=group_size, dev=dev, prune=prune, do_owq=do_owq, outlier_path=outlier_path)
        self.method = 'awq'

        self.clip_asym = kwargs.get('clip_asym', True)
        if self.clip_asym:
            print("Clipping asymmetrically")
        else:
            print("Clipping symmetrically")


    def run(self, nsamples=128, seqlen=512, no_zero_point=False):
        q_config = {
            "zero_point": not no_zero_point,  # by default True
            "q_group_size": self.group_size,  # whether to use group quantization
        }
        print("Quantization config:", q_config)
        awq_results = run_awq(self.model, self.tokenizer, q_config=q_config, arch=self.arch, clip_asym=self.clip_asym, n_samples=nsamples, seqlen=seqlen, do_owq=self.do_owq, outlier=self.outlier)
        self.load_model(device_map='cpu', dtype=self.dtype)
        # self.model = simple_dispatch_model(self.model, self.device_map)
        self.model = dispatch_model(self.model, self.device_map)
        apply_awq(self.model, awq_results, q_config=q_config, arch=self.arch, clip_asym=self.clip_asym, do_owq=self.do_owq, outlier=self.outlier)
        torch.cuda.empty_cache()
        gc.collect()
        
        # target_path_list = ["model.layers.15.self_attn.v_proj", "model.layers.23.mlp.down_proj"]
        target_path_list = ["model.layers.7.self_attn.q_proj", 
                            "model.layers.25.self_attn.k_proj", 
                            "model.layers.15.self_attn.v_proj", 
                            "model.layers.30.self_attn.o_proj", 
                            "model.layers.12.mlp.down_proj",
                            "model.layers.22.mlp.gate_proj",
                            "model.layers.6.mlp.up_proj"]
        
        from datasets import load_dataset
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = self.tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        inputs = trainenc.input_ids[:1, :1].to(self.model.device)
        for target_path in target_path_list:
            # # 캡처 버퍼
            save_list = {}

            # 1) 원하는 모듈 경로 지정: 예) 5번째 디코더 레이어의 MLP down_proj
            target_module = self.model.get_submodule(target_path)

            # 2) forward hook: Linear의 '입력'은 hook의 input[0]
            def save_input_hook(module, inputs, output):
                x = inputs[0].detach()          # (batch, seq, hidden)
                save_list['activations'] = x

            hook_handle = target_module.register_forward_hook(save_input_hook)

            with torch.inference_mode():
                self.model(inputs, use_cache=False)      # or model.generate(**tok(...))
            
            # 4) 저장 (예: PyTorch tensor 리스트로 저장)
            import os
            save_path = os.path.join('/NAS/SJ/nsgaquant/save', '_'.join(target_path.split('.')) + f'_{int(self.arch["linear"]["self_attn.q_proj"][0])}bit_activations.pth')
            torch.save(save_list, save_path)
            
            # 훅 해제(선택)
            hook_handle.remove()
        exit()
        