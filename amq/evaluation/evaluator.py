import math
from copy import deepcopy

import warnings
warnings.simplefilter("ignore")

from utils.func import get_hfmodel, get_quantization_proxy, get_bits_usage, clean_up, getsubattr, setsubattr, getblock
from utils.data import get_loader
from utils.eval import eval_loss, eval_ppl, get_logits
from quantization.model import get_quantized_model

class Evaluator:
    def __init__(self,  
                 config,
                 accelerator,
                 model_id='',
                 quantization_proxy_paths=[],
                 bits_range=[],
                 group_size=128,
                 datasets=['wikitext2'],
                 seed=0,
                 seqlen=2048,
                 n_sample=128,
                 device_map='auto',
                 dtype='auto',
                 search = True,
                 **kwargs):
        
        self.model_id = model_id
        self.model = None
        self.device_map = device_map
        self.dtype = dtype
        self.config = config
        self.seqlen = seqlen
        self.group_size = group_size
        self.search = search

        if self.search:
            self.train_loaders = {dataset: accelerator.prepare(get_loader(dataset, model=model_id, n_sample=n_sample, train=True, seed=seed, seqlen=seqlen)) for dataset in datasets}
            self.dense_logits = {dataset: None for dataset in self.train_loaders.keys()}
        
            print(f'Obtaining dense logits')
            model = get_hfmodel(model_id, dtype=dtype, device_map=device_map)
            self.dense_logits = {dataset: get_logits(model, loader) for dataset, loader in self.train_loaders.items()}
            del model
            clean_up()
        
            print(f'Loading quantization proxies')
            self.quantization_proxies = get_quantization_proxy(quantization_proxy_paths, device_map)
            self.bits_range = bits_range
            assert len(self.bits_range) == len(self.quantization_proxies), f'Number of bits range and quantization proxies must be the same'

            self.model = deepcopy(self.quantization_proxies[-1])
        else:
            self.test_loaders = {dataset: accelerator.prepare(get_loader(dataset, model=model_id, train=False, seqlen=seqlen)) for dataset in datasets}

            print(f'Loading model')
            self.model = get_hfmodel(model_id, dtype=dtype, device_map=device_map)
            clean_up()

        accelerator.wait_for_everyone()

    def sample(self, arch, method='awq'):
        if self.search:
            for linear, linear_bits in arch['linear'].items():
                for blk_idx, bits in enumerate(linear_bits):
                    flag = False
                    for q_bits, q_model in zip(self.bits_range, self.quantization_proxies):
                        if math.isclose(int(bits), q_bits) and q_bits > 0:
                            setsubattr(getblock(self.model, self.config)[blk_idx], linear, getsubattr(getblock(q_model, self.config)[blk_idx], linear))
                            flag = True
                    if not flag:
                        raise NotImplementedError(f'{linear}: {bits} is not available')
        else:
            # TODO: Implement Quantization method(AWQ, GPTQ, OWQ)
            self.model = get_quantized_model(self.model, method, arch, get_bits_usage(arch, self.config, self.group_size), self.group_size, self.config, self.dev)
            self.model.eval()
            self.model.config_use_cache = False

        return self.model

    def eval(self, accelerator, architecture):
        metric_list = dict()

        if self.search:
            for dataset, loader in self.train_loaders.items():
                metric_list[dataset] = eval_loss(model=self.sample(architecture), accelerator=accelerator, loader=loader, dense_logits_list=self.dense_logits[dataset], seqlen=self.seqlen)
        else:
            for dataset, loader in self.test_loaders.items():
                metric_list[dataset] = eval_ppl(model=self.model, accelerator=accelerator, loader=loader, seqlen=self.seqlen)

        bits_usage = get_bits_usage(architecture, self.config, self.group_size)
        clean_up()

        return metric_list, bits_usage