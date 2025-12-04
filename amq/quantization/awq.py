from .base import BASE
from .awq_utils.pre_quant import run_awq, apply_awq
from utils.func import clean_up, get_hfmodel

class AWQ(BASE):
    def __init__(self, model, tokenizer, method, arch, avg_bits, group_size=128, config=None, dev='cuda', **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, method=method, arch=arch, avg_bits=avg_bits, group_size=group_size, config=config, dev=dev)
        self.method = 'awq'

        self.clip_asym = kwargs.get('clip_asym', True)
        self.zero_point = kwargs.get('zero_point', True)

        self.device_map = kwargs.get('device_map', 'auto')

    def run(self, nsamples=128, seqlen=512):
        q_config = {
            "zero_point": self.zero_point,  # by default True
            "q_group_size": self.group_size,  # whether to use group quantization
        }

        print("Quantization config:", q_config)

        awq_results = run_awq(model=self.model, tokenizer=self.tokenizer, q_config=q_config, arch=self.arch, clip_asym=self.clip_asym, n_samples=nsamples, seqlen=seqlen)

        # Reload clean model (awq_orig style)
        model_path = self.model.config._name_or_path
        dtype = self.model.dtype

        del self.model
        clean_up()

        self.model = get_hfmodel(model_path, dtype=dtype, device_map=self.device_map)
        self.model = self.model.to(self.dev)
        self.model.eval()

        apply_awq(model=self.model, awq_results=awq_results, q_config=q_config, arch=self.arch, clip_asym=self.clip_asym)
        
        clean_up()