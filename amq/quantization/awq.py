from .base import BASE
from .awq_utils.pre_quant import run_awq, apply_awq
from utils.func import clean_up

class AWQ(BASE):
    def __init__(self, model, config, arch, group_size=128, dev='cuda', **kwargs):
        super().__init__(model, config, arch, group_size=group_size, dev=dev)
        self.method = 'awq'

        self.clip_asym = kwargs.get('clip_asym', True)
        self.no_zero_point = kwargs.get('no_zero_point', False)


    def run(self, nsamples=128, seqlen=512, no_zero_point=False):
        q_config = {
            "zero_point": not no_zero_point,  # by default True
            "q_group_size": self.group_size,  # whether to use group quantization
            "clip_asym": self.clip_asym,
        }

        print("Quantization config:", q_config)

        awq_results = run_awq(self.model, self.tokenizer, q_config=q_config, arch=self.arch, clip_asym=self.clip_asym, n_samples=nsamples, seqlen=seqlen)
        apply_awq(self.model, awq_results, q_config=q_config, arch=self.arch, clip_asym=self.clip_asym)
        
        clean_up()