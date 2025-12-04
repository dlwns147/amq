from .awq import AWQ
from .gptq import GPTQ
from .owq import OWQ

from utils.func import clean_up
from accelerate import dispatch_model

METHOD = {
    'awq': AWQ,
    'gptq': GPTQ,
    'owq': OWQ
}

def get_quantized_model(model, method, arch, avg_bits, group_size=128, config=None, dev='cuda', **kwargs):
    method = METHOD[method](model=model, config=config, group_size=group_size, dev=dev, arch=arch, **kwargs)    
    method.run()

    del method
    clean_up()

    return model