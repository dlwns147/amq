from .awq import AWQ
from .gptq import GPTQ
from .owq import OWQ

from utils.func import clean_up

METHOD = {
    'awq': AWQ,
    'gptq': GPTQ,
    'owq': OWQ
}

def get_quantized_model(model, tokenizer, method, arch, avg_bits, group_size=128, config=None, dev='cuda', **kwargs):
    quantizer = METHOD[method](model=model, tokenizer=tokenizer, method=method, arch=arch, avg_bits=avg_bits, group_size=group_size, config=config, dev=dev, **kwargs)    
    quantizer.run()

    clean_up()

    return quantizer.model