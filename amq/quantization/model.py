from .awq import AWQ
from .gptq import GPTQ
from .qeft import QEFT
from amq.utils import clean_up

from accelerate import dispatch_model

METHOD = {
    'gptq': GPTQ,
    'awq': AWQ,
    'qeft': QEFT
}

def get_quantized_model(method, arch, model_name, device_map, dtype='auto', group_size=128, config=None, dev='cuda', prune=False, do_owq=False, outlier_path=None, **kwargs):
    method = METHOD[method](model_name=model_name, config=config, device_map=device_map, dtype=dtype, group_size=group_size, dev=dev, arch=arch, prune=prune, do_owq=do_owq, outlier_path=outlier_path, **kwargs)

    if prune:
        print('Pruning the model')
        method.prune_model()
        
    method.run()    
    model = dispatch_model(method.model, method.device_map)
    del method
    clean_up()

    return model