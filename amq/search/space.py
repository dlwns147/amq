import numpy as np
from utils.func import get_bits_usage
from tqdm import tqdm
import math


class SearchSpace:
    def __init__(self, 
                config,
                n_block,
                n_linear,
                group_size=128,
                pass_linear_list=[],
                bits_range=[2, 3, 4],
                ):

        self.config = config
        self.n_block = n_block  # number of blocks
        self.n_linear = n_linear
        self.bits_range = bits_range
        self.pass_linear_list = pass_linear_list
        self.group_size = group_size
        self.rand_size = len(bits_range)
        
        self.pass_linear_idx_list = []
        for pass_linear in self.pass_linear_list:
            blk, linear = pass_linear.split('.', maxsplit=1)
            linear_idx = self.config['linear'].index(linear)
            self.pass_linear_idx_list.append(int(blk) + self.n_block * linear_idx)
            
        self.pass_linear_idx_list.sort()
        print(f'self.pass_linear_idx_list : {self.pass_linear_idx_list}')

    def sample(self, n_samples=1, nb=None, bits=None, pool=[]):
        """ randomly sample a architecture"""
        nb = self.n_block if nb is None else nb
        bits = self.bits_range if bits is None else bits
        
        data = []
        for n in tqdm(range(n_samples), desc='Sampling'):
            while True:
                prob = np.random.rand(self.rand_size)

                bits_prob = prob[np.array([self.bits_range.index(_x) for _x in bits])]

                q_list = np.random.choice(bits, size=nb, p=bits_prob / bits_prob.sum(), replace=True).tolist()
                k_list = np.random.choice(bits, size=nb, p=bits_prob / bits_prob.sum(), replace=True).tolist()
                v_list = np.random.choice(bits, size=nb, p=bits_prob / bits_prob.sum(), replace=True).tolist()
                o_list = np.random.choice(bits, size=nb, p=bits_prob / bits_prob.sum(), replace=True).tolist()
                gate_list = np.random.choice(bits, size=nb, p=bits_prob / bits_prob.sum(), replace=True).tolist()
                up_list = np.random.choice(bits, size=nb, p=bits_prob / bits_prob.sum(), replace=True).tolist()
                down_list = np.random.choice(bits, size=nb, p=bits_prob / bits_prob.sum(), replace=True).tolist()

                for pass_linear in self.pass_linear_list:
                    blk, linear = pass_linear.split('.')[0], pass_linear.split('.')[-1]
                    blk = int(blk)

                    if linear == 'q_proj':
                        q_list[blk] = max(bits)
                    elif linear == 'k_proj':
                        k_list[blk] = max(bits)
                    elif linear == 'v_proj':
                        v_list[blk] = max(bits)
                    elif linear == 'o_proj':
                        o_list[blk] = max(bits)
                    elif linear == 'gate_proj':
                        gate_list[blk] = max(bits)
                    elif linear == 'up_proj':
                        up_list[blk] = max(bits)
                    elif linear == 'down_proj':
                        down_list[blk] = max(bits)
                    else:
                        raise NotImplementedError(f"linear : {linear}")

                new_arch = {'linear': {'self_attn.q_proj': q_list, 'self_attn.k_proj': k_list, 'self_attn.v_proj': v_list, 'self_attn.o_proj': o_list, 'mlp.gate_proj': gate_list, 'mlp.up_proj': up_list, 'mlp.down_proj': down_list}}
                bits_usage = get_bits_usage(new_arch, self.config, self.group_size)
                if (new_arch not in data) and \
                    (new_arch not in pool) and \
                    (math.isclose(bits_usage, self.bits_range[0] + (32 / self.group_size)) or bits_usage > (self.bits_range[0] + (32 / self.group_size))) and \
                    (math.isclose(bits_usage, self.bits_range[-1] + (32 / self.group_size)) or bits_usage < (self.bits_range[-1] + (32 / self.group_size))):
                    break
                
            data.append(new_arch)
        return data

    def initialize(self, n_doe, pool=[]):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = []
        for bit in self.bits_range:
            data.append(self.sample(n_samples=1, bits=[bit])[0])
            n_doe -= 1
        data.extend(self.sample(n_samples=n_doe, bits=self.bits_range, pool=pool))
        return data

    def encode(self, architecture):
        q_encode = np.array([self.bits_range.index(_x) for _x in architecture['linear']['self_attn.q_proj']])
        k_encode = np.array([self.bits_range.index(_x) for _x in architecture['linear']['self_attn.k_proj']])
        v_encode = np.array([self.bits_range.index(_x) for _x in architecture['linear']['self_attn.v_proj']])
        o_encode = np.array([self.bits_range.index(_x) for _x in architecture['linear']['self_attn.o_proj']])
        gate_encode = np.array([self.bits_range.index(_x) for _x in architecture['linear']['mlp.gate_proj']])
        up_encode = np.array([self.bits_range.index(_x) for _x in architecture['linear']['mlp.up_proj']])
        down_encode = np.array([self.bits_range.index(_x) for _x in architecture['linear']['mlp.down_proj']])

        return np.concatenate((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode))
    
    def decode(self, x):
        x_reshape = x.reshape(self.n_linear, self.n_block)
        return {
                    'linear': {
                        'self_attn.q_proj': np.array(self.bits_range)[x_reshape[0]].tolist(),
                        'self_attn.k_proj': np.array(self.bits_range)[x_reshape[1]].tolist(),
                        'self_attn.v_proj': np.array(self.bits_range)[x_reshape[2]].tolist(),
                        'self_attn.o_proj': np.array(self.bits_range)[x_reshape[3]].tolist(),
                        'mlp.gate_proj': np.array(self.bits_range)[x_reshape[4]].tolist(),
                        'mlp.up_proj': np.array(self.bits_range)[x_reshape[5]].tolist(),
                        'mlp.down_proj': np.array(self.bits_range)[x_reshape[6]].tolist(),
                    },
                }
    
    def encode_predictor(self, architecture):
        q_encode = np.array([self.bits_range.index(_x) for blk_idx, _x in enumerate(architecture['linear']['self_attn.q_proj']) if f'{blk_idx}.self_attn.q_proj' not in self.pass_linear_list])
        k_encode = np.array([self.bits_range.index(_x) for blk_idx, _x in enumerate(architecture['linear']['self_attn.k_proj']) if f'{blk_idx}.self_attn.k_proj' not in self.pass_linear_list])
        v_encode = np.array([self.bits_range.index(_x) for blk_idx, _x in enumerate(architecture['linear']['self_attn.v_proj']) if f'{blk_idx}.self_attn.v_proj' not in self.pass_linear_list])
        o_encode = np.array([self.bits_range.index(_x) for blk_idx, _x in enumerate(architecture['linear']['self_attn.o_proj']) if f'{blk_idx}.self_attn.o_proj' not in self.pass_linear_list])
        gate_encode = np.array([self.bits_range.index(_x) for blk_idx, _x in enumerate(architecture['linear']['mlp.gate_proj']) if f'{blk_idx}.mlp.gate_proj' not in self.pass_linear_list])
        up_encode = np.array([self.bits_range.index(_x) for blk_idx, _x in enumerate(architecture['linear']['mlp.up_proj']) if f'{blk_idx}.mlp.up_proj' not in self.pass_linear_list])
        down_encode = np.array([self.bits_range.index(_x) for blk_idx, _x in enumerate(architecture['linear']['mlp.down_proj']) if f'{blk_idx}.mlp.down_proj' not in self.pass_linear_list])

        return np.concatenate((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode))
    
    def decode_encode_predictor(self, x): # x : (batch_size, dim)
        return np.delete(x, self.pass_linear_idx_list, axis=-1)