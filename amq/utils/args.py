import argparse


def get_base_parser() -> argparse.ArgumentParser:
    """Default argument parser for AMQ"""
    parser = argparse.ArgumentParser(
        description='AMQ: AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    return parser

def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add model-related arguments"""
    group = parser.add_argument_group('Model Configuration')
    
    group.add_argument('--model_path', type=str, required=True, default='meta-llama',
                       help='Path to the model directory')
    group.add_argument('--model_name', type=str, required=True, default='Llama-2-7b-hf',
                       help='Name of the model (e.g., Llama-2-7b-hf)')
    group.add_argument('--config', type=str, default='amq/configs/llama.json',
                       help='Path to model configuration JSON file')
    # group.add_argument('--dtype', type=str, default='float16', 
    #                    choices=['float16', 'float', 'fp16', 'bfloat16', 'bfloat', 'bf16', 'auto'],
    #                    help='Data type for model weights')
    group.add_argument('--quantization_proxy_paths', type=str, nargs='+', default=[], 
                       help='Paths to quantization proxies')
    group.add_argument('--gpu_id', type=str, default='0',
                       help='id of available gpus')
    
    return parser

def add_quantization_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add quantization-related arguments"""
    group = parser.add_argument_group('Quantization Configuration')
    
    # Quantization method parameters
    group.add_argument('--method', type=str, default='hqq',
                       choices=['awq', 'gptq', 'hqq'],
                       help='Quantization method(s) to use')
    group.add_argument('--group_size', type=int, default=128,
                       help='Group size for quantization (128 for per-channel)')

    # Candidates Selection parameters
    group.add_argument('--prefer', type=str, nargs='+', default=[],
                       help='Preference for candidate selection (e.g., "metric#0.0 bits#3.0")')
    group.add_argument('--target_bits', type=float, default=3.0,
                       help='Target bit-width')

    # General parameters
    group.add_argument('--load', type=str, default=None,
                       help='Path to load search results. file format must be end with *.iter_*.stats')

    return parser

def add_search_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add search-related arguments"""
    group = parser.add_argument_group('Search Configuration')

    # Search Space Pruning parameters
    group.add_argument('--sensitivity_threshold', type=float, default=2.0,
                       help='Threshold for sensitivity-based pruning. The layer whose sensitivity is larger than threshold * median sensitivity of all layers will be pruned.')
    group.add_argument('--sensitivity_datasets', type=str, default='wikitext2',
                       choices=['wikitext2', 'c4', 'ptb'],
                       help='Datasets for sensitivity analysis')
    group.add_argument('--sensitivity_n_sample', type=int, default=128,
                       help='Number of samples for sensitivity analysis')
    group.add_argument('--sensitivity_seqlen', type=int, default=2048,
                       help='Sequence length for sensitivity analysis')

    # Qulity Predictor parameters
    group.add_argument('--predictor', type=str, default='rbf',
                       choices=['mlp', 'rbf'],
                       help='Surrogate model for quality prediction (mlp or rbf)')

    # Iterative Search and Update parameters
    group.add_argument('--iterations', type=int, default=200,
                       help='Total number of search iterations')
    group.add_argument('--n_doe', type=int, default=250,
                       help='Number of samples for Design of Experiments (DOE) Pretraining Data')
    group.add_argument('--n_iter', type=int, default=50,
                       help='Number of high-fidelity evaluations per iteration')
    group.add_argument('--max_value', type=float, default=1.0,
                       help='Maximum value for search space in genetic algorithm')
    
    # Genetic Algorithm parameters
    group.add_argument('--subset_pop_size', type=int, default=100,
                       help='Population size for subset selection')
    group.add_argument('--ga_pop_size', type=int, default=200,
                       help='Population size for genetic algorithm')
    group.add_argument('--crossover_prob', type=float, default=0.9,
                       help='Crossover probability in genetic algorithm')
    group.add_argument('--mut_prob', type=float, default=0.1,
                       help='Mutation probability in genetic algorithm')

    # General parameters
    group.add_argument('--save_iter', type=int, default=1,
                       help='Save results every n iterations')
    group.add_argument('--save_path', type=str, default=None,
                       help='Path to save results')
    group.add_argument('--result_file', type=str, default='results.txt',
                       help='File name to save results')
    group.add_argument('--resume_path', type=str, default=None,
                       help='Path to resume search from checkpoint')
    
    return parser

def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add data-related arguments"""
    group = parser.add_argument_group('Data Configuration')
    
    group.add_argument('--dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'c4', 'ptb'],
                       help='Dataset for calibration and evaluation')
    group.add_argument('--n_sample', type=int, default=128,
                       help='Number of samples for calibration')
    group.add_argument('--seqlen', type=int, default=2048,
                       help='Sequence length for calibration')
    group.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')
    
    return parser

def parse_args(mode: str = 'search') -> argparse.Namespace:
    """
    Parse arguments based on the mode
    
    Args:
        mode: 'search', 'quantization'
    
    Returns:
        Parsed arguments
    """

    parser = get_base_parser()
    parser = add_model_args(parser)
    parser = add_data_args(parser)

    # TODO: add sensitivity analysis arguments

    if mode == 'search':
        parser = add_search_args(parser)
    elif mode == 'quantization':
        parser = add_quantization_args(parser)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'search', 'quantization'")

    return parser.parse_args()


