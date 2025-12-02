# AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models

This is the official repository for **AMQ**, accepted as an **oral paper** at *EMNLP 2025 Main Conference*.

📄 Paper: https://arxiv.org/abs/2509.12019 

### **🚀 The code is constantly being updated.**

AMQ is an automated mixed-precision quantization library for Large Language Models (LLMs). 
It uses multi-objective optimization to find the optimal balance between model performance and efficiency.

## Key Features

- **Multiple Quantization Methods**: Support for AWQ, GPTQ, QEFT, and more
- **Multi-objective Optimization**: NSGA-II based search algorithm
- **Surrogate Models**: Efficient exploration through MLP, GP, and RBF
- **Layer-wise Sensitivity Analysis**: Measure quantization sensitivity per layer
- **Automated Mixed-precision Search**: Automatic exploration of optimal bit configurations

## Installation

### Installation via pip

```bash
pip install -e .
```

### Installation via requirements

```bash
pip install -r requirements.txt
```

### Developer Mode Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Measure Layer Sensitivity

```python
from amq.evaluation import Evaluator
from amq.quantization import AWQ

# Load model and measure sensitivity
evaluator = Evaluator(model_name="Llama-2-7b-hf")
sensitivity = evaluator.measure_sensitivity()
```

### 2. Mixed-precision Search

```python
from amq.search import Optimizer
from amq.predictor import PredictorFactory

# Run search
optimizer = Optimizer(
    model_name="Llama-2-7b-hf",
    predictor_type="mlp",
    iterations=300
)
results = optimizer.search()
```

### 3. Evaluate Search Results

```python
from amq.evaluation import Evaluator

# Evaluate results
evaluator = Evaluator(model_name="Llama-2-7b-hf")
metrics = evaluator.evaluate_candidates(results)
```

## Usage Examples

### Command Line Interface

#### Measure Layer Sensitivity
```bash
CUDA_VISIBLE_DEVICES=0 python examples/linear_sensitivity.py \
    --model_name Llama-2-7b-hf \
    --method hqq \
    --n_sample 128 \
    --config configs/llama.json
```

#### Mixed-precision Search
```bash
CUDA_VISIBLE_DEVICES=0 python examples/search_model.py \
    --model_name Llama-2-7b-hf \
    --method hqq \
    --predictor mlp \
    --iterations 300 \
    --config configs/llama.json
```

#### Evaluate Search Results
```bash
CUDA_VISIBLE_DEVICES=0 python examples/post_search.py \
    --model_name Llama-2-7b-hf \
    --config configs/llama.json \
    --expr path/to/search/results
```

## Package Structure

```
amq/
├── quantization/          # Quantization method implementations
│   ├── awq.py            # AWQ quantization
│   ├── gptq.py           # GPTQ quantization
│   └── qeft.py           # QEFT quantization
├── search/               # Search algorithms
│   ├── optimizer.py      # Optimization engine
│   └── space.py          # Search space definition
├── predictor/            # Surrogate models
│   ├── mlp.py            # MLP predictor
│   ├── gp.py             # Gaussian Process
│   └── rbf.py            # RBF network
├── evaluation/           # Evaluation modules
│   └── evaluator.py      # Model evaluator
└── utils/                # Utility functions
```

## Supported Models

- Llama 2 (7B, 13B, 70B)
- Mistral
- Qwen2
- OPT

## Dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers == 4.45.2
- HQQ == 0.2.2
- pymoo == 0.6.1.3
- See requirements.txt for more

## Configuration Files

Model-specific configuration files are located in the `configs/` directory:

- `configs/llama.json` - Llama model configuration
- `configs/mistral.json` - Mistral model configuration
- `configs/qwen2.json` - Qwen2 model configuration

## License

This project is licensed under the Apache License 2.0. 
See the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{lee-etal-2025-amq,
    title = "{AMQ}: Enabling {A}uto{ML} for Mixed-precision Weight-Only Quantization of Large Language Models",
    author = "Lee, Sangjun  and
      Woo, Seung-taek  and
      Jin, Jun-gyu  and
      Lee, Changhun  and
      Park, Eunhyeok",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1799/",
    doi = "10.18653/v1/2025.emnlp-main.1799",
    pages = "35520--35538",
    ISBN = "979-8-89176-332-6",
}

```

## Contributing

Contributions are welcome! Please submit a Pull Request or open an issue.

## Contact

If you have any questions or feedback, please open an issue.

