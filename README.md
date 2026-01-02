# AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models

<p align="center">
  <img width="256" height="256" alt="대표이미지1" src="https://github.com/user-attachments/assets/1fbb9aee-0156-4cdc-a74c-4896f90d0c69" />
</p>

This is the official repository for **AMQ**, accepted as an **oral paper** at *EMNLP 2025 Main Conference*.

📄 Paper: https://arxiv.org/abs/2509.12019 

AMQ is an automated mixed-precision quantization library for Large Language Models (LLMs). 
It uses multi-objective optimization to find the optimal balance between model performance and efficiency.

## Key Features

- **Multiple Quantization Methods**: Support for AWQ, GPTQ, OWQ, and more
- **Multi-objective Optimization**: NSGA-II based search algorithm
- **Surrogate Models**: Efficient exploration through MLP and RBF
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

## Usage Examples

### 0. Prepare the Quantization proxy
```bash
bash scripts/amq_quantiztion_proxy.sh 0
```

### 1. Measure Layer Sensitivity

```bash
bash scripts/amq_sensitivity.sh 0
```

### 2. Mixed-precision Search

```bash
bash scripts/amq_search.sh 0
```

### 3. Evaluate Search Results

```bash
bash scripts/amq_quantization_proxy
```

## Supported Models

- Llama 2 (7B, 13B, 70B)
- Mistral
- Qwen2

## Dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers == 4.45.2
- HQQ >= 0.2.0
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
@inproceedings{lee2025amq,
  title={Amq: Enabling automl for mixed-precision weight-only quantization of large language models},
  author={Lee, Sangjun and Woo, Seung-taek and Jin, Jun-gyu and Lee, Changhun and Park, Eunhyeok},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={35520--35538},
  year={2025}
}

```

## Contributing

Contributions are welcome! Please submit a Pull Request or open an issue.

## Contact

If you have any questions or feedback, please open an issue.

