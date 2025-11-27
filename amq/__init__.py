"""
AMQ: Automated Mixed-precision Quantization

A Python package for automated mixed-precision quantization of large language models
using multi-objective optimization.
"""

__version__ = "0.1.0"

from amq import quantization
from amq import search
from amq import predictor
from amq import evaluation
from amq import utils

__all__ = [
    "quantization",
    "search",
    "predictor",
    "evaluation",
    "utils",
]

