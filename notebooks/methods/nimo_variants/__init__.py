"""
NIMO Variants Package

This package contains different implementations of NIMO (Nonlinear Interpretable Model).
Each variant is implemented as a separate module with a consistent interface.

Available variants:
- baseline: Adaptive Ridge Logistic Regression with Lightning
- variant: Original NIMO Implementation with IRLS-Loop
"""

from .baseline import run_nimo_baseline
from .variant import run_nimo_variant
from .nimo import run_nimo
from .nimoTransformer_NN import run_nimo as run_nimo_transformer

__all__ = ['run_nimo_baseline', 'run_nimo_variant', 'run_nimo', 'run_nimo_transformer'] 