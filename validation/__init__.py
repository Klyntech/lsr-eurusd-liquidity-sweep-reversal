"""
Validation package for LSR-EH1 Strategy.

Provides walk-forward testing infrastructure to evaluate strategy
robustness and detect overfitting through rolling out-of-sample analysis.
"""

from validation.walk_forward import WalkForwardFold, WalkForwardRunner

__all__ = ["WalkForwardFold", "WalkForwardRunner"]
