"""
Public utilities shared by knowledge editing methods.

Ported from:
- nethook: https://github.com/kmeng01/rome  (MIT License)
- compute_z / layer_stats: https://github.com/TrustedLLM/UnKE
"""

from trainer.edit.utils.nethook import (
    Trace,
    TraceDict,
    StopForward,
    get_module,
    get_parameter,
    replace_module,
    set_requires_grad,
    recursive_copy,
)

__all__ = [
    "Trace",
    "TraceDict",
    "StopForward",
    "get_module",
    "get_parameter",
    "replace_module",
    "set_requires_grad",
    "recursive_copy",
]
