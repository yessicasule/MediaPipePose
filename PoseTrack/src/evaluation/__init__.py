"""
src/evaluation/__init__.py
Convenience imports for the evaluation package.
"""

from .h36m_loader import parse_h36m_file, iter_h36m_dataset, GTAngles
from .metrics import (
    evaluate_framework,
    compute_joint_metrics,
    print_metrics_table,
    metrics_to_dict,
    JOINTS,
    JOINT_LABELS,
    FrameworkMetrics,
    JointMetrics,
)
