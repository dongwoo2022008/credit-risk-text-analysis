"""
Utility modules for credit risk analysis
"""

from .data_loader import (
    load_raw_data,
    encode_target,
    prepare_structured_features,
    split_data,
    scale_features,
    get_credit_score_groups,
    get_text_length_groups,
    load_and_prepare_data
)

from .evaluator import (
    evaluate_model,
    evaluate_multiple_models,
    compare_models,
    calculate_threshold_metrics,
    calculate_uncertainty_metrics,
    analyze_error_cases,
    print_evaluation_summary
)

__all__ = [
    # data_loader
    'load_raw_data',
    'encode_target',
    'prepare_structured_features',
    'split_data',
    'scale_features',
    'get_credit_score_groups',
    'get_text_length_groups',
    'load_and_prepare_data',
    # evaluator
    'evaluate_model',
    'evaluate_multiple_models',
    'compare_models',
    'calculate_threshold_metrics',
    'calculate_uncertainty_metrics',
    'analyze_error_cases',
    'print_evaluation_summary'
]
