"""Training utilities: feature extraction and synthetic data for chunk-level bot risk."""

from poker44.training.features import FEATURE_VERSION, N_FEATURES, featurize_chunk

__all__ = ["FEATURE_VERSION", "N_FEATURES", "featurize_chunk"]
