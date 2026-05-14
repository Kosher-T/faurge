"""
CLAP utilities for the Fabian Orchestrator.

Provides audio/text encoding via the frozen LAION CLAP model,
used for similarity scoring in the reflection loop and for
providing latent embeddings to Genesis.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def load_clap_model(model_dir: Optional[Path] = None, device: str = "cpu"):
    """
    Load the frozen CLAP model from local cache or Hugging Face.
    Sets model to eval mode and returns (model, processor).
    """
    raise NotImplementedError


def encode_audio(audio: np.ndarray, sr: int = 48000,
                 model=None, processor=None, device: str = "cpu") -> np.ndarray:
    """
    Encode audio waveform through CLAP audio tower.
    Returns 512D float32 embedding.
    """
    raise NotImplementedError


def encode_text(text: str, model=None, processor=None,
                device: str = "cpu") -> np.ndarray:
    """
    Encode text through CLAP text tower.
    Returns 512D float32 embedding.
    """
    raise NotImplementedError


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two CLAP embeddings.
    """
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))
