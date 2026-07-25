"""Sequence models for dynamic-sign (temporal ASL) recognition — research track.

The static fingerspelling classifier ([src/models.py](src/models.py), a CNN over a single
hand crop) cannot represent motion. Real ASL signs are *trajectories*. This module models
a sign as a sequence of MediaPipe hand landmarks:

    (T, 63) = T frames x 21 landmarks x (x, y, z)

A bidirectional GRU over *normalized* landmark sequences is lightweight (no CNN), runs at
webcam frame rates on CPU, and is the honest starting point for "actually ASL, not
fingerspelling." Pair with:

  - ``src/scripts/capture_landmark_sequences.py``  — collect labeled clips
  - ``train_sequence.py``                          — train + held-out test evaluation
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # torch-free environments can still use normalize_landmarks
    torch = None
    nn = None

NUM_LANDMARKS = 21
FEATURES_PER_FRAME = NUM_LANDMARKS * 3  # 63


def normalize_landmarks(seq) -> np.ndarray:
    """Make landmark sequences translation- and scale-invariant.

    ``seq``: (T, 63) or (T, 21, 3). Returns (T, 63). Each frame is centered on the wrist
    (landmark 0) and scaled by the wrist->middle-finger-MCP (landmark 9) distance, so the
    model learns hand *shape/motion* rather than where the hand happens to sit in frame.
    """
    seq = np.asarray(seq, dtype=np.float32).reshape(-1, NUM_LANDMARKS, 3)
    wrist = seq[:, 0:1, :]                                    # (T, 1, 3)
    centered = seq - wrist
    scale = np.linalg.norm(centered[:, 9, :], axis=1)         # (T,)
    scale = np.where(scale < 1e-6, 1.0, scale)[:, None, None]  # (T, 1, 1)
    normed = centered / scale
    return normed.reshape(seq.shape[0], FEATURES_PER_FRAME)


if torch is not None:

    class SignSequenceModel(nn.Module):
        """Bidirectional-GRU classifier over (B, T, 63) landmark sequences.

        Mean-pools over time so it is robust to clip length / padding.
        """

        def __init__(self, num_classes: int, hidden: int = 128,
                     num_layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.gru = nn.GRU(
                FEATURES_PER_FRAME, hidden, num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0, bidirectional=True,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, num_classes),
            )

        def forward(self, x):
            out, _ = self.gru(x)        # (B, T, 2*hidden)
            pooled = out.mean(dim=1)    # temporal mean-pool
            return self.head(pooled)

else:  # pragma: no cover
    SignSequenceModel = None
