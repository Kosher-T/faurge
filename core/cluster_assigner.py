"""
core/cluster_assigner.py
Nearest-centroid cluster assignment for Ursula's cluster conditioning input.

Used by Fabian (BakeOrchestrator) at inference time to determine which
voice cluster the source audio belongs to. Falls back to "unknown" if
the audio is too far from all known cluster centroids.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class ClusterAssigner:
    """
    Assigns a (K+1)-D one-hot cluster vector to a 67D metrics vector
    via nearest-centroid matching.

    K = number of clusters (default 8).
    Index K is reserved for "unknown".
    """

    def __init__(
        self,
        centroids_path: Optional[Path] = None,
        n_clusters: int = 8,
        threshold: Optional[float] = None,
    ):
        self.n_clusters = n_clusters
        self._centroids: Optional[np.ndarray] = None
        self._threshold: Optional[float] = threshold

        if centroids_path is not None and centroids_path.exists():
            self._load_centroids(centroids_path)

    def _load_centroids(self, path: Path) -> None:
        import json
        with open(path, "r") as f:
            data = json.load(f)

        centroids = []
        for i in range(self.n_clusters):
            key = f"cluster_{i}"
            if key in data:
                centroids.append(data[key]["centroid_67d"])
                if self._threshold is None:
                    self._threshold = data[key].get("threshold")
            else:
                raise ValueError(
                    f"Missing centroid for cluster_{i} in {path}"
                )

        self._centroids = np.array(centroids, dtype=np.float64)

    @property
    def is_ready(self) -> bool:
        return self._centroids is not None

    @property
    def threshold(self) -> float:
        if self._threshold is not None:
            return self._threshold
        return float("inf")

    def assign(self, metrics_67d: np.ndarray) -> Tuple[int, bool]:
        """
        Determine cluster for a 67D metrics vector.

        Returns:
            (cluster_id, is_unknown):
                cluster_id in [0, K) for known, K for unknown.
                is_unknown = True when distance exceeds threshold.
        """
        if not self.is_ready:
            return (self.n_clusters, True)

        diffs = self._centroids - metrics_67d.reshape(1, -1)
        mses = np.mean(diffs ** 2, axis=1)
        nearest = int(np.argmin(mses))
        min_mse = float(mses[nearest])

        if min_mse > self.threshold:
            return (self.n_clusters, True)

        return (nearest, False)

    def to_onehot(self, cluster_id: int) -> np.ndarray:
        """
        Build (K+1)-D one-hot vector.
        cluster_id = K means "unknown".
        """
        onehot = np.zeros(self.n_clusters + 1, dtype=np.float32)
        if 0 <= cluster_id <= self.n_clusters:
            onehot[cluster_id] = 1.0
        else:
            onehot[self.n_clusters] = 1.0
        return onehot

    def assign_and_encode(self, metrics_67d: np.ndarray) -> np.ndarray:
        """Convenience: assign + to_onehot in one call."""
        cluster_id, _ = self.assign(metrics_67d)
        return self.to_onehot(cluster_id)
