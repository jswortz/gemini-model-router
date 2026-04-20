from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from router.config_loader import ClassifierCfg, expand, load_anchors


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = x / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class EmbedAnchorsClassifier:
    def __init__(self, cfg: ClassifierCfg):
        self.cfg = cfg
        self.cache_dir = expand(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._labels: list[str] = []
        self._centroids: np.ndarray | None = None
        self._load_or_build()

    def _lazy_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.model)
        return self._model

    def _anchor_fingerprint(self, anchors: dict[str, list[str]]) -> str:
        h = hashlib.sha256()
        h.update(self.cfg.model.encode())
        for label in sorted(anchors):
            h.update(label.encode())
            for a in anchors[label]:
                h.update(b"\x00")
                h.update(a.encode())
        return h.hexdigest()[:16]

    def _cache_paths(self, fp: str) -> tuple[Path, Path]:
        return self.cache_dir / f"centroids-{fp}.npy", self.cache_dir / f"labels-{fp}.json"

    def _load_or_build(self) -> None:
        anchors = load_anchors(self.cfg.anchors_file)
        fp = self._anchor_fingerprint(anchors)
        npy, jsn = self._cache_paths(fp)
        if npy.exists() and jsn.exists():
            self._centroids = np.load(npy)
            self._labels = json.loads(jsn.read_text())
            return
        # build
        labels = sorted(anchors.keys())
        model = self._lazy_model()
        centroids = []
        for label in labels:
            embs = model.encode(anchors[label], normalize_embeddings=True, show_progress_bar=False)
            centroid = np.asarray(embs).mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            centroids.append(centroid)
        self._centroids = np.vstack(centroids).astype(np.float32)
        self._labels = labels
        np.save(npy, self._centroids)
        jsn.write_text(json.dumps(labels))

    def rebuild(self) -> None:
        # invalidate cache by sweeping files for this model and re-running build
        for f in self.cache_dir.glob("centroids-*.npy"):
            f.unlink(missing_ok=True)
        for f in self.cache_dir.glob("labels-*.json"):
            f.unlink(missing_ok=True)
        self._centroids = None
        self._labels = []
        self._load_or_build()

    def classify(self, prompt: str) -> dict[str, float]:
        assert self._centroids is not None
        emb = self._lazy_model().encode(
            [prompt], normalize_embeddings=True, show_progress_bar=False
        )
        emb = np.asarray(emb[0], dtype=np.float32)
        sims = self._centroids @ emb  # cosine since both normalized
        probs = _softmax(sims, self.cfg.softmax_temp)
        return {label: float(p) for label, p in zip(self._labels, probs, strict=True)}
