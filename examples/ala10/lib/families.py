"""Canonical H0--H3 helical-family atlas for neutral Ala10.

The canonical Monte Carlo sample alone defines the strict-helical manifold
(at least 70% of the nine backbone phi/psi pairs in the alpha region) and its
four recurrent families: strict frames are embedded as (cos, sin) features of
the backbone angles and clustered by k-means (30 initializations, seed
20260719); labels follow decreasing canonical population.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

MODEL_SEED = 20260719
FAMILY_COUNT = 4
PRIMARY_HELIX_THRESHOLD = 0.70


@dataclass
class CanonicalModel:
    model: KMeans
    remap: np.ndarray
    backbone: np.ndarray
    strict_features: np.ndarray
    strict_labels: np.ndarray
    populations: np.ndarray
    medoids: np.ndarray
    radii_90: np.ndarray
    pca: PCA


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have finite positive mass")
    return weights / total


def helicity(angles: np.ndarray, names: np.ndarray) -> np.ndarray:
    degrees = np.rad2deg(np.asarray(angles))
    phi = degrees[:, names == "phi"]
    psi = degrees[:, names == "psi"]
    alpha = (
        (phi >= -100.0)
        & (phi <= -30.0)
        & (psi >= -80.0)
        & (psi <= -5.0)
    )
    return np.mean(alpha, axis=1)


def torus_embedding(angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles)
    return np.concatenate((np.cos(angles), np.sin(angles)), axis=1).astype(
        np.float32, copy=False
    )


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return np.maximum(
        np.sum(np.square(left), axis=1)[:, None]
        + np.sum(np.square(right), axis=1)[None, :]
        - 2.0 * left @ right.T,
        0.0,
    )


def fit_canonical_model(
    canonical_angles: np.ndarray,
    names: np.ndarray,
) -> CanonicalModel:
    backbone = np.isin(names, ("phi", "psi"))
    strict = helicity(canonical_angles, names) >= PRIMARY_HELIX_THRESHOLD
    features = torus_embedding(canonical_angles[strict][:, backbone])
    model = KMeans(
        n_clusters=FAMILY_COUNT,
        n_init=30,
        max_iter=500,
        random_state=MODEL_SEED,
    ).fit(features)
    original_counts = np.bincount(model.labels_, minlength=FAMILY_COUNT)
    order = np.argsort(-original_counts)
    remap = np.empty(FAMILY_COUNT, dtype=np.int64)
    remap[order] = np.arange(FAMILY_COUNT)
    labels = remap[model.labels_]
    populations = np.bincount(labels, minlength=FAMILY_COUNT).astype(float)
    populations /= len(labels)

    ordered_centers = model.cluster_centers_[order]
    distance_to_center = squared_distances(features, ordered_centers)
    medoids = np.empty((FAMILY_COUNT, features.shape[1]), dtype=np.float32)
    radii_90 = np.empty(FAMILY_COUNT, dtype=np.float64)
    for family in range(FAMILY_COUNT):
        members = np.flatnonzero(labels == family)
        medoid_index = members[np.argmin(distance_to_center[members, family])]
        medoids[family] = features[medoid_index]
        local_distance = np.sqrt(
            np.sum(np.square(features[members] - medoids[family]), axis=1)
        )
        radii_90[family] = float(np.quantile(local_distance, 0.90))

    pca = PCA(n_components=2, random_state=MODEL_SEED).fit(features)
    return CanonicalModel(
        model=model,
        remap=remap,
        backbone=backbone,
        strict_features=features,
        strict_labels=labels,
        populations=populations,
        medoids=medoids,
        radii_90=radii_90,
        pca=pca,
    )


def classify_strict(
    coordinates: np.ndarray,
    names: np.ndarray,
    canonical: CanonicalModel,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates)
    strict = helicity(coordinates, names) >= PRIMARY_HELIX_THRESHOLD
    features = torus_embedding(coordinates[strict][:, canonical.backbone])
    labels = (
        canonical.remap[canonical.model.predict(features)]
        if len(features)
        else np.empty(0, dtype=np.int64)
    )
    return strict, labels
