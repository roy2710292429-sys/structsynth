from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import CandidateUnit, Segment


@dataclass
class SpectralConfig:
    components: int = 8
    cluster_count: int = 8
    activation_quantile: float = 0.80
    neighbor_fraction: float = 0.35
    mask_same_question: bool = True
    normalize_affinity: bool = True
    random_state: int = 17


@dataclass
class SpectralResult:
    matrix: np.ndarray
    null_matrix: np.ndarray
    contrast: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    selected_indices: np.ndarray
    clusters: np.ndarray


def _l2_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _segment_matrix(segments: list[Segment]) -> np.ndarray:
    if len(segments) < 2:
        raise ValueError("At least two segments are required for spectral induction.")
    embeddings = [segment.embedding for segment in segments]
    if any(vector is None for vector in embeddings):
        raise ValueError("Every segment must have an embedding.")
    matrix = np.asarray(embeddings, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Segment embeddings contain NaN or infinity.")
    return matrix


def _structural_similarity(segments: list[Segment]) -> np.ndarray:
    """Soft role/operation similarity, robust to paraphrased surface wording."""
    count = len(segments)
    similarity = np.zeros((count, count), dtype=float)
    token_sets: list[set[str]] = []
    for segment in segments:
        label = "|".join(
            value
            for value in (segment.role, segment.operation, segment.abstraction)
            if value
        )
        compact = "".join(
            character.lower()
            for character in label
            if character.isalnum() and not character.isspace()
        )
        if not compact:
            token_sets.append(set())
            continue
        tokens = {compact[index : index + 2] for index in range(len(compact) - 1)}
        tokens.update(label.lower().split())
        token_sets.append(tokens)

    for left in range(count):
        if not token_sets[left]:
            continue
        for right in range(left + 1, count):
            if not token_sets[right]:
                continue
            union = token_sets[left] | token_sets[right]
            score = (
                len(token_sets[left] & token_sets[right]) / len(union)
                if union
                else 0.0
            )
            similarity[left, right] = similarity[right, left] = score
    return similarity


def _sparsify_neighbors(
    similarity: np.ndarray,
    segments: list[Segment],
    neighbor_fraction: float,
    mask_same_question: bool,
) -> np.ndarray:
    """Keep a per-row local neighborhood instead of applying one global threshold."""
    count = len(similarity)
    question_ids = np.asarray([segment.question_id for segment in segments])
    eligible = np.ones((count, count), dtype=bool)
    np.fill_diagonal(eligible, False)
    if mask_same_question:
        eligible &= question_ids[:, None] != question_ids[None, :]
    fraction = float(np.clip(neighbor_fraction, 0.10, 0.90))
    directed = np.zeros_like(similarity)
    for row in range(count):
        candidates = np.flatnonzero(eligible[row])
        if not len(candidates):
            continue
        keep = max(1, int(np.ceil(len(candidates) * fraction)))
        ranked = candidates[np.argsort(similarity[row, candidates])[::-1][:keep]]
        directed[row, ranked] = similarity[row, ranked]
    mask = (directed != 0) | (directed.T != 0)
    sparse = np.where(mask, similarity, 0.0)
    return 0.5 * (sparse + sparse.T)


def _kmeans(
    points: np.ndarray,
    n_clusters: int,
    *,
    n_init: int = 20,
    max_iter: int = 150,
    random_state: int = 17,
) -> np.ndarray:
    """Deterministic k-means++ with a few restarts and no compiled ML dependency."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("K-means requires a non-empty two-dimensional array.")
    n_clusters = max(1, min(n_clusters, len(points)))
    if n_clusters == 1:
        return np.zeros(len(points), dtype=int)

    rng = np.random.default_rng(random_state)
    best_labels: np.ndarray | None = None
    best_inertia = math.inf

    for _ in range(max(1, n_init)):
        centers = np.empty((n_clusters, points.shape[1]), dtype=float)
        first = int(rng.integers(0, len(points)))
        centers[0] = points[first]
        closest = np.sum((points - centers[0]) ** 2, axis=1)
        for center_index in range(1, n_clusters):
            weights = np.maximum(closest, 1e-12)
            cumulative = np.cumsum(weights)
            position = float(rng.random() * cumulative[-1])
            chosen = int(np.searchsorted(cumulative, position, side="right"))
            chosen = min(chosen, len(points) - 1)
            centers[center_index] = points[chosen]
            distance = np.sum((points - centers[center_index]) ** 2, axis=1)
            closest = np.minimum(closest, distance)

        labels = np.full(len(points), -1, dtype=int)
        for _iteration in range(max_iter):
            distances = np.sum(
                (points[:, None, :] - centers[None, :, :]) ** 2, axis=2
            )
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            for cluster_id in range(n_clusters):
                members = np.flatnonzero(labels == cluster_id)
                if len(members):
                    centers[cluster_id] = points[members].mean(axis=0)
                else:
                    # Re-seed an empty cluster at the currently worst-fit point.
                    assigned_distance = distances[np.arange(len(points)), labels]
                    replacement = int(np.argmax(assigned_distance))
                    centers[cluster_id] = points[replacement]

        distances = np.sum(
            (points[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )
        labels = np.argmin(distances, axis=1)
        inertia = float(np.min(distances, axis=1).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    assert best_labels is not None
    return best_labels


def build_contrast(
    segments: list[Segment],
    config: SpectralConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build M, its degree-corrected null N, and A=2 sym(M)-sym(N).

    M combines semantic similarity with soft role/operation similarity. It keeps
    only each segment's strongest cross-question neighbors, so paraphrases need
    not pass a single global similarity threshold. N preserves observed row
    strengths but removes pairwise communities.
    """
    embeddings = _l2_rows(_segment_matrix(segments))
    semantic = embeddings @ embeddings.T
    semantic = 0.5 * (semantic + semantic.T)
    structural = _structural_similarity(segments)
    has_structure = bool(np.any(structural))
    similarity = (
        0.68 * semantic + 0.32 * structural if has_structure else semantic
    )
    similarity = _sparsify_neighbors(
        similarity,
        segments,
        config.neighbor_fraction,
        config.mask_same_question,
    )

    strength = np.maximum(similarity.sum(axis=1), 1e-8)
    total = max(float(strength.sum()), 1e-8)
    null = np.outer(strength, strength) / total
    np.fill_diagonal(null, 0.0)
    null = 0.5 * (null + null.T)
    contrast = 2.0 * similarity - null
    contrast = 0.5 * (contrast + contrast.T)

    if config.normalize_affinity:
        degree = np.sum(np.abs(contrast), axis=1)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
        contrast = inv_sqrt[:, None] * contrast * inv_sqrt[None, :]
        contrast = 0.5 * (contrast + contrast.T)
    return similarity, null, contrast


def decompose_and_cluster(
    segments: list[Segment],
    config: SpectralConfig | None = None,
) -> SpectralResult:
    config = config or SpectralConfig()
    count = len(segments)
    components = max(1, min(config.components, count - 1))
    matrix, null, contrast = build_contrast(segments, config)
    eigenvalues, eigenvectors = np.linalg.eigh(contrast)
    order = np.argsort(eigenvalues)[::-1]
    # Using the leading positive spectrum keeps the contrast's semantic signal.
    positive = [int(index) for index in order if eigenvalues[index] > 1e-10]
    chosen = positive[:components]
    if len(chosen) < components:
        remainder = [int(index) for index in order if int(index) not in chosen]
        chosen.extend(remainder[: components - len(chosen)])
    chosen = chosen[:components]
    selected_vectors = eigenvectors[:, chosen]
    selected_values = eigenvalues[chosen]

    row_scores = np.max(np.abs(selected_vectors), axis=1)
    threshold = float(np.quantile(row_scores, config.activation_quantile))
    selected = np.flatnonzero(row_scores >= threshold)
    if len(selected) < 2:
        selected = np.argsort(row_scores)[-min(count, max(2, components)) :]

    spectral_rows = _l2_rows(selected_vectors[selected])
    clusters_wanted = max(1, min(config.cluster_count, len(selected)))
    if clusters_wanted == 1:
        labels = np.zeros(len(selected), dtype=int)
    else:
        labels = _kmeans(
            spectral_rows,
            clusters_wanted,
            n_init=20,
            random_state=config.random_state,
        )

    return SpectralResult(
        matrix=matrix,
        null_matrix=null,
        contrast=contrast,
        eigenvalues=selected_values,
        eigenvectors=selected_vectors,
        selected_indices=selected,
        clusters=labels,
    )


def candidates_from_spectrum(
    segments: list[Segment],
    result: SpectralResult,
) -> list[CandidateUnit]:
    candidates: list[CandidateUnit] = []
    row_scores = np.max(np.abs(result.eigenvectors), axis=1)
    for cluster_id in sorted(set(int(label) for label in result.clusters)):
        local = np.flatnonzero(result.clusters == cluster_id)
        indices = result.selected_indices[local]
        indices = indices[np.argsort(row_scores[indices])[::-1]]
        chosen_segments = [segments[int(index)] for index in indices[:12]]
        if not chosen_segments:
            continue
        relative_orders = []
        for segment in chosen_segments:
            siblings = [item for item in segments if item.question_id == segment.question_id]
            denominator = max(1, len(siblings) - 1)
            relative_orders.append((segment.order - 1) / denominator)
        activation = float(np.mean(row_scores[indices]))
        candidates.append(
            CandidateUnit(
                id=f"U{len(candidates) + 1:02d}",
                label=f"候选语义单元{len(candidates) + 1}",
                definition="待 LLM 归纳",
                segment_ids=[segment.id for segment in chosen_segments],
                exemplars=[segment.text for segment in chosen_segments[:5]],
                activation=activation,
                order_hint=float(np.mean(relative_orders)),
            )
        )
    return sorted(candidates, key=lambda candidate: (candidate.order_hint, candidate.id))


def candidates_from_m_only(
    segments: list[Segment],
    cluster_count: int = 8,
    random_state: int = 17,
) -> list[CandidateUnit]:
    embeddings = _l2_rows(_segment_matrix(segments))
    count = max(1, min(cluster_count, len(segments)))
    if count == 1:
        labels = np.zeros(len(segments), dtype=int)
    else:
        labels = _kmeans(embeddings, count, n_init=20, random_state=random_state)

    candidates: list[CandidateUnit] = []
    for cluster_id in sorted(set(int(label) for label in labels)):
        indices = np.flatnonzero(labels == cluster_id)
        selected = [segments[int(index)] for index in indices[:12]]
        if not selected:
            continue
        relative_orders = []
        for segment in selected:
            siblings = [item for item in segments if item.question_id == segment.question_id]
            denominator = max(1, len(siblings) - 1)
            relative_orders.append((segment.order - 1) / denominator)
        candidates.append(
            CandidateUnit(
                id=f"U{len(candidates) + 1:02d}",
                label=f"候选语义单元{len(candidates) + 1}",
                definition="待 LLM 归纳",
                segment_ids=[segment.id for segment in selected],
                exemplars=[segment.text for segment in selected[:5]],
                activation=float(len(selected)),
                order_hint=float(np.mean(relative_orders)),
            )
        )
    return sorted(candidates, key=lambda candidate: (candidate.order_hint, candidate.id))


def spectral_diagnostics(
    segments: list[Segment],
    result: SpectralResult,
) -> dict[str, Any]:
    top_by_component = []
    for component in range(result.eigenvectors.shape[1]):
        vector = result.eigenvectors[:, component]
        top = np.argsort(np.abs(vector))[::-1][:8]
        top_by_component.append(
            {
                "eigenvalue": float(result.eigenvalues[component]),
                "segments": [
                    {
                        "id": segments[int(index)].id,
                        "text": segments[int(index)].text,
                        "activation": float(abs(vector[int(index)])),
                    }
                    for index in top
                ],
            }
        )
    return {
        "n_segments": len(segments),
        "n_components": int(result.eigenvectors.shape[1]),
        "n_high_activation_segments": int(len(result.selected_indices)),
        "n_clusters": int(len(set(int(x) for x in result.clusters))),
        "top_by_component": top_by_component,
        "matrix_stats": {
            "M_mean_offdiag": float(
                result.matrix.sum() / max(1, len(segments) * (len(segments) - 1))
            ),
            "N_mean_offdiag": float(
                result.null_matrix.sum()
                / max(1, len(segments) * (len(segments) - 1))
            ),
            "A_min": float(np.min(result.contrast)),
            "A_max": float(np.max(result.contrast)),
        },
    }


def safe_finite(value: float, digits: int = 6) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(value, digits)
