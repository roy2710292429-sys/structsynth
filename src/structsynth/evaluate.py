from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .llm import LLM
from .models import GeneratedQuestion, Template
from .pipeline import judge_questions


def _cosine_diversity(vectors: list[list[float]]) -> float:
    if len(vectors) < 2:
        return 0.0
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    similarity = matrix @ matrix.T
    upper = similarity[np.triu_indices(len(vectors), k=1)]
    return float(np.clip(1.0 - np.mean(upper), 0.0, 2.0) / 2.0)


def _lexical_diversity(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    normalized = [set(text) for text in texts]
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            union = normalized[left] | normalized[right]
            total += 1.0 - (
                len(normalized[left] & normalized[right]) / max(1, len(union))
            )
            pairs += 1
    return total / max(1, pairs)


def evaluate_arms(
    llm: LLM,
    template: Template,
    generations: dict[str, list[GeneratedQuestion]],
) -> dict[str, Any]:
    flat = [question for group in generations.values() for question in group]
    judgments: dict[str, dict[str, Any]] = {}
    # Judge in small blocks to reduce prompt size and isolate malformed responses.
    for start in range(0, len(flat), 20):
        judgments.update(judge_questions(llm, template, flat[start : start + 20]))

    all_vectors = llm.embed([question.text for question in flat])
    vector_by_id = {
        question.id: vector for question, vector in zip(flat, all_vectors)
    }
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    detail: list[dict[str, Any]] = []
    for question in flat:
        judgment = judgments[question.id]
        structural_similarity = min(
            1.0, max(0.0, float(judgment.get("structural_similarity", 0.0)))
        )
        cognitive_coverage = min(
            1.0, max(0.0, float(judgment.get("cognitive_coverage", 0.0)))
        )
        depth_alignment = min(
            1.0, max(0.0, float(judgment.get("depth_alignment", 0.0)))
        )
        row = {
            **question.to_dict(),
            "isomorphic": judgment["isomorphic"],
            "same_family": bool(judgment.get("same_family", False)),
            "structural_similarity": structural_similarity,
            "cognitive_coverage": cognitive_coverage,
            "depth_score": int(judgment.get("depth_score", 0) or 0),
            "depth_alignment": depth_alignment,
            "coverage": judgment["coverage"],
            "matched_slot_ids": judgment.get("matched_slot_ids", []),
            "missing_slot_ids": judgment.get("missing_slot_ids", []),
            "distorted_slot_ids": judgment.get("distorted_slot_ids", []),
            "judge_reason": judgment.get("reason", ""),
        }
        by_arm[question.arm].append(row)
        detail.append(row)

    aggregate: dict[str, Any] = {}
    for arm, rows in sorted(by_arm.items()):
        aggregate[arm] = {
            "n": len(rows),
            "isomorphism_rate": float(np.mean([row["isomorphic"] for row in rows])),
            "family_match_rate": float(
                np.mean([row["same_family"] for row in rows])
            ),
            "structural_similarity": float(
                np.mean([row["structural_similarity"] for row in rows])
            ),
            "slot_coverage": float(np.mean([row["coverage"] for row in rows])),
            "cognitive_coverage": float(
                np.mean([row["cognitive_coverage"] for row in rows])
            ),
            "reasoning_depth_alignment": float(
                np.mean([row["depth_alignment"] for row in rows])
            ),
            "embedding_diversity": _cosine_diversity(
                [vector_by_id[row["id"]] for row in rows]
            ),
            "lexical_diversity": _lexical_diversity([row["text"] for row in rows]),
            "unique_question_rate": len({row["text"].strip() for row in rows})
            / max(1, len(rows)),
        }
    return {
        "template_id": template.name,
        "warning": (
            "LLM judge scores are proxies. Structural similarity intentionally tolerates "
            "surface paraphrase; validate depth alignment and family membership with "
            "blind domain-expert review before making claims."
        ),
        "aggregate": aggregate,
        "details": detail,
    }
