from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .llm import LLM, extract_json
from .models import CandidateUnit, GeneratedQuestion, SeedQuestion, Segment, Template, TemplateSlot
from .prompts import (
    JUDGE_SYSTEM,
    PROFILE_SYSTEM,
    SEGMENT_SYSTEM,
    SLOT_SYSTEM,
    generation_system,
    generation_user,
    judge_user,
    profile_user,
    segment_user,
    slot_user,
)


def segment_questions(llm: LLM, seeds: list[SeedQuestion]) -> list[Segment]:
    payload = [{"id": seed.id, "text": seed.text} for seed in seeds]
    response = extract_json(
        llm.chat(SEGMENT_SYSTEM, segment_user(payload), json_mode=True)
    )
    rows = response.get("questions")
    if not isinstance(rows, list):
        raise ValueError("Segmentation response is missing a questions list.")

    by_id = {str(row.get("question_id")): row for row in rows if isinstance(row, dict)}
    segments: list[Segment] = []
    for seed in seeds:
        row = by_id.get(seed.id)
        rows_for_question = row.get("segments") if row else None
        if not isinstance(rows_for_question, list) or not rows_for_question:
            raise ValueError(f"Question {seed.id} was not segmented correctly.")
        for order, segment_row in enumerate(rows_for_question, start=1):
            if isinstance(segment_row, dict):
                clean = str(segment_row.get("text", "")).strip()
                role = str(segment_row.get("role", "")).strip()
                operation = str(segment_row.get("operation", "")).strip()
                abstraction = str(segment_row.get("abstraction", "")).strip()
            else:
                clean = str(segment_row).strip()
                role = operation = abstraction = ""
            if not clean:
                raise ValueError(f"Question {seed.id} has an empty segment.")
            segments.append(
                Segment(
                    id=f"{seed.id}:s{order:02d}",
                    question_id=seed.id,
                    order=order,
                    text=clean,
                    role=role,
                    operation=operation,
                    abstraction=abstraction,
                )
            )
    return segments


def attach_embeddings(llm: LLM, segments: list[Segment]) -> None:
    vectors = llm.embed([segment.text for segment in segments])
    if len(vectors) != len(segments):
        raise ValueError(
            f"Embedding count mismatch: got {len(vectors)} for {len(segments)} segments."
        )
    if not vectors or not vectors[0]:
        raise ValueError("Embedding API returned empty vectors.")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("Embedding API returned inconsistent vector dimensions.")
    for segment, vector in zip(segments, vectors):
        segment.embedding = vector


def induce_reasoning_profile(
    llm: LLM,
    seeds: list[SeedQuestion],
) -> dict[str, Any]:
    payload = [
        {
            "id": seed.id,
            "text": seed.text,
            "type": seed.type,
            "difficulty": seed.difficulty,
        }
        for seed in seeds
    ]
    response = extract_json(
        llm.chat(PROFILE_SYSTEM, profile_user(payload), json_mode=True)
    )
    try:
        depth = int(response.get("depth_level", 3))
    except (TypeError, ValueError):
        depth = 3
    response["depth_level"] = max(1, min(5, depth))
    operations = response.get("operations", [])
    response["operations"] = (
        [str(item) for item in operations] if isinstance(operations, list) else []
    )
    return response


def induce_template(
    llm: LLM,
    candidates: list[CandidateUnit],
    *,
    source: str,
    reasoning_profile: dict[str, Any] | None = None,
) -> Template:
    candidate_payload = [candidate.to_dict() for candidate in candidates]
    response = extract_json(
        llm.chat(
            SLOT_SYSTEM,
            slot_user(candidate_payload, reasoning_profile=reasoning_profile),
            json_mode=True,
        )
    )
    rows = response.get("slots")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Template induction returned no slots.")

    ordered = sorted(
        (row for row in rows if isinstance(row, dict)),
        key=lambda row: (float(row.get("order", 999)), str(row.get("id", ""))),
    )
    slots: list[TemplateSlot] = []
    known_unit_ids = {candidate.id for candidate in candidates}
    for row in ordered:
        raw_sources = row.get("source_unit_ids", [])
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        source_ids = [
            str(unit_id)
            for unit_id in raw_sources
            if str(unit_id) in known_unit_ids
        ]
        if not source_ids:
            continue
        slots.append(
            TemplateSlot(
                id=f"S{len(slots) + 1:02d}",
                name=str(row.get("name") or f"槽位{len(slots) + 1}"),
                definition=str(row.get("definition") or ""),
                required=bool(row.get("required", True)),
                source_unit_ids=source_ids,
            )
        )
    if not slots:
        raise ValueError("Template induction returned only malformed slots.")
    invariants = response.get("invariants", [])
    if not isinstance(invariants, list):
        invariants = []
    return Template(
        name=str(response.get("name") or "自动归纳模板"),
        description=str(response.get("description") or ""),
        slots=slots,
        invariants=[str(item) for item in invariants],
        reasoning_profile=reasoning_profile or {},
        source=source,
    )


def generate_arm(
    llm: LLM,
    arm: str,
    seeds: list[SeedQuestion],
    count: int,
    *,
    template: Template | None,
    candidates: list[CandidateUnit] | None,
    temperature: float = 0.8,
) -> list[GeneratedQuestion]:
    seed_payload = [
        {"id": seed.id, "text": seed.text, "type": seed.type} for seed in seeds
    ]
    template_payload = template.to_dict() if template else None
    candidate_payload = (
        [candidate.to_dict() for candidate in candidates] if candidates else None
    )
    response = extract_json(
        llm.chat(
            generation_system(arm),
            generation_user(
                arm,
                seed_payload,
                count,
                template_payload,
                candidate_payload,
            ),
            temperature=temperature,
            json_mode=True,
        )
    )
    rows = response.get("questions")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError(
            f"Generator for {arm} returned {len(rows) if isinstance(rows, list) else 'no'} "
            f"questions; expected {count}."
        )
    generated: list[GeneratedQuestion] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or not str(row.get("text", "")).strip():
            raise ValueError(f"Generator for {arm} returned malformed question {index}.")
        raw_slots = row.get("slot_texts", {})
        slot_texts = (
            {str(key): str(value) for key, value in raw_slots.items()}
            if isinstance(raw_slots, dict)
            else {}
        )
        try:
            depth_level = int(row.get("depth_level"))
        except (TypeError, ValueError):
            depth_level = None
        outline = row.get("reasoning_outline", [])
        generated.append(
            GeneratedQuestion(
                id=f"{arm}:g{index:03d}",
                arm=arm,
                text=str(row["text"]).strip(),
                slot_texts=slot_texts,
                question_type=str(row.get("type") or "unknown"),
                depth_level=depth_level,
                reasoning_outline=(
                    [str(item) for item in outline]
                    if isinstance(outline, list)
                    else []
                ),
            )
        )
    return generated


def judge_questions(
    llm: LLM,
    template: Template,
    questions: list[GeneratedQuestion],
) -> dict[str, dict[str, Any]]:
    payload = [
        {
            "id": question.id,
            "text": question.text,
            "slot_texts": question.slot_texts,
            "depth_level": question.depth_level,
            "reasoning_outline": question.reasoning_outline,
        }
        for question in questions
    ]
    response = extract_json(
        llm.chat(
            JUDGE_SYSTEM,
            judge_user(template.to_dict(), payload),
            temperature=0.0,
            json_mode=True,
        )
    )
    rows = response.get("results")
    if not isinstance(rows, list):
        raise ValueError("Judge response is missing a results list.")
    results: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("question_id")) in {
            question.id for question in questions
        }:
            row["coverage"] = min(1.0, max(0.0, float(row.get("coverage", 0.0))))
            row["isomorphic"] = bool(row.get("isomorphic", False))
            results[str(row["question_id"])] = row
    missing = [question.id for question in questions if question.id not in results]
    if missing:
        raise ValueError(f"Judge omitted generated questions: {', '.join(missing)}")
    return results


def serialize_segments(segments: list[Segment]) -> list[dict[str, Any]]:
    return [segment.to_dict(include_embedding=False) for segment in segments]


def serialize_generations(generations: list[GeneratedQuestion]) -> list[dict[str, Any]]:
    return [asdict(question) for question in generations]


def serialize_template(template: Template) -> dict[str, Any]:
    return template.to_dict()
