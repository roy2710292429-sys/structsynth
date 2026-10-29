import json

from structsynth.evaluate import evaluate_arms
from structsynth.llm import MockLLM, extract_json
from structsynth.models import SeedQuestion
from structsynth.pipeline import (
    attach_embeddings,
    generate_arm,
    induce_reasoning_profile,
    induce_template,
    segment_questions,
)
from structsynth.spectral import (
    candidates_from_spectrum,
    decompose_and_cluster,
)


def test_extract_json_handles_fences_and_prefix():
    assert extract_json('```json\n{"ok": true}\n```') == {"ok": True}
    assert extract_json('Result: {"ok": 1}') == {"ok": 1}


def test_full_mock_pipeline_is_runnable():
    llm = MockLLM()
    seeds = [
        SeedQuestion("q1", "若 x>0，求 x+1/x 的最小值，并说明理由。"),
        SeedQuestion("q2", "设 a>0，求 a+4/a 的最小值，并给出依据。"),
        SeedQuestion("q3", "当 t>0 时，求 t+9/t 的最小值，并解释判断过程。"),
    ]
    segments = segment_questions(llm, seeds)
    assert len(segments) >= 6
    attach_embeddings(llm, segments)
    result = decompose_and_cluster(segments)
    candidates = candidates_from_spectrum(segments, result)
    assert candidates
    profile = induce_reasoning_profile(llm, seeds)
    assert 1 <= profile["depth_level"] <= 5
    assert profile["operations"]
    template = induce_template(
        llm,
        candidates,
        source="spectral",
        reasoning_profile=profile,
    )
    assert [slot.id for slot in template.slots] == [
        f"S{index:02d}" for index in range(1, len(template.slots) + 1)
    ]
    assert all(slot.source_unit_ids for slot in template.slots)
    assert template.reasoning_profile["operations"]
    generated = generate_arm(
        llm,
        "spectral",
        seeds,
        3,
        template=template,
        candidates=candidates,
    )
    assert len(generated) == 3
    assert all(question.slot_texts for question in generated)
    evaluation = evaluate_arms(llm, template, {"spectral": generated})
    assert evaluation["aggregate"]["spectral"]["n"] == 3
    assert "reasoning_depth_alignment" in evaluation["aggregate"]["spectral"]
    assert "structural_similarity" in evaluation["aggregate"]["spectral"]
    json.dumps(evaluation, ensure_ascii=False)
