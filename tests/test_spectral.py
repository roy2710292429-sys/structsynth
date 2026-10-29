import numpy as np

from structsynth.llm import MockLLM
from structsynth.models import Segment
from structsynth.spectral import (
    SpectralConfig,
    build_contrast,
    candidates_from_spectrum,
    decompose_and_cluster,
)


def _segments() -> list[Segment]:
    texts = [
        "已知函数",
        "在给定区间",
        "求最大值",
        "设另一个函数",
        "在指定范围",
        "判断最小值",
        "给定表达式",
        "分析单调性",
        "说明理由",
    ]
    llm = MockLLM()
    vectors = llm.embed(texts)
    return [
        Segment(
            id=f"q{index // 3 + 1}:s{index % 3 + 1}",
            question_id=f"q{index // 3 + 1}",
            order=index % 3 + 1,
            text=text,
            embedding=vector,
        )
        for index, (text, vector) in enumerate(zip(texts, vectors))
    ]


def test_contrast_is_symmetric_and_masks_same_question():
    segments = _segments()
    config = SpectralConfig(mask_same_question=True, normalize_affinity=False)
    matrix, null, contrast = build_contrast(segments, config)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(null, null.T)
    assert np.allclose(contrast, 2.0 * matrix - null)
    assert np.allclose(np.diag(matrix), 0.0)
    question_ids = np.array([segment.question_id for segment in segments])
    assert np.count_nonzero(
        matrix[question_ids[:, None] == question_ids[None, :]]
    ) == 0


def test_spectral_candidates_have_traceable_segments():
    segments = _segments()
    result = decompose_and_cluster(
        segments,
        SpectralConfig(components=3, cluster_count=3, activation_quantile=0.5),
    )
    candidates = candidates_from_spectrum(segments, result)
    assert candidates
    known_ids = {segment.id for segment in segments}
    for candidate in candidates:
        assert candidate.segment_ids
        assert set(candidate.segment_ids).issubset(known_ids)
