from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .evaluate import evaluate_arms
from .llm import LLM, MockLLM, OpenAICompatibleLLM
from .models import SeedQuestion
from .pipeline import (
    attach_embeddings,
    generate_arm,
    induce_reasoning_profile,
    induce_template,
    segment_questions,
    serialize_generations,
    serialize_segments,
)
from .spectral import (
    SpectralConfig,
    candidates_from_m_only,
    candidates_from_spectrum,
    decompose_and_cluster,
    spectral_diagnostics,
)

ProgressCallback = Callable[[str, int, str], None]


@dataclass
class APISettings:
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    chat_model: str = ""
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    timeout: float = 120.0


@dataclass
class ExperimentSettings:
    provider: str = "mock"
    count: int = 8
    components: int = 8
    clusters: int = 8
    activation_quantile: float = 0.80
    neighbor_fraction: float = 0.35
    temperature: float = 0.80
    api: APISettings | None = None


def build_llm(settings: ExperimentSettings) -> LLM:
    if settings.provider == "mock":
        return MockLLM()
    if settings.provider != "openai":
        raise ValueError(f"Unsupported provider: {settings.provider}")
    api = settings.api or APISettings()
    if api.api_key:
        if not api.chat_model or not api.embedding_model:
            raise ValueError("Chat and embedding model names are required.")
        embedding_base_url = api.embedding_base_url or api.base_url
        same_origin = embedding_base_url.rstrip("/") == api.base_url.rstrip("/")
        embedding_api_key = api.embedding_api_key or (
            api.api_key if same_origin else ""
        )
        if not embedding_api_key:
            raise ValueError(
                "Chat and embedding use different endpoints; "
                "an embedding API key is required."
            )
        return OpenAICompatibleLLM(
            base_url=api.base_url,
            api_key=api.api_key,
            chat_model=api.chat_model,
            embedding_model=api.embedding_model,
            timeout=api.timeout,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
        )
    return OpenAICompatibleLLM.from_env()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def markdown_report(evaluation: dict[str, Any]) -> str:
    lines = [
        "# StructSynth 评测结果",
        "",
        "> 这是管线验收报告。小样本或离线 mock 的结果不能代表生成质量。",
        "",
        "| 条件 | n | 结构族命中 | 连续结构相似 | 思维深度对齐 | 槽位覆盖 | 向量多样性 | 唯一题率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, metrics in evaluation["aggregate"].items():
        lines.append(
            f"| {arm} | {metrics['n']} | {metrics['family_match_rate']:.3f} | "
            f"{metrics['structural_similarity']:.3f} | "
            f"{metrics['reasoning_depth_alignment']:.3f} | "
            f"{metrics['slot_coverage']:.3f} | "
            f"{metrics['embedding_diversity']:.3f} | "
            f"{metrics['unique_question_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 结构族命中、连续结构相似与思维深度对齐来自 blind-style LLM judge，仍需人工盲评校准。",
            "- 连续结构相似度与思维深度对齐允许题面灵活变化，不按专有名词或数字重合度计算。",
            "- 多样性是两两余弦距离的均值，越高只代表越不相似，不代表越有价值。",
            "- 若三组共用一个由谱方法归纳的模板，judge 可能偏向谱组；正式实验需独立参考标注或留出集。",
        ]
    )
    return "\n".join(lines) + "\n"


def _notify(
    callback: ProgressCallback | None,
    stage: str,
    percent: int,
    detail: str,
) -> None:
    if callback:
        callback(stage, max(0, min(100, percent)), detail)


def run_experiment(
    seeds: list[SeedQuestion],
    settings: ExperimentSettings,
    *,
    output_dir: Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if len(seeds) < 3:
        raise ValueError("At least three seed questions are required.")
    llm = build_llm(settings)
    result: dict[str, Any] = {
        "provider": settings.provider,
        "settings": {
            "count": settings.count,
            "components": settings.components,
            "clusters": settings.clusters,
            "activation_quantile": settings.activation_quantile,
            "neighbor_fraction": settings.neighbor_fraction,
            "temperature": settings.temperature,
        },
    }

    _notify(progress, "segment", 8, "正在把种子题切成结构语段")
    segments = segment_questions(llm, seeds)
    attach_embeddings(llm, segments)
    segments_json = serialize_segments(segments)
    result["segments"] = segments_json
    if output_dir:
        write_json(output_dir / "segments.json", segments_json)

    _notify(progress, "profile", 24, "正在归纳思维深度、依赖链和主要瓶颈")
    reasoning_profile = induce_reasoning_profile(llm, seeds)
    result["reasoning_profile"] = reasoning_profile

    _notify(progress, "spectral", 31, "正在构造语义与角色结构的多视图矩阵 M")
    spectral_config = SpectralConfig(
        components=settings.components,
        cluster_count=settings.clusters,
        activation_quantile=settings.activation_quantile,
        neighbor_fraction=settings.neighbor_fraction,
    )
    spectral_result = decompose_and_cluster(segments, spectral_config)
    spectral_candidates = candidates_from_spectrum(segments, spectral_result)
    if not spectral_candidates:
        raise ValueError("Spectral clustering produced no candidate semantic units.")
    spectral_json = {
        "config": asdict(spectral_config),
        "diagnostics": spectral_diagnostics(segments, spectral_result),
        "candidates": [candidate.to_dict() for candidate in spectral_candidates],
    }
    result["spectral"] = spectral_json
    if output_dir:
        write_json(output_dir / "spectral.json", spectral_json)

    _notify(progress, "matrix_baseline", 42, "正在构造仅使用 M 的对照语段组")
    matrix_candidates = candidates_from_m_only(
        segments,
        cluster_count=min(settings.clusters, max(2, len(segments) // 3)),
    )

    _notify(progress, "template", 52, "正在把候选语段归纳成软模板槽位")
    spectral_template = induce_template(
        llm,
        spectral_candidates,
        source="spectral",
        reasoning_profile=reasoning_profile,
    )
    matrix_template = induce_template(
        llm,
        matrix_candidates,
        source="matrix",
        reasoning_profile=reasoning_profile,
    )
    templates_json = {
        "spectral": spectral_template.to_dict(),
        "matrix": matrix_template.to_dict(),
    }
    result["templates"] = templates_json
    if output_dir:
        write_json(output_dir / "templates.json", templates_json)

    generations = {}
    _notify(progress, "generate_baseline", 61, "正在生成普通仿写基线")
    generations["baseline"] = generate_arm(
        llm,
        "baseline",
        seeds,
        settings.count,
        template=None,
        candidates=None,
        temperature=settings.temperature,
    )
    _notify(progress, "generate_matrix", 70, "正在生成 M-only 对照组")
    generations["matrix"] = generate_arm(
        llm,
        "matrix",
        seeds,
        settings.count,
        template=matrix_template,
        candidates=matrix_candidates,
        temperature=settings.temperature,
    )
    _notify(progress, "generate_spectral", 79, "正在生成谱模板条件组")
    generations["spectral"] = generate_arm(
        llm,
        "spectral",
        seeds,
        settings.count,
        template=spectral_template,
        candidates=spectral_candidates,
        temperature=settings.temperature,
    )
    generation_rows = {
        arm: serialize_generations(rows) for arm, rows in generations.items()
    }
    result["generations"] = generation_rows
    if output_dir:
        write_json(output_dir / "generations.json", generation_rows)

    _notify(progress, "evaluate", 89, "正在盲评结构同构与槽位覆盖")
    evaluation = evaluate_arms(llm, spectral_template, generations)
    result["evaluation"] = evaluation
    if output_dir:
        write_json(output_dir / "evaluation.json", evaluation)
        (output_dir / "report.md").write_text(
            markdown_report(evaluation), encoding="utf-8"
        )

    _notify(progress, "done", 100, "实验完成")
    return result
