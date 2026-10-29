from __future__ import annotations

import json
import mimetypes
import os
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .experiment import APISettings, ExperimentSettings, run_experiment
from .models import SeedQuestion

STATIC_DIR = Path(__file__).with_name("web_static")
MAX_REQUEST_BYTES = 2 * 1024 * 1024
DOWNLOAD_FILES = {
    "segments.json",
    "spectral.json",
    "templates.json",
    "generations.json",
    "evaluation.json",
    "report.md",
}


@dataclass
class Job:
    id: str
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    message: str = "等待启动"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    output_dir: str = ""
    logs: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None

    def snapshot(self, include_result: bool = True) -> dict[str, Any]:
        result = {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "logs": self.logs[-40:],
            "error": self.error,
        }
        if include_result and self.status == "complete":
            result["result"] = self.result
            result["downloads"] = sorted(DOWNLOAD_FILES)
        return result


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, output_dir: Path | None = None) -> Job:
        job_id = uuid.uuid4().hex[:12]
        if output_dir is None:
            output_dir = Path.cwd() / "outputs" / "ui" / job_id
        job = Job(id=job_id, output_dir=str(output_dir))
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = time.time()

    def append_log(self, job_id: str, stage: str, percent: int, detail: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.stage = stage
            job.progress = percent
            job.message = detail
            job.updated_at = time.time()
            job.logs.append(
                {
                    "time": time.strftime("%H:%M:%S"),
                    "stage": stage,
                    "percent": percent,
                    "detail": detail,
                }
            )


JOBS = JobStore()


def _json_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    value: Any,
) -> None:
    body = json.dumps(value, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    _json_response(handler, status, {"error": message})


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw_length = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length.") from exc
    if length <= 0:
        raise ValueError("Request body is empty.")
    if length > MAX_REQUEST_BYTES:
        raise ValueError("Request body is too large.")
    body = handler.rfile.read(length)
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON body must be an object.")
    return value


def _parse_seeds(rows: Any) -> list[SeedQuestion]:
    if not isinstance(rows, list):
        raise ValueError("seeds must be a JSON list.")
    if len(rows) < 3:
        raise ValueError("请至少输入 3 道种子题。")
    if len(rows) > 500:
        raise ValueError("一次最多输入 500 道种子题。")
    seeds = []
    for index, row in enumerate(rows):
        if isinstance(row, str):
            row = {"text": row}
        if not isinstance(row, dict) or not str(row.get("text", "")).strip():
            raise ValueError(f"第 {index + 1} 道种子题没有有效文本。")
        seeds.append(SeedQuestion.from_dict(row, index))
    return seeds


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数。") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间。")
    return parsed


def _bounded_float(
    value: Any, name: str, minimum: float, maximum: float
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字。") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间。")
    return parsed


def _settings_from_payload(payload: dict[str, Any]) -> ExperimentSettings:
    provider = str(payload.get("provider", "mock"))
    if provider not in {"mock", "openai"}:
        raise ValueError("provider 只支持 mock 或 openai。")
    api_payload = payload.get("api")
    if api_payload is not None and not isinstance(api_payload, dict):
        raise ValueError("api must be an object.")
    api_data = api_payload or {}
    api = APISettings(
        base_url=str(
            api_data.get("chat_base_url")
            or api_data.get("base_url")
            or os.getenv("QS_LLM_BASE_URL", "https://api.openai.com/v1")
        ),
        api_key=str(
            api_data.get("chat_api_key")
            or api_data.get("api_key")
            or os.getenv("QS_LLM_API_KEY", "")
        ),
        chat_model=str(api_data.get("chat_model") or os.getenv("QS_CHAT_MODEL", "")),
        embedding_model=str(
            api_data.get("embedding_model")
            or os.getenv("QS_EMBEDDING_MODEL", "")
        ),
        embedding_base_url=str(
            api_data.get("embedding_base_url")
            or os.getenv("QS_EMBEDDING_BASE_URL", "")
        ),
        embedding_api_key=str(
            api_data.get("embedding_api_key")
            or os.getenv("QS_EMBEDDING_API_KEY", "")
        ),
        timeout=_bounded_float(
            api_data.get("timeout", os.getenv("QS_LLM_TIMEOUT", 120)),
            "API timeout",
            10,
            600,
        ),
    )
    if provider == "openai":
        if not api.api_key:
            raise ValueError("真实 API 模式需要聊天 API Key。")
        if not api.chat_model:
            raise ValueError("真实 API 模式需要 Chat model。")
        if not api.embedding_model:
            raise ValueError("真实 API 模式需要 Embedding model。")
        embedding_base = api.embedding_base_url or api.base_url
        if (
            embedding_base.rstrip("/") != api.base_url.rstrip("/")
            and not api.embedding_api_key
        ):
            raise ValueError("聊天与嵌入使用不同接口时，需要独立的 Embedding API Key。")
    return ExperimentSettings(
        provider=provider,
        count=_bounded_int(payload.get("count", 8), "生成数量", 1, 30),
        components=_bounded_int(payload.get("components", 8), "特征维数", 1, 64),
        clusters=_bounded_int(payload.get("clusters", 8), "聚类数", 1, 64),
        activation_quantile=_bounded_float(
            payload.get("activation_quantile", 0.8),
            "激活分位点",
            0.5,
            0.99,
        ),
        neighbor_fraction=_bounded_float(
            payload.get("neighbor_fraction", 0.35),
            "相似连接比例",
            0.1,
            0.9,
        ),
        temperature=_bounded_float(
            payload.get("temperature", 0.8), "Temperature", 0, 2
        ),
        api=api if provider == "openai" else None,
    )


def _run_job(job_id: str, seeds: list[SeedQuestion], settings: ExperimentSettings) -> None:
    output_dir = Path(JOBS.get(job_id).output_dir)  # type: ignore[union-attr]
    output_dir.mkdir(parents=True, exist_ok=True)
    JOBS.update(job_id, status="running", stage="starting", message="正在启动实验")

    def progress(stage: str, percent: int, detail: str) -> None:
        JOBS.append_log(job_id, stage, percent, detail)

    try:
        result = run_experiment(
            seeds,
            settings,
            output_dir=output_dir,
            progress=progress,
        )
        JOBS.update(
            job_id,
            status="complete",
            stage="done",
            progress=100,
            message="实验完成",
            result=result,
        )
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        JOBS.append_log(job_id, "failed", 0, error_text)
        JOBS.update(
            job_id,
            status="failed",
            stage="failed",
            message="实验失败",
            error=error_text,
        )
        (output_dir / "error.log").write_text(
            "".join(traceback.format_exception_only(type(exc), exc)),
            encoding="utf-8",
        )


class Handler(BaseHTTPRequestHandler):
    server_version = "StructSynthUI/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[ui] {self.address_string()} {format % args}")

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path in {"/", "/index.html"}:
            self._serve_file(STATIC_DIR / "index.html")
            return
        if path.startswith("/assets/"):
            name = path.removeprefix("/assets/")
            if "/" in name or name.startswith("."):
                _error(self, HTTPStatus.NOT_FOUND, "Not found.")
                return
            self._serve_file(STATIC_DIR / name)
            return
        if path == "/api/config":
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ui_version": "0.3.0",
                    "api_schema": 3,
                    "base_url": os.getenv("QS_LLM_BASE_URL", ""),
                    "chat_model": os.getenv("QS_CHAT_MODEL", ""),
                    "embedding_base_url": os.getenv("QS_EMBEDDING_BASE_URL", ""),
                    "embedding_model": os.getenv("QS_EMBEDDING_MODEL", ""),
                    "has_env_chat_key": bool(os.getenv("QS_LLM_API_KEY")),
                    "has_env_embedding_key": bool(
                        os.getenv("QS_EMBEDDING_API_KEY")
                    ),
                },
            )
            return
        if path == "/api/examples":
            example = Path.cwd() / "examples" / "demo_seeds.json"
            if example.exists():
                _json_response(
                    self,
                    HTTPStatus.OK,
                    json.loads(example.read_text(encoding="utf-8")),
                )
            else:
                _error(self, HTTPStatus.NOT_FOUND, "示例题文件不存在。")
            return
        if path.startswith("/api/runs/"):
            parts = path.strip("/").split("/")
            if len(parts) == 3:
                job = JOBS.get(parts[2])
                if not job:
                    _error(self, HTTPStatus.NOT_FOUND, "任务不存在。")
                    return
                _json_response(self, HTTPStatus.OK, job.snapshot())
                return
            if len(parts) == 5 and parts[3] == "files":
                self._serve_download(parts[2], parts[4])
                return
            if len(parts) == 4 and parts[3] == "archive":
                self._serve_archive(parts[2])
                return
        _error(self, HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path != "/api/runs":
            _error(self, HTTPStatus.NOT_FOUND, "Not found.")
            return
        try:
            payload = _read_json(self)
            seeds = _parse_seeds(payload.get("seeds"))
            settings = _settings_from_payload(payload)
            job = JOBS.create()
        except ValueError as exc:
            _error(self, HTTPStatus.BAD_REQUEST, str(exc))
            return
        thread = threading.Thread(
            target=_run_job,
            args=(job.id, seeds, settings),
            daemon=True,
            name=f"structsynth-{job.id}",
        )
        thread.start()
        _json_response(
            self,
            HTTPStatus.ACCEPTED,
            {"id": job.id, "status": "queued"},
        )

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            _error(self, HTTPStatus.NOT_FOUND, "Not found.")
            return
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _serve_download(self, job_id: str, name: str) -> None:
        job = JOBS.get(job_id)
        if not job or job.status != "complete" or name not in DOWNLOAD_FILES:
            _error(self, HTTPStatus.NOT_FOUND, "文件不存在。")
            return
        path = Path(job.output_dir) / name
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header(
            "Content-Disposition", f'attachment; filename="structsynth-{name}"'
        )
        self.end_headers()
        self.wfile.write(content)

    def _serve_archive(self, job_id: str) -> None:
        job = JOBS.get(job_id)
        if not job or job.status != "complete":
            _error(self, HTTPStatus.NOT_FOUND, "任务结果不存在。")
            return
        archive = Path(job.output_dir) / "structsynth-results.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in sorted(DOWNLOAD_FILES):
                path = Path(job.output_dir) / name
                if path.exists():
                    bundle.write(path, arcname=name)
        content = archive.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(content)))
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="structsynth-{job_id}.zip"',
        )
        self.end_headers()
        self.wfile.write(content)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"StructSynth UI: http://{host}:{port}")
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        server.server_close()
