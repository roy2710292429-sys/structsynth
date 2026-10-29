from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Protocol

import numpy as np


class LLM(Protocol):
    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"LLM did not return a JSON object: {text[:300]!r}")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from the LLM.")
    return value


class OpenAICompatibleLLM:
    """Dependency-free client with independent chat and embedding endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        chat_model: str,
        embedding_model: str,
        timeout: float = 120.0,
        *,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Chat API key is empty.")
        if not embedding_api_key:
            raise ValueError("Embedding API key is empty.")
        self.chat_base_url = base_url.rstrip("/")
        self.chat_api_key = api_key
        self.embedding_base_url = (embedding_base_url or base_url).rstrip("/")
        self.embedding_api_key = embedding_api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLM":
        missing = [
            name
            for name in ("QS_LLM_API_KEY", "QS_CHAT_MODEL", "QS_EMBEDDING_MODEL")
            if not os.getenv(name)
        ]
        if missing:
            raise ValueError(
                "Missing environment variables: " + ", ".join(missing)
            )
        return cls(
            base_url=os.getenv("QS_LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ["QS_LLM_API_KEY"],
            chat_model=os.environ["QS_CHAT_MODEL"],
            embedding_model=os.environ["QS_EMBEDDING_MODEL"],
            timeout=float(os.getenv("QS_LLM_TIMEOUT", "120")),
            embedding_base_url=os.getenv("QS_EMBEDDING_BASE_URL"),
            embedding_api_key=(
                os.getenv("QS_EMBEDDING_API_KEY") or os.environ["QS_LLM_API_KEY"]
            ),
        )

    def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        base_url: str,
        api_key: str,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP {exc.code}: {body[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach LLM API: {exc.reason}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("LLM API returned a non-object response.")
        return data

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            data = self._post(
                "chat/completions",
                payload,
                base_url=self.chat_base_url,
                api_key=self.chat_api_key,
            )
        except RuntimeError as exc:
            # Some compatible providers reject response_format. Retry only that case.
            if json_mode and "400" in str(exc):
                payload.pop("response_format", None)
                data = self._post(
                    "chat/completions",
                    payload,
                    base_url=self.chat_base_url,
                    api_key=self.chat_api_key,
                )
            else:
                raise
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat response: {data!r}") from exc
        if not isinstance(content, str):
            raise RuntimeError("Chat response content is not text.")
        return content

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._post(
            "embeddings",
            {"model": self.embedding_model, "input": texts},
            base_url=self.embedding_base_url,
            api_key=self.embedding_api_key,
        )
        try:
            rows = sorted(data["data"], key=lambda row: row["index"])
            return [list(map(float, row["embedding"])) for row in rows]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected embedding response: {data!r}") from exc


class MockLLM:
    """Offline deterministic adapter for smoke tests, not a quality benchmark."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        dimensions = 96
        vectors: list[list[float]] = []
        for text in texts:
            vector = np.zeros(dimensions, dtype=float)
            compact = re.sub(r"\s+", "", text)
            grams = [compact[i : i + 2] for i in range(max(1, len(compact) - 1))]
            for gram in grams:
                digest = hashlib.sha256(gram.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            norm = float(np.linalg.norm(vector))
            vectors.append((vector / norm if norm else vector).tolist())
        return vectors

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        if "TASK:SEGMENT" in user:
            return json.dumps(self._segment(user), ensure_ascii=False)
        if "TASK:INDUCE_REASONING_PROFILE" in user:
            return json.dumps(self._profile(user), ensure_ascii=False)
        if "TASK:INDUCE_TEMPLATE" in user:
            return json.dumps(self._induce(user), ensure_ascii=False)
        if "TASK:GENERATE" in user:
            return json.dumps(self._generate(user), ensure_ascii=False)
        if "TASK:JUDGE" in user:
            return json.dumps(self._judge(user), ensure_ascii=False)
        raise ValueError("MockLLM does not recognize this prompt.")

    @staticmethod
    def _tagged_json(user: str, tag: str) -> Any:
        match = re.search(
            rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
            user,
            flags=re.S,
        )
        if not match:
            raise ValueError(f"Missing <{tag}> block.")
        return json.loads(match.group(1))

    def _segment(self, user: str) -> dict[str, Any]:
        questions = self._tagged_json(user, "seed_questions")
        results = []
        for question in questions:
            text = question["text"].rstrip("。！？!?")
            chunks = self._split_clauses(text)
            if len(chunks) == 1:
                chunks = [chunk for chunk in re.split(r"(?=且|并|若|则|求|问)", text) if chunk]
            rows = []
            for index, chunk in enumerate(chunks[:8], start=1):
                if index == 1:
                    role, operation, abstraction = (
                        "初始条件",
                        "提取",
                        "给出题目可用的初始信息",
                    )
                elif any(word in chunk for word in ("求", "判断", "分析")):
                    role, operation, abstraction = (
                        "目标",
                        "推导",
                        "求出或判断目标性质",
                    )
                elif index == len(chunks):
                    role, operation, abstraction = (
                        "验证",
                        "验证",
                        "为结论提供理由或检查结果",
                    )
                else:
                    role, operation, abstraction = (
                        "中间操作",
                        "转换",
                        "连接已知条件与目标",
                    )
                rows.append(
                    {
                        "text": chunk,
                        "role": role,
                        "operation": operation,
                        "abstraction": abstraction,
                    }
                )
            results.append({"question_id": question["id"], "segments": rows})
        return {"questions": results}

    @staticmethod
    def _split_clauses(text: str) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        depth = 0
        opening = "([{（【"
        closing = ")]}）】"
        for character in text:
            if character in opening:
                depth += 1
            elif character in closing:
                depth = max(0, depth - 1)
            if character in "，,；;。！？!?" and depth == 0:
                chunk = "".join(current).strip(" ，,；;。！？!?")
                if chunk:
                    chunks.append(chunk)
                current = []
            else:
                current.append(character)
        chunk = "".join(current).strip(" ，,；;。！？!?")
        if chunk:
            chunks.append(chunk)
        return chunks

    def _profile(self, user: str) -> dict[str, Any]:
        seeds = self._tagged_json(user, "seed_questions")
        average_length = (
            sum(len(str(row.get("text", ""))) for row in seeds) / max(1, len(seeds))
        )
        depth = 3 if average_length < 80 else 4
        return {
            "depth_level": depth,
            "depth_label": "中等偏深",
            "abstraction_level": "具体到抽象",
            "operations": ["条件转译", "多步推导", "结果验证"],
            "dependency_chain": [
                "读取并抽象条件",
                "选择关系或方法",
                "执行关键推导",
                "验证目标是否满足",
            ],
            "bottleneck": "把条件转化为可用于推导的关系",
            "verification_burden": "需要说明结论成立的依据",
            "surface_variation": ["更换数值", "更换情境对象"],
            "depth_invariants": [
                "至少需要两段相互依赖的推理",
                "结论必须经过验证或说明依据",
            ],
        }

    def _induce(self, user: str) -> dict[str, Any]:
        candidates = self._tagged_json(user, "candidates")
        slots = []
        invariants = []
        for index, candidate in enumerate(candidates, start=1):
            exemplar = (candidate.get("exemplars") or ["语义内容"])[0]
            slots.append(
                {
                    "source_unit_ids": [candidate["id"]],
                    "name": f"语义槽位{index}",
                    "definition": f"表达与“{exemplar[:30]}”相近的结构角色",
                    "required": index <= max(3, len(candidates) - 2),
                    "order": index,
                }
            )
        if candidates:
            invariants.append("槽位数量与顺序保持稳定")
        return {
            "name": "自动归纳题模板",
            "description": "由候选语义单元归纳的离线演示模板",
            "slots": slots,
            "invariants": invariants,
        }

    def _generate(self, user: str) -> dict[str, Any]:
        seeds = self._tagged_json(user, "seed_questions")
        condition = self._tagged_json(user, "condition")
        count_match = re.search(r"生成\s*(\d+)\s*道", user)
        count = int(count_match.group(1)) if count_match else 3
        template = condition.get("template") or {}
        slots = template.get("slots") or []
        base = seeds[0]["text"] if seeds else "给定条件"
        rows = []
        for index in range(count):
            slot_texts = {}
            for slot_index, slot in enumerate(slots):
                source = (seeds[(index + slot_index) % len(seeds)]["text"] if seeds else base)
                slot_texts[str(slot["id"])] = source[:14]
            text = (
                f"【演示题 {index + 1}】在给定条件“{base[:22]}”的情形下，"
                f"请判断目标关系是否成立，并给出简要依据。"
            )
            rows.append(
                {
                    "text": text,
                    "slot_texts": slot_texts,
                    "type": "演示同构题",
                    "difficulty": ["easy", "medium", "hard"][index % 3],
                    "depth_level": 3,
                    "reasoning_outline": [
                        "抽取已知关系",
                        "执行一步推导",
                        "检查目标约束",
                    ],
                }
            )
        return {"questions": rows}

    def _judge(self, user: str) -> dict[str, Any]:
        template = self._tagged_json(user, "reference_template")
        questions = self._tagged_json(user, "generated_questions")
        required = [
            str(slot["id"]) for slot in template.get("slots", []) if slot.get("required", True)
        ]
        results = []
        for question in questions:
            present = {str(key) for key in question.get("slot_texts", {})}
            matched = [slot_id for slot_id in required if slot_id in present]
            missing = [slot_id for slot_id in required if slot_id not in present]
            coverage = len(matched) / len(required) if required else 0.0
            results.append(
                {
                    "question_id": question["id"],
                    "isomorphic": coverage >= 0.75,
                    "same_family": coverage >= 0.6,
                    "structural_similarity": coverage,
                    "cognitive_coverage": coverage,
                    "depth_score": int(question.get("depth_level") or 3),
                    "depth_alignment": 1.0 - abs(
                        int(question.get("depth_level") or 3)
                        - int(template.get("reasoning_profile", {}).get("depth_level", 3))
                    )
                    / 4.0,
                    "matched_slot_ids": matched,
                    "missing_slot_ids": missing,
                    "distorted_slot_ids": [],
                    "coverage": coverage,
                    "reason": "离线演示判定器按显式槽位存在性检查。",
                }
            )
        return {"results": results}
