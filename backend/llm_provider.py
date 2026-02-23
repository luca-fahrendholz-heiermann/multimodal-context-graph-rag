from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class LlmJsonResult:
    status: str
    raw_response: str | None
    warnings: list[str]


def _collect_gemini_text(response_payload: dict[str, Any]) -> str | None:
    candidates = response_payload.get("candidates") or []
    if not candidates:
        return None

    parts = (candidates[0].get("content") or {}).get("parts") or []
    extracted: list[str] = []
    for part in parts:
        text = part.get("text") if isinstance(part, dict) else None
        if isinstance(text, str) and text.strip():
            extracted.append(text.strip())

    if extracted:
        return "\n".join(extracted)
    return None


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=20) as response:
        response_body = response.read().decode("utf-8")
    return json.loads(response_body)


def _http_error_warning(provider_name: str, exc: error.HTTPError) -> str:
    response_excerpt = ""
    try:
        response_body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        response_body = ""

    if response_body:
        try:
            parsed = json.loads(response_body)
            if isinstance(parsed, dict):
                error_payload = parsed.get("error")
                if isinstance(error_payload, dict):
                    message = error_payload.get("message")
                    if isinstance(message, str) and message.strip():
                        response_excerpt = message.strip()
        except json.JSONDecodeError:
            response_excerpt = response_body[:200]

    if response_excerpt:
        return f"{provider_name} request failed with HTTP {exc.code}: {response_excerpt}"

    return f"{provider_name} request failed with HTTP {exc.code}."


def _collect_openai_text(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []

    collected: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue

        direct_text = part.get("text")
        if isinstance(direct_text, str) and direct_text.strip():
            collected.append(direct_text.strip())
            continue

        value = part.get("value")
        if isinstance(value, str) and value.strip():
            collected.append(value.strip())

    return collected


def _extract_openai_text(response_payload: dict[str, Any]) -> str | None:
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = response_payload.get("output")
    if isinstance(output, list):
        extracted: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            extracted.extend(_collect_openai_text(item.get("content")))

        if extracted:
            return "\n".join(extracted)

    return None




def _detect_image_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"

def _classify_prompt(text: str, labels: list[str]) -> str:
    return (
        "Classify the user text into exactly one label from this allow-list: "
        f"{', '.join(labels)}. "
        "Return only JSON with this exact schema: "
        '{"label":"<one-allowed-label>","confidence":<float-between-0-and-1>}. '\
        f"User text: {text}"
    )


def classify_with_openai(api_key: str, model: str, text: str, labels: list[str]) -> LlmJsonResult:
    try:
        payload = {
            "model": model,
            "input": _classify_prompt(text, labels),
            "max_output_tokens": 200,
            "text": {"format": {"type": "json_object"}},
        }
        response_payload = _post_json(
            "https://api.openai.com/v1/responses",
            payload,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
    except error.HTTPError as exc:
        return LlmJsonResult(
            status="error",
            raw_response=None,
            warnings=[_http_error_warning("OpenAI", exc)],
        )
    except Exception:
        return LlmJsonResult(
            status="error",
            raw_response=None,
            warnings=["OpenAI request failed before receiving a response."],
        )

    extracted_text = _extract_openai_text(response_payload)
    if extracted_text:
        return LlmJsonResult(status="success", raw_response=extracted_text, warnings=[])

    return LlmJsonResult(
        status="error",
        raw_response=json.dumps(response_payload),
        warnings=["OpenAI response did not contain extractable text content."],
    )


def classify_with_gemini(api_key: str, model: str, text: str, labels: list[str]) -> LlmJsonResult:
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    try:
        payload = {
            "contents": [{"parts": [{"text": _classify_prompt(text, labels)}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": 200,
            },
        }
        response_payload = _post_json(
            endpoint,
            payload,
            {"Content-Type": "application/json"},
        )
    except error.HTTPError as exc:
        return LlmJsonResult(
            status="error",
            raw_response=None,
            warnings=[_http_error_warning("Gemini", exc)],
        )
    except Exception:
        return LlmJsonResult(
            status="error",
            raw_response=None,
            warnings=["Gemini request failed before receiving a response."],
        )

    candidates = response_payload.get("candidates") or []
    if candidates:
        parts = (candidates[0].get("content") or {}).get("parts") or []
        if parts and isinstance(parts[0].get("text"), str):
            return LlmJsonResult(
                status="success",
                raw_response=parts[0]["text"],
                warnings=[],
            )

    return LlmJsonResult(
        status="error",
        raw_response=json.dumps(response_payload),
        warnings=["Gemini response did not contain JSON text content."],
    )


def generate_text_with_openai(api_key: str, model: str, prompt: str, max_output_tokens: int = 400) -> LlmJsonResult:
    try:
        payload = {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }
        response_payload = _post_json(
            "https://api.openai.com/v1/responses",
            payload,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
    except error.HTTPError as exc:
        return LlmJsonResult(
            status="error",
            raw_response=None,
            warnings=[_http_error_warning("OpenAI", exc)],
        )
    except Exception:
        return LlmJsonResult(
            status="error",
            raw_response=None,
            warnings=["OpenAI request failed before receiving a response."],
        )

    extracted_text = _extract_openai_text(response_payload)
    if extracted_text:
        return LlmJsonResult(status="success", raw_response=extracted_text, warnings=[])

    return LlmJsonResult(
        status="error",
        raw_response=json.dumps(response_payload),
        warnings=["OpenAI response did not contain extractable text content."],
    )


def generate_text_with_gemini(api_key: str, model: str, prompt: str, max_output_tokens: int = 400) -> LlmJsonResult:
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
            },
        }
        response_payload = _post_json(
            endpoint,
            payload,
            {"Content-Type": "application/json"},
        )
    except error.HTTPError as exc:
        return LlmJsonResult(
            status="error",
            raw_response=None,
            warnings=[_http_error_warning("Gemini", exc)],
        )
    except Exception:
        return LlmJsonResult(
            status="error",
            raw_response=None,
            warnings=["Gemini request failed before receiving a response."],
        )

    extracted_text = _collect_gemini_text(response_payload)
    if extracted_text:
        return LlmJsonResult(status="success", raw_response=extracted_text, warnings=[])

    return LlmJsonResult(
        status="error",
        raw_response=json.dumps(response_payload),
        warnings=["Gemini response did not contain text content."],
    )


def describe_image_with_openai(api_key: str, model: str, image_bytes: bytes, prompt: str, mime_type: str | None = None) -> LlmJsonResult:
    import base64

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    image_mime_type = mime_type or _detect_image_mime_type(image_bytes)
    try:
        payload = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:{image_mime_type};base64,{image_b64}"},
                    ],
                }
            ],
            "max_output_tokens": 350,
        }
        response_payload = _post_json(
            "https://api.openai.com/v1/responses",
            payload,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
    except error.HTTPError as exc:
        return LlmJsonResult(status="error", raw_response=None, warnings=[_http_error_warning("OpenAI", exc)])
    except Exception:
        return LlmJsonResult(
            status="error",
            raw_response=None,
            warnings=["OpenAI image description request failed before receiving a response."],
        )

    extracted_text = _extract_openai_text(response_payload)
    if extracted_text:
        return LlmJsonResult(status="success", raw_response=extracted_text, warnings=[])

    return LlmJsonResult(
        status="error",
        raw_response=json.dumps(response_payload),
        warnings=["OpenAI image description response did not contain extractable text."],
    )


def describe_image_with_gemini(api_key: str, model: str, image_bytes: bytes, prompt: str, mime_type: str | None = None) -> LlmJsonResult:
    import base64

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    image_mime_type = mime_type or _detect_image_mime_type(image_bytes)
    try:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": image_mime_type, "data": image_b64}},
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 350,
            },
        }
        response_payload = _post_json(endpoint, payload, {"Content-Type": "application/json"})
    except error.HTTPError as exc:
        return LlmJsonResult(status="error", raw_response=None, warnings=[_http_error_warning("Gemini", exc)])
    except Exception:
        return LlmJsonResult(
            status="error",
            raw_response=None,
            warnings=["Gemini image description request failed before receiving a response."],
        )

    extracted_text = _collect_gemini_text(response_payload)
    if extracted_text:
        return LlmJsonResult(status="success", raw_response=extracted_text, warnings=[])

    return LlmJsonResult(
        status="error",
        raw_response=json.dumps(response_payload),
        warnings=["Gemini image description response did not contain text content."],
    )
