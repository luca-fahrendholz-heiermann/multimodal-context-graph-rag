from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

from backend.llm_provider import describe_image_with_gemini, describe_image_with_openai

SYSTEM_PROMPT_DE = (
    "Du bist ein präziser Bildanalyst.\n"
    "1) Beschreibe das Bild objektiv und strukturiert.\n"
    "2) Analysiere Inhalte, Beziehungen, mögliche Intentionen und Auffälligkeiten.\n"
    "3) Trenne klar zwischen Beobachtung und Interpretation.\n"
    "4) Wenn du spekulierst, markiere es explizit als SPEKULATION.\n"
    "5) Nutze kurze Sätze und klare Bullet Points.\n"
)

USER_TASK_DE = (
    "Bitte beschreibe und analysiere dieses Bild.\n"
    "Gib erst eine kompakte Inhaltsbeschreibung.\n"
    "Gib dann eine detaillierte Analyse.\n"
    "Gib zuletzt eine Liste offener Fragen, die man mit Kontext klären müsste.\n"
)


@dataclass(frozen=True)
class ImagePayload:
    data_url: str
    mime: str


@dataclass(frozen=True)
class VisionAnalysisResult:
    combined_text: str | None
    description_text: str | None
    analysis_text: str | None
    open_questions: list[str]
    warnings: list[str]
    meta: dict


def load_image_as_data_url(image_path: str) -> ImagePayload:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/jpeg"

    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    return ImagePayload(data_url=data_url, mime=mime)


def _split_vision_sections(text: str) -> tuple[str | None, str | None, list[str]]:
    if not text.strip():
        return None, None, []

    lines = [line.strip() for line in text.splitlines()]
    if not lines:
        return None, None, []

    description_lines: list[str] = []
    analysis_lines: list[str] = []
    question_lines: list[str] = []
    current = "description"

    for line in lines:
        normalized = line.lower().strip(" :")
        if normalized.startswith("beschreibung") or normalized.startswith("inhaltsbeschreibung"):
            current = "description"
            continue
        if normalized.startswith("analyse"):
            current = "analysis"
            continue
        if normalized.startswith("offene fragen"):
            current = "questions"
            continue

        if current == "description":
            description_lines.append(line)
        elif current == "analysis":
            analysis_lines.append(line)
        else:
            cleaned = line.lstrip("-•* ").strip()
            if cleaned:
                question_lines.append(cleaned)

    description = "\n".join(item for item in description_lines if item).strip() or None
    analysis = "\n".join(item for item in analysis_lines if item).strip() or None
    return description, analysis, question_lines


def analyze_image_bytes_with_provider(*, filename: str, content_bytes: bytes, provider: str | None = None, api_key: str | None = None) -> VisionAnalysisResult:
    warnings: list[str] = []
    configured_provider = (provider or os.getenv("RAG_IMAGE_DESCRIPTION_PROVIDER") or "").strip().lower()
    prompt = f"{SYSTEM_PROMPT_DE}\n\n{USER_TASK_DE}"

    if configured_provider in {"none", "disabled"}:
        return VisionAnalysisResult(
            combined_text=None,
            description_text=None,
            analysis_text=None,
            open_questions=[],
            warnings=warnings,
            meta={"enabled": False},
        )

    if configured_provider == "":
        if (api_key or os.getenv("OPENAI_API_KEY") or "").strip():
            configured_provider = "openai"
        else:
            warnings.append("Image description skipped: no vision provider or API key configured.")
            return VisionAnalysisResult(
                combined_text=None,
                description_text=None,
                analysis_text=None,
                open_questions=[],
                warnings=warnings,
                meta={"enabled": False},
            )

    if configured_provider in {"openai", "chatgpt"}:
        resolved_api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        model = (os.getenv("RAG_IMAGE_DESCRIPTION_OPENAI_MODEL") or "gpt-4.1-mini").strip()
        if not resolved_api_key:
            warnings.append("Image description skipped: OPENAI_API_KEY not configured.")
            return VisionAnalysisResult(None, None, None, [], warnings, {"enabled": True, "provider": "openai", "model": model})
        response = describe_image_with_openai(api_key=resolved_api_key, model=model, image_bytes=content_bytes, prompt=prompt)
        provider_name = "openai"
    elif configured_provider == "gemini":
        resolved_api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        model = (os.getenv("RAG_IMAGE_DESCRIPTION_GEMINI_MODEL") or "gemini-1.5-flash").strip()
        if not resolved_api_key:
            warnings.append("Image description skipped: GEMINI_API_KEY not configured.")
            return VisionAnalysisResult(None, None, None, [], warnings, {"enabled": True, "provider": "gemini", "model": model})
        response = describe_image_with_gemini(api_key=resolved_api_key, model=model, image_bytes=content_bytes, prompt=prompt)
        provider_name = "gemini"
    else:
        warnings.append(f"Image description skipped: unsupported provider '{configured_provider}'.")
        return VisionAnalysisResult(None, None, None, [], warnings, {"enabled": True, "provider": configured_provider})

    warnings.extend(response.warnings)
    combined_text = (response.raw_response or "").strip() if response.status == "success" else None
    if not combined_text:
        return VisionAnalysisResult(None, None, None, [], warnings, {"enabled": True, "provider": provider_name})

    description, analysis, questions = _split_vision_sections(combined_text)
    meta = {
        "enabled": True,
        "provider": provider_name,
        "description_text": description,
        "analysis_text": analysis,
        "open_questions": questions,
    }
    return VisionAnalysisResult(combined_text, description, analysis, questions, warnings, meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "gemini"], default=None)
    parser.add_argument("--image", required=True, help="Pfad zur Bilddatei")
    args = parser.parse_args()

    image_bytes = Path(args.image).read_bytes()
    result = analyze_image_bytes_with_provider(filename=args.image, content_bytes=image_bytes, provider=args.provider)
    if result.combined_text:
        print(result.combined_text)
    else:
        print("Keine Analyse erzeugt.")

    if result.warnings:
        print("\nWarnungen:")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
