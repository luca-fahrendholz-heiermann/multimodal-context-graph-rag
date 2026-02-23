import json

from backend import app


def test_store_upload_fallback_markdown_uses_normalized_artifact_for_images(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    normalized_dir = upload_dir / "artifacts" / "normalized"
    upload_dir.mkdir(parents=True)
    normalized_dir.mkdir(parents=True)

    stored_filename = "scan_20260101T000000Z.png"
    (upload_dir / stored_filename).write_bytes(b"\x89PNG\r\n\x1a\n")
    (normalized_dir / f"{stored_filename}.json").write_text(
        json.dumps(
            {
                "canonical_text": "Ein Diagramm mit Balken und Beschriftungen.",
                "chunks": [
                    {"text": "Bildinhalt: Quartalsumsatz Q1-Q4."},
                    {"text": "Legende zeigt Produktlinien A und B."},
                ],
            }
        ),
        encoding="utf-8",
    )

    captured: dict = {}

    def fake_store_markdown_artifact(filename: str, markdown: str, warnings=None):
        captured["filename"] = filename
        captured["markdown"] = markdown
        captured["warnings"] = warnings or []

    monkeypatch.setattr("backend.app.store_markdown_artifact", fake_store_markdown_artifact)
    monkeypatch.setattr("backend.app.ingestion.UPLOAD_DIR", upload_dir)
    monkeypatch.setattr("backend.app.ingestion.NORMALIZED_DIR", normalized_dir)

    recovered, warnings = app._store_upload_fallback_markdown(stored_filename, ".png")

    assert recovered is True
    assert warnings == []
    assert captured["filename"] == stored_filename
    assert "Source file reference" in captured["markdown"]
    assert "Quartalsumsatz" in captured["markdown"]
    assert any("fallback indexed normalized ingestion content" in item.lower() for item in captured["warnings"])


def test_store_upload_fallback_markdown_returns_error_when_normalized_missing(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    normalized_dir = upload_dir / "artifacts" / "normalized"
    upload_dir.mkdir(parents=True)
    normalized_dir.mkdir(parents=True)

    stored_filename = "scan_20260101T000000Z.png"
    (upload_dir / stored_filename).write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr("backend.app.ingestion.UPLOAD_DIR", upload_dir)
    monkeypatch.setattr("backend.app.ingestion.NORMALIZED_DIR", normalized_dir)

    recovered, warnings = app._store_upload_fallback_markdown(stored_filename, ".png")

    assert recovered is False
    assert any("requires normalized ingestion output" in warning.lower() for warning in warnings)
