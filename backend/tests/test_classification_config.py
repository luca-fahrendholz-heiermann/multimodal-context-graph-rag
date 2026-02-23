import json

from backend.classification_config import load_label_whitelist


def test_classification_labels_loader_uses_configured_file(tmp_path, monkeypatch):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(["sales", "support", "support", " "]), encoding="utf-8")
    monkeypatch.setenv("CLASSIFICATION_LABELS_PATH", str(labels_path))

    whitelist = load_label_whitelist()

    assert whitelist.labels == ["sales", "support"]
    assert whitelist.source == str(labels_path)
