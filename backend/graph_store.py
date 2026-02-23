from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from uuid import uuid4

from backend import ingestion

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency fallback
    nx = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _graph_store_path() -> Path:
    return ingestion.UPLOAD_DIR / "graph_store.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _default_store() -> dict:
    return {
        "updated_at": None,
        "active_graph_id": None,
        "graphs": [],
    }


def _new_version(name: str = "Initial Draft", base_version_id: str | None = None) -> dict:
    now = _utc_now()
    version_id = f"v_{uuid4().hex[:10]}"
    return {
        "version_id": version_id,
        "name": name,
        "status": "draft",
        "base_version_id": base_version_id,
        "created_at": now,
        "updated_at": now,
        "nodes": [],
        "edges": [],
        "ui_meta": {
            "layout_positions": {},
            "layout_algo": "spring_layout",
            "layout_seed": 42,
            "layout_updated_at": now,
        },
    }


def _new_graph(name: str) -> dict:
    now = _utc_now()
    graph_id = f"g_{uuid4().hex[:10]}"
    first_version = _new_version(name="Initial Draft")
    return {
        "graph_id": graph_id,
        "name": name,
        "created_at": now,
        "updated_at": now,
        "active_version_id": first_version["version_id"],
        "versions": [first_version],
    }


def load_store() -> dict:
    path = _graph_store_path()
    if not path.exists():
        store = _default_store()
        seed_graph = _new_graph("Default Graph")
        store["graphs"].append(seed_graph)
        store["active_graph_id"] = seed_graph["graph_id"]
        store["updated_at"] = _utc_now()
        _atomic_write_json(path, store)
        return store
    return json.loads(path.read_text(encoding="utf-8"))


def save_store(store: dict) -> dict:
    store["updated_at"] = _utc_now()
    _atomic_write_json(_graph_store_path(), store)
    return store


def list_graphs() -> dict:
    store = load_store()
    graphs = []
    for graph in store.get("graphs", []):
        graphs.append(
            {
                "graph_id": graph["graph_id"],
                "name": graph.get("name") or graph["graph_id"],
                "created_at": graph.get("created_at"),
                "updated_at": graph.get("updated_at"),
                "active_version_id": graph.get("active_version_id"),
                "version_count": len(graph.get("versions", [])),
            }
        )
    return {
        "active_graph_id": store.get("active_graph_id"),
        "graphs": graphs,
    }


def create_graph(name: str | None = None) -> dict:
    store = load_store()
    graph_name = (name or "New Graph").strip() or "New Graph"
    graph = _new_graph(graph_name)
    store.setdefault("graphs", []).append(graph)
    store["active_graph_id"] = graph["graph_id"]
    save_store(store)
    return graph


def _find_graph(store: dict, graph_id: str) -> dict | None:
    for graph in store.get("graphs", []):
        if graph.get("graph_id") == graph_id:
            return graph
    return None


def _find_version(graph: dict, version_id: str) -> dict | None:
    for version in graph.get("versions", []):
        if version.get("version_id") == version_id:
            return version
    return None


def list_versions(graph_id: str) -> dict:
    store = load_store()
    graph = _find_graph(store, graph_id)
    if graph is None:
        raise KeyError("graph_not_found")
    versions = [
        {
            "version_id": version["version_id"],
            "name": version.get("name") or version["version_id"],
            "status": version.get("status") or "draft",
            "base_version_id": version.get("base_version_id"),
            "created_at": version.get("created_at"),
            "updated_at": version.get("updated_at"),
            "edge_count": len(version.get("edges", [])),
            "node_count": len(version.get("nodes", [])),
        }
        for version in graph.get("versions", [])
    ]
    return {
        "graph_id": graph_id,
        "active_version_id": graph.get("active_version_id"),
        "versions": versions,
    }


def create_draft(graph_id: str, from_version_id: str | None = None) -> dict:
    store = load_store()
    graph = _find_graph(store, graph_id)
    if graph is None:
        raise KeyError("graph_not_found")

    base_version = _find_version(graph, from_version_id or graph.get("active_version_id"))
    if base_version is None:
        raise KeyError("version_not_found")

    draft = copy.deepcopy(base_version)
    draft["version_id"] = f"v_{uuid4().hex[:10]}"
    draft["name"] = f"Draft from {base_version['version_id']}"
    draft["status"] = "draft"
    draft["base_version_id"] = base_version["version_id"]
    draft["created_at"] = _utc_now()
    draft["updated_at"] = draft["created_at"]

    graph.setdefault("versions", []).append(draft)
    graph["active_version_id"] = draft["version_id"]
    graph["updated_at"] = _utc_now()
    save_store(store)
    return draft


def commit_version(graph_id: str, version_id: str) -> dict:
    store = load_store()
    graph = _find_graph(store, graph_id)
    if graph is None:
        raise KeyError("graph_not_found")
    version = _find_version(graph, version_id)
    if version is None:
        raise KeyError("version_not_found")

    version["status"] = "committed"
    version["updated_at"] = _utc_now()
    graph["active_version_id"] = version_id
    graph["updated_at"] = _utc_now()
    save_store(store)
    return version


def rollback_active_version(graph_id: str, version_id: str) -> dict:
    store = load_store()
    graph = _find_graph(store, graph_id)
    if graph is None:
        raise KeyError("graph_not_found")
    version = _find_version(graph, version_id)
    if version is None:
        raise KeyError("version_not_found")

    graph["active_version_id"] = version_id
    graph["updated_at"] = _utc_now()
    save_store(store)
    return version


def _rag_documents() -> list[dict]:
    path = ingestion.UPLOAD_DIR / "rag_index.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    documents = payload.get("documents") if isinstance(payload, dict) else {}
    if not isinstance(documents, dict):
        return []

    result: list[dict] = []
    for stored_filename, doc in documents.items():
        metadata = doc.get("metadata") if isinstance(doc, dict) else {}
        label = str((metadata or {}).get("source_filename") or stored_filename)
        result.append({"doc_id": stored_filename, "label": label})
    result.sort(key=lambda item: item["label"].lower())
    return result


def _compute_layout(nodes: list[dict], edges: list[dict], ui_meta: dict | None) -> tuple[dict[str, list[float]], str, int]:
    seed = int((ui_meta or {}).get("layout_seed") or 42)
    previous = (ui_meta or {}).get("layout_positions") or {}
    previous_pos = {
        key: (float(value[0]), float(value[1]))
        for key, value in previous.items()
        if isinstance(value, (list, tuple)) and len(value) == 2
    }

    if not nodes:
        return {}, "spring_layout", seed

    if nx is None:
        spacing = 180
        return {
            node["node_id"]: [float((idx % 5) * spacing), float((idx // 5) * spacing)]
            for idx, node in enumerate(nodes)
        }, "grid_fallback", seed

    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(node["node_id"])
    for edge in edges:
        graph.add_edge(edge["source_node_id"], edge["target_node_id"])

    algo = "spring_layout"
    positions = None

    try:
        if len(nodes) <= 8:
            positions = nx.kamada_kawai_layout(graph)
            algo = "kamada_kawai_layout"
        elif len(nodes) <= 150:
            positions = nx.spring_layout(
                graph,
                seed=seed,
                pos=previous_pos or None,
                iterations=120,
                k=max(0.5, 2.4 / max(1, len(nodes) ** 0.5)),
            )
            algo = "spring_layout"
        else:
            positions = nx.spring_layout(graph, seed=seed, pos=previous_pos or None, iterations=80, k=0.8)
            algo = "spring_layout"
    except Exception:
        positions = nx.spring_layout(graph, seed=seed, pos=previous_pos or None)
        algo = "spring_layout"

    if positions is None:
        positions = nx.spring_layout(graph, seed=seed, pos=previous_pos or None)

    # optional Graphviz for better overlap handling when available
    try:
        from networkx.drawing.nx_agraph import graphviz_layout

        if len(nodes) <= 80:
            positions = graphviz_layout(graph, prog="sfdp")
            algo = "graphviz_layout"
    except Exception:
        pass

    layout_positions = {
        node_id: [float(point[0]), float(point[1])]
        for node_id, point in positions.items()
    }
    return layout_positions, algo, seed


def get_graph_view(graph_id: str, version_id: str) -> dict:
    store = load_store()
    graph = _find_graph(store, graph_id)
    if graph is None:
        raise KeyError("graph_not_found")
    version = _find_version(graph, version_id)
    if version is None:
        raise KeyError("version_not_found")

    nodes = list(version.get("nodes") or [])
    edges = list(version.get("edges") or [])
    ui_meta = dict(version.get("ui_meta") or {})
    layout_positions, algo, seed = _compute_layout(nodes, edges, ui_meta)

    ui_meta.update(
        {
            "layout_positions": layout_positions,
            "layout_algo": algo,
            "layout_seed": seed,
            "layout_updated_at": _utc_now(),
        }
    )
    version["ui_meta"] = ui_meta
    version["updated_at"] = _utc_now()
    save_store(store)

    return {
        "graph_id": graph_id,
        "version_id": version_id,
        "layout_algo": algo,
        "layout_seed": seed,
        "nodes": nodes,
        "edges": edges,
        "layout_positions": layout_positions,
        "document_options": _rag_documents(),
    }




def resolve_graph_document_ids(graph_id: str, version_id: str) -> list[str]:
    store = load_store()
    graph = _find_graph(store, graph_id)
    if graph is None:
        raise KeyError("graph_not_found")
    version = _find_version(graph, version_id)
    if version is None:
        raise KeyError("version_not_found")

    doc_ids: list[str] = []
    for node in version.get("nodes", []):
        doc_id = str(node.get("doc_id") or "").strip()
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def add_edge(
    graph_id: str,
    version_id: str,
    *,
    from_doc_id: str,
    to_doc_id: str,
    relation_type: str,
    note: str | None,
) -> dict:
    store = load_store()
    graph = _find_graph(store, graph_id)
    if graph is None:
        raise KeyError("graph_not_found")
    version = _find_version(graph, version_id)
    if version is None:
        raise KeyError("version_not_found")

    from_doc = from_doc_id.strip()
    to_doc = to_doc_id.strip()
    rel_type = relation_type.strip() or "references"

    if not from_doc or not to_doc:
        raise ValueError("doc_ids_required")

    nodes = version.setdefault("nodes", [])
    node_index = {node["doc_id"]: node for node in nodes}

    if from_doc not in node_index:
        node = {
            "node_id": f"n_{uuid4().hex[:10]}",
            "doc_id": from_doc,
            "label": from_doc,
        }
        nodes.append(node)
        node_index[from_doc] = node

    if to_doc not in node_index:
        node = {
            "node_id": f"n_{uuid4().hex[:10]}",
            "doc_id": to_doc,
            "label": to_doc,
        }
        nodes.append(node)
        node_index[to_doc] = node

    edge = {
        "edge_id": f"e_{uuid4().hex[:10]}",
        "source_node_id": node_index[from_doc]["node_id"],
        "target_node_id": node_index[to_doc]["node_id"],
        "type": rel_type,
        "note": (note or "").strip() or None,
        "created_at": _utc_now(),
    }

    version.setdefault("edges", []).append(edge)
    version["status"] = "draft"
    version["updated_at"] = _utc_now()
    graph["active_version_id"] = version_id
    graph["updated_at"] = _utc_now()
    save_store(store)

    return get_graph_view(graph_id, version_id)
