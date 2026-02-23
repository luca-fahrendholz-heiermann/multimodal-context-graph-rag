# Architecture

## Components and interfaces

- **Frontend UI** (`frontend/`): Intake, Chat, Evidence Viewer, Graph interactions.
- **API Layer** (`backend/app.py`): REST endpoints for ingestion, query, classification, graph workflows.
- **Ingestion Layer** (`backend/ingestion.py`): upload handling, email composition/inbox ingestion, format parsing.
- **Processing Layer** (`backend/chunking.py`, `backend/docling_integration.py`, `backend/classification.py`): conversion, chunking, embeddings, optional classification.
- **Retrieval Layer** (`backend/vector_store.py`, `backend/chroma_store.py`): vector search with storage fallback path.
- **Graph Layer** (`backend/graph_store.py`): document relation graph and versions.
- **LLM Layer** (`backend/llm_provider.py`, `backend/rag.py`): external providers (OpenAI/Gemini) plus local fallback in demo mode.

## Data flow

1. **Ingestion**: user uploads files or composes emails in UI.
2. **Processing**: backend validates input, optionally converts document formats, chunks text, computes embeddings, stores metadata.
3. **Indexing**: embeddings + chunk metadata are persisted for retrieval; graph relations can be added.
4. **Query**: query endpoint embeds query, searches store, assembles evidence context.
5. **Answer**: backend uses provider response when available/valid, otherwise local extractive fallback.
6. **UI rendering**: answer + evidence + warnings are shown in chat/evidence views.

## Mermaid diagram

```mermaid
flowchart TD
  subgraph Client
    UI[Frontend UI]
  end

  subgraph Backend
    API[FastAPI API]
    ING[Ingestion]
    PROC[Processing]
    RET[Retrieval]
    GRAPH[Graph Layer]
    RAG[RAG Orchestrator]
  end

  subgraph Storage
    VDB[(Vector Store / Chroma or JSON fallback)]
    GDB[(Graph data)]
  end

  subgraph Model
    REAL[LLM Provider: OpenAI/Gemini]
    STUB[Stub/Local Fallback]
  end

  UI --> API
  API --> ING --> PROC --> VDB
  API --> RET --> VDB
  API --> GRAPH --> GDB
  API --> RAG
  RAG --> REAL
  RAG --> STUB
  RAG --> RET
  RAG --> UI
```


## Mermaid diagram: Project Relation Graph + Scoped Retrieval

```mermaid
flowchart TD
  User[User im Chat]
  UI[Frontend: Chat + Graph Builder]
  API[Backend API]

  subgraph Indexing[Indexing & Persistenz]
    ING[Ingestion: Upload / E-Mail]
    CH[Chunking + Embeddings]
    VDB[(Vector DB: Embeddings + Chunks)]
    DDB[(Document DB: Metadata + Sources)]
    GDB[(Graph DB: Project Relation Graphs)]
    GB[Graph Builder aus selektierten Dokumenten]
  end

  subgraph RetrievalModes[Retrieval Modi]
    ALL[Global Retrieval über alle Dokumente]
    SCOPE[Graph-Scoped Retrieval über ausgewählten Projektgraphen]
    NEIGH[Graph Traversal / k-hop Nachbarschaft]
    FILTER[Dokument-Filter aus Graph-Knotenmenge]
    EVID[Evidence Ranking + Zitate]
  end

  subgraph Answer[Antworterzeugung]
    RAG[RAG Orchestrator]
    LLM[LLM Provider oder Local Fallback]
  end

  User --> UI --> API

  API --> ING --> CH
  CH --> VDB
  CH --> DDB

  API --> GB --> GDB
  GB --> DDB

  API --> ALL
  ALL --> VDB
  ALL --> DDB

  API --> SCOPE --> GDB
  SCOPE --> NEIGH --> FILTER
  FILTER --> VDB
  FILTER --> DDB

  ALL --> EVID
  FILTER --> EVID
  EVID --> RAG --> LLM --> UI
```
