# Vector Database Decision (für dieses Projekt)

## Entscheidung

Für dieses Repository wird **Chroma** als Ziel-Vector-DB empfohlen und im Backend bereits über eine Chroma-Bridge genutzt.

## Begründung für dieses Projekt

Dieses Repo ist laut README ein **Portfolio-Prototyp** und **kein Production-System**. Dafür passt Chroma am besten:

- Python-native und schnell integrierbar.
- Persistenz ohne eigene, komplexe Server-Infrastruktur.
- Sehr guter Fit für typische RAG-MVPs mit LangChain-/Python-Workflows.
- Deutlich "datenbankiger" als der aktuelle JSON-Index und damit ein sinnvoller nächster Schritt.

## Einordnung im Markt (praxisnah)

- Forschung/Experimente: meist **FAISS**.
- Python MVP/Prototyp: meist **Chroma**.
- Produktion Self-Hosted: oft **Qdrant** oder **Weaviate**.
- Produktion Managed Cloud: oft **Pinecone**.

## Empfehlung für die nächste Ausbaustufe

1. **Jetzt:** Chroma ist als persistente Vector-DB im Backend integriert.
2. **Später (Production):** auf Qdrant migrieren, wenn Multi-User, Skalierung und Betriebssicherheit zentral werden.

## Konkrete Leitplanken

- Chroma als Standard für lokale Entwicklung und Demo.
- Klare Abstraktion im Code (`vector_store` Interface) beibehalten, damit ein späterer Wechsel auf Qdrant mit geringem Refactor möglich bleibt.
- Relevante Metadaten (source, filename, timestamp, labels) von Anfang an als Filterfelder modellieren.
